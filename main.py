import dotenv
dotenv.load_dotenv()
import openai, os
openai.api_key = os.getenv("OPENAI_API_KEY")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import sys
import binascii
from model.DoubleTake_MDM import doubleTake_MDM
from utils.fixseed import fixseed
import numpy as np
import torch
from utils.parser_util import generate_args
from utils.model_util import load_model
from utils import dist_util
from data_loaders.get_data import get_dataset_loader
from data_loaders.humanml.scripts.motion_process import recover_from_ric
from utils.sampling_utils import unfold_sample_arb_len, single_take_arb_len
from visualize import Joints2SMPL



def load_dataset(args, n_frames):
    if args.dataset == 'babel':
        args.num_frames = (args.min_seq_len, args.max_seq_len)
    else:
        args.num_frames = n_frames
    data = get_dataset_loader(name=args.dataset,
                              batch_size=args.batch_size,
                              num_frames=args.num_frames,
                              split='val',
                              load_mode='text_only',
                              short_db=args.short_db,
                              cropping_sampler=args.cropping_sampler)
    data.fixed_length = n_frames
    return data


def calc_frame_colors(handshake_size, blend_size, step_sizes, lengths):
    for ii, step_size in enumerate(step_sizes):
        if ii == 0:
            frame_colors = ['orange'] * (step_size - handshake_size - blend_size) + \
                           ['blue'] * blend_size + \
                           ['purple'] * (handshake_size // 2)
            continue
        if ii == len(step_sizes) - 1:
            frame_colors += ['purple'] * (handshake_size // 2) + \
                            ['blue'] * blend_size + \
                            ['orange'] * (lengths[ii] - handshake_size - blend_size)
            continue
        frame_colors += ['purple'] * (handshake_size // 2) + ['blue'] * blend_size + \
                        ['orange'] * (lengths[ii] - 2 * handshake_size - 2 * blend_size) + \
                        ['blue'] * blend_size + \
                        ['purple'] * (handshake_size // 2)
    return frame_colors

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"],
                   allow_headers=["*"])

server_data = {}

@app.on_event('startup')
def init_data():
    old_argv = sys.argv
    sys.argv = [old_argv[0]] + ["--model_path", "./save/my_humanml_trans_enc_512/model000200000.pt",
                     "--handshake_size", "20",
                     "--blend_len", "10"]
    args = generate_args()
    sys.argv = old_argv
    fixseed(args.seed)
    args.fps = 20
    args.n_frames = 100
    dist_util.setup_dist(args.device)
    assert (args.double_take)
    args.num_samples = 1
    args.batch_size = 1  # Sampling a single batch from the testset, with exactly args.num_samples
    data = load_dataset(args, args.n_frames)
    model, diffusion = load_model(args, data, dist_util.dev(), ModelClass=doubleTake_MDM)
    server_data["args"] = args
    server_data["model"] = model
    server_data["diffusion"] = diffusion
    server_data["data"] = data
    server_data["j2s"] = Joints2SMPL(device="cuda")
    return server_data

def prompt2motion(prompt, args, model, diffusion, data):
    model_kwargs = {'y': {
        'mask': torch.ones((1, 1, 1, args.n_frames)), #196 is humanml max frames number
        'lengths': torch.tensor([args.n_frames]),
        'text': [prompt],
        'tokens': [''],
        'scale': torch.ones(1)*2.5
    }}
    if args.guidance_param != 1:
        model_kwargs['y']['scale'] = torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param
    model_kwargs['y'] = {key: val.to(dist_util.dev()) if torch.is_tensor(val) else val for key, val in model_kwargs['y'].items()}
    sample = single_take_arb_len(args, diffusion, model, model_kwargs, model_kwargs['y']['lengths'][0])
    sample = unfold_sample_arb_len(sample, args.handshake_size, [args.n_frames], args.n_frames, model_kwargs)
    n_joints = 22
    sample = data.dataset.t2m_dataset.inv_transform(sample.cpu().permute(0, 2, 3, 1)).float()
    sample = recover_from_ric(sample, n_joints)
    sample = sample.view(-1, *sample.shape[2:])[0].cpu().numpy()
    return sample

@app.get("/mld_pos/")
async def mld_pos(prompt: str):
    try:
        prompt = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system",
                       "content": "translate to english without any explanation. If it's already in english, just repeat it."},
                      {"role": "user", "content": prompt}],
        )["choices"][0]["message"]["content"]
    except:
        pass
    joints = prompt2motion(prompt, server_data["args"], server_data["model"], server_data["diffusion"], server_data["data"])
    return {"positions": binascii.b2a_base64(joints.flatten().astype(np.float32).tobytes()).decode("utf-8"),
            "dtype": "float32",
            "fps": server_data["args"].fps,
            "mode": "xyz",
            "n_frames": joints.shape[0],
            "n_joints": 22}


@app.get("/mld_angle/")
async def mld_angle(prompt: str, do_translation: bool = True):
    fps = 20
    if do_translation:
        try:
            prompt = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "system",
                           "content": "translate to english without any explanation. If it's already in english, just repeat it."},
                          {"role": "user", "content": prompt}],
                timeout=10,
            )["choices"][0]["message"]["content"]
        except:
            pass
    joints = prompt2motion(prompt, server_data["args"], server_data["model"], server_data["diffusion"], server_data["data"])
    if ((joints[:, 1, 0] > joints[:, 2, 0]) & (joints[:, 13, 0] > joints[:, 14, 0]) & (
            joints[:, 9, 1] > joints[:, 0, 1])).sum() / joints.shape[0] > 0.85:
        rotations, root_pos = server_data["j2s"](joints, step_size=1e-2, num_iters=150, optimizer="adam")
    else:
        rotations, root_pos = server_data["j2s"](joints, step_size=2e-2, num_iters=25, optimizer="lbfgs")
    return {"root_positions": binascii.b2a_base64(
        root_pos.flatten().astype(np.float32).tobytes()).decode("utf-8"),
            "rotations": binascii.b2a_base64(rotations.flatten().astype(np.float32).tobytes()).decode("utf-8"),
            "dtype": "float32",
            "fps": server_data["args"].fps,
            "mode": "axis_angle",
            "n_frames": joints.shape[0],
            "n_joints": 24}
