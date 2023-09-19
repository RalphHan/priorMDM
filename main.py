import random
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
from utils.sampling_utils import single_take_arb_len
from visualize import Joints2SMPL
import requests
import json
import aiohttp
import asyncio
from collections import defaultdict
from ordered_set import OrderedSet


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
    sys.argv = [old_argv[0]] + ["--model_path", "./save/my_humanml_trans_enc_512/model000200000.pt"]
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


def prompt2motion(prompt, args, model, diffusion, data, priors=None, do_refine=False, want_number: int = 1):
    if priors is not None:
        assert len(priors) == want_number
        if not do_refine:
            lengths = [prior.shape[0] for prior in priors]
            max_frames = max(lengths)
        else:
            lengths = [min(args.n_frames, prior.shape[0]) for prior in priors]
            max_frames = max(lengths)
        st_frames = [random.randint(0, prior.shape[0] - l) for l, prior in zip(lengths, priors)]

        priors = [(prior[st:st + l] - data.dataset.mean) / data.dataset.std for st, l, prior in
                  zip(st_frames, lengths, priors)]
        priors = [np.concatenate((prior, np.zeros((max_frames - prior.shape[0], prior.shape[1]), dtype=prior.dtype)),
                                 axis=0).transpose(1, 0)[None, :, None] for prior in priors]
        priors = np.concatenate(priors, axis=0)
        priors = torch.tensor(priors, dtype=torch.float32).to(dist_util.dev())
    elif not do_refine:
        lengths = [args.n_frames] * want_number
        max_frames = args.n_frames
    else:
        raise
    model_kwargs = {'y': {
        'mask': torch.ones((want_number, 1, 1, max_frames)),  # 196 is humanml max frames number
        'lengths': torch.tensor(lengths),
        'text': [prompt] * want_number,
        'tokens': [''],
        'scale': torch.ones(want_number) * (args.guidance_param if not do_refine else args.guidance_param * 0.5)
    }}
    model_kwargs['y'] = {key: val.to(dist_util.dev()) if torch.is_tensor(val) else val for key, val in
                         model_kwargs['y'].items()}
    samples = single_take_arb_len(args, diffusion, model, model_kwargs, max_frames, prior=priors,
                                  do_refine=do_refine)
    n_joints = 22
    samples = data.dataset.t2m_dataset.inv_transform(samples.cpu().permute(0, 2, 3, 1)).float()
    samples = recover_from_ric(samples, n_joints)
    samples = samples.view(-1, *samples.shape[2:]).cpu().numpy()

    vec1 = np.cross(samples[:, :, 14] - samples[:, :, 12], samples[:, :, 13] - samples[:, :, 12])
    vec2 = samples[:, :, 15] - samples[:, :, 12]
    cosine = (vec1 * vec2).sum(-1) / (np.linalg.norm(vec1, axis=-1) * np.linalg.norm(vec2, axis=-1) + 1e-7)
    for i in range(want_number):
        if (samples[i, :lengths[i], 9, 1] > samples[i, :lengths[i], 0, 1]).sum() / lengths[i] > 0.85 \
                and (cosine[i, :lengths[i]] > -0.2).sum() / lengths[i] < 0.5:
            samples[i, :, :, 0] = -samples[i, :, :, 0]
    samples = [sample[:l] for sample, l in zip(samples, lengths)]
    return samples


def translation(prompt):
    try:
        prompt = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system",
                       "content": "translate to english without any explanation. If it's already in english, just repeat it. "
                                  "If get a <motion> without a subject, transfer it to: 'A person is <motion>'"},
                      {"role": "user", "content": prompt}],
            timeout=10,
        )["choices"][0]["message"]["content"]
    except:
        pass
    return prompt


async def fetch(session, **kwargs):
    async with session.get(**kwargs) as response:
        data = await response.json()
    return OrderedSet([x["motion_id"] for x in data])


async def search(prompt, want_number=1, get_h3d=True):
    async with aiohttp.ClientSession() as session:
        t2t_request = fetch(session, url=os.getenv("T2T_SERVER") + "/result/",
                            params={"query": prompt, "fs_weight": 0.1, "max_num": want_number * 4 * 4})
        t2m_request = fetch(session, url=os.getenv("T2M_SERVER") + "/result/",
                            params={"query": prompt, "max_num": want_number * 4})
        weights = [1.0, 0.5]
        ranks = await asyncio.gather(*[t2t_request, t2m_request])
    min_length = min([len(rank) for rank in ranks])
    for i in range(len(ranks)):
        ranks[i] = ranks[i][:min_length]
    total_rank = defaultdict(float)
    min_rank = defaultdict(lambda :min_length)
    total_id = set()
    for rank in ranks:
        total_id |= rank
    for rank, weight in zip(ranks, weights):
        rank = {x: i for i, x in enumerate(rank)}
        for x in total_id:
            total_rank[x] += rank.get(x, min_length) * weight/sum(weights)
            min_rank[x] = min(min_rank[x], rank.get(x, min_length))
    final_fank={}
    for x in total_id:
        final_fank[x] = (total_rank[x]*2 + min_rank[x])/3
    final_fank = sorted(final_fank.items(), key=lambda x: x[1])
    motion_ids = [x[0] for x in final_fank]
    assert motion_ids
    want_ids = []
    while len(want_ids) < want_number:
        want_ids.extend(motion_ids)
    want_ids = want_ids[:want_number]
    motions = []
    for want_id in want_ids:
        if get_h3d:
            motions.append(np.load(f"database/{want_id}.npy"))
        else:
            with open(f"motion_database/{want_id}.json") as f:
                motions.append(json.load(f))
    return motions


@app.get("/position/")
async def position(prompt: str, do_translation: bool = True, do_search: bool = True, do_refine: bool = True,
                   want_number: int = 1):
    assert 1 <= want_number <= 4
    if do_translation:
        prompt = translation(prompt)
    priors = (await search(prompt, want_number)) if do_search else None
    all_joints = prompt2motion(prompt, server_data["args"], server_data["model"], server_data["diffusion"],
                               server_data["data"], priors=priors, do_refine=do_refine,
                               want_number=want_number)
    return [{"positions": binascii.b2a_base64(joints.flatten().astype(np.float32).tobytes()).decode("utf-8"),
             "dtype": "float32",
             "fps": server_data["args"].fps,
             "mode": "xyz",
             "n_frames": joints.shape[0],
             "n_joints": 22} for joints in all_joints]


@app.get("/angle/")
async def angle(prompt: str, do_translation: bool = True, do_search: bool = True, do_refine: bool = True,
                want_number: int = 1):
    assert 1 <= want_number <= 4
    if do_translation:
        prompt = translation(prompt)
    priors = (await search(prompt, want_number, get_h3d=do_refine)) if do_search else None
    if do_search and not do_refine:
        return {"clips": priors}
    all_joints = prompt2motion(prompt, server_data["args"], server_data["model"], server_data["diffusion"],
                               server_data["data"], priors=priors, do_refine=do_refine,
                               want_number=want_number)
    all_rotations, all_root_pos = server_data["j2s"](all_joints, step_size=2e-2, num_iters=25, optimizer="lbfgs")
    return {"clips": [{"root_positions": binascii.b2a_base64(
        root_pos.flatten().astype(np.float32).tobytes()).decode("utf-8"),
                       "rotations": binascii.b2a_base64(rotations.flatten().astype(np.float32).tobytes()).decode(
                           "utf-8"),
                       "dtype": "float32",
                       "fps": server_data["args"].fps,
                       "mode": "axis_angle",
                       "n_frames": rotations.shape[0],
                       "n_joints": 24} for rotations, root_pos in zip(all_rotations, all_root_pos)]}
