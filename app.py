import sys

import gradio as gr
import numpy as np
import torch
from data_loaders.tensors import collate
from model.comMDM import ComMDM
from utils.fixseed import fixseed
import os
from utils.parser_util import generate_multi_args
from utils.model_util import create_model_and_diffusion, load_model
from utils import dist_util
from data_loaders.humanml.scripts.motion_process import recover_from_ric2
import data_loaders.humanml.utils.paramUtil as paramUtil
from data_loaders.humanml.utils.plot_script import plot_3d_motion
import shutil
import uuid

import utils.rotation_conversions as geometry
from data_loaders.get_data import get_dataset_loader

def load_dataset(args, max_frames, n_frames):
    data = get_dataset_loader(name=args.multi_dataset,
                              batch_size=args.batch_size,
                              num_frames=max_frames,
                              split='validation',  # test
                              load_mode='text')  # for GT vis
    data.fixed_length = n_frames
    return data

def init():
    print(f"generating samples")
    sys.argv.extend(["--model_path", "./save/my_pw3d_text/model000500000.pt"])
    args = generate_multi_args()
    print(args)
    fixseed(args.seed)

    args.guidance_param = 1.  # Hard coded - higher values will work but will limit diversity.
    args.max_frames = 120
    args.fps = 20
    args.n_frames = 120
    dist_util.setup_dist(args.device)

    args.num_samples = 1
    args.batch_size = args.num_samples
    print('Loading dataset...')
    data = load_dataset(args, args.max_frames, args.n_frames)
    args.total_num_samples = args.num_samples * args.num_repetitions
    print("Creating model and diffusion...")
    if not args.sample_gt:
        model, diffusion = load_model(args, data, dist_util.dev(), ModelClass=ComMDM)
    else:
        model, diffusion = create_model_and_diffusion(args, data, ModelClass=ComMDM)

    return args,model,diffusion,data


def prompt2video(prompt,args, model, diffusion, data):
    collate_args = [{'inp': torch.zeros(args.n_frames), 'tokens': None, 'lengths': args.n_frames, 'text':prompt}]
    _, model_kwargs = collate(collate_args)

    the_uuid = str(uuid.uuid4())

    out_path = os.path.join(os.path.dirname(args.model_path),'gradio_videos',the_uuid)

    if args.guidance_param != 1:
        model_kwargs['y']['scale'] = torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param

    model_kwargs['y'] = {key: val.to(dist_util.dev()) if torch.is_tensor(val) else val for key, val in
                         model_kwargs['y'].items()}
    if not args.sample_gt:
        sample_fn = diffusion.p_sample_loop
        sample = sample_fn(
            model,
            (args.batch_size, model.njoints, model.nfeats, args.n_frames + 1),
            clip_denoised=False,
            model_kwargs=model_kwargs,
            skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
            init_image=None,
            progress=True,
            predict_two_person=True,
            dump_steps=None,
            noise=None,
            const_noise=False,
        )
        sample, sample1 = sample
    else:
        sample1 = model_kwargs['y']['other_motion'].cpu()

    _, sample = torch.split(sample, [1, sample.shape[-1] - 1], dim=-1)
    _, sample1 = torch.split(sample1, [1, sample1.shape[-1] - 1], dim=-1)
    # Recover XYZ *positions* from HumanML3D vector representation
    if model.data_rep == 'hml_vec':
        n_joints = 22 if sample.shape[1] == 263 else 21
        sample = data.dataset.t2m_dataset.inv_transform(sample.cpu().permute(0, 2, 3, 1)).float()[0,0]
        sample1 = data.dataset.t2m_dataset.inv_transform(sample1.cpu().permute(0, 2, 3, 1)).float()[0, 0]
        vec=(sample1[...,:3]-sample[...,:3])
        sample1[...,:3]=sample[...,:3]+torch.tanh(vec)
        sample = recover_from_ric2(sample, n_joints).numpy()
        sample1 = recover_from_ric2(sample1, n_joints).numpy()

    os.makedirs(out_path)

    print(f"saving visualizations to [{out_path}]...")
    skeleton = paramUtil.kit_kinematic_chain if args.dataset == 'kit' else paramUtil.t2m_kinematic_chain
    save_file = 'sample.mp4'
    animation_save_path = os.path.join(out_path, save_file)
    plot_3d_motion(animation_save_path, skeleton, sample, dataset=args.dataset, title=prompt, fps=args.fps,
                   vis_mode='gt' if args.sample_gt else 'default',
                   joints2=sample1)  # , captions=captions)
    return animation_save_path


args, model, diffusion, data = init()
demo = gr.Interface(
    lambda prompt: prompt2video(prompt, args, model, diffusion, data),
    [gr.Textbox("the first person wraps both arms around the other person's neck.")],
    [gr.Video(format="mp4",autoplay=True)],
)
demo.launch(server_name='0.0.0.0',server_port=7861)
