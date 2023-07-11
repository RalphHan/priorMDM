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
from data_loaders.humanml.scripts.motion_process import recover_from_ric
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
    sys.argv.extend(["--model_path", "./save/pw3d_text/model000100000.pt"])
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

    all_motions = []
    all_lengths = []
    all_text = []
    all_captions = []

    the_uuid = str(uuid.uuid4())

    out_path = os.path.join(os.path.dirname(args.model_path),'gradio_videos',the_uuid)

    for rep_i in range(args.num_repetitions):
        print(f'### Sampling [repetitions #{rep_i}]')

        # add CFG scale to batch
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

        canon0, sample = torch.split(sample, [1, sample.shape[-1] - 1], dim=-1)
        canon0 = data.dataset.t2m_dataset.rebuilt_canon(canon0[:, :4, 0, 0])
        canon1, sample1 = torch.split(sample1, [1, sample1.shape[-1] - 1], dim=-1)
        canon1 = data.dataset.t2m_dataset.rebuilt_canon(canon1[:, :4, 0, 0])
        diff_trans = canon1[:, -3:] - canon0[:, -3:]
        _rot0 = geometry.rotation_6d_to_matrix(canon0[:, :6])
        _rot1 = geometry.rotation_6d_to_matrix(canon1[:, :6])
        diff_rot = torch.matmul(_rot0, _rot1.permute(0, 2, 1)).float().cpu()

        # Recover XYZ *positions* from HumanML3D vector representation
        if model.data_rep == 'hml_vec':
            n_joints = 22 if sample.shape[1] == 263 else 21
            sample = data.dataset.t2m_dataset.inv_transform(sample.cpu().permute(0, 2, 3, 1)).float()
            sample = recover_from_ric(sample, n_joints)
            sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)

            if sample1 is not None:
                sample1 = data.dataset.t2m_dataset.inv_transform(sample1.cpu().permute(0, 2, 3, 1)).float()
                sample1 = recover_from_ric(sample1, n_joints)
                sample1 = torch.matmul(diff_rot.view(-1, 1, 1, 1, 3, 3), sample1.unsqueeze(-1)).squeeze(-1)
                sample1 = sample1.view(-1, *sample1.shape[2:]).permute(0, 2, 3, 1).cpu().numpy()
                sample1 += diff_trans.view(-1, 1, 3, 1).cpu().numpy()

        text_key = 'text'
        all_text += model_kwargs['y'][text_key]
        all_captions += model_kwargs['y'][text_key]

        all_motions.append(sample.cpu().numpy())
        all_lengths.append(model_kwargs['y']['lengths'].cpu().numpy())

        print(f"created {len(all_motions) * args.batch_size} samples")

    all_motions = np.concatenate(all_motions, axis=0)
    all_motions = all_motions[:args.total_num_samples]  # [bs, njoints, 6, seqlen]
    all_text = all_text[:args.total_num_samples]
    all_lengths = np.concatenate(all_lengths, axis=0)[:args.total_num_samples]

    os.makedirs(out_path)

    npy_path = os.path.join(out_path, 'results.npy')
    print(f"saving results file to [{npy_path}]")
    np.save(npy_path,
            {'motion': sample, 'other_motion': sample1, 'text': all_text, 'lengths': all_lengths,
             'num_samples': args.num_samples, 'num_repetitions': args.num_repetitions})
    with open(npy_path.replace('.npy', '.txt'), 'w') as fw:
        fw.write('\n'.join(all_text))
    with open(npy_path.replace('.npy', '_len.txt'), 'w') as fw:
        fw.write('\n'.join([str(l) for l in all_lengths]))

    print(f"saving visualizations to [{out_path}]...")
    skeleton = paramUtil.kit_kinematic_chain if args.dataset == 'kit' else paramUtil.t2m_kinematic_chain

    if args.sample_gt:
        args.num_repetitions = 1

    for sample_i in range(args.num_samples):
        for rep_i in range(args.num_repetitions):

            caption = all_text[rep_i * args.batch_size + sample_i]
            length = all_lengths[rep_i * args.batch_size + sample_i] - 1
            motion = all_motions[rep_i * args.batch_size + sample_i].transpose(2, 0, 1)[:length]
            motion1 = None
            if sample1 is not None:
                motion1 = sample1[sample_i].transpose(2, 0, 1)[:length]
            save_file = 'sample{:02d}_rep{:02d}.mp4'.format(sample_i, rep_i)
            animation_save_path = os.path.join(out_path, save_file)
            print(f'[({sample_i}) "{caption}" | Rep #{rep_i} | -> {save_file}]')
            plot_3d_motion(animation_save_path, skeleton, motion, dataset=args.dataset, title=caption, fps=args.fps,
                           vis_mode='gt' if args.sample_gt else 'default',
                           joints2=motion1)  # , captions=captions)
            return animation_save_path


args, model, diffusion, data = init()
demo = gr.Interface(
    lambda prompt: prompt2video(prompt, args, model, diffusion, data),
    [gr.Textbox("A Capoeira practice. One is kicking and the other is avoiding the kick.")],
    [gr.Video(format="mp4",autoplay=True)],
)
demo.launch(server_name='0.0.0.0',server_port=7861)
