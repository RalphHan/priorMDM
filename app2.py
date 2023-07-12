import sys
import uuid
import gradio as gr
from model.DoubleTake_MDM import doubleTake_MDM
from utils.fixseed import fixseed
import os
import numpy as np
import torch
from utils.parser_util import generate_args
from utils.model_util import load_model
from utils import dist_util
from data_loaders.get_data import get_dataset_loader
from data_loaders.humanml.scripts.motion_process import recover_from_ric
import data_loaders.humanml.utils.paramUtil as paramUtil
from data_loaders.humanml.utils.plot_script import plot_3d_motion
from utils.sampling_utils import unfold_sample_arb_len, double_take_arb_len

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

def init():
    print(f"generating samples")
    sys.argv.extend(["--model_path","./save/my_humanml_trans_enc_512/model000200000.pt",
                     "--handshake_size","20",
                     "--blend_len","10"])
    args = generate_args()
    sys.argv.extend(["--model_path", "./save/pw3d_text/model000100000.pt"])
    fixseed(args.seed)

    args.fps = 30 if args.dataset == 'babel' else 20
    args.n_frames = 150
    dist_util.setup_dist(args.device)
    assert (args.double_take)
    args.num_samples=2
    args.batch_size = 2  # Sampling a single batch from the testset, with exactly args.num_samples

    print('Loading dataset...')
    data = load_dataset(args, args.n_frames)

    print("Creating model and diffusion...")
    model, diffusion = load_model(args, data, dist_util.dev(), ModelClass=doubleTake_MDM)

    return args, model, diffusion, data

def prompt2video(prompt,args, model, diffusion, data):
    total_num_samples = args.num_samples * args.num_repetitions
    model_kwargs = {'y': {
        'mask': torch.ones((2, 1, 1, 196)), #196 is humanml max frames number
        'lengths': torch.tensor([args.n_frames,95]),
        'text': [prompt,"A person makes a long leap forward"],
        'tokens': [''],
        'scale': torch.ones(2)*2.5
    }}

    all_motions = []
    all_lengths = []
    all_text = []
    all_captions = []

    for rep_i in range(args.num_repetitions):
        print(f'### Sampling [repetitions #{rep_i}]')
        if args.guidance_param != 1:
            model_kwargs['y']['scale'] = torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param
        model_kwargs['y'] = {key: val.to(dist_util.dev()) if torch.is_tensor(val) else val for key, val in model_kwargs['y'].items()}

        max_arb_len = model_kwargs['y']['lengths'].max()
        min_arb_len = 2 * args.handshake_size + 2*args.blend_len + 10

        for ii, len_s in enumerate(model_kwargs['y']['lengths']):
            if len_s > max_arb_len:
                model_kwargs['y']['lengths'][ii] = max_arb_len
            if len_s < min_arb_len:
                model_kwargs['y']['lengths'][ii] = min_arb_len
        samples_per_rep_list, samples_type = double_take_arb_len(args, diffusion, model, model_kwargs, max_arb_len)

        step_sizes = np.zeros(len(model_kwargs['y']['lengths']), dtype=int)
        for ii, len_i in enumerate(model_kwargs['y']['lengths']):
            if ii == 0:
                step_sizes[ii] = len_i
                continue
            step_sizes[ii] = step_sizes[ii-1] + len_i - args.handshake_size

        final_n_frames = step_sizes[-1]

        for sample_i, samples_type_i in zip(samples_per_rep_list, samples_type):

            sample = unfold_sample_arb_len(sample_i, args.handshake_size, step_sizes, final_n_frames, model_kwargs)

            # Recover XYZ *positions* from HumanML3D vector representation
            if model.data_rep == 'hml_vec':
                n_joints = 22 if sample.shape[1] == 263 else 21
                sample = data.dataset.t2m_dataset.inv_transform(sample.cpu().permute(0, 2, 3, 1)).float()
                sample = recover_from_ric(sample, n_joints)
                sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)
            if args.dataset == 'babel':
                from data_loaders.amass.transforms import SlimSMPLTransform
                transform = SlimSMPLTransform(batch_size=args.batch_size, name='SlimSMPLTransform', ename='smplnh', normalization=True)

                all_feature = sample #[bs, nfeats, 1, seq_len]
                all_feature_squeeze = all_feature.squeeze(2) #[bs, nfeats, seq_len]
                all_feature_permutes = all_feature_squeeze.permute(0, 2, 1) #[bs, seq_len, nfeats]
                splitted = torch.split(all_feature_permutes, all_feature.shape[0]) #[list of [seq_len,nfeats]]
                sample_list = []
                for seq in splitted[0]:
                    all_features = seq
                    Datastruct = transform.SlimDatastruct
                    datastruct = Datastruct(features=all_features)
                    sample = datastruct.joints

                    sample_list.append(sample.permute(1, 2, 0).unsqueeze(0))
                sample = torch.cat(sample_list)
            else:
                rot2xyz_pose_rep = 'xyz' if model.data_rep in ['xyz', 'hml_vec'] else model.data_rep
                if args.dataset == 'babel':
                    rot2xyz_pose_rep = 'rot6d'
                rot2xyz_mask = None

                sample = model.rot2xyz(x=sample, mask=rot2xyz_mask, pose_rep=rot2xyz_pose_rep, glob=True, translation=True,
                                       jointstype='smpl', vertstrans=True, betas=None, beta=0, glob_rot=None,
                                       get_rotations_back=False)

            text_key = 'text' if 'text' in model_kwargs['y'] else 'action_text'

            all_text += model_kwargs['y'][text_key]
            all_captions += model_kwargs['y'][text_key]

            all_motions.append(sample.cpu().numpy())
            all_lengths.append(model_kwargs['y']['lengths'].cpu().numpy())

            print(f"created {len(all_motions) * args.batch_size} samples")

    # param update for unfolding visualization
    # out of for rep_i
    old_num_samples = args.num_samples
    n_frames = final_n_frames

    num_repetitions = args.num_repetitions

    all_motions = np.concatenate(all_motions, axis=0)
    all_motions = all_motions[:total_num_samples]  # [bs, njoints, 6, seqlen]
    all_text = all_text[:total_num_samples]
    all_lengths = [n_frames] * num_repetitions

    the_uuid = str(uuid.uuid4())
    out_path = os.path.join(os.path.dirname(args.model_path), 'gradio_videos', the_uuid)
    os.makedirs(out_path)

    npy_path = os.path.join(out_path, 'results.npy')
    frame_colors = calc_frame_colors(args.handshake_size, args.blend_len, step_sizes, model_kwargs['y']['lengths'])
    print(f"saving results file to [{npy_path}]")
    np.save(npy_path,
            {'motion': all_motions, 'text': all_text, 'lengths': all_lengths,
             'num_samples': args.num_samples, 'num_repetitions': num_repetitions, 'frame_colors': frame_colors})
    with open(npy_path.replace('.npy', '.txt'), 'w') as fw:
        fw.write('\n'.join(all_text))
    with open(npy_path.replace('.npy', '_len.txt'), 'w') as fw:
        fw.write('\n'.join([str(l) for l in all_lengths]))

    print(f"saving visualizations to [{out_path}]...")
    skeleton = paramUtil.kit_kinematic_chain if args.dataset == 'kit' else paramUtil.t2m_kinematic_chain
    if args.dataset == 'babel':
        skeleton = paramUtil.t2m_kinematic_chain
    for sample_i in range(args.num_samples):
        for rep_i, samples_type_i in zip(range(num_repetitions), samples_type):
            caption = [f'{samples_type_i} {all_text[0]}'] * (model_kwargs['y']['lengths'][0] - int(args.handshake_size/2))
            for ii in range(1, old_num_samples):
                caption += [f'{samples_type_i} {all_text[ii]}'] * (int(model_kwargs['y']['lengths'][ii])-args.handshake_size)
            caption += [f'{samples_type_i} {all_text[ii]}'] * (int(args.handshake_size/2))
            length = all_lengths[rep_i*args.batch_size + sample_i]
            motion = all_motions[rep_i*args.batch_size + sample_i].transpose(2, 0, 1)[:length]
            save_file = 'sample{:02d}_rep{:02d}.mp4'.format(sample_i, rep_i)
            animation_save_path = os.path.join(out_path, save_file)
            print(f'[({sample_i}) "{set(caption)}" | Rep #{rep_i} | -> {save_file}]')
            plot_3d_motion(animation_save_path, skeleton, motion, dataset=args.dataset, title=caption, fps=args.fps,
                           vis_mode='gt' if args.sample_gt else 'unfold_arb_len', handshake_size=args.handshake_size,
                           blend_size=args.blend_len,step_sizes=step_sizes, lengths=model_kwargs['y']['lengths'])
            # Credit for visualization: https://github.com/EricGuo5513/text-to-motion
            return animation_save_path


args, model, diffusion, data = init()
demo = gr.Interface(
    lambda prompt: prompt2video(prompt, args, model, diffusion, data),
    [gr.Textbox("A person sits while crossing legs")],
    [gr.Video(format="mp4",autoplay=True)],
)
demo.launch(server_name='0.0.0.0',server_port=7862)
