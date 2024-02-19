import setproctitle

setproctitle.setproctitle("IK")
import binascii
import json
import numpy as np
import torch
from .visualize import Joints2SMPL
import os
from tqdm import tqdm
from data_loaders.humanml.common.quaternion import qbetween_np
import sys
import random
from .txt2npy import _read_skeleton
import traceback

batch_size = 8


def is_OK(aid):
    return not (50 <= aid <= 60 or aid >= 106)


def worker(worker_id, n_workers):
    videos = [video for video in os.listdir("skeletons") if is_OK(int(video[17:20]))]
    videos = sorted(videos)
    random.seed(12321)
    random.shuffle(videos)

    block_size = (len(videos) + n_workers - 1) // n_workers
    motions = videos[worker_id * block_size:(worker_id + 1) * block_size]

    j2s = Joints2SMPL(device=f"cuda:{worker_id % torch.cuda.device_count()}")
    os.makedirs("skeletons_pose", exist_ok=True)
    batch = []
    for motion_file in tqdm(motions):
        if os.path.exists(f"skeletons_pose/{motion_file.replace('.skeleton', '.json')}"):
            try:
                with open(f"skeletons_pose/{motion_file.replace('.npy', '.json')}") as f:
                    json.load(f)["fps"]
                continue
            except:
                pass
        try:
            joints = _read_skeleton(f"skeletons/{motion_file}")['skel_body0'].astype(np.float32)
            '''Put on Floor'''
            floor_height = joints.min(axis=0).min(axis=0)[1]
            joints[..., 1] -= floor_height

            '''XZ at origin'''
            joints[..., [0, 2]] -= joints[0, 0, [0, 2]]

            '''All initially face Z+'''
            root_pos_init = joints[0]
            r_hip, l_hip, sdr_r, sdr_l = [16, 12, 8, 4]
            across1 = root_pos_init[r_hip] - root_pos_init[l_hip]
            across2 = root_pos_init[sdr_r] - root_pos_init[sdr_l]
            across = across1 + across2
            across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]
            forward_init = np.cross(np.float32([[0, 1, 0]]), across, axis=-1)
            forward_init = forward_init / np.sqrt((forward_init ** 2).sum(axis=-1))[..., np.newaxis]
            target = np.float32([[0, 0, 1]])
            root_quat_init = qbetween_np(forward_init, target).astype(np.float32)
            root_quat_init = np.ones(joints.shape[:-1] + (4,), dtype=np.float32) * root_quat_init
            joints = qrot_np(root_quat_init, joints)
            print(joints.shape, joints.dtype)
        except:
            traceback.print_exc()
            with open("failed.txt", "a") as f:
                f.write(f"{motion_file}\n")
            continue
        batch.append((motion_file, joints))
        if len(batch) < batch_size and motion_file != motions[-1]:
            continue
        try:
            all_rotations, all_root_pos = j2s([x[1] for x in batch], step_size=2e-2, num_iters=30, optimizer="lbfgs")
            for rotations, root_pos, file in zip(all_rotations, all_root_pos, [x[0] for x in batch]):
                with open(f"skeletons_pose/{file.replace('.skeleton', '.json')}", "w") as f:
                    json.dump({"root_positions": binascii.b2a_base64(
                        root_pos.flatten().astype(np.float32).tobytes()).decode("utf-8"),
                               "rotations": binascii.b2a_base64(
                                   rotations.flatten().astype(np.float32).tobytes()).decode(
                                   "utf-8"),
                               "dtype": "float32",
                               "fps": 30,
                               "mode": "axis_angle",
                               "n_frames": rotations.shape[0],
                               "n_joints": 24}, f, indent=4)
        except:
            traceback.print_exc()
            with open("failed.txt", "a") as f:
                for x in batch:
                    f.write(f"{x[0]}\n")
        batch.clear()


if __name__ == '__main__':
    worker(int(sys.argv[1]), int(sys.argv[2]))
