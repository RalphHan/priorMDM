import setproctitle

setproctitle.setproctitle("IK")
import binascii
import json
import numpy as np
import torch
from visualize import Joints2SMPL
import os
from tqdm import tqdm
from data_loaders.humanml.scripts.motion_process import recover_from_ric
import sys
import random

batch_size = 1


def worker(worker_id, n_workers):
    motions = sorted(os.listdir("database/"))
    random.seed(12321)
    random.shuffle(motions)
    j2s = Joints2SMPL(device=f"cuda:{worker_id % torch.cuda.device_count()}", use_collision=True)
    block_size = (len(motions) + n_workers - 1) // n_workers
    start = worker_id * block_size
    end = start + block_size
    motions = motions[start:end]
    os.makedirs("motion_database3/", exist_ok=True)
    batch = []
    for motion_file in tqdm(motions):
        if os.path.exists(f"motion_database3/{motion_file.replace('.npy', '.json')}"):
            try:
                with open(f"motion_database3/{motion_file.replace('.npy', '.json')}") as f:
                    json.load(f)["fps"]
                continue
            except:
                pass
        joints = torch.tensor(np.load(f"database/{motion_file}"))
        n_joints = 22
        joints = recover_from_ric(joints, n_joints).numpy()
        vec1 = np.cross(joints[:, 14] - joints[:, 12], joints[:, 13] - joints[:, 12])
        vec2 = joints[:, 15] - joints[:, 12]
        cosine = (vec1 * vec2).sum(-1) / (np.linalg.norm(vec1, axis=-1) * np.linalg.norm(vec2, axis=-1) + 1e-7)
        if (joints[:, 9, 1] > joints[:, 0, 1]).sum() / joints.shape[0] > 0.85 \
                and (cosine > -0.2).sum() / joints.shape[0] < 0.5:
            joints[..., 0] = -joints[..., 0]
        with open(f"motion_database/{motion_file.replace('.npy', '.json')}") as f:
            json_data = json.load(f)
            refine = np.frombuffer(binascii.a2b_base64(json_data["rotations"]),
                                   dtype=json_data["dtype"]).reshape(-1, 24*3)

        batch.append((motion_file, joints, refine))
        if len(batch) < batch_size and motion_file != motions[-1]:
            continue
        # try:
        all_rotations, all_root_pos = j2s([x[1] for x in batch], step_size=2e-2, num_iters=30, optimizer="lbfgs",refine=[x[2] for x in batch])
        for rotations, root_pos, file in zip(all_rotations, all_root_pos, [x[0] for x in batch]):
            with open(f"motion_database3/{file.replace('.npy', '.json')}", "w") as f:
                json.dump({"root_positions": binascii.b2a_base64(
                    root_pos.flatten().astype(np.float32).tobytes()).decode("utf-8"),
                           "rotations": binascii.b2a_base64(
                               rotations.flatten().astype(np.float32).tobytes()).decode(
                               "utf-8"),
                           "dtype": "float32",
                           "fps": 30 if file.startswith("mixamo_") else 20,
                           "mode": "axis_angle",
                           "n_frames": rotations.shape[0],
                           "n_joints": 24}, f, indent=4)
        # except:
        #     with open("failed.txt", "a") as f:
        #         for x in batch:
        #             f.write(f"{x[0]}\n")
        batch.clear()


if __name__ == '__main__':
    worker(int(sys.argv[1]), int(sys.argv[2]))
