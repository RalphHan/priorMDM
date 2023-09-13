import binascii
import json
import numpy as np
import torch
from visualize import Joints2SMPL
import os
from tqdm import tqdm
from data_loaders.humanml.scripts.motion_process import recover_from_ric
import torch.multiprocessing as mp

n_workers = torch.cuda.device_count()

def worker(worker_id):
    files = os.listdir("database/")
    motions = []
    for file in sorted(files):
        try:
            int(file.split('.')[0])
            motions.append(file)
        except:
            pass
    j2s = Joints2SMPL(device=f"cuda:{worker_id}")
    block_size = (len(motions) + n_workers - 1) // n_workers
    start = worker_id * block_size
    end = start + block_size
    os.makedirs("motion_database/", exist_ok=True)
    for motion_file in tqdm(motions[start:end]):
        joints = torch.tensor(np.load(f"database/{motion_file}"))
        n_joints = 22
        joints = recover_from_ric(joints, n_joints).numpy()
        vec1 = np.cross(joints[:, 14] - joints[:, 12], joints[:, 13] - joints[:, 12])
        vec2 = joints[:, 15] - joints[:, 12]
        cosine = (vec1 * vec2).sum(-1) / (np.linalg.norm(vec1, axis=-1) * np.linalg.norm(vec2, axis=-1) + 1e-7)
        if (joints[:, 9, 1] > joints[:, 0, 1]).sum() / joints.shape[0] > 0.85 \
                and (cosine > -0.2).sum() / joints.shape[0] < 0.5:
            joints[..., 0] = -joints[..., 0]
        rotations, root_pos = j2s([joints], step_size=2e-2, num_iters=30, optimizer="lbfgs")
        rotations = rotations[0]
        root_pos = root_pos[0]
        with open(f"motion_database/{motion_file.replace('.npy', '.json')}", "w") as f:
            json.dump({"root_positions": binascii.b2a_base64(
                root_pos.flatten().astype(np.float32).tobytes()).decode("utf-8"),
                       "rotations": binascii.b2a_base64(rotations.flatten().astype(np.float32).tobytes()).decode(
                           "utf-8"),
                       "dtype": "float32",
                       "fps": 20,
                       "mode": "axis_angle",
                       "n_frames": rotations.shape[0],
                       "n_joints": 24}, f, indent=4)


if __name__ == '__main__':
    mp.set_start_method('spawn')
    processes = []
    for i in range(n_workers):
        p = mp.Process(target=worker, args=(i,))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    print("Done!")
