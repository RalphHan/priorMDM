import numpy as np
import os
import torch
import h5py
from .joints2smpl import config
from .joints2smpl.smplify import SMPLify3D
import scipy.ndimage.filters as filters
import utils.rotation_conversions as geometry


class Joints2SMPL:

    def __init__(self, device, use_collision=False):
        self.device = torch.device(device)
        self.joint_category = "AMASS"
        self.use_collision = use_collision
        if self.use_collision:
            from smplx import SMPL
        else:
            from .joints2smpl.my_smpl import MySMPL as SMPL
        model_path = os.path.join(config.SMPL_MODEL_DIR, "smpl")
        smplmodel = SMPL(model_path, gender="neutral", ext="pkl").to(self.device)
        self.file = h5py.File(config.SMPL_MEAN_FILE, 'r')
        self.smplify = SMPLify3D(smplxmodel=smplmodel,
                                 joints_category=self.joint_category,
                                 use_collision=self.use_collision,
                                 device=self.device)

    def __call__(self, input_joints, step_size=1e-2, num_iters=150, optimizer="adam", refine=None):
        if refine is not None:
            assert self.use_collision
        lengths = [joints.shape[0] for joints in input_joints]
        input_joints = np.concatenate(input_joints, axis=0)
        input_joints = torch.from_numpy(input_joints)
        n_frames = input_joints.size(0)
        pred_pose = torch.from_numpy(self.file['pose'][:]).unsqueeze(0).repeat(n_frames, 1).float().to(
            self.device)
        pred_betas = torch.from_numpy(self.file['shape'][:]).unsqueeze(0).repeat(n_frames, 1).float().to(
            self.device)
        pred_cam_t = torch.Tensor([0.0, 0.0, 0.0]).unsqueeze(0).to(self.device)
        keypoints_3d = input_joints.to(self.device).float()
        if refine is not None:
            refine = np.concatenate(refine, axis=0)
            refine = torch.from_numpy(refine).to(self.device).float()

        pose, _ = self.smplify(
            pred_pose.detach(),
            pred_betas.detach(),
            pred_cam_t.detach(),
            keypoints_3d.detach(),
            step_size=step_size,
            num_iters=num_iters,
            optimizer=optimizer,
            refine=refine
        )
        all_thetas = pose.reshape(n_frames, 24, 3)
        pointer = 0
        ret_thetas = []
        ret_root_loc = []
        for i in range(len(lengths)):
            thetas = all_thetas[pointer:pointer + lengths[i]]
            thetas = geometry.matrix_to_rotation_6d(geometry.axis_angle_to_matrix(thetas)).cpu().numpy()
            thetas = torch.tensor(filters.gaussian_filter1d(thetas, 1.5, axis=0, mode='nearest'))
            thetas = geometry.matrix_to_axis_angle(geometry.rotation_6d_to_matrix(thetas)).numpy()
            ret_thetas.append(thetas)
            ret_root_loc.append(keypoints_3d[pointer:pointer + lengths[i], 0].cpu().numpy())
            pointer += lengths[i]
        return ret_thetas, ret_root_loc
