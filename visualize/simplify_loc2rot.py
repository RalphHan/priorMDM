import numpy as np
import os
import torch
import h5py
from .joints2smpl import config
from .joints2smpl.my_smpl import MySMPL
from .joints2smpl.smplify import SMPLify3D
import scipy.ndimage.filters as filters
import utils.rotation_conversions as geometry


class Joints2SMPL:

    def __init__(self, device):
        self.device = torch.device(device)
        self.num_joints = 22
        self.joint_category = "AMASS"
        self.fix_foot = False
        model_path = os.path.join(config.SMPL_MODEL_DIR, "smpl")
        smplmodel = MySMPL(model_path, gender="neutral", ext="pkl").to(self.device)
        self.file = h5py.File(config.SMPL_MEAN_FILE, 'r')
        self.smplify = SMPLify3D(smplxmodel=smplmodel,
                                 joints_category=self.joint_category,
                                 device=self.device)

    def __call__(self, input_joints, step_size=1e-2, num_iters=150, optimizer="adam"):
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
        confidence_input = torch.ones(self.num_joints)
        if self.fix_foot:
            confidence_input[7] = 1.5
            confidence_input[8] = 1.5
            confidence_input[10] = 1.5
            confidence_input[11] = 1.5

        pose, _, _ = self.smplify(
            pred_pose.detach(),
            pred_betas.detach(),
            pred_cam_t.detach(),
            keypoints_3d.detach(),
            conf_3d=confidence_input.to(self.device),
            step_size=step_size,
            num_iters=num_iters,
            optimizer=optimizer
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
