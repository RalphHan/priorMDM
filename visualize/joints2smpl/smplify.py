import torch
from .customloss import (camera_fitting_loss_3d,
                         body_fitting_loss_3d,
                         )
from .prior import MaxMixturePrior
from . import config


@torch.no_grad()
def guess_init_3d(model_joints,
                  j3d,
                  joints_category="orig"):
    """Initialize the camera translation via triangle similarity, by using the torso joints        .
    :param model_joints: SMPL model with pre joints
    :param j3d: 25x3 array of Kinect Joints
    :returns: 3D vector corresponding to the estimated camera translation
    """
    # get the indexed four
    gt_joints = ['RHip', 'LHip', 'RShoulder', 'LShoulder']
    gt_joints_ind = [config.JOINT_MAP[joint] for joint in gt_joints]

    if joints_category == "orig":
        joints_ind_category = [config.JOINT_MAP[joint] for joint in gt_joints]
    elif joints_category == "AMASS":
        joints_ind_category = [config.AMASS_JOINT_MAP[joint] for joint in gt_joints]
    else:
        print("NO SUCH JOINTS CATEGORY!")

    sum_init_t = (j3d[:, joints_ind_category] - model_joints[:, gt_joints_ind]).sum(dim=1)
    init_t = sum_init_t / 4.0
    return init_t


# SMPLIfy 3D
class SMPLify3D():
    """Implementation of SMPLify, use 3D joints."""

    def __init__(self,
                 smplxmodel,
                 joints_category="orig",
                 device=torch.device('cuda:0'),
                 ):
        self.device = device
        self.pose_prior = MaxMixturePrior(prior_folder=config.GMM_MODEL_DIR,
                                          num_gaussians=8,
                                          dtype=torch.float32).to(device)
        self.smpl = smplxmodel

        self.model_faces = smplxmodel.faces_tensor.view(-1)
        self.joints_category = joints_category
        self.smpl_index = config.amass_smpl_idx
        self.corr_index = config.amass_idx

    def __call__(self, init_pose, init_betas, init_cam_t, j3d, conf_3d=1.0, step_size=1e-2, num_iters=100, optimizer='adam'):
        search_tree = None
        pen_distance = None
        filter_faces = None
        body_pose = init_pose[:, 3:].detach().clone()
        global_orient = init_pose[:, :3].detach().clone()
        betas = init_betas.detach().clone()

        # use guess 3d to get the initial
        smpl_output = self.smpl(global_orient=global_orient,
                                body_pose=body_pose,
                                betas=betas)
        model_joints = smpl_output.joints

        init_cam_t = guess_init_3d(model_joints, j3d, self.joints_category).unsqueeze(1).detach()
        camera_translation = init_cam_t.clone()

        preserve_pose = init_pose[:, 3:].detach().clone()
        # -------------Step 1: Optimize camera translation and body orientation--------
        # Optimize only camera translation and body orientation
        body_pose.requires_grad = False
        betas.requires_grad = False
        global_orient.requires_grad = True
        camera_translation.requires_grad = True

        camera_opt_params = [global_orient, camera_translation]
        if optimizer == 'adam':
            cam_steps = 20
            camera_optimizer = torch.optim.Adam(camera_opt_params, lr=step_size, betas=(0.9, 0.999))
        elif optimizer == 'lbfgs':
            cam_steps = 10
            camera_optimizer = torch.optim.LBFGS(camera_opt_params, max_iter=cam_steps,
                                                 lr=step_size, line_search_fn='strong_wolfe')
        else:
            raise NotImplementedError

        for i in range(cam_steps):
            def closure():
                camera_optimizer.zero_grad()
                smpl_output = self.smpl(global_orient=global_orient,
                                        body_pose=body_pose,
                                        betas=betas)
                model_joints = smpl_output.joints
                loss = camera_fitting_loss_3d(model_joints[:, self.smpl_index], camera_translation,
                                              init_cam_t, j3d[:, self.corr_index], self.joints_category)
                loss.backward()
                return loss
            camera_optimizer.step(closure)

        # Fix camera translation after optimizing camera
        # --------Step 2: Optimize body joints --------------------------
        # Optimize only the body pose and global orientation of the body
        body_pose.requires_grad = True
        global_orient.requires_grad = True
        camera_translation.requires_grad = True

        # --- if we use the sequence, fix the shape
        betas.requires_grad = True
        body_opt_params = [body_pose, betas, global_orient, camera_translation]

        if optimizer == 'adam':
            body_optimizer = torch.optim.Adam(body_opt_params, lr=step_size, betas=(0.9, 0.999))
            pose_preserve_weight = 0.0
        elif optimizer == 'lbfgs':
            body_optimizer = torch.optim.LBFGS(body_opt_params, max_iter=num_iters,
                                                 lr=step_size, line_search_fn='strong_wolfe')
            pose_preserve_weight = 5.0
        else:
            raise NotImplementedError

        for _ in range(num_iters):
            def closure():
                body_optimizer.zero_grad()
                smpl_output = self.smpl(global_orient=global_orient,
                                        body_pose=body_pose,
                                        betas=betas)
                model_joints = smpl_output.joints

                loss = body_fitting_loss_3d(body_pose, preserve_pose, betas, model_joints[:, self.smpl_index],
                                            camera_translation,
                                            j3d[:, self.corr_index], self.pose_prior,
                                            joints3d_conf=conf_3d,
                                            joint_loss_weight=600.0,
                                            pose_preserve_weight=pose_preserve_weight,
                                            use_collision=False,
                                            model_vertices=None, model_faces=self.model_faces,
                                            search_tree=search_tree, pen_distance=pen_distance,
                                            filter_faces=filter_faces)
                loss.backward()
                return loss
            body_optimizer.step(closure)

        pose = torch.cat([global_orient, body_pose], dim=-1)

        return pose.detach(), betas.detach(), camera_translation.detach()
