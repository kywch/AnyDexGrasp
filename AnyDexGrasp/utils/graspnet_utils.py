import copy

import torch
import numpy as np

import MinkowskiEngine as ME

from ..models.minkowski_graspnet import MinkowskiGraspNet
from .pt_utils import batch_viewpoint_params_to_matrix
from .np_utils import transform_point_cloud


def parse_preds(end_points, max_grasp_width):
    batch_size = end_points["stage3_grasp_features"].shape[0]

    ## load preds
    before_generator = end_points["before_generator"]  # (B, Ns, 256)
    point_features = end_points["point_features"]  # (B, Ns, 512)
    coords = end_points["sinput"].C  # (\Sigma Ni, 4)
    objectness_pred = end_points["stage1_objectness_pred"]  # (Sigma Ni, 2)
    objectness_mask = torch.argmax(objectness_pred, dim=1).bool()  # (\Sigma Ni,)
    seed_xyz = end_points["stage2_seed_xyz"]  # (B, Ns, 3)
    seed_inds = end_points["stage2_seed_inds"]  # (B, Ns)
    grasp_view_xyz = end_points["stage2_view_xyz"]  # (B, Ns, 3)
    grasp_view_inds = end_points["stage2_view_inds"]
    grasp_view_scores = end_points["stage2_view_scores"]
    grasp_scores = end_points["stage3_grasp_scores"]  # (B, Ns, A, D)
    grasp_features_two_finger = end_points["stage3_grasp_features"].view(
        grasp_scores.size()[0], grasp_scores.size()[1], -1
    )  # (B, Ns, 3 + C)
    grasp_widths = max_grasp_width * end_points["stage3_normalized_grasp_widths"]  # (B, Ns, A, D)
    grasp_widths[grasp_widths > max_grasp_width] = max_grasp_width

    grasp_preds = []
    grasp_features = []
    grasp_vdistance_list = []
    for i in range(batch_size):
        cloud_mask_i = coords[:, 0] == i
        seed_inds_i = seed_inds[i]
        objectness_mask_i = objectness_mask[cloud_mask_i][seed_inds_i]  # (Ns,)

        if objectness_mask_i.any() is False:
            continue

        seed_xyz_i = seed_xyz[i]  # [objectness_mask_i]  # (Ns', 3)
        point_features_i = point_features[i]  # [objectness_mask_i]

        seed_inds_i = seed_inds_i  # [objectness_mask_i]
        before_generator_i = before_generator[i]  # [objectness_mask_i]
        grasp_view_xyz_i = grasp_view_xyz[i]  # [objectness_mask_i]  # (Ns', 3)
        grasp_view_inds_i = grasp_view_inds[i]  # [objectness_mask_i]
        grasp_view_scores_i = grasp_view_scores[i]  # [objectness_mask_i]
        grasp_scores_i = grasp_scores[i]  # [objectness_mask_i]  # (Ns', A, D)
        grasp_widths_i = grasp_widths[i]  # [objectness_mask_i] # (Ns', A, D)

        Ns, A, D = grasp_scores_i.size()
        grasp_features_two_finger_i = grasp_features_two_finger[i]  # [objectness_mask_i] # (Ns', 3 + C)
        grasp_scores_i_A_D = copy.deepcopy(grasp_scores_i).view(Ns, -1)

        grasp_scores_i = torch.minimum(grasp_scores_i[:, :24, :], grasp_scores_i[:, 24:, :])
        seed_inds_i = seed_inds_i.view(Ns, -1)
        grasp_view_inds_i = grasp_view_inds_i.view(Ns, -1)
        grasp_view_scores_i = grasp_view_scores_i.view(Ns, -1)

        grasp_scores_i, grasp_angles_class_i = torch.max(grasp_scores_i, dim=1)  # (Ns', D), (Ns', D)
        grasp_angles_i = (grasp_angles_class_i.float() - 12) / 24 * np.pi  # (Ns', topk, D)

        # grasp width & vdistance
        grasp_angles_class_i = grasp_angles_class_i.unsqueeze(1)  # (Ns', 1, D)
        grasp_widths_pos_i = torch.gather(grasp_widths_i, 1, grasp_angles_class_i).squeeze(1)  # (Ns', D)
        grasp_widths_neg_i = torch.gather(grasp_widths_i, 1, grasp_angles_class_i + 24).squeeze(1)  # (Ns', D)

        ## slice preds by grasp score/depth
        # grasp score & depth
        grasp_scores_i, grasp_depths_class_i = torch.max(grasp_scores_i, dim=1, keepdims=True)  # (Ns', 1), (Ns', 1)
        grasp_depths_i = (grasp_depths_class_i.float() + 1) * 0.01  # (Ns'*topk, 1)
        grasp_depths_i -= 0.01
        grasp_depths_i[grasp_depths_class_i == 0] = 0.005

        grasp_angles_i = torch.gather(grasp_angles_i, 1, grasp_depths_class_i)  # (Ns', 1)
        grasp_widths_pos_i = torch.gather(grasp_widths_pos_i, 1, grasp_depths_class_i)  # (Ns', 1)
        grasp_widths_neg_i = torch.gather(grasp_widths_neg_i, 1, grasp_depths_class_i)  # (Ns', 1)

        # convert to rotation matrix
        rotation_matrices_i = batch_viewpoint_params_to_matrix(-grasp_view_xyz_i, grasp_angles_i.squeeze(1))

        # # adjust gripper centers
        grasp_widths_i = grasp_widths_pos_i + grasp_widths_neg_i
        rotation_matrices_i = rotation_matrices_i.view(Ns, 9)

        # merge preds
        grasp_preds.append(
            torch.cat([grasp_scores_i, grasp_widths_i, grasp_depths_i, rotation_matrices_i, seed_xyz_i], axis=1)
        )  # (Ns, 15)
        grasp_features.append(
            torch.cat(
                [
                    grasp_scores_i_A_D,
                    grasp_features_two_finger_i,
                    before_generator_i,
                    point_features_i,
                    grasp_view_inds_i,
                    grasp_view_scores_i,
                    seed_inds_i,
                    grasp_angles_i * 24 / np.pi + 12,
                    grasp_depths_i,
                ],
                axis=1,
            )
        )  # (Ns'*3, A, D)

    return grasp_preds, grasp_features


def flip_ggarray(ggarray):
    ggarray_rotations = ggarray[:, 4:13].reshape((-1, 3, 3))
    tcp_x_axis_on_base_frame = ggarray_rotations[:, 1, 1]
    if_flip = [False for _ in range(len(ggarray))]
    for ids, y_x in enumerate(tcp_x_axis_on_base_frame):
        if y_x < 0:
            ggarray_rotations[ids, :3, 0:2] = -ggarray_rotations[ids, :3, 0:2]
            if_flip[ids] = True
    ggarray[:, 4:13] = ggarray_rotations.reshape((-1, 9))
    return ggarray, if_flip


def get_grasp_features(grasp_features_array):
    # TODO: check where these numbers come from
    grasp_features = dict()
    grasp_features["grasp_angles"] = int(grasp_features_array[-3] + 0.1)
    grasp_features["grasp_depths"] = int(grasp_features_array[-2] * 100 + 0.1)
    grasp_features["stage3_grasp_scores"] = grasp_features_array[:240].tolist()
    grasp_features["grasp_preds_features"] = grasp_features_array[240 : 240 + 480].tolist()
    grasp_features["stage3_grasp_features"] = grasp_features_array[240 + 480 : 240 + 480 + 512].tolist()
    grasp_features["before_generator"] = grasp_features_array[240 + 480 + 512 : 240 + 480 + 512 + 512].tolist()
    grasp_features["point_features"] = grasp_features_array[
        240 + 480 + 512 + 512 : 240 + 480 + 512 + 512 + 512
    ].tolist()
    grasp_features["point_id"] = int(grasp_features_array[-4])
    grasp_features["if_flip"] = bool(int(grasp_features_array[-1]))
    grasp_features["view_inds"] = bool(int(grasp_features_array[-6]))
    grasp_features["view_score"] = bool(int(grasp_features_array[-5]))
    return grasp_features


class GraspNetRunner:
    def __init__(
        self,
        camera,
        graspnet_path,
        num_depths=5,
        num_seed=2048,
        half_views=False,
        max_grasp_width=0.1,
        min_grasp_width=0.02,
    ):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._load_graspnet(graspnet_path, num_depths, num_seed, half_views)

        self.max_grasp_width = max_grasp_width
        self.min_grasp_width = min_grasp_width

        # NOTE: wrist camera changes the location
        # So use the invariant infos like width, height, fx, cx, and do NOT store camera info
        # self.camera = camera
        # self.world_to_pixel_mat = camera.world_to_pixel_mat
        # self.camera_to_world_mat = camera.camera_to_world_mat
        # self.camera_to_pixel_mat = camera.camera_to_pixel_mat

        self.camera_scale = camera.scale

        xmap = np.arange(camera.width)
        ymap = np.arange(camera.height)
        _xmap, _ymap = np.meshgrid(xmap, ymap)
        self._points_x_norm = (_xmap - camera.cx) / camera.fx
        self._points_y_norm = (_ymap - camera.cy) / camera.fy

    def _load_graspnet(self, graspnet_path, num_depths, num_seed, half_views):
        self.network = MinkowskiGraspNet(
            num_depth=num_depths, num_seed=num_seed, is_training=False, half_views=half_views
        )
        self.network.to(self.device)
        self.network.eval()

        # Get the number of params
        num_params = sum(p.numel() for p in self.network.parameters() if p.requires_grad)
        print(f"Number of parameters: {num_params}")

        checkpoint = torch.load(graspnet_path)
        self.network.load_state_dict(checkpoint["model_state_dict"])

    def get_grasp(self, depth_map, augment_mat=None, voxel_size=0.005, flip=False):
        # NOTE: in the original code, color map was only used for visuzlizing with open3d

        points_z = depth_map / self.camera_scale
        points_x = self._points_x_norm * points_z
        points_y = self._points_y_norm * points_z

        # The distances from the wrist camera
        mask = (points_z > 0.3) & (points_z < 0.6)
        points = np.stack([points_x, points_y, points_z], axis=-1)
        points = points[mask].astype(np.float32)
        assert len(points) > 0, "No points detected"

        if augment_mat is not None:
            points = transform_point_cloud(points, augment_mat).astype(np.float32)
        else:
            augment_mat = np.eye(4)

        points = torch.from_numpy(points)
        coords = np.ascontiguousarray(points / voxel_size, dtype=int)
        # Upd Note. API change.
        _, idxs = ME.utils.sparse_quantize(coords, return_index=True)
        coords = coords[idxs]
        points = points[idxs]
        coords_batch, points_batch = ME.utils.sparse_collate([coords], [points])

        sinput = ME.SparseTensor(points_batch, coords_batch, device=self.device)
        end_points = {"sinput": sinput, "point_clouds": [sinput.F]}
        with torch.no_grad():
            end_points = self.network(end_points)
            preds, grasp_features = parse_preds(end_points, self.max_grasp_width)
            if len(preds) == 0:
                print("No grasp detected")
                # Check the return vars
                return None, points.cuda(), None, [sinput]
            else:
                preds = preds[0]

        # filter
        if flip:
            augment_mat[:, 0] = -augment_mat[:, 0]
        augment_mat_tensor = torch.tensor(
            copy.deepcopy(np.linalg.inv(augment_mat).astype(np.float32)), device=self.device
        )
        rotation = augment_mat_tensor[:3, :3].reshape((-1)).repeat((preds.size()[0], 1)).view((preds.size()[0], 3, 3))
        translation = augment_mat_tensor[:3, 3]

        preds[:, 12:15] = torch.matmul(rotation, preds[:, 12:15].view((-1, 3, 1))).view(-1, 3) + translation
        pose_rotation = torch.matmul(rotation, preds[:, 3:12].view((-1, 3, 3)))
        if flip:
            preds[:, 12] = -preds[:, 12]
            pose_rotation[:, 0, :] = -pose_rotation[:, 0, :]
            pose_rotation[:, :, 1] = -pose_rotation[:, :, 1]
        preds[:, 3:12] = pose_rotation.view((-1, 9))

        # CHECK the hardcoded numbers
        # Something like ... preserves the grasp poses that are within a 30-degree angle with the vertical pose
        mask = (preds[:, 9] > 0.85) & (preds[:, 1] < self.max_grasp_width) & (preds[:, 1] > self.min_grasp_width)

        # The second mask preserves the grasp poses within the workspace of the robot.
        # workspace_mask = (preds[:, 12] > -0.25) & (preds[:, 12] < 0.25) & (preds[:, 13] > -0.20) & (preds[:, 13] < 0.05)
        # NOTE: preds[:, 12] and preds[:, 13] are in camera coordinates, so the above numbers are a bit wrong.

        preds = preds[mask]
        grasp_features = grasp_features[0][mask]
        if len(preds) == 0:
            print("No grasp detected after grasp width filtering")
            return None, points.cuda(), None, [sinput]

        points = points.cuda()
        heights = 0.03 * torch.ones([preds.shape[0], 1]).cuda()
        object_ids = -1 * torch.ones([preds.shape[0], 1]).cuda()
        ggarray = torch.cat([preds[:, 0:2], heights, preds[:, 2:15], preds[:, 15:16], object_ids], axis=-1)

        return ggarray, points, grasp_features, [sinput]

    def _get_augment_mat(self, flip=False):
        flip_mat = np.eye(4)
        # Flipping along the YZ plane
        if flip:
            flip_mat = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

        # Rotation along up-axis/Z-axis
        rot_angle = (np.random.random() * np.pi / 3) - np.pi / 6  # -30 ~ +30 degree
        c, s = np.cos(rot_angle), np.sin(rot_angle)
        rot_mat = np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

        # Translation along X/Y/Z-axis
        offset_x = np.random.random() * 0.1 - 0.05  # -0.05 ~ 0.05
        offset_y = np.random.random() * 0.1 - 0.05  # -0.05 ~ 0.05
        trans_mat = np.array([[1, 0, 0, offset_x], [0, 1, 0, offset_y], [0, 0, 1, 0], [0, 0, 0, 1]])

        aug_mat = np.dot(trans_mat, np.dot(rot_mat, flip_mat).astype(np.float32)).astype(np.float32)
        return aug_mat

    def get_ggarray_features(self, depth_map, num_augment=0):
        augment_mat, ggarray = np.eye(4), None
        while ggarray is None:
            ggarray, points_down, grasp_features, sinput = self.get_grasp(depth_map, augment_mat)

            # This gets used when ggarray is None
            augment_mat = self._get_augment_mat()

        # Augment point cloud
        for i in range(num_augment):
            flip = i % 2
            augment_mat = self._get_augment_mat(flip)

            ggarray2, _, grasp_features2, sinput2 = self.get_grasp(
                depth_map,
                augment_mat,
                flip=flip,
            )

            if ggarray2 is None:
                continue

            sinput.append(sinput2[0])
            ggarray = torch.cat([ggarray, ggarray2], axis=0)
            grasp_features = torch.cat([grasp_features, grasp_features2], axis=0)

        return ggarray, points_down, grasp_features, sinput
