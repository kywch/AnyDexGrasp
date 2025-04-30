import os
import copy
import json
import time
import argparse
import datetime
from collections import OrderedDict

import torch
import numpy as np
from scipy.spatial.transform import Rotation

import cv2
import open3d as o3d
from PIL import Image

from graspnetAPI import GraspGroup
import MinkowskiEngine as ME

import robosuite.utils.camera_utils as CU

from AnyDexGrasp.models.minkowski_graspnet import MinkowskiGraspNet
from AnyDexGrasp.utils.pt_utils import batch_viewpoint_params_to_matrix
from AnyDexGrasp.utils.collision_detector import ModelFreeCollisionDetectorMultifinger
from AnyDexGrasp.utils.np_utils import transform_point_cloud

from robosuite_env import make_robosuite_env, RobosuiteCameraInfo, execute_grasp
from InspireHandR_grasp import GRASP_TYPES, InspireHandRGraspGroup

DEBUG = False

MAX_GRASP_WIDTH = 0.1
MIN_GRASP_WIDTH = 0.02
BATCH_SIZE = 1
CALIB = False
GRIPPER_TOTAL_LEN = 0.155
FLANGE_TOTAL_LEN = 0.055
NUM_OF_INSPIREHAND_DEPTH = 8
NUM_OF_INSPIREHAND_TYPE = 4
INSPIREHANDR_VOXEL_GRID = 0.002
POINTCLOUD_AUGMENT_NUM = 10

GRAB_SITE_OFFSET = np.array([-0.01, 0, 0])  # in the grip frame


def parse_preds(end_points, use_v2=True):
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
    grasp_widths = MAX_GRASP_WIDTH * end_points["stage3_normalized_grasp_widths"]  # (B, Ns, A, D)
    grasp_widths[grasp_widths > MAX_GRASP_WIDTH] = MAX_GRASP_WIDTH

    grasp_preds = []
    grasp_features = []
    grasp_vdistance_list = []
    for i in range(BATCH_SIZE):
        cloud_mask_i = coords[:, 0] == i
        seed_inds_i = seed_inds[i]
        objectness_mask_i = objectness_mask[cloud_mask_i][seed_inds_i]  # (Ns,)

        if objectness_mask_i.any() == False:
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


class GraspNetRunner:
    def __init__(
        self,
        camera,
        graspnet_path,
        num_depths=5,
        num_seed=2048,
        half_views=False,
    ):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._load_graspnet(graspnet_path, num_depths, num_seed, half_views)

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
            preds, grasp_features = parse_preds(end_points)
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
        # mask = (preds[:, 9] > 0.85) & (preds[:, 1] < MAX_GRASP_WIDTH) & (preds[:, 1] > MIN_GRASP_WIDTH)

        # The second mask preserves the grasp poses within the workspace of the robot.
        # workspace_mask = (preds[:, 12] > -0.25) & (preds[:, 12] < 0.25) & (preds[:, 13] > -0.20) & (preds[:, 13] < 0.05)

        # NOTE: The below is from robot_inspire.py  ... A bit different
        mask = (preds[:, 9] > 0.92) & (preds[:, 1] < MAX_GRASP_WIDTH) & (preds[:, 1] > MIN_GRASP_WIDTH)
        workspace_mask = (
            (preds[:, 12] > -0.25) & (preds[:, 12] < 0.25) & (preds[:, 13] > -0.205) & (preds[:, 13] < 0.03)
        )

        preds = preds[workspace_mask & mask]
        grasp_features = grasp_features[0][workspace_mask & mask]
        if len(preds) == 0:
            print("No grasp detected after masking")
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


def save_grasp_information(
    two_fingers_ggarray,
    two_fingers_ggarray_object_ids,
    InspireHandR_ggarray,
    two_fingers_ggarray_source,
    InspireHandR_ggarray_source,
    two_fingers_ggarray_object_ids_source,
    two_fingers_grasp_used,
    InspireHandR_grasp_used,
    grasp_features_used,
    mat_pose,
    colors_saved,
    depths_saved,
    grasp_features,
    two_fingers_source_grasp_features,
    before_collision,
    after_collision,
):
    save_path = cfgs.save_information_path

    timeStamp = datetime.datetime.now().timestamp()
    timeArray = time.localtime(timeStamp)
    otherStyleTime = time.strftime("%Y-%m-%d_%H-%M-%S", timeArray)
    save_path = os.path.join(
        save_path, GRASP_TYPES[str(int(InspireHandR_grasp_used.grasp_type))]["name"], otherStyleTime
    )

    information = OrderedDict()
    two_fingers_ggarray_proposals = []
    InspireHandR_ggarray_proposals = []
    for idx, tfg in enumerate(two_fingers_ggarray):
        two_fingers_ggarray_proposals.append(
            [float(tfg.score), float(tfg.width), float(tfg.height), float(tfg.depth)]
            + np.array(tfg.rotation_matrix).reshape((-1)).tolist()
            + list(tfg.translation.tolist())
            + [float(two_fingers_ggarray_object_ids[idx])]
        )
        InspireHandR_ggarray_proposals.append(list(InspireHandR_ggarray[idx].get_array_grasp()))

    two_fingers_ggarray_source_saved = []
    InspireHandR_ggarray_source_saved = []
    for idx, tfg in enumerate(two_fingers_ggarray_source):
        two_fingers_ggarray_source_saved.append(
            [float(tfg.score), float(tfg.width), float(tfg.height), float(tfg.depth)]
            + np.array(tfg.rotation_matrix).reshape((-1)).tolist()
            + list(tfg.translation.tolist())
            + [float(two_fingers_ggarray_object_ids_source[idx])]
        )
        InspireHandR_ggarray_source_saved.append(list(InspireHandR_ggarray_source[idx].get_array_grasp()))

    tfg = two_fingers_grasp_used
    two_fingers_array = (
        [float(tfg.score), float(tfg.width), float(tfg.height), float(tfg.depth)]
        + np.array(tfg.rotation_matrix).reshape((-1)).tolist()
        + list(tfg.translation.reshape(-1).tolist())
        + [float(InspireHandR_grasp_used.object_id)]
    )
    grasp_features_used = get_grasp_features(grasp_features_used)
    information["two_fingers_pose"] = list(two_fingers_array)
    information["InspiredHandR_pose"] = list(InspireHandR_grasp_used.get_array_grasp())
    information["two_fingers_pose_angle_type"] = grasp_features_used["grasp_angles"]
    information["two_fingers_pose_depth_type"] = grasp_features_used["grasp_depths"]

    information["two_fingers_pose_AD"] = grasp_features_used["stage3_grasp_scores"]
    information["grasp_preds_features"] = grasp_features_used["grasp_preds_features"]
    information["two_fingers_pose_features"] = grasp_features_used["stage3_grasp_features"]
    information["two_fingers_pose_features_before_generator"] = grasp_features_used["before_generator"]
    information["point_features"] = grasp_features_used["point_features"]

    information["point_id"] = grasp_features_used["point_id"]
    information["if_flip"] = grasp_features_used["if_flip"]
    information["InspiredHandR_pose_finger_type"] = int(InspireHandR_grasp_used.grasp_type + 0.1)
    information["InspiredHandR_pose_depth_type"] = (
        int(InspireHandR_grasp_used.depth * 100 + 0.1) - grasp_features_used["grasp_depths"]
    )
    information["before_collision"] = before_collision
    information["after_collision"] = after_collision
    information["base_2_tcp1"] = np.array(mat_pose[0]).tolist()
    information["base_2_tcp1_backup"] = np.array(mat_pose[1]).tolist()
    information["tcp_2_gripper"] = np.array(mat_pose[2]).tolist()
    information["base_2_TwoFingersGripper_pose"] = np.array(mat_pose[3]).tolist()
    information["tcp_2_camera"] = np.array(mat_pose[4]).tolist()
    information["base_2_tcp_ready"] = np.array(mat_pose[5]).tolist()

    information["camera_internal"] = [[631.119, 363.884], [919.835, 919.61]]

    # NOTE: is this manual?
    restart = False
    print(
        "Is the grasping successful? press 1 successfully, press 2 failed, restart grasping and press 3, exit press 4\n"
    )
    if_success = input("The result is: ")
    while True:
        if if_success == "1":
            information["result"] = True
            break
        elif if_success == "2":
            information["result"] = False
            break
        elif if_success == "3":
            restart = True
            break
        elif if_success == "4":
            exit()
        else:
            if_success = input("Re-enter the result: ")
    if restart:
        return
    information["two_fingers_ggarray_proposals"] = two_fingers_ggarray_proposals
    information["InspireHandR_ggarray_proposals"] = InspireHandR_ggarray_proposals
    information["two_fingers_ggarray_informations_proposals"] = np.array(grasp_features).tolist()

    information["two_fingers_ggarray_source"] = two_fingers_ggarray_source_saved
    information["InspireHandR_ggarray_source_saved"] = InspireHandR_ggarray_source_saved

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    np.save(os.path.join(save_path, "two_fingers_ggarray_informations_source.npy"), two_fingers_source_grasp_features)

    color_path = os.path.join(save_path, "color.png")
    depth_path = os.path.join(save_path, "depth.png")

    cv2.imwrite(color_path, (cv2.cvtColor(colors_saved, cv2.COLOR_RGB2BGR) * 255.0).astype(np.float32))
    cv2.imwrite(depth_path, depths_saved)
    json_path = os.path.join(save_path, "information.json")

    json_file = json.dumps(information, indent=4)
    with open(json_path, "w") as handle:
        handle.write(json_file)
    print("Saved successfully")


def load_meshes_pointcloud(path):
    meshes_pcls = dict()
    meshes_pcl_path = os.path.join(
        path, "meshes/source_pointclouds/voxel_size_" + str(int(INSPIREHANDR_VOXEL_GRID * 1000))
    )
    for type in os.listdir(meshes_pcl_path):
        type_path = os.path.join(meshes_pcl_path, type)
        for name in os.listdir(type_path):
            width = name[:-4]
            name_path = os.path.join(type_path, name)
            meshes_pcl = o3d.io.read_point_cloud(name_path)
            meshes_pcls[type + "_" + width] = meshes_pcl
    return meshes_pcls


# def robot_grasp(cfgs):
#     net = get_net(cfgs.checkpoint_path)
#     robot = get_robot(cfgs.robot_ip, robot_debug=True, gripper_type="InspireHandR", global_cam=cfgs.global_camera)
#     fail = 0
#     existing_shm_color = shared_memory.SharedMemory(name="realsense_color")
#     existing_shm_depth = shared_memory.SharedMemory(name="realsense_depth")
#     meshes_pcls = load_meshes_pointcloud(cfgs.inspire_mesh_json_path)

#     try:
#         v = 0.01
#         a = 0.01
#         if cfgs.global_camera:
#             robot.movej(
#                 robot.throwj2, acc=a * 2, vel=v * 3
#             )  # this v and a are anguler, so it should be larger than translational
#             t1 = time.time()
#             depths = get_depth(existing_shm_depth)
#             depths_saved = copy.deepcopy(depths)
#             colors_saved = np.copy(np.ndarray((720, 1280, 3), dtype=np.float32, buffer=existing_shm_color.buf))
#             ggarray, cloud, points_down, grasp_features = get_grasp(net, depths, existing_shm_color)
#             t3 = time.time()
#             print(f"Net Time:{t3 - t1}")

#         else:
#             robot.movel(
#                 robot.ready_pose(), acc=a * 2, vel=v * 3
#             )  # this v and a are anguler, so it should be larger than translational

#         while True:
#             if not cfgs.global_camera:
#                 t1 = time.time()
#                 robot.movel(
#                     robot.ready_pose(), acc=a * 10, vel=v * 10, wait=True
#                 )  # this v and a are anguler, so it should be larger than translational
#                 time.sleep(0.5)
#                 print("movel")
#                 depths = get_depth(existing_shm_depth)
#                 depths_saved = copy.deepcopy(depths)
#                 colors_saved = np.copy(np.ndarray((720, 1280, 3), dtype=np.float32, buffer=existing_shm_color.buf))
#                 ggarray, cloud, points_down, grasp_features = get_grasp(net, depths, existing_shm_color)
#                 t3 = time.time()
#                 print(f"Net Time:{t3 - t1}")

#             if ggarray is None:
#                 fail = fail + 1
#                 if not cfgs.global_camera:
#                     while robot.is_program_running():
#                         robot.stopj(acc=10.0 * a)
#                     robot.movel(
#                         robot.ready_pose(), acc=a * 10, vel=v * 10
#                     )  # this v and a are anguler, so it should be larger than translational
#                     time.sleep(0.1)
#                 else:
#                     depths = get_depth(existing_shm_depth)
#                     ggarray, cloud, points_down, grasp_features = get_grasp(net, depths, existing_shm_color)
#                     time.sleep(0.1)
#                 if DEBUG:
#                     frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
#                     sphere = o3d.geometry.TriangleMesh.create_sphere(0.002, 20).translate([0, 0, 0.490])
#                     o3d.visualization.draw_geometries([cloud, frame, sphere])
#                 continue

#             ########## PROCESS GRASPS ##########
#             # collision detection
#             ggarray = ggarray.cpu().numpy()
#             source_index = ggarray[:, 0].argsort()
#             ggarray = ggarray[source_index][::-1][:500]

#             # Prevent the robot arm from crossing the border,
#             ggarray, if_flip = flip_ggarray(ggarray)

#             grasp_features = grasp_features.cpu().numpy()
#             two_fingers_source_grasp_features = copy.deepcopy(grasp_features)
#             grasp_features = grasp_features[source_index][::-1][:500]
#             grasp_features = np.c_[grasp_features, if_flip]
#             assert cfgs.grasp_type >= 1 and cfgs.grasp_type <= 8
#             InspireHandR_types = np.random.randint(cfgs.grasp_type, cfgs.grasp_type + 1, (len(ggarray),))
#             InspireHandR_depths = (np.random.randint(0, 4, (len(ggarray),))) * 0.01
#             for ids, grasp_type in enumerate(InspireHandR_types):
#                 if grasp_type == 5:
#                     InspireHandR_depths[ids] = InspireHandR_depths[ids] + 0.02

#             two_fingers_ggarray = GraspGroup(ggarray)
#             two_fingers_ggarray_object_ids_source = ggarray[:, 16]
#             two_fingers_ggarray_source = copy.deepcopy(two_fingers_ggarray)

#             InspireHandR_ggarray = InspireHandRGraspGroup()
#             InspireHandR_ggarray.set_grasp_min_width(MIN_GRASP_WIDTH)
#             InspireHandR_ggarray.from_graspgroup(two_fingers_ggarray, InspireHandR_types, cfgs.inspire_mesh_json_path)
#             InspireHandR_ggarray.object_ids = two_fingers_ggarray_object_ids_source
#             InspireHandR_ggarray.depths = InspireHandR_ggarray.depths + InspireHandR_depths
#             InspireHandR_ggarray_source = copy.deepcopy(InspireHandR_ggarray)

#             index_filter_by_z_axis = InspireHandR_ggarray.filter_grasp_group_by_z_axis(0.4)
#             two_fingers_ggarray = two_fingers_ggarray[index_filter_by_z_axis]
#             two_fingers_ggarray_object_ids = two_fingers_ggarray_object_ids_source[index_filter_by_z_axis]
#             grasp_features = grasp_features[index_filter_by_z_axis]

#             before_collision = [
#                 copy.deepcopy(np.array(InspireHandR_ggarray.grasp_group_array).tolist()),
#                 copy.deepcopy(np.array(two_fingers_ggarray.grasp_group_array).tolist()),
#                 copy.deepcopy(np.array(grasp_features).tolist()),
#             ]
#             if len(InspireHandR_ggarray) == 0:
#                 print("No grasp detected after filter")
#                 if cfgs.global_camera:
#                     ggarray, cloud, points_down, grasp_features = get_grasp(net, depths, existing_shm_color)
#                 if DEBUG:
#                     frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
#                     sphere = o3d.geometry.TriangleMesh.create_sphere(0.002, 20).translate([0, 0, 0.490])
#                     o3d.visualization.draw_geometries([cloud, frame, sphere])
#                 continue

#             start_time = time.time()
#             approach_distance = 0.04
#             mfcdetector = ModelFreeCollisionDetectorInspireHandR(points_down.cpu().numpy(), voxel_size=0.001)
#             InspireHandR_ggarray, two_fingers_ggarray, empty_mask, min_width_index = mfcdetector.detect(
#                 InspireHandR_ggarray,
#                 two_fingers_ggarray,
#                 cfgs.inspire_mesh_json_path,
#                 meshes_pcls,
#                 min_grasp_width=MIN_GRASP_WIDTH,
#                 VoxelGrid=INSPIREHANDR_VOXEL_GRID,
#                 DEBUG=False,
#                 approach_dist=approach_distance,
#                 collision_thresh=1,
#                 adjust_gripper_centers=False,
#             )

#             # proposals
#             InspireHandR_ggarray = InspireHandR_ggarray[empty_mask]
#             two_fingers_ggarray = two_fingers_ggarray[empty_mask]
#             two_fingers_ggarray_object_ids = two_fingers_ggarray_object_ids[min_width_index][empty_mask]
#             grasp_features = grasp_features[min_width_index][empty_mask]

#             after_collision = [
#                 copy.deepcopy(np.array(InspireHandR_ggarray.grasp_group_array).tolist()),
#                 copy.deepcopy(np.array(two_fingers_ggarray.grasp_group_array).tolist()),
#                 copy.deepcopy(np.array(grasp_features).tolist()),
#             ]

#             if len(InspireHandR_ggarray) == 0:
#                 print("No Grasp detected after collision detection!")
#                 fail = fail + 1
#                 if DEBUG:
#                     frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
#                     sphere = o3d.geometry.TriangleMesh.create_sphere(0.002, 20).translate([0, 0, 0.490])
#                     o3d.visualization.draw_geometries([cloud, frame, sphere])
#                 if not cfgs.global_camera:
#                     while robot.is_program_running():
#                         robot.stopj(acc=10.0 * a)
#                     robot.movel(
#                         robot.ready_pose(), acc=a * 10, vel=v * 10
#                     )  # this v and a are anguler, so it should be larger than translational
#                 else:
#                     t1 = time.time()
#                     depths = get_depth(existing_shm_depth)
#                     depths_saved = copy.deepcopy(depths)
#                     colors_saved = np.copy(np.ndarray((720, 1280, 3), dtype=np.float32, buffer=existing_shm_color.buf))
#                     ggarray, cloud, points_down, grasp_features = get_grasp(net, depths, existing_shm_color)
#                     t3 = time.time()
#                     print(f"Net Time:{t3 - t1}")
#                 continue

#             # sort
#             index_score = np.argsort(InspireHandR_ggarray.scores)[::-1]
#             two_fingers_ggarray_object_ids = two_fingers_ggarray_object_ids[index_score][0:10]
#             InspireHandR_ggarray = InspireHandR_ggarray[index_score][0:10]
#             two_fingers_ggarray = two_fingers_ggarray[index_score][0:10]
#             grasp_features = grasp_features[index_score][0:10]

#             InspireHandR_grasp_used = InspireHandR_ggarray[0]
#             two_fingers_grasp_used = two_fingers_ggarray[0]
#             grasp_features_used = grasp_features[0]

#             print(
#                 "picked by scores rotations, translations: ",
#                 InspireHandR_grasp_used.rotation_matrix,
#                 InspireHandR_grasp_used.translation,
#                 two_fingers_grasp_used.translation,
#                 two_fingers_grasp_used.rotation_matrix,
#             )
#             print("grasp score:", InspireHandR_grasp_used.score, two_fingers_grasp_used.score)
#             print("grasp width:", InspireHandR_grasp_used.width, two_fingers_grasp_used.width)
#             print("grasp depth:", InspireHandR_grasp_used.depth, two_fingers_grasp_used.depth)
#             print("grasp type:", InspireHandR_grasp_used.grasp_type)
#             print("grasp angle:", InspireHandR_grasp_used.angle)
#             t4 = time.time()

#             ####################################
#             if DEBUG:
#                 InspireHandR_pose = InspireHandR_grasp_used.load_mesh(
#                     cfgs.inspire_mesh_json_path, two_fingers_grasp_used
#                 )
#                 frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
#                 sphere = o3d.geometry.TriangleMesh.create_sphere(0.002, 20).translate([0, 0, 0.490])
#                 InspireHandR_pose.paint_uniform_color([1, 0, 0])
#                 meshes_pointclouds = InspireHandR_grasp_used.load_mesh_pointclouds(
#                     cfgs.inspire_mesh_json_path, two_fingers_grasp_used, voxel_size=0.002
#                 )
#                 voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(input=meshes_pointclouds, voxel_size=0.002)
#                 scene_cloud = o3d.geometry.PointCloud()
#                 scene_cloud.points = o3d.utility.Vector3dVector(points_down.cpu().numpy())
#                 scene_cloud = scene_cloud.voxel_down_sample(0.001)
#                 ps = scene_cloud.points
#                 ps = o3d.utility.Vector3dVector(ps)
#                 output = voxel_grid.check_if_included(ps)
#                 o3d.visualization.draw_geometries(
#                     [InspireHandR_pose, cloud, sphere, frame, two_fingers_grasp_used.to_open3d_geometry()]
#                 )
#             gripper_time = 0.5
#             print("angle: ", InspireHandR_grasp_used.angle)
#             robot.open_gripper(InspireHandR_grasp_used.angle)
#             mat_pose = robot.grasp_and_throw(
#                 InspireHandR_grasp_used,
#                 two_fingers_grasp_used,
#                 cloud,
#                 cfgs.inspire_mesh_json_path,
#                 acc=a * 2,
#                 vel=v * 3,
#                 approach_dist=approach_distance,
#                 execute_grasp=True,
#                 use_ready_pose=True,
#                 gripper_time=gripper_time,
#             )
#             while robot.is_program_running():
#                 pass

#             t45 = time.time()

#             save_grasp_information(
#                 two_fingers_ggarray,
#                 two_fingers_ggarray_object_ids,
#                 InspireHandR_ggarray,
#                 two_fingers_ggarray_source,
#                 InspireHandR_ggarray_source,
#                 two_fingers_ggarray_object_ids_source,
#                 two_fingers_grasp_used,
#                 InspireHandR_grasp_used,
#                 grasp_features_used,
#                 mat_pose,
#                 colors_saved,
#                 depths_saved,
#                 grasp_features,
#                 two_fingers_source_grasp_features,
#                 before_collision,
#                 after_collision,
#             )
#             if cfgs.global_camera:
#                 t1 = time.time()
#                 depths = get_depth(existing_shm_depth)
#                 depths_saved = copy.deepcopy(depths)
#                 colors_saved = np.copy(np.ndarray((720, 1280, 3), dtype=np.float32, buffer=existing_shm_color.buf))
#                 ggarray, cloud, points_down, grasp_features = get_grasp(net, depths, existing_shm_color)
#                 t3 = time.time()
#                 print(f"Net Time:{t3 - t1}")

#             t5 = time.time()
#             print(f"ready time:{t5 - t45}")
#             print(f"Exec Time:{t5 - t4}")
#             mpph = 3600 / (t5 - t1)
#             print(f"\033[1;31mMPPH:{mpph}\033[0m\n--------------------")
#     finally:
#         robot.close()
#         existing_shm_depth.close()
#         if DEBUG:
# existing_shm_color.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Using the downloaded checkpoint
    parser.add_argument("--checkpoint_path", default="logs/model/checkpoint.tar.18", help="Model checkpoint path")
    parser.add_argument(
        "--save_information_path",
        default="logs/data/decision_model/inspire/obj140/collect",
        help="inspire model result information path",
    )
    parser.add_argument(
        "--inspire_mesh_json_path",
        default="generate_mesh_and_pointcloud/inspire_urdf",
        help="InspireHandR meshes and json path",  # generated
    )
    parser.add_argument("--grasp_type", default=8, help="the data of Inspire grasp type, 1~8")
    # parser.add_argument('--robot_ip', required=True, help='Robot IP')
    parser.add_argument("--use_graspnet_v2", default=True, help="Whether to use graspnet v2 format")
    # parser.add_argument("--half_views", action="store_true", help="Use only half views in network.")
    parser.add_argument("--global_camera", action="store_true", help="Use the settings for global camera.")
    cfgs = parser.parse_args()

    assert cfgs.grasp_type >= 1 and cfgs.grasp_type <= 8

    # Contact-centric grasp representation
    meshes_pcls = load_meshes_pointcloud(cfgs.inspire_mesh_json_path)

    # Setup env
    camera_name = "robot0_eye_in_hand"
    robot_env = make_robosuite_env(task="Lift", camera_name=camera_name)
    obs_dict = robot_env.reset()

    camera = RobosuiteCameraInfo(robot_env.sim, camera_name, camera_height=720, camera_width=1280)
    camera.camera_to_world_mat
    graspnet_runner = GraspNetRunner(camera, cfgs.checkpoint_path)

    # The initial pose
    ref_id = robot_env.sim.model.site_name2id("gripper0_right_grip_site")
    eef_pos = robot_env.sim.data.site_xpos[ref_id]
    eef_ori_mat = robot_env.sim.data.site_xmat[ref_id].reshape((3, 3))
    eef_ori_aa = Rotation.from_matrix(eef_ori_mat).as_rotvec()

    search_pose = np.zeros(7)  # OSC_POSE
    search_pose[:3] = eef_pos + np.array([0.10, -0.20, 0.10])
    search_pose[3:6] = eef_ori_aa
    search_pose[6] = 1

    # Get the hand out of the camera view
    for _ in range(50):
        obs_dict, _, _, _ = robot_env.step(search_pose)

    # check the image
    Image.fromarray(obs_dict["{}_image".format(camera_name)][::-1]).show()

    input("Press Enter to continue...")

    while True:
        # Running the grasp net
        color_map = obs_dict["{}_image".format(camera_name)][::-1] / 255.0
        depth_map = CU.get_real_depth_map(
            sim=robot_env.sim, depth_map=obs_dict["{}_depth".format(camera_name)][::-1]
        ).squeeze()

        camera.update_mappings()  # Check camera.camera_to_world_mat
        ggarray, points_down, grasp_features, _ = graspnet_runner.get_grasp(depth_map)

        if DEBUG:
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
            sphere = o3d.geometry.TriangleMesh.create_sphere(0.002, 20).translate([0, 0, 0.490])
            scene_cloud = o3d.geometry.PointCloud()
            scene_cloud.points = o3d.utility.Vector3dVector(points_down.cpu().numpy())
            scene_cloud = scene_cloud.voxel_down_sample(0.001)
            scene_cloud.paint_uniform_color([0.8, 0.8, 0.8])
            o3d.visualization.draw_plotly([scene_cloud, frame, sphere], width=1024, height=640)

        if ggarray is None or len(ggarray) == 0:
            print("No grasp detected this. Trying again")
            continue

        ########## PROCESS GRASPS ##########
        # collision detection
        ggarray = ggarray.cpu().numpy()
        source_index = ggarray[:, 0].argsort()
        ggarray = ggarray[source_index][::-1][:500]

        # Prevent the robot arm from crossing the border,
        # ggarray, if_flip = flip_ggarray(ggarray)
        if_flip = [False for _ in range(len(ggarray))]

        grasp_features = grasp_features.cpu().numpy()
        two_fingers_source_grasp_features = copy.deepcopy(grasp_features)
        grasp_features = grasp_features[source_index][::-1][:500]
        grasp_features = np.c_[grasp_features, if_flip]

        # Randomize depths, BUT NOT grasp type, for data collection
        # InspireHandR_types = np.random.randint(cfgs.grasp_type, cfgs.grasp_type + 1, (len(ggarray),))
        InspireHandR_types = np.array([cfgs.grasp_type] * len(ggarray))
        InspireHandR_depths = (np.random.randint(0, 4, (len(ggarray),))) * 0.01
        for ids, grasp_type in enumerate(InspireHandR_types):
            if grasp_type == 5:  # Medium_Wrap
                InspireHandR_depths[ids] = InspireHandR_depths[ids] + 0.02

        # graspnet returns two finger grasps
        two_fingers_ggarray = GraspGroup(ggarray)
        two_fingers_ggarray_object_ids_source = ggarray[:, 16]
        two_fingers_ggarray_source = copy.deepcopy(two_fingers_ggarray)

        InspireHandR_ggarray = InspireHandRGraspGroup()
        InspireHandR_ggarray.set_grasp_min_width(MIN_GRASP_WIDTH)
        InspireHandR_ggarray.from_graspgroup(two_fingers_ggarray, InspireHandR_types, cfgs.inspire_mesh_json_path)
        InspireHandR_ggarray.object_ids = two_fingers_ggarray_object_ids_source
        InspireHandR_ggarray.depths = InspireHandR_ggarray.depths + InspireHandR_depths
        InspireHandR_ggarray_source = copy.deepcopy(InspireHandR_ggarray)

        index_filter_by_z_axis = InspireHandR_ggarray.filter_grasp_group_by_z_axis(0.4)
        two_fingers_ggarray = two_fingers_ggarray[index_filter_by_z_axis]
        two_fingers_ggarray_object_ids = two_fingers_ggarray_object_ids_source[index_filter_by_z_axis]
        grasp_features = grasp_features[index_filter_by_z_axis]

        before_collision = [
            copy.deepcopy(np.array(InspireHandR_ggarray.grasp_group_array).tolist()),
            copy.deepcopy(np.array(two_fingers_ggarray.grasp_group_array).tolist()),
            copy.deepcopy(np.array(grasp_features).tolist()),
        ]
        if len(InspireHandR_ggarray) == 0:
            print("No grasp detected after filter")
            continue

        # start_time = time.time()
        approach_distance = 0.04
        mfcdetector = ModelFreeCollisionDetectorMultifinger(points_down.cpu().numpy(), voxel_size=0.001)
        InspireHandR_ggarray, two_fingers_ggarray, empty_mask, min_width_index = mfcdetector.detect(
            InspireHandR_ggarray,
            two_fingers_ggarray,
            cfgs.inspire_mesh_json_path,
            meshes_pcls,
            min_grasp_width=MIN_GRASP_WIDTH,
            VoxelGrid=INSPIREHANDR_VOXEL_GRID,
            approach_dist=approach_distance,
            collision_thresh=1,
            adjust_gripper_centers=False,
            DEBUG=False,
        )

        # proposals
        InspireHandR_ggarray = InspireHandR_ggarray[empty_mask]
        two_fingers_ggarray = two_fingers_ggarray[empty_mask]
        two_fingers_ggarray_object_ids = two_fingers_ggarray_object_ids[min_width_index][empty_mask]
        grasp_features = grasp_features[min_width_index][empty_mask]

        after_collision = [
            copy.deepcopy(np.array(InspireHandR_ggarray.grasp_group_array).tolist()),
            copy.deepcopy(np.array(two_fingers_ggarray.grasp_group_array).tolist()),
            copy.deepcopy(np.array(grasp_features).tolist()),
        ]

        if len(InspireHandR_ggarray) == 0:
            print("No grasp detected after collision detection")
            continue

        break

    # sort
    index_score = np.argsort(InspireHandR_ggarray.scores)[::-1]
    two_fingers_ggarray_object_ids = two_fingers_ggarray_object_ids[index_score][0:10]
    InspireHandR_ggarray = InspireHandR_ggarray[index_score][0:10]
    two_fingers_ggarray = two_fingers_ggarray[index_score][0:10]
    grasp_features = grasp_features[index_score][0:10]

    InspireHandR_grasp_used = InspireHandR_ggarray[0]
    two_fingers_grasp_used = two_fingers_ggarray[0]
    grasp_features_used = grasp_features[0]

    print(
        "picked by scores rotations, translations: ",
        InspireHandR_grasp_used.rotation_matrix,
        InspireHandR_grasp_used.translation,
        two_fingers_grasp_used.translation,
        two_fingers_grasp_used.rotation_matrix,
    )
    print("grasp score:", InspireHandR_grasp_used.score, two_fingers_grasp_used.score)
    print("grasp width:", InspireHandR_grasp_used.width, two_fingers_grasp_used.width)
    print("grasp depth:", InspireHandR_grasp_used.depth, two_fingers_grasp_used.depth)
    print("grasp type:", InspireHandR_grasp_used.grasp_type)
    print("grasp angle:", InspireHandR_grasp_used.angle)

    if DEBUG:
        InspireHandR_pose = InspireHandR_grasp_used.load_mesh(cfgs.inspire_mesh_json_path, two_fingers_grasp_used)
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
        sphere = o3d.geometry.TriangleMesh.create_sphere(0.002, 20).translate([0, 0, 0.490])
        InspireHandR_pose.paint_uniform_color([1, 0, 0])
        meshes_pointclouds = InspireHandR_grasp_used.load_mesh_pointclouds(
            cfgs.inspire_mesh_json_path, two_fingers_grasp_used, voxel_size=0.002
        )
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(input=meshes_pointclouds, voxel_size=0.002)
        scene_cloud = o3d.geometry.PointCloud()
        scene_cloud.points = o3d.utility.Vector3dVector(points_down.cpu().numpy())
        scene_cloud = scene_cloud.voxel_down_sample(0.001)
        scene_cloud.paint_uniform_color([0.8, 0.8, 0.8])
        ps = scene_cloud.points
        ps = o3d.utility.Vector3dVector(ps)
        output = voxel_grid.check_if_included(ps)
        # o3d.visualization.draw_plotly(
        #     [InspireHandR_pose, scene_cloud, sphere, frame, two_fingers_grasp_used.to_open3d_geometry()],
        #     width=1024, height=640
        # )

        # transformation
        geoms = [InspireHandR_pose, scene_cloud, sphere, frame, two_fingers_grasp_used.to_open3d_geometry()]

        transformed = []
        for geom in geoms:
            geom_copy = copy.deepcopy(geom)
            geom_copy.transform(camera.camera_to_world_mat)
            transformed.append(geom_copy)

        o3d.visualization.draw_plotly(transformed, width=1024, height=640)

    execute_grasp(robot_env, camera, InspireHandR_grasp_used, two_fingers_grasp_used, grab_site_offset=GRAB_SITE_OFFSET)

    # Below is the data required to train the grasp decision models
    """
    end_points = {}
    end_points["two_fingers_pose_depth_type"] = batch_data_label["two_fingers_pose_depth_type"]
    end_points["multifinger_pose_finger_type"] = batch_data_label["multifinger_pose_finger_type"]
    end_points["multifinger_pose_depth_type"] = batch_data_label["multifinger_pose_depth_type"]
    end_points["grasp_preds_features"] = batch_data_label["grasp_preds_features"]
    end_points["if_flip"] = batch_data_label["if_flip"]
    end_points["result"] = batch_data_label["result"]  # Grasp success 1, fail 0
    """

    # TODO: after executing grasp, save relevant information for training
    # Repeat this for 1000 trials per grasp type
    # The env should have multiple objects with a clean-up task
    # Sampling diverse geometry is the key

    print()

    # pass

    # t0 = time.time()
    # try:
    #     robot_grasp(cfgs)
    # finally:
    #     tn = time.time()
    #     print(f"total time:{tn - t0}")
