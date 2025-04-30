import os
import sys
import copy
import json
import time
import random
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

import robosuite.utils.camera_utils as CU

from AnyDexGrasp.models import minkowski_graspnet
from AnyDexGrasp.utils.collision_detector import ModelFreeCollisionDetectorMultifinger

from robosuite_env import make_robosuite_env, RobosuiteCameraInfo, execute_grasp
from InspireHandR_grasp import InspireHandRGraspGroup

# TODO: pull out shared functions?
from collect_rs_inspire_grasp_data import flip_ggarray, get_grasp_features, GraspNetRunner, load_meshes_pointcloud

# NOTE: This is to fix the error in get_inspire_model, torch.load()
sys.modules["minkowski_graspnet"] = minkowski_graspnet


DEBUG = False
RANDOM_GRASP = False

MAX_GRASP_WIDTH = 0.1
MIN_GRASP_WIDTH = 0.01
BATCH_SIZE = 1
CALIB = False
GRIPPER_TOTAL_LEN = 0.155
FLANGE_TOTAL_LEN = 0.055
INSPIREHANDR_DEFAULT_DEPTH = 0.000
NUM_OF_INSPIRE_DEPTH = 4
NUM_OF_INSPIRE_TYPE = 8
POINTCLOUD_AUGMENT_NUM = 10
INSPIREHANDR_VOXEL_GRID = 0.003

GRAB_SITE_OFFSET = np.zeros(3)  # np.array([-0.01, 0.02, 0])  # in the grip frame


def get_graspgroup_features(grasp_features_array, sinput):
    grasp_features = dict()
    grasp_features["grasp_angles"] = (grasp_features_array[:, -3] + 0.1).astype(int)
    grasp_features["grasp_depths"] = (grasp_features_array[:, -2] * 100 + 0.1).astype(int)
    grasp_features["stage3_grasp_scores"] = grasp_features_array[:, :240]
    grasp_features["grasp_preds_features"] = grasp_features_array[:, 240 : 240 + 480]
    grasp_features["stage3_grasp_features"] = grasp_features_array[:, 240 + 480 : 240 + 480 + 512]
    grasp_features["before_generator"] = grasp_features_array[:, 240 + 480 + 512 : 240 + 480 + 512 + 512]
    grasp_features["point_features"] = grasp_features_array[:, 240 + 480 + 512 + 512 : 240 + 480 + 512 + 512 + 512]
    grasp_features["point_id"] = (grasp_features_array[:, -4] + 0.1).astype(int)
    grasp_features["if_flip"] = grasp_features_array[:, -1]
    grasp_features["view_inds"] = (grasp_features_array[:, -6] + 0.1).astype(int)
    grasp_features["view_score"] = grasp_features_array[:, -5]
    grasp_features["sinput"] = sinput

    grasp_preds_features_rot = np.zeros(grasp_features["grasp_preds_features"].shape, dtype=np.float32)
    new_type = grasp_features["grasp_angles"]
    for idx, if_flip in enumerate(grasp_features["if_flip"]):
        if if_flip:
            first_half_scores = copy.deepcopy(grasp_features["grasp_preds_features"][idx, :120])
            last_half_scores = copy.deepcopy(grasp_features["grasp_preds_features"][idx, 120:240])
            first_half_widths = copy.deepcopy(grasp_features["grasp_preds_features"][idx, 240:360])
            last_half_widths = copy.deepcopy(grasp_features["grasp_preds_features"][idx, 360:480])
            grasp_features["grasp_preds_features"][idx, :120] = last_half_scores
            grasp_features["grasp_preds_features"][idx, 120:240] = first_half_scores
            grasp_features["grasp_preds_features"][idx, 240:360] = last_half_widths
            grasp_features["grasp_preds_features"][idx, 360:480] = first_half_widths
        grasp_preds_features_rot[idx, : 240 - new_type[idx] * 5] = grasp_features["grasp_preds_features"][
            idx, new_type[idx] * 5 : 240
        ]
        grasp_preds_features_rot[idx, 240 - new_type[idx] * 5 : 240] = grasp_features["grasp_preds_features"][
            idx, 0 : new_type[idx] * 5
        ]
        grasp_preds_features_rot[idx, 240 : 480 - new_type[idx] * 5] = grasp_features["grasp_preds_features"][
            idx, 240 + new_type[idx] * 5 : 480
        ]
        grasp_preds_features_rot[idx, 480 - new_type[idx] * 5 : 480] = grasp_features["grasp_preds_features"][
            idx, 240 : 240 + new_type[idx] * 5
        ]
    grasp_features["grasp_preds_features"] = grasp_preds_features_rot

    return grasp_features


# NOTE: A bit different from collect_data's save_grasp_information()
def save_grasp_information(
    two_fingers_ggarray,
    two_fingers_ggarray_object_ids,
    InspireHandR_ggarray,
    two_fingers_ggarray_source,
    InspireHandR_ggarray_source,
    two_fingers_ggarray_object_ids_source,
    two_fingers_grasp_used,
    InspireHandR_grasp_used,
    grasp_informations_used,
    mat_pose,
    colors_saved,
    depths_saved,
    grasp_informations,
):
    save_path = cfgs.save_information_path

    timeStamp = datetime.datetime.now().timestamp()
    timeArray = time.localtime(timeStamp)
    otherStyleTime = time.strftime("%Y-%m-%d_%H-%M-%S", timeArray)
    save_path = os.path.join(
        save_path, grasp_types[str(int(InspireHandR_grasp_used.grasp_type))]["name"], otherStyleTime
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
    information["two_fingers_pose"] = list(two_fingers_array)
    information["InspiredHandR_pose"] = list(InspireHandR_grasp_used.get_array_grasp())
    information["two_fingers_pose_angle_type"] = int(grasp_informations_used[0] + 0.1)
    information["two_fingers_pose_depth_type"] = int(grasp_informations_used[1] * 100 + 0.1)

    information["two_fingers_pose_depth_type"] = grasp_features_used["grasp_depths"]
    information["point_id"] = grasp_features_used["point_id"]
    information["if_flip"] = grasp_features_used["if_flip"]
    information["InspiredHandR_pose_finger_type"] = int(InspireHandR_grasp_used.grasp_type + 0.1)
    information["InspiredHandR_pose_depth_type"] = (
        int(InspireHandR_grasp_used.depth * 100 + 0.1) - grasp_features_used["grasp_depths"]
    )

    information["two_fingers_pose_AD"] = grasp_features_used["stage3_grasp_scores"]
    information["two_fingers_pose_features_grasp_preds"] = grasp_features_used["grasp_preds_features"]
    information["two_fingers_pose_feature_and_AD"] = np.array(grasp_informations_used[2:]).tolist()
    information["two_fingers_pose_features"] = grasp_features_used["stage3_grasp_features"]
    information["two_fingers_pose_features_before_generator"] = grasp_features_used["before_generator"]
    information["point_features"] = grasp_features_used["point_features"]

    information["base_2_tcp1"] = np.array(mat_pose[0]).tolist()
    information["base_2_tcp1_backup"] = np.array(mat_pose[1]).tolist()
    information["tcp_2_gripper"] = np.array(mat_pose[2]).tolist()
    information["base_2_TwoFingersGripper_pose"] = np.array(mat_pose[3]).tolist()
    information["tcp_2_camera"] = np.array(mat_pose[4]).tolist()
    information["base_2_tcp_ready"] = np.array(mat_pose[5]).tolist()

    information["camera_internal"] = [[629.535, 351.636], [912.897, 912.258]]

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    color_path = os.path.join(save_path, "color.png")
    depth_path = os.path.join(save_path, "depth.png")
    from PIL import Image

    cv2.imwrite(color_path, (cv2.cvtColor(colors_saved, cv2.COLOR_RGB2BGR) * 255.0).astype(np.float32))
    cv2.imwrite(depth_path, depths_saved)
    json_path = os.path.join(save_path, "information.json")

    json_file = json.dumps(information, indent=4)
    with open(json_path, "w") as handle:
        handle.write(json_file)
    print("Saved successfully")


def get_inspire_model(inspire_models_path, model_input_dim=480):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    inspire_model_path = os.path.join(inspire_models_path, str(model_input_dim))

    # Sort the keys (grasp types), from 1 to 8
    sub_models = os.listdir(inspire_model_path)
    sub_models.sort()

    inspire_models = dict()
    for model_class in sub_models:
        model_classs_type = os.path.join(inspire_model_path, model_class)
        model_files = os.listdir(model_classs_type)
        inspire_model_type_path = os.path.join(model_classs_type, model_files[0])  # Just use 1 file
        inspire_net = torch.load(inspire_model_type_path)

        inspire_model = minkowski_graspnet.MultifingerGraspSuccessPredictor(input_num=model_input_dim)
        # NOTE: torch.load() needed -- sys.modules["minkowski_graspnet"] = minkowski_graspnet
        inspire_model.load_state_dict(inspire_net.state_dict())
        inspire_model.to(device)
        inspire_model.eval()
        inspire_models[model_class] = inspire_model

    return inspire_models


def get_inspire_depth_type(inspire_models, grasp_features_dic, ggarray, grasp_features):
    if RANDOM_GRASP:
        inspire_type = np.random.randint(0, NUM_OF_INSPIRE_TYPE, (len(ggarray),))
        inspire_depth = np.random.randint(0, 4, (len(ggarray),))
        inspire_depth[inspire_type == 5] = inspire_depth[inspire_type == 5] + 2
        inspire_depth = inspire_depth * 0.01
        scores = ggarray[:, 0]
        return inspire_depth, inspire_type, scores, ggarray, grasp_features

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for k, v in grasp_features_dic.items():
        if k == "point_id" or k == "sinput":
            continue
        grasp_features_dic[k] = torch.tensor(copy.deepcopy(v), device=device)

    inspire_depth_type_scores = []
    # for model_type, inspire_model in inspire_models.items():
    #     if model_type == "240":
    #         continue
    #         model_input = torch.cat([grasp_features_dic["stage3_grasp_scores"]], dim=1)
    #     elif model_type == "480":

    print("Use the models with 480 ...")
    for model_class, sub_inspire_model in inspire_models.items():
        with torch.no_grad():
            grasp_pred = sub_inspire_model(grasp_features_dic["grasp_preds_features"])  # (B, 1， NUM_OF_TWO_FINGER_DEPTH*NUM_OF_INSPIRE_DEPTH)
            grasp_pred = grasp_pred.view(grasp_pred.shape[0], 5 * NUM_OF_INSPIRE_DEPTH)

        two_fingers_depth = grasp_features_dic["grasp_depths"]  # (B, )
        base = torch.tensor(
            np.array([[i for i in range(NUM_OF_INSPIRE_DEPTH)] for _ in range(grasp_pred.size()[0])]), device=device
        )  # (B, NUM_OF_INSPIRE_DEPTH)
        select_index = (two_fingers_depth).view(-1, 1) * NUM_OF_INSPIRE_DEPTH + base
        inspire_depth_type_scores.append(grasp_pred.gather(1, select_index))

    inspire_depth_type_scores = torch.cat(inspire_depth_type_scores, axis=1).view(
        -1
    )  # (B, NUM_OF_INSPIRE_DEPTH*NUM_OF_INSPIRE_TYPE)
    scores, index = inspire_depth_type_scores.topk(min(3000, inspire_depth_type_scores.size()[0]))
    pose_index = (index / (NUM_OF_INSPIRE_DEPTH * NUM_OF_INSPIRE_TYPE)).long()
    ggarray = torch.tensor(copy.deepcopy(ggarray), device=device)[pose_index]
    grasp_features = torch.tensor(copy.deepcopy(grasp_features), device=device)[pose_index]
    inspire_depth = ((index % (NUM_OF_INSPIRE_DEPTH * NUM_OF_INSPIRE_TYPE)) % NUM_OF_INSPIRE_DEPTH).int()
    inspire_type = ((index % (NUM_OF_INSPIRE_DEPTH * NUM_OF_INSPIRE_TYPE)) / NUM_OF_INSPIRE_DEPTH).int()

    # Manual offset for the grasp type 5 (Medium_Wrap)
    inspire_depth[inspire_type == 5] = inspire_depth[inspire_type == 5] + 2
    inspire_depth = inspire_depth * 0.01

    return (
        inspire_depth.cpu().numpy(),
        inspire_type.cpu().numpy() + 1,
        scores.detach().cpu().numpy(),
        ggarray.cpu().numpy(),
        grasp_features.cpu().numpy(),
    )


def select_grasp_type(inspire_gg):
    score_sorted_idx = np.argsort(inspire_gg.scores)[::-1]
    select_gg_types = {i: [] for i in range(1, NUM_OF_INSPIRE_TYPE + 1)}

    for idx in score_sorted_idx:
        grasp_type = int(inspire_gg.grasp_types[idx])
        if grasp_type in [1]:  # Ring grasp has the lowest success rate, so ...
            max_num = 10
        else:
            max_num = 50

        if len(select_gg_types[grasp_type]) < max_num:
            select_gg_types[grasp_type].append(idx)
    gg_type_id = []
    for gg_type in select_gg_types.values():
        gg_type_id += gg_type
    return gg_type_id


# def robot_grasp(cfgs):
#     net = get_net(cfgs.checkpoint_path, use_v2=cfgs.use_graspnet_v2)
#     robot = get_robot(cfgs.robot_ip, robot_debug=True, gripper_type="InspireHandR", global_cam=cfgs.global_camera)
#     inspire_models = get_inspire_model(cfgs.inspire_model_path)
#     fail = 0
#     existing_shm_color = shared_memory.SharedMemory(name="realsense_color")
#     existing_shm_depth = shared_memory.SharedMemory(name="realsense_depth")
#     meshes_pcls = load_meshes_pointcloud(cfgs.inspire_mesh_json_path)

#     try:
#         v = 0.07
#         a = 0.07
#         if cfgs.global_camera:
#             robot.movej(
#                 robot.throwj2, acc=a * 2, vel=v * 3
#             )  # this v and a are anguler, so it should be larger than translational
#             t1 = time.time()
#             depths = get_depth(existing_shm_depth)
#             depths_saved = copy.deepcopy(depths)
#             colors_saved = np.copy(np.ndarray((720, 1280, 3), dtype=np.float32, buffer=existing_shm_color.buf))
#             ggarray, cloud, points_down, grasp_features, sinput = get_ggarray_features(
#                 existing_shm_depth, existing_shm_color, net
#             )
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
#                     robot.ready_pose(), acc=a * 2, vel=v * 3, wait=True
#                 )  # this v and a are anguler, so it should be larger than translational
#                 time.sleep(0.5)
#                 print("movel")
#                 time.sleep(0.7)
#                 depths = get_depth(existing_shm_depth)
#                 depths_saved = copy.deepcopy(depths)
#                 colors_saved = np.copy(np.ndarray((720, 1280, 3), dtype=np.float32, buffer=existing_shm_color.buf))
#                 t1 = time.time()
#                 ggarray, cloud, points_down, grasp_features, sinput = get_ggarray_features(
#                     existing_shm_depth, existing_shm_color, net
#                 )
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
#                     depths_saved = copy.deepcopy(depths)
#                     colors_saved = np.copy(np.ndarray((720, 1280, 3), dtype=np.float32, buffer=existing_shm_color.buf))
#                     ggarray, cloud, points_down, grasp_features, sinput = get_ggarray_features(
#                         existing_shm_depth, existing_shm_color, net
#                     )
#                     time.sleep(0.1)
#                 if DEBUG:
#                     frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
#                     sphere = o3d.geometry.TriangleMesh.create_sphere(0.002, 20).translate([0, 0, 0.490])
#                     o3d.visualization.draw_geometries([cloud, frame, sphere])
#                 continue

#             ########## PROCESS GRASPS ##########
#             # collision detection
#             ggarray = ggarray.cpu().numpy()

#             # Prevent the robot arm from crossing the border,
#             grasp_features = grasp_features.cpu().numpy()
#             two_fingers_source_grasp_features = copy.deepcopy(grasp_features)

#             ggarray, if_flip = flip_ggarray(ggarray)
#             grasp_features = np.c_[grasp_features, if_flip]
#             ggarray = ggarray[~np.array(if_flip)]
#             grasp_features = grasp_features[~np.array(if_flip)]

#             source_index = ggarray[:, 0].argsort()
#             ggarray = ggarray[source_index][::-1][:1000]
#             grasp_features = grasp_features[source_index][::-1][:1000]
#             t_multi = time.time()
#             grasp_features_dic = get_graspgroup_features(grasp_features, sinput)

#             inspire_depth, inspire_type, scores, ggarray, grasp_features = get_inspire_depth_type(
#                 inspire_models, grasp_features_dic, ggarray, grasp_features=grasp_features
#             )

#             if not RANDOM_GRASP:
#                 score_thresh = 0.85
#                 mask = scores > score_thresh
#                 ggarray = ggarray[mask]
#                 grasp_features = grasp_features[mask]
#                 inspire_depth = inspire_depth[mask]
#                 inspire_type = inspire_type[mask]
#                 scores = scores[mask]

#             if len(ggarray) == 0:
#                 print("There is no grasp that score greater than 0.9 ")
#                 if cfgs.global_camera:
#                     depths = get_depth(existing_shm_depth)
#                     depths_saved = copy.deepcopy(depths)
#                     colors_saved = np.copy(np.ndarray((720, 1280, 3), dtype=np.float32, buffer=existing_shm_color.buf))
#                     ggarray, cloud, points_down, grasp_features, sinput = get_ggarray_features(
#                         existing_shm_depth, existing_shm_color, net
#                     )
#                 if DEBUG:
#                     frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
#                     sphere = o3d.geometry.TriangleMesh.create_sphere(0.002, 20).translate([0, 0, 0.490])
#                     o3d.visualization.draw_geometries([cloud, frame, sphere])
#                 continue

#             two_fingers_ggarray = GraspGroup(ggarray)
#             two_fingers_ggarray_object_ids_source = ggarray[:, 16]
#             two_fingers_ggarray_source = copy.deepcopy(two_fingers_ggarray)

#             InspireHandR_ggarray = InspireHandRGraspGroup()
#             InspireHandR_ggarray.set_grasp_min_width(MIN_GRASP_WIDTH)
#             InspireHandR_ggarray.from_graspgroup(two_fingers_ggarray, inspire_type, cfgs.inspire_mesh_json_path)
#             InspireHandR_ggarray.object_ids = two_fingers_ggarray_object_ids_source
#             InspireHandR_ggarray.scores = scores
#             InspireHandR_ggarray.depths = InspireHandR_ggarray.depths + inspire_depth + INSPIREHANDR_DEFAULT_DEPTH
#             InspireHandR_ggarray_source = copy.deepcopy(InspireHandR_ggarray)

#             index_filter_by_z_axis = InspireHandR_ggarray.filter_grasp_group_by_z_axis(0.4)
#             two_fingers_ggarray = two_fingers_ggarray[index_filter_by_z_axis]
#             two_fingers_ggarray_object_ids = two_fingers_ggarray_object_ids_source[index_filter_by_z_axis]
#             grasp_features = grasp_features[index_filter_by_z_axis]

#             if len(InspireHandR_ggarray) == 0:
#                 print("No grasp detected after filter")
#                 if cfgs.global_camera:
#                     ggarray, cloud, points_down, grasp_features, sinput = get_ggarray_features(
#                         existing_shm_depth, existing_shm_color, net
#                     )
#                 if DEBUG:
#                     frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
#                     sphere = o3d.geometry.TriangleMesh.create_sphere(0.002, 20).translate([0, 0, 0.490])
#                     o3d.visualization.draw_geometries([cloud, frame, sphere])
#                 continue

#             index_score = np.argsort(InspireHandR_ggarray.scores)[::-1]
#             InspireHandR_ggarray = InspireHandR_ggarray[index_score]
#             two_fingers_ggarray = two_fingers_ggarray[index_score]
#             grasp_features = grasp_features[index_score]
#             two_fingers_ggarray_object_ids = two_fingers_ggarray_object_ids[index_score]

#             index_type = select_grasp_type(InspireHandR_ggarray)
#             InspireHandR_ggarray = InspireHandR_ggarray[index_type]
#             two_fingers_ggarray = two_fingers_ggarray[index_type]
#             grasp_features = grasp_features[index_type]
#             two_fingers_ggarray_object_ids = two_fingers_ggarray_object_ids[index_type]

#             approach_distance = 0.06
#             start_time = time.time()
#             mfcdetector = ModelFreeCollisionDetectorMultifinger(points_down.cpu().numpy(), voxel_size=0.001)
#             InspireHandR_ggarray, two_fingers_ggarray, empty_mask, min_width_index = mfcdetector.detect(
#                 InspireHandR_ggarray,
#                 two_fingers_ggarray,
#                 cfgs.inspire_mesh_json_path,
#                 meshes_pcls,
#                 min_grasp_width=MIN_GRASP_WIDTH,
#                 VoxelGrid=INSPIREHANDR_VOXElGRID,
#                 DEBUG=False,
#                 approach_dist=approach_distance,
#                 collision_thresh=0,
#                 adjust_gripper_centers=True,
#             )

#             # proposals
#             InspireHandR_ggarray = InspireHandR_ggarray[empty_mask]
#             two_fingers_ggarray = two_fingers_ggarray[empty_mask]
#             two_fingers_ggarray_object_ids = two_fingers_ggarray_object_ids[min_width_index][empty_mask]
#             grasp_features = grasp_features[min_width_index][empty_mask]

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
#                     ggarray, cloud, points_down, grasp_features, sinput = get_ggarray_features(
#                         existing_shm_depth, existing_shm_color, net
#                     )
#                     t3 = time.time()
#                     print(f"Net Time:{t3 - t1}")
#                 continue

#             # sort
#             index_score = np.argsort(InspireHandR_ggarray.scores)[::-1][:10]
#             two_fingers_ggarray_object_ids = two_fingers_ggarray_object_ids[index_score]
#             InspireHandR_ggarray = InspireHandR_ggarray[index_score]
#             two_fingers_ggarray = two_fingers_ggarray[index_score]
#             grasp_features = grasp_features[index_score]

#             idx = random.randint(0, len(index_score) - 1)
#             InspireHandR_grasp_used = InspireHandR_ggarray[idx]
#             two_fingers_grasp_used = two_fingers_ggarray[idx]
#             grasp_features_used = grasp_features[idx]

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
#             print(f"Collision Processing Time:{t4 - t3}")
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

#             gripper_time = 0.4
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
#             if cfgs.global_camera:
#                 robot.movej(
#                     robot.throwj2, acc=a * 4, vel=v * 5.5
#                 )  # this v and a are anguler, so it should be larger than translational
#                 robot.open_gripper(InspireHandR_grasp_used.angle)

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
#             )

#             if cfgs.global_camera:
#                 t1 = time.time()
#                 depths = get_depth(existing_shm_depth)
#                 depths_saved = copy.deepcopy(depths)
#                 colors_saved = np.copy(np.ndarray((720, 1280, 3), dtype=np.float32, buffer=existing_shm_color.buf))
#                 ggarray, cloud, points_down, grasp_features, sinput = get_ggarray_features(
#                     existing_shm_depth, existing_shm_color, net
#                 )
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
#             existing_shm_color.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", default="logs/model/checkpoint.tar.18", help="Model checkpoint path")
    parser.add_argument(
        "--inspire_model_path", default="logs/model/inspire_model/obj140", help="inspire model checkpoint path"
    )
    parser.add_argument(
        "--save_information_path",
        default="logs/data/inspire/inspire_test/obj140",
        help="inspire model result information path",
    )
    parser.add_argument(
        "--inspire_mesh_json_path",
        default="generate_mesh_and_pointcloud/inspire_urdf",
        help="InspireHandR meshes and json path",
    )
    # parser.add_argument('--robot_ip', required=True, help='Robot IP')
    parser.add_argument("--use_graspnet_v2", default=True, help="Whether to use graspnet v2 format")
    # parser.add_argument('--half_views', action='store_true', help='Use only half views in network.')
    parser.add_argument("--global_camera", action="store_true", help="Use the settings for global camera.")
    cfgs = parser.parse_args()

    inspire_models = get_inspire_model(cfgs.inspire_model_path)
    meshes_pcls = load_meshes_pointcloud(cfgs.inspire_mesh_json_path)

    # Setup env
    camera_name = "robot0_eye_in_hand"
    robot_env = make_robosuite_env(task="Lift", camera_name=camera_name)
    obs_dict = robot_env.reset()

    camera = RobosuiteCameraInfo(robot_env.sim, camera_name, camera_height=720, camera_width=1280)
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

    while True:
        # Running the grasp net
        color_map = obs_dict["{}_image".format(camera_name)][::-1] / 255.0
        depth_map = CU.get_real_depth_map(
            sim=robot_env.sim, depth_map=obs_dict["{}_depth".format(camera_name)][::-1]
        ).squeeze()

        camera.update_mappings()  # Check camera.camera_to_world_mat

        # NOTE: check what does augmentation do?
        ggarray, points_down, grasp_features, sinput = graspnet_runner.get_ggarray_features(depth_map, num_augment=3)

        if DEBUG:
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
            sphere = o3d.geometry.TriangleMesh.create_sphere(0.002, 20).translate([0, 0, 0.490])
            scene_cloud = o3d.geometry.PointCloud()
            scene_cloud.points = o3d.utility.Vector3dVector(points_down.cpu().numpy())
            scene_cloud = scene_cloud.voxel_down_sample(0.001)
            scene_cloud.paint_uniform_color([0.8, 0.8, 0.8])
            o3d.visualization.draw_plotly([scene_cloud, frame, sphere], width=1024, height=640)

        ########## PROCESS GRASPS ##########
        # collision detection
        ggarray = ggarray.cpu().numpy()

        # Prevent the robot arm from crossing the border,
        grasp_features = grasp_features.cpu().numpy()
        two_fingers_source_grasp_features = copy.deepcopy(grasp_features)

        # flip, but filter the flipped grasps out?
        ggarray, if_flip = flip_ggarray(ggarray)
        grasp_features = np.c_[grasp_features, if_flip]
        ggarray = ggarray[~np.array(if_flip)]
        grasp_features = grasp_features[~np.array(if_flip)]

        source_index = ggarray[:, 0].argsort()
        ggarray = ggarray[source_index][::-1][:1000]
        grasp_features = grasp_features[source_index][::-1][:1000]
        t_multi = time.time()
        grasp_features_dic = get_graspgroup_features(grasp_features, sinput)

        inspire_depth, inspire_type, scores, ggarray, grasp_features = get_inspire_depth_type(
            inspire_models, grasp_features_dic, ggarray, grasp_features=grasp_features
        )

        if not RANDOM_GRASP:
            score_thresh = 0.85
            mask = scores > score_thresh
            ggarray = ggarray[mask]
            grasp_features = grasp_features[mask]
            inspire_depth = inspire_depth[mask]
            inspire_type = inspire_type[mask]
            scores = scores[mask]

        if len(ggarray) == 0:
            print(f"There is no grasp that score greater than {score_thresh}")
            continue

        two_fingers_ggarray = GraspGroup(ggarray)
        two_fingers_ggarray_object_ids_source = ggarray[:, 16]
        two_fingers_ggarray_source = copy.deepcopy(two_fingers_ggarray)

        InspireHandR_ggarray = InspireHandRGraspGroup()
        InspireHandR_ggarray.set_grasp_min_width(MIN_GRASP_WIDTH)
        InspireHandR_ggarray.from_graspgroup(two_fingers_ggarray, inspire_type, cfgs.inspire_mesh_json_path)
        InspireHandR_ggarray.object_ids = two_fingers_ggarray_object_ids_source
        InspireHandR_ggarray.scores = scores
        InspireHandR_ggarray.depths = InspireHandR_ggarray.depths + inspire_depth + INSPIREHANDR_DEFAULT_DEPTH
        InspireHandR_ggarray_source = copy.deepcopy(InspireHandR_ggarray)

        index_filter_by_z_axis = InspireHandR_ggarray.filter_grasp_group_by_z_axis(0.4)
        two_fingers_ggarray = two_fingers_ggarray[index_filter_by_z_axis]
        two_fingers_ggarray_object_ids = two_fingers_ggarray_object_ids_source[index_filter_by_z_axis]
        grasp_features = grasp_features[index_filter_by_z_axis]

        if len(InspireHandR_ggarray) == 0:
            print("No grasp detected after z-axis filter")
            continue

        # Sort by scores in each type
        index_type_score = select_grasp_type(InspireHandR_ggarray)
        InspireHandR_ggarray = InspireHandR_ggarray[index_type_score]
        two_fingers_ggarray = two_fingers_ggarray[index_type_score]
        grasp_features = grasp_features[index_type_score]
        two_fingers_ggarray_object_ids = two_fingers_ggarray_object_ids[index_type_score]

        print("Start collision detection ...")
        approach_distance = 0.06
        start_time = time.time()
        mfcdetector = ModelFreeCollisionDetectorMultifinger(points_down.cpu().numpy(), voxel_size=0.001)
        InspireHandR_ggarray, two_fingers_ggarray, empty_mask, min_width_index = mfcdetector.detect(
            InspireHandR_ggarray,
            two_fingers_ggarray,
            cfgs.inspire_mesh_json_path,
            meshes_pcls,
            min_grasp_width=MIN_GRASP_WIDTH,
            VoxelGrid=INSPIREHANDR_VOXEL_GRID,
            DEBUG=False,
            approach_dist=approach_distance,
            collision_thresh=0,
            adjust_gripper_centers=True,
        )
        print(f"Collision Detection Time:{time.time() - start_time}")

        # proposals after collision detection
        InspireHandR_ggarray = InspireHandR_ggarray[empty_mask]
        two_fingers_ggarray = two_fingers_ggarray[empty_mask]
        two_fingers_ggarray_object_ids = two_fingers_ggarray_object_ids[min_width_index][empty_mask]
        grasp_features = grasp_features[min_width_index][empty_mask]

        if len(InspireHandR_ggarray) == 0:
            print("No Grasp detected after collision detection!")
            continue

        # Final selection
        index_score = np.argsort(InspireHandR_ggarray.scores)[::-1][:10]
        two_fingers_ggarray_object_ids = two_fingers_ggarray_object_ids[index_score]
        InspireHandR_ggarray = InspireHandR_ggarray[index_score]
        two_fingers_ggarray = two_fingers_ggarray[index_score]
        grasp_features = grasp_features[index_score]

        # Select one grasp in random among the top
        idx = random.randint(0, len(index_score) - 1)
        InspireHandR_grasp_used = InspireHandR_ggarray[idx]
        two_fingers_grasp_used = two_fingers_ggarray[idx]
        grasp_features_used = grasp_features[idx]

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

        execute_grasp(
            robot_env, camera, InspireHandR_grasp_used, two_fingers_grasp_used, grab_site_offset=GRAB_SITE_OFFSET
        )

        # TODO: The env should have multiple objects with a clean-up task
        # This script is to evaluate the trained models

        print()
