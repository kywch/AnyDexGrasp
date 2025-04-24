import os
from re import sub
import sys
import numpy as np
from PIL import Image
import scipy.io as scio
import json
import torch
import random
import shutil
from scipy.spatial.transform import Rotation
import copy

TORCH_MAJOR = int(torch.__version__.split(".")[0])
TORCH_MINOR = int(torch.__version__.split(".")[1])

if TORCH_MAJOR == 1 and TORCH_MINOR < 8:
    from torch._six import container_abcs
else:
    import collections.abc as container_abcs

from torch.utils.data import Dataset
from tqdm import tqdm
from ..utils.np_utils import (
    transform_point_cloud,
    remove_invisible_grasp_points,
    create_point_cloud_from_depth_image,
    get_workspace_mask,
)
import MinkowskiEngine as ME

MAX_GRIPPER_WIDTH = 0.1
MAX_MU = 1.0
MIN_MU = 0.1
MAX_GRASP_SCORE = np.log(MAX_MU / MIN_MU)
CONVERT_TO_NEW_DEIVCE = [
    "object_poses_list",
    "grasp_points_list",
    "grasp_widths_list",
    "grasp_labels_list",
    "grasp_heatmap_list",
    "grasp_view_heatmap_list",
    "grasp_heatmap_raw_list",
    "grasp_collision_list",
]
DEPTH_FACTOR = 1000.0


class CameraInfo:
    def __init__(self, width, height, fx, fy, cx, cy, scale):
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.scale = scale


class MultifingerDataset(Dataset):
    def __init__(self, root, multifinger_type="Inspire", dataset_type="train", train_type=0, num_multifinger_type=1):
        self.root = root
        self.multifinger_type = multifinger_type
        self.dataset_type = dataset_type
        self.train_type = train_type
        self.num_multifinger_type = num_multifinger_type

        if self.dataset_type == "train":
            information_json_file_name = os.path.join(root.split("train_sr")[0], "obj140.json")
        elif self.dataset_type == "test":
            information_json_file_name = os.path.join(root.split("test_sr")[0], "test_single_point.json")
        else:
            raise ValueError('dataset type must be "test" or "train"')

        with open(information_json_file_name) as f:
            self.informations = json.load(f)

        true_number, false_number, grasp_type, pose_excluded = self.get_true_false_rate()
        print(self.dataset_type, " num, true, false, rate: ", true_number, false_number, true_number / false_number)
        print("data grasp type distribution: \n", grasp_type)

        datas = list(self.informations.keys())
        self.data = []
        data_num = 0
        for dt in datas:
            if dt not in pose_excluded:
                self.data.append(dt)
                data_num += 1
                if self.dataset_type == "train" and data_num == 100:
                    break
        print("len of dataset is:", len(self.data))
        random.shuffle(self.data)

    def get_true_false_rate(self):
        true_number = 0
        false_number = 0
        grasp_type = [[0, 0, 0] for _ in range(4)]
        pose_excluded = {""}
        print(self.train_type)
        for k, v in self.informations.items():
            if self.multifinger_type == "Inspire":
                gripper_type = v["InspiredHandR_pose_finger_type"]
                depth_type = v["InspiredHandR_pose_depth_type"]
            elif self.multifinger_type == "DH3":
                gripper_type = v["DH3_pose_finger_type"]
                depth_type = v["DH3_pose_depth_type"]
            elif self.multifinger_type == "Allegro":
                gripper_type = v["Allegro_pose_finger_type"]
                depth_type = v["Allegro_pose_depth_type"]

            if int(gripper_type) != int(self.train_type):
                pose_excluded.add(k)
                continue

            result = v["result"]
            if result:
                grasp_type[depth_type][0] += 1
            else:
                grasp_type[depth_type][1] += 1
            grasp_type[depth_type][2] += 1
            result = v["result"]
            if result:
                true_number += 1
            else:
                false_number += 1
        return true_number, false_number, grasp_type, pose_excluded

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item_dir = self.data[index]

        information = self.informations[item_dir]

        ret_dict = {}

        ret_dict["two_fingers_pose_angle_type"] = np.array([information["two_fingers_pose_angle_type"]], dtype=np.int32)
        ret_dict["two_fingers_pose_depth_type"] = np.array([information["two_fingers_pose_depth_type"]], dtype=np.int32)
        if self.multifinger_type == "Inspire":
            ret_dict["multifinger_pose_finger_type"] = np.array(
                [information["InspiredHandR_pose_finger_type"]], dtype=np.int32
            )
            ret_dict["multifinger_pose_depth_type"] = np.array(
                [information["InspiredHandR_pose_depth_type"]], dtype=np.int32
            )

        elif self.multifinger_type == "DH3":
            ret_dict["multifinger_pose_finger_type"] = np.array([information["DH3_pose_finger_type"]], dtype=np.int32)
            ret_dict["multifinger_pose_depth_type"] = np.array([information["DH3_pose_depth_type"]], dtype=np.int32)
        elif self.multifinger_type == "Allegro":
            ret_dict["multifinger_pose_finger_type"] = np.array(
                [information["Allegro_pose_finger_type"]], dtype=np.int32
            )
            ret_dict["multifinger_pose_depth_type"] = np.array([information["Allegro_pose_depth_type"]], dtype=np.int32)

        ret_dict["if_flip"] = np.array([information["if_flip"]]).astype(np.int32)  # 1 for Flip and 0 for not
        grasp_preds_features = np.array(information["grasp_preds_features"], dtype=np.float32)[:480]  # [480]

        if ret_dict["if_flip"]:
            first_half_scores = copy.deepcopy(grasp_preds_features[:120])
            last_half_scores = copy.deepcopy(grasp_preds_features[120:240])
            first_half_widths = copy.deepcopy(grasp_preds_features[240:360])
            last_half_widths = copy.deepcopy(grasp_preds_features[360:480])
            grasp_preds_features[:120] = last_half_scores
            grasp_preds_features[120:240] = first_half_scores
            grasp_preds_features[240:360] = last_half_widths
            grasp_preds_features[360:480] = first_half_widths

        new_type = 12 + information["two_fingers_pose_angle_type"] * 2
        grasp_preds_features_rot = np.zeros(grasp_preds_features.shape, dtype=np.float32)
        grasp_preds_features_rot[: 240 - new_type * 5] = grasp_preds_features[new_type * 5 : 240]
        grasp_preds_features_rot[240 - new_type * 5 : 240] = grasp_preds_features[0 : new_type * 5]
        grasp_preds_features_rot[240 : 480 - new_type * 5] = grasp_preds_features[240 + new_type * 5 : 480]
        grasp_preds_features_rot[480 - new_type * 5 : 480] = grasp_preds_features[240 : 240 + new_type * 5]

        ret_dict["grasp_preds_features"] = grasp_preds_features_rot[:480]  # [480]

        ret_dict["result"] = np.array([information["result"]]).astype(np.int32)  # 1 for True and 0 for False
        return ret_dict


def collate_fn(batch):
    if type(batch[0]).__module__ == "numpy":
        return [torch.from_numpy(b) for b in batch]
    elif isinstance(batch[0], container_abcs.Sequence):
        return [[torch.from_numpy(sample) for sample in b] for b in batch]
    elif isinstance(batch[0], container_abcs.Mapping):
        ret_dict = {key: collate_fn([d[key] for d in batch]) for key in batch[0]}

        for key in ret_dict.keys():
            if not key in ["coords", "feats", "sinput"]:
                ret_dict[key] = torch.tensor([d.numpy() for d in ret_dict[key]])
        return ret_dict
    raise TypeError("batch must contain tensors, dicts or lists; found {}".format(type(batch[0])))


def convert_data_to_device(data, device):
    if isinstance(data, container_abcs.Sequence):
        return [convert_data_to_device(data[i], device) for i in range(len(data))]
    else:
        return data.to(device)


def convert_data_to_gpu(data):
    ret_dict = dict()
    for key in data.keys():
        ret_dict[key] = convert_data_to_device(data[key], "cuda:0")
    return ret_dict
