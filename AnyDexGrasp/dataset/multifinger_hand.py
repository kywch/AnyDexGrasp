import copy
from collections.abc import Sequence, Mapping

import numpy as np
import torch
from torch.utils.data import Dataset


def filter_json_data(data, gripper_type, grasp_type, num_depth=4):
    filtered_data = {}

    if gripper_type.lower() == "inspire":
        finger_type_key = "InspiredHandR_pose_finger_type"
        depth_type_key = "InspiredHandR_pose_depth_type"
    elif gripper_type.lower() == "dh3":
        finger_type_key = "DH3_pose_finger_type"
        depth_type_key = "DH3_pose_depth_type"
    elif gripper_type.lower() == "allegro":
        finger_type_key = "Allegro_pose_finger_type"
        depth_type_key = "Allegro_pose_depth_type"
    else:
        raise ValueError(f"Unknown gripper type: {gripper_type}")

    trial_stats = {}
    for i in range(num_depth):
        trial_stats[i] = {
            "count": 0,
            "success": 0,
            "fail": 0,
        }

    for trial_key, trial_dict in data.items():
        # Keep the trial data if both gripper_type and grasp_type match
        if (
            finger_type_key in trial_dict
            and trial_dict[finger_type_key] == grasp_type
            and trial_dict[depth_type_key] in trial_stats
        ):
            trial_dict["multifinger_pose_finger_type"] = trial_dict[finger_type_key]
            trial_dict["multifinger_pose_depth_type"] = trial_dict[depth_type_key]
            filtered_data[trial_key] = trial_dict

            trial_stats[trial_dict["multifinger_pose_depth_type"]]["count"] += 1
            if trial_dict["result"]:
                trial_stats[trial_dict["multifinger_pose_depth_type"]]["success"] += 1
            else:
                trial_stats[trial_dict["multifinger_pose_depth_type"]]["fail"] += 1

    return filtered_data, trial_stats


class MultifingerDataset(Dataset):
    def __init__(self, data_dict):
        # Assume that data_dict is filtered, and using all the keys
        self.data = data_dict
        self.data_keys = list(data_dict.keys())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item_key = self.data_keys[index]
        information = self.data[item_key]

        ret_dict = {}

        ret_dict["two_fingers_pose_angle_type"] = np.array([information["two_fingers_pose_angle_type"]], dtype=np.int32)
        ret_dict["two_fingers_pose_depth_type"] = np.array([information["two_fingers_pose_depth_type"]], dtype=np.int32)

        # Use the standardized keys directly, as created by filter_json_data
        ret_dict["multifinger_pose_finger_type"] = np.array(
            [information["multifinger_pose_finger_type"]], dtype=np.int32
        )
        ret_dict["multifinger_pose_depth_type"] = np.array([information["multifinger_pose_depth_type"]], dtype=np.int32)

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

        # Use the angle type directly (The provided data has 0-11 range, and theoretically 0-23)
        # The original calculation '12 + angle_type * 2' seems incorrect (suspected by gemini),
        # and will result in out-of-index errors for the angle_type >= 19
        # new_type = 12 + information["two_fingers_pose_angle_type"] * 2  # original

        # NOTE: this may affect the model performance
        new_angle_type = information["two_fingers_pose_angle_type"]

        # The code performs the cyclic rotation independently on the first 240 elements (scores) and the next 240 elements (widths).
        # This rotation aims to create a canonical representation of the features relative to the predicted grasp angle (two_fingers_pose_angle_type).
        # By rotating the features so that the data corresponding to the predicted angle always starts at index 0 (within the score and width blocks),
        # the subsequent network layers might learn patterns more easily, as they don't need to be invariant to the absolute angle index.
        grasp_preds_features_rot = np.zeros(grasp_preds_features.shape, dtype=np.float32)
        grasp_preds_features_rot[: 240 - new_angle_type * 5] = grasp_preds_features[new_angle_type * 5 : 240]
        grasp_preds_features_rot[240 - new_angle_type * 5 : 240] = grasp_preds_features[0 : new_angle_type * 5]
        grasp_preds_features_rot[240 : 480 - new_angle_type * 5] = grasp_preds_features[240 + new_angle_type * 5 : 480]
        grasp_preds_features_rot[480 - new_angle_type * 5 : 480] = grasp_preds_features[240 : 240 + new_angle_type * 5]

        ret_dict["grasp_preds_features"] = grasp_preds_features_rot[:480]  # [480]

        ret_dict["result"] = np.array([information["result"]]).astype(np.int32)  # 1 for True and 0 for False
        return ret_dict


def collate_fn(batch):
    if type(batch[0]).__module__ == "numpy":
        return [torch.from_numpy(b) for b in batch]
    elif isinstance(batch[0], Sequence):
        return [[torch.from_numpy(sample) for sample in b] for b in batch]
    elif isinstance(batch[0], Mapping):
        ret_dict = {key: collate_fn([d[key] for d in batch]) for key in batch[0]}

        for key in ret_dict.keys():
            if key not in ["coords", "feats", "sinput"]:
                ret_dict[key] = torch.tensor(np.array([d.numpy() for d in ret_dict[key]]))
        return ret_dict
    raise TypeError("batch must contain tensors, dicts or lists; found {}".format(type(batch[0])))


def convert_data_to_device(data, device):
    """
    Recursively moves tensors within nested structures (lists, tuples, dicts)
    to the specified device. Skips non-tensor elements.
    """
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, Mapping): # Handle dictionaries
        return {k: convert_data_to_device(v, device) for k, v in data.items()}
    elif isinstance(data, Sequence) and not isinstance(data, str): # Handle lists/tuples but not strings
        return [convert_data_to_device(item, device) for item in data]
    else:
        # Return data as is if it's not a tensor, dict, or sequence (e.g., int, float, str)
        return data


def convert_data_to_gpu(data, device=None):
    """
    Moves all tensor values within a dictionary (potentially nested) to the
    specified device (defaults to 'cuda:0').
    """
    if device is None:
        # Default to the first CUDA device if available, otherwise CPU
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        # Convert string device name to torch.device object
        device = torch.device(device)

    # Ensure data is a dictionary before iterating
    if not isinstance(data, Mapping):
        raise TypeError(f"Expected data to be a dictionary (Mapping), but got {type(data)}")

    # Use the recursive function to handle nested structures
    return convert_data_to_device(data, device)
