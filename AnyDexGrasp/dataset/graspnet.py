import os
import collections.abc as container_abcs

import numpy as np
from PIL import Image
import scipy.io as scio

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

import MinkowskiEngine as ME

from ..utils.collision_detector import CollisionType
from ..utils.np_utils import (
    transform_point_cloud,
    remove_invisible_grasp_points,
    create_point_cloud_from_depth_image,
    get_workspace_mask,
)

MAX_GRIPPER_WIDTH = 0.08
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
]
BOX_LIST = [192, 193]
IGNORED_LABELS = [18]
IGNORED_SCENES = [248, 351, 352, 353, 354, 365, 392, 393, 394]


class CameraInfo:
    def __init__(self, width, height, fx, fy, cx, cy, scale):
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.scale = scale


class GraspNetVoxelizationDataset(Dataset):
    def __init__(
        self,
        root,
        valid_obj_idxs=None,
        grasp_labels=None,
        camera="realsense",
        split="train",
        voxel_size=0.005,
        heatmap="scene",
        heatmap_th=0.6,
        view_heatmap_th=0.6,
        score_as_heatmap=False,
        score_as_view_heatmap=False,
        remove_outlier=False,
        remove_invisible=False,
        augment=False,
        load_label=True,
        centralize_points=False,
    ):
        self.root = root
        self.split = split
        self.voxel_size = voxel_size
        self.heatmap = heatmap
        self.heatmap_th = heatmap_th
        self.view_heatmap_th = view_heatmap_th
        self.score_as_heatmap = score_as_heatmap
        self.score_as_view_heatmap = score_as_view_heatmap
        self.remove_outlier = remove_outlier
        self.remove_invisible = remove_invisible
        self.valid_obj_idxs = valid_obj_idxs
        self.grasp_labels = grasp_labels
        self.camera = camera
        self.augment = augment
        self.load_label = load_label
        self.centralize_points = centralize_points

        assert self.camera in ["kinect", "realsense"]
        assert self.split in [
            "train",
            "test",
            "test_seen",
            "test_similar",
            "test_novel",
            "ablation_train",
            "ablation_val",
        ]
        assert self.heatmap in ["scene", "object", "collision"]

        if split == "train":
            # self.sceneIds = list( range(100) )  # Ouf of memory for 128G ram
            self.sceneIds = list(range(10))
        elif split == "test":
            self.sceneIds = list(range(100, 190))
        elif split == "test_seen":
            self.sceneIds = list(range(100, 130))
        elif split == "test_similar":
            self.sceneIds = list(range(130, 160))
        elif split == "test_novel":
            self.sceneIds = list(range(160, 190))
        elif split == "ablation_train":
            self.sceneIds = [x for x in range(100) if x % 10 != 0]
        elif split == "ablation_val":
            self.sceneIds = [x for x in range(100) if x % 10 == 0]
        self.sceneIds = ["scene_{}".format(str(x).zfill(4)) for x in self.sceneIds]

        self.colorpath = []
        self.depthpath = []
        self.labelpath = []
        self.metapath = []
        self.scenename = []
        self.frameid = []
        self.viewgraspnesspath = []
        self.collision_labels = {}
        self.pointgraspness = {}
        self.visibleid = {}

        for i, x in enumerate(tqdm(self.sceneIds, desc="Loading data path and collision labels...")):
            for img_num in range(256):
                self.colorpath.append(os.path.join(root, "scenes", x, camera, "rgb", str(img_num).zfill(4) + ".png"))
                self.depthpath.append(os.path.join(root, "scenes", x, camera, "depth", str(img_num).zfill(4) + ".png"))
                self.labelpath.append(os.path.join(root, "scenes", x, camera, "label", str(img_num).zfill(4) + ".png"))
                self.metapath.append(os.path.join(root, "scenes", x, camera, "meta", str(img_num).zfill(4) + ".mat"))
                self.scenename.append(x.strip())
                self.frameid.append(img_num)
                self.viewgraspnesspath.append(
                    os.path.join(
                        root,
                        "pre_generated_label_combinescore",
                        x,
                        camera,
                        str(img_num).zfill(4) + "_view_graspness.npy",
                    )
                )

            if self.load_label:
                collision_labels = np.load(
                    os.path.join(root, "fric_collision_label", x.strip(), "collision_labels.npz")
                )
                self.collision_labels[x.strip()] = {}
                for i in range(len(collision_labels)):
                    self.collision_labels[x.strip()][i] = collision_labels["arr_{}".format(i)]
                self.pointgraspness[x.strip()] = np.load(
                    os.path.join(root, "pre_generated_label_combinescore", x, camera, "point_graspness.npy"),
                    allow_pickle=True,
                )
                self.visibleid[x.strip()] = np.load(
                    os.path.join(root, "pre_generated_label_combinescore", x, camera, "visibleid.npy"),
                    allow_pickle=True,
                )

    def __len__(self):
        return len(self.colorpath)

    def scene_list(self):
        return self.scenename

    def remove_objects(self, point_idxs, seg, obj_idxs, poses, collision_labels, max_num_object=3):
        # shuffle indices
        num_objs = np.random.randint(max_num_object) + 1
        indices = np.arange(len(obj_idxs))
        np.random.shuffle(indices)
        # pick object idxs
        indices_to_keep = list()
        for idx in indices:
            obj_idx = obj_idxs[idx]
            mask = seg == obj_idx
            if mask.sum() >= 50 and obj_idx in self.valid_obj_idxs:
                indices_to_keep.append(idx)
        num_objs = min(num_objs, len(indices_to_keep))
        indices_to_keep = indices_to_keep[:num_objs]
        # remove objects
        obj_idxs_to_keep = obj_idxs[indices_to_keep]
        point_idxs_to_keep = [point_idxs[seg == 0]]
        seg_to_keep = [seg[seg == 0]]
        poses_to_keep = list()
        collision_labels_to_keep = list()
        for idx in indices_to_keep:
            poses_to_keep.append(poses[idx])
            collision_labels_to_keep.append(collision_labels[idx])
        for obj_idx in obj_idxs_to_keep:
            mask = seg == obj_idx
            point_idxs_to_keep.append(point_idxs[mask])
            seg_to_keep.append(seg[mask])
        point_idxs_to_keep = np.concatenate(point_idxs_to_keep, axis=0)
        seg_to_keep = np.concatenate(seg_to_keep, axis=0)

        return point_idxs_to_keep, obj_idxs_to_keep, poses_to_keep, collision_labels_to_keep

    def centralize_data(self, point_clouds, object_poses_list=None):
        # Translate points to center
        center = (point_clouds.max(axis=0) + point_clouds.min(axis=0)) / 2
        trans_mat = np.array([[1, 0, 0, -center[0]], [0, 1, 0, -center[1]], [0, 0, 1, -center[2]], [0, 0, 0, 1]])

        point_clouds = transform_point_cloud(point_clouds, trans_mat, "4x4")
        if object_poses_list is None:
            return point_clouds

        for i in range(len(object_poses_list)):
            item = object_poses_list[i]
            ones = np.ones(item.shape[1])[np.newaxis, :]
            item_ = np.concatenate([item, ones], axis=0)
            item_transformed = np.dot(trans_mat, item_).astype(np.float32)
            object_poses_list[i] = item_transformed[:3, :]
        return point_clouds, object_poses_list

    def augment_data(self, point_clouds, object_poses_list):
        flip_mat = np.identity(4)
        # Flipping along the YZ plane
        if np.random.random() > 0.5:
            flip_mat = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

        # Rotation along up-axis/Z-axis
        rot_angle = (np.random.random() * np.pi / 3) - np.pi / 6  # -30 ~ +30 degree
        c, s = np.cos(rot_angle), np.sin(rot_angle)
        rot_mat = np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]])

        # Translation along X/Y/Z-axis
        offset_x = np.random.random() * 0.4 - 0.2  # -0.2 ~ 0.2
        offset_y = np.random.random() * 0.4 - 0.2  # -0.2 ~ 0.2
        # for one-shot grasp
        offset_z = np.random.random() * 0.3 - 0.1  # -0.1 ~ 0.2
        # # for tracking prior
        trans_mat = np.array([[1, 0, 0, offset_x], [0, 1, 0, offset_y], [0, 0, 1, offset_z], [0, 0, 0, 1]])

        aug_mat = np.dot(trans_mat, np.dot(rot_mat, flip_mat).astype(np.float32)).astype(np.float32)
        point_clouds = transform_point_cloud(point_clouds, aug_mat, "4x4")
        for i in range(len(object_poses_list)):
            item = object_poses_list[i]
            ones = np.ones(item.shape[1])[np.newaxis, :]
            item_ = np.concatenate([item, ones], axis=0)
            item_transformed = np.dot(aug_mat, item_).astype(np.float32)
            object_poses_list[i] = item_transformed[:3, :]
        return point_clouds, object_poses_list

    def __getitem__(self, index):
        if self.load_label:
            return self._get_data_label(index)
        else:
            return self._get_data(index)

    def _get_data(self, index, return_raw_cloud=False):
        # load data
        color = np.array(Image.open(self.colorpath[index]), dtype=np.float32) / 255.0
        depth = np.array(Image.open(self.depthpath[index]))
        seg = np.array(Image.open(self.labelpath[index]))
        meta = scio.loadmat(self.metapath[index])
        scene = self.scenename[index]

        # parse metadata
        poses_mat = meta["poses"]
        intrinsic = meta["intrinsic_matrix"]
        factor_depth = meta["factor_depth"]
        camerainfo = CameraInfo(
            1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth
        )

        # generate cloud
        cloud = create_point_cloud_from_depth_image(depth, camerainfo, organized=True)

        # get valid points
        depth_mask = depth > 0
        seg_mask = seg > 0
        if self.remove_outlier:
            root = self.root
            camera_split = self.camera
            camera_poses = np.load(os.path.join(root, "scenes", scene, camera_split, "camera_poses.npy"))
            align_mat = np.load(os.path.join(root, "scenes", scene, camera_split, "cam0_wrt_table.npy"))
            trans = np.dot(align_mat, camera_poses[self.frameid[index]])
            workspace_mask = get_workspace_mask(cloud, seg, trans=trans, organized=True, outlier=0.02)
            mask = depth_mask & workspace_mask
        else:
            mask = depth_mask
        cloud_masked = cloud[mask]
        color_masked = color[mask]
        if return_raw_cloud:
            return cloud_masked, color_masked

        # shuffle points
        idxs = np.arange(cloud_masked.shape[0])
        np.random.shuffle(idxs)
        cloud_masked = cloud_masked[idxs]
        color_masked = color_masked[idxs]

        # centralize data
        if self.centralize_points:
            cloud_masked = self.centralize_data(cloud_masked)

        # voxelization
        # Upd Note. Make coords contiguous.
        coords = np.ascontiguousarray(cloud_masked / self.voxel_size, dtype=np.int32)
        _, idxs = ME.utils.sparse_quantize(coords, return_index=True)
        coords = coords[idxs]
        cloud_voxeled = cloud_masked[idxs]
        color_voxeled = color_masked[idxs]

        ret_dict = {}
        ret_dict["coords"] = coords.astype(np.int32)
        ret_dict["feats"] = cloud_voxeled.astype(np.float32)
        ret_dict["point_clouds"] = cloud_voxeled.astype(np.float32)
        ret_dict["cloud_colors"] = color_voxeled.astype(np.float32)

        return ret_dict

    def _get_data_label(self, index):
        # load data
        color = np.array(Image.open(self.colorpath[index]), dtype=np.float32) / 255.0
        depth = np.array(Image.open(self.depthpath[index]))
        seg = np.array(Image.open(self.labelpath[index]))
        meta = scio.loadmat(self.metapath[index])
        scene = self.scenename[index]
        frameid = self.frameid[index]
        image_pointgraspness = self.pointgraspness[scene][frameid]
        image_visibleid = self.visibleid[scene][frameid]
        image_viewgraspness = np.load(self.viewgraspnesspath[index], allow_pickle=True).item()
        collision_labels = self.collision_labels[scene]

        # parse metadata
        obj_idxs = meta["cls_indexes"].flatten().astype(np.int32)
        poses_mat = meta["poses"]
        intrinsic = meta["intrinsic_matrix"]
        factor_depth = meta["factor_depth"]
        poses = [poses_mat[:, :, i] for i in range(poses_mat.shape[-1])]
        camerainfo = CameraInfo(
            1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth
        )

        # generate cloud
        cloud = create_point_cloud_from_depth_image(depth, camerainfo, organized=True)

        # get valid points
        depth_mask = depth > 0
        if self.remove_outlier:
            root = self.root
            camera_split = self.camera
            camera_poses = np.load(os.path.join(root, "scenes", scene, camera_split, "camera_poses.npy"))
            align_mat = np.load(os.path.join(root, "scenes", scene, camera_split, "cam0_wrt_table.npy"))
            trans = np.dot(align_mat, camera_poses[frameid])
            workspace_mask = get_workspace_mask(cloud, seg, trans=trans, organized=True, outlier=0.02)
            mask = depth_mask & workspace_mask
        else:
            mask = depth_mask
        cloud_masked = cloud[mask]
        color_masked = color[mask]
        seg_masked = seg[mask]

        # shuffle points
        idxs = np.arange(cloud_masked.shape[0])
        np.random.shuffle(idxs)
        cloud_masked = cloud_masked[idxs]
        color_masked = color_masked[idxs]
        seg_masked = seg_masked[idxs]

        # centralize data
        if self.centralize_points:
            cloud_masked, poses = self.centralize_data(cloud_masked, poses)

        # augment data
        if self.augment:
            cloud_masked, poses = self.augment_data(cloud_masked, poses)

        # voxelization
        # Upd Note. Make coords contiguous.
        coords = np.ascontiguousarray(cloud_masked / self.voxel_size, dtype=np.int32)
        _, idxs = ME.utils.sparse_quantize(coords, return_index=True)

        # remove objects randomly
        if np.random.uniform() < 0.2 and int(scene[-3:]) not in IGNORED_SCENES:
            idxs, obj_idxs, poses, collision_labels = self.remove_objects(
                idxs, seg_masked[idxs], obj_idxs, poses, collision_labels, max_num_object=3
            )
        coords = coords[idxs]
        cloud_voxeled = cloud_masked[idxs]
        color_voxeled = color_masked[idxs]
        seg_voxeled = seg_masked[idxs]
        objectness_label = seg_voxeled.copy()
        objectness_label[objectness_label > 1] = 1

        # merge grasp labels
        object_poses_list = []
        grasp_points_list = []
        grasp_widths_list = []
        grasp_labels_list = []
        grasp_heatmap_list = []
        grasp_view_heatmap_list = []
        # grasp_heatmap_raw_list = []
        ret_obj_idxs = []
        for i, obj_idx in enumerate(obj_idxs):
            if obj_idx not in self.valid_obj_idxs:
                continue
            if (seg_voxeled == obj_idx).sum() < 50:
                continue
            points, data = self.grasp_labels[obj_idx][:2]
            collision = collision_labels[i] > 0

            widths = data[:, :, :, :, 0]
            width_mask = (widths > 0) & (widths < MAX_GRIPPER_WIDTH)
            scores = data[:, :, :, :, 1].copy()
            scores[collision | ~width_mask] = 0

            # remove invisible grasp points
            if self.remove_invisible:
                visible_idxs = image_visibleid[str(obj_idx)]
                points = points[visible_idxs]
                widths = widths[visible_idxs]
                scores = scores[visible_idxs]
                collision = collision[visible_idxs]

            # generate heatmap
            if self.score_as_heatmap:
                grasp_heatmap = scores.copy()
                grasp_mask = grasp_heatmap > 0
                grasp_heatmap[grasp_mask] = np.log(MAX_MU / grasp_heatmap[grasp_mask])
                grasp_heatmap[~grasp_mask] = 0
                grasp_heatmap /= MAX_GRASP_SCORE
                grasp_heatmap = grasp_heatmap.reshape([grasp_heatmap.shape[0], -1])
                grasp_heatmap = grasp_heatmap.mean(axis=-1)
            else:
                grasp_heatmap = image_pointgraspness[str(obj_idx) + "_pointwise_graspness"]
            if self.score_as_view_heatmap:
                grasp_view_heatmap = scores.copy()
                grasp_view_mask = grasp_view_heatmap > 0
                grasp_view_heatmap[grasp_view_mask] = np.log(MAX_MU / grasp_view_heatmap[grasp_view_mask])
                grasp_view_heatmap[~grasp_view_mask] = 0
                grasp_view_heatmap /= MAX_GRASP_SCORE
                grasp_view_heatmap = grasp_view_heatmap.reshape(
                    [grasp_view_heatmap.shape[0], grasp_view_heatmap.shape[1], -1]
                )
                grasp_view_heatmap = grasp_view_heatmap.max(axis=-1)
            else:
                grasp_view_heatmap = image_viewgraspness[str(obj_idx) + "_viewwise_graspness"]

            # add to list
            object_poses_list.append(poses[i].astype(np.float32))
            grasp_points_list.append(points.astype(np.float32))
            grasp_widths_list.append(widths.astype(np.float32))
            grasp_labels_list.append(scores.astype(np.float32))
            grasp_heatmap_list.append(grasp_heatmap.astype(np.float32))
            grasp_view_heatmap_list.append(grasp_view_heatmap.astype(np.float32))
            ret_obj_idxs.append(obj_idx)

        ret_dict = {}
        ret_dict["coords"] = coords.astype(np.int32)
        ret_dict["feats"] = cloud_voxeled.astype(np.float32)
        ret_dict["point_clouds"] = cloud_voxeled.astype(np.float32)
        ret_dict["cloud_colors"] = color_voxeled.astype(np.float32)
        ret_dict["objectness_label"] = objectness_label.astype(np.int64)
        ret_dict["object_poses_list"] = object_poses_list
        ret_dict["grasp_points_list"] = grasp_points_list
        ret_dict["grasp_widths_list"] = grasp_widths_list
        ret_dict["grasp_labels_list"] = grasp_labels_list
        ret_dict["grasp_heatmap_list"] = grasp_heatmap_list
        ret_dict["grasp_view_heatmap_list"] = grasp_view_heatmap_list

        return ret_dict


def load_grasp_labels(root):
    obj_names = list(range(88))
    valid_obj_idxs = []
    grasp_labels = {}
    for i, obj_name in enumerate(tqdm(obj_names, desc="Loading fric representation labels...")):
        # if obj_name in IGNORED_LABELS: continue
        valid_obj_idxs.append(obj_name + 1)  # here align with label png
        label = np.load(os.path.join(root, "fric_rep", "{}_labels_small.npz".format(str(obj_name).zfill(3))))
        data = label["data"].astype(np.float32)
        data[(data[:, :, :, :, 1] > MAX_MU), 1] = 0
        data[(data[:, :, :, :, 1] > 0) & (data[:, :, :, :, 1] <= MIN_MU), 1] = MIN_MU

        grasp_labels[obj_name + 1] = [label["points"].astype(np.float32), data]

    return valid_obj_idxs, grasp_labels


def collate_fn(batch):
    if type(batch[0]).__module__ == "numpy":
        return [torch.from_numpy(b) for b in batch]
    elif isinstance(batch[0], container_abcs.Sequence):
        return [[torch.from_numpy(sample) for sample in b] for b in batch]
    elif isinstance(batch[0], container_abcs.Mapping):
        ret_dict = {key: collate_fn([d[key] for d in batch]) for key in batch[0]}
        coords_batch = ret_dict["coords"]
        feats_batch = ret_dict["feats"]
        if "objectness_label" in ret_dict:
            labels_batch = ret_dict["objectness_label"]
            coords_batch, feats_batch, labels_batch = ME.utils.sparse_collate(coords_batch, feats_batch, labels_batch)
            ret_dict["objectness_label"] = labels_batch
        else:
            coords_batch, feats_batch = ME.utils.sparse_collate(coords_batch, feats_batch)
        ret_dict["coords"] = coords_batch
        ret_dict["feats"] = feats_batch
        return ret_dict

    raise TypeError("batch must contain tensors, dicts or lists; found {}".format(type(batch[0])))


def multigpu_collate_fn(batch):
    def split_data_list(data_list):
        import torch

        num_devices = torch.cuda.device_count()
        num_samples_per_device = int((len(data_list) + num_devices - 1) / float(num_devices))
        splited_lists = list()
        for i in range(num_devices):
            begin = i * num_samples_per_device
            end = min((i + 1) * num_samples_per_device, len(data_list))
            splited_lists.append(data_list[begin:end])
        return splited_lists

    if type(batch[0]).__module__ == "numpy":
        return [torch.from_numpy(b) for b in batch]
    elif isinstance(batch[0], container_abcs.Sequence):
        return [[torch.from_numpy(sample) for sample in b] for b in batch]
    elif isinstance(batch[0], container_abcs.Mapping):
        ret_dicts = {key: split_data_list(collate_fn([d[key] for d in batch])) for key in batch[0]}
        num_splits = len(ret_dicts["coords"])
        ret_dicts = [{key: ret_dicts[key][i] for key in ret_dicts} for i in range(num_splits)]
        for i in range(num_splits):
            coords_batch = ret_dicts[i]["coords"]
            feats_batch = ret_dicts[i]["feats"]
            if "objectness_label" in ret_dicts[i]:
                labels_batch = ret_dicts[i]["objectness_label"]
                coords_batch, feats_batch, labels_batch = ME.utils.sparse_collate(
                    coords_batch, feats_batch, labels_batch
                )
                ret_dicts[i]["objectness_label"] = labels_batch
            else:
                coords_batch, feats_batch = ME.utils.sparse_collate(coords_batch, feats_batch)
            ret_dicts[i]["coords"] = coords_batch
            ret_dicts[i]["feats"] = feats_batch
        return ret_dicts

    raise TypeError("batch must contain tensors, dicts or lists; found {}".format(type(batch[0])))


def convert_data_to_device(data, device):
    if isinstance(data, container_abcs.Mapping):
        return {key: convert_data_to_device(data[key], device) for key in data}
    elif isinstance(data, container_abcs.Sequence):
        return [convert_data_to_device(data[i], device) for i in range(len(data))]
    else:
        return data.to(device)


def convert_data_to_gpu(data):
    def convert_data_to_device(data, device):
        if isinstance(data, container_abcs.Sequence):
            return [convert_data_to_device(data[i], device) for i in range(len(data))]
        else:
            return data.to(device)

    devices = list(range(torch.cuda.device_count()))
    src_dev = devices[0]
    if len(devices) > 1:
        dist_devs = devices[1:]
    else:
        dist_devs = devices

    ret_dict = dict()
    for key in data:
        if key in CONVERT_TO_NEW_DEIVCE:
            dist_data_list = []
            for i in range(len(data[key])):
                dst_dev = dist_devs[i % len(dist_devs)]
                dist_data_list.append(convert_data_to_device(data[key][i], dst_dev))
            ret_dict[key] = dist_data_list
        else:
            ret_dict[key] = convert_data_to_device(data[key], src_dev)
    return ret_dict


if __name__ == "__main__":
    root = "logs/data/representation_model/graspnet_v1_newformat/"
    valid_obj_idxs, grasp_labels = load_grasp_labels(root)
    train_dataset = GraspNetVoxelizationDataset(
        root, valid_obj_idxs, grasp_labels, split="train", voxel_size=0.005, remove_outlier=True, remove_invisible=True
    )
    print(len(train_dataset))

    end_points = train_dataset[233]
    coords = end_points["coords"]
    cloud = end_points["point_clouds"]
    seg = end_points["objectness_label"]
    print(cloud.shape)
    print(seg.shape)
    print((seg > 0).sum())
    print(np.unique(coords[:, 0]))
    print(np.unique(coords[:, 1]))
    print(np.unique(coords[:, 2]))
    for i in range(256):
        end_points = train_dataset[i * 100]
