import numpy as np

import torch
import torch.nn as nn

from knn_pytorch.knn_modules import knn
from ..utils.pt_utils import (
    huber_loss,
    focal_loss,
    transform_point_cloud,
    batch_viewpoint_params_to_matrix,
    generate_grasp_views,
)

MAX_GRASP_WIDTH = 0.08
MAX_MU = 1.0
MIN_MU = 0.1
MAX_GRASP_SCORE = torch.log(torch.Tensor([MAX_MU / MIN_MU])).item()
DISTANCE_THRESH = 0.005
VIEW_ANGLE_DISTANCE_THRESH = 0.4  # (max nearest angle distance in 300 views is 0.20019782(~11.46 degrees))


def process_grasp_labels(end_points):
    clouds = end_points["point_clouds"]  # [(Ni,3),]
    src_dev = clouds[0].device
    objectness_label = end_points["objectness_label"]  # (\Sigma Ni,)
    coords = end_points["coords"]  # (\Sigma Ni, 1+3)

    batch_grasp_points = []
    batch_grasp_heatmap = []
    batch_grasp_views = []
    batch_grasp_views_rot = []
    batch_grasp_views_heatmap = []
    batch_grasp_labels = []
    batch_objectness_mask = []
    batch_grasp_widths = []
    stage1_graspness_mask = []
    stage2_graspness_mask = []
    stage2_grasp_view_mask = []
    for i in range(len(clouds)):
        poses = end_points["object_poses_list"][i]  # [(3, 4),]
        dst_dev = poses[0].device
        cloud = clouds[i].to(dst_dev)  # (Ni, 3)
        seed_inds = end_points["stage2_seed_inds"][i].to(dst_dev)  # (Ns,)
        cloud_mask = coords[:, 0] == i  # (Ni,)
        objectness_mask = objectness_label[cloud_mask].to(dst_dev)  # (Ni,)
        objectness_mask = objectness_mask[seed_inds].bool()  # (Ns,)

        # get merged grasp points for label computation
        grasp_points_merged = []
        grasp_heatmap_merged = []
        grasp_views_merged = []
        grasp_views_rot_merged = []
        grasp_views_heatmap_merged = []
        grasp_labels_merged = []
        grasp_widths_merged = []
        stage2_grasp_view_mask_merged = []
        for obj_idx, pose in enumerate(poses):
            grasp_points = end_points["grasp_points_list"][i][obj_idx]  # (Np,3)
            grasp_heatmap = end_points["grasp_heatmap_list"][i][obj_idx]  # (Np)
            grasp_labels = end_points["grasp_labels_list"][i][obj_idx]  # (Np, V, A, D)
            grasp_views_heatmap = end_points["grasp_view_heatmap_list"][i][obj_idx]  # (Np, V)
            grasp_widths = end_points["grasp_widths_list"][i][obj_idx]  # (Np, V, A, D)
            # grasp_heatmap_raw = end_points['grasp_heatmap_raw_list'][i][obj_idx] #(Np)
            num_grasp_points, V, A, D = grasp_labels.size()
            # transform grasp points and views
            grasp_views = generate_grasp_views(V).to(pose.device)  # (V, 3)
            grasp_points_trans = transform_point_cloud(grasp_points, pose, "3x4")
            grasp_views_trans = transform_point_cloud(grasp_views, pose[:3, :3], "3x3")
            # transform grasp view rotation
            angles = torch.zeros(grasp_views.size(0), dtype=grasp_views.dtype, device=grasp_views.device)
            grasp_views_rot = batch_viewpoint_params_to_matrix(-grasp_views, angles)  # (V, 3, 3)
            grasp_views_rot_trans = torch.matmul(pose[:3, :3], grasp_views_rot)  # (V, 3, 3)

            # assign views & compute view masks
            grasp_views_ = grasp_views.transpose(0, 1).contiguous().unsqueeze(0)
            grasp_views_trans_ = grasp_views_trans.transpose(0, 1).contiguous().unsqueeze(0)
            view_inds = knn(grasp_views_trans_, grasp_views_, k=1).squeeze() - 1
            grasp_views_trans = torch.index_select(grasp_views_trans, 0, view_inds)  # (V, 3)
            view_angle_dists = torch.sum(grasp_views * grasp_views_trans, dim=-1)  # (V)
            stage2_grasp_view_mask_i = view_angle_dists > np.cos(VIEW_ANGLE_DISTANCE_THRESH)  # (V)
            stage2_grasp_view_mask_i = stage2_grasp_view_mask_i.unsqueeze(0).expand(num_grasp_points, -1)  # (Np, V)
            grasp_views_trans = grasp_views_trans.unsqueeze(0).expand(num_grasp_points, -1, -1)  # (Np, V, 3)
            grasp_views_rot_trans = torch.index_select(grasp_views_rot_trans, 0, view_inds)  # (V, 3, 3)
            grasp_views_rot_trans = grasp_views_rot_trans.unsqueeze(0).expand(
                num_grasp_points, -1, -1, -1
            )  # (Np, V, 3, 3)
            grasp_labels = torch.index_select(grasp_labels, 1, view_inds)  # (Np, V, A, D)
            grasp_views_heatmap = torch.index_select(grasp_views_heatmap, 1, view_inds)  # (Np, V)
            grasp_widths = torch.index_select(grasp_widths, 1, view_inds)  # (Np, V, A, D)
            # add to list
            grasp_points_merged.append(grasp_points_trans)
            grasp_heatmap_merged.append(grasp_heatmap)
            grasp_views_merged.append(grasp_views_trans)
            grasp_views_rot_merged.append(grasp_views_rot_trans)
            grasp_views_heatmap_merged.append(grasp_views_heatmap)
            grasp_labels_merged.append(grasp_labels)
            grasp_widths_merged.append(grasp_widths)
            stage2_grasp_view_mask_merged.append(stage2_grasp_view_mask_i)
            # grasp_heatmap_raw_merged.append(grasp_heatmap_raw)
        # concat list
        grasp_points_merged = torch.cat(grasp_points_merged, dim=0)  # (Np', 3)
        grasp_heatmap_merged = torch.cat(grasp_heatmap_merged, dim=0)  # (Np')
        grasp_views_merged = torch.cat(grasp_views_merged, dim=0)  # (Np', V, 3)
        grasp_views_rot_merged = torch.cat(grasp_views_rot_merged, dim=0)  # (Np', V, 3, 3)
        grasp_labels_merged = torch.cat(grasp_labels_merged, dim=0)  # (Np', V, A, D)
        grasp_views_heatmap_merged = torch.cat(grasp_views_heatmap_merged, dim=0)  # (Np', V)
        grasp_widths_merged = torch.cat(grasp_widths_merged, dim=0)  # (Np', V, A, D)
        stage2_grasp_view_mask_merged = torch.cat(stage2_grasp_view_mask_merged, dim=0)  # (Np', V)
        # grasp_heatmap_raw_merged = torch.cat(grasp_heatmap_raw_merged, dim=0) #(Np')

        # compute nearest neighbors
        cloud_ = cloud.transpose(0, 1).contiguous().unsqueeze(0)  # (1,3,Ni)
        grasp_points_merged_ = grasp_points_merged.transpose(0, 1).contiguous().unsqueeze(0)  # (1,3,Np')
        nn_inds = knn(grasp_points_merged_, cloud_, k=1).squeeze() - 1  # (Ni)
        # compute graspness mask
        grasp_points_merged = torch.index_select(grasp_points_merged, 0, nn_inds)  # (Ni, 3)
        point_dists = torch.norm(cloud - grasp_points_merged, dim=1)
        stage1_graspness_mask_i = point_dists <= DISTANCE_THRESH  # (Ni)

        # assign grasp points to scene points
        grasp_points_merged = torch.index_select(grasp_points_merged, 0, seed_inds)  # (Ns, 3)
        grasp_heatmap_merged = torch.index_select(grasp_heatmap_merged, 0, nn_inds)  # (Ni)
        grasp_views_merged = torch.index_select(grasp_views_merged, 0, nn_inds[seed_inds])  # (Ns, V, 3)
        grasp_views_rot_merged = torch.index_select(grasp_views_rot_merged, 0, nn_inds[seed_inds])  # (Ns, V, 3, 3)
        grasp_labels_merged = torch.index_select(grasp_labels_merged, 0, nn_inds[seed_inds])  # (Ns, V, A, D)
        grasp_views_heatmap_merged = torch.index_select(grasp_views_heatmap_merged, 0, nn_inds[seed_inds])  # (Ns, V)
        grasp_widths_merged = torch.index_select(grasp_widths_merged, 0, nn_inds[seed_inds])  # (Ns, V, A, D)
        # grasp_heatmap_raw_merged = torch.index_select(grasp_heatmap_raw_merged, 0, nn_inds) #(Ni)
        stage2_graspness_mask_i = torch.index_select(stage1_graspness_mask_i, 0, seed_inds)  # (Ns)
        stage2_grasp_view_mask_merged = torch.index_select(
            stage2_grasp_view_mask_merged, 0, nn_inds[seed_inds]
        )  # (Ns, V)

        # add to batch
        batch_grasp_points.append(grasp_points_merged.to(src_dev))
        batch_grasp_heatmap.append(grasp_heatmap_merged.to(src_dev))
        batch_grasp_views.append(grasp_views_merged.to(src_dev))
        batch_grasp_views_rot.append(grasp_views_rot_merged.to(src_dev))
        batch_grasp_labels.append(grasp_labels_merged.to(src_dev))
        batch_grasp_views_heatmap.append(grasp_views_heatmap_merged.to(src_dev))
        batch_objectness_mask.append(objectness_mask.to(src_dev))
        batch_grasp_widths.append(grasp_widths_merged.to(src_dev))
        # batch_grasp_heatmap_raw.append(grasp_heatmap_raw_merged.to(src_dev))
        stage1_graspness_mask.append(stage1_graspness_mask_i.to(src_dev))
        stage2_graspness_mask.append(stage2_graspness_mask_i.to(src_dev))
        stage2_grasp_view_mask.append(stage2_grasp_view_mask_merged.to(src_dev))

    # concat batch
    batch_grasp_points = torch.stack(batch_grasp_points, dim=0)  # (B, Ns, 3)
    batch_grasp_heatmap = torch.cat(batch_grasp_heatmap, dim=0)  # (\Sigma Ni)
    batch_grasp_views = torch.stack(batch_grasp_views, dim=0)  # (B, Ns, V, 3)
    batch_grasp_views_rot = torch.stack(batch_grasp_views_rot, dim=0)  # (B, Ns, V, 3, 3)
    batch_grasp_labels = torch.stack(batch_grasp_labels, dim=0)  # (B, Ns, V, A, D)
    batch_grasp_views_heatmap = torch.stack(batch_grasp_views_heatmap, dim=0)  # (B, Ns, V)
    batch_objectness_mask = torch.stack(batch_objectness_mask, dim=0)  # (B, Ns)
    batch_grasp_widths = torch.stack(batch_grasp_widths, dim=0)  # (B, Ns, V, A, D)
    # batch_grasp_heatmap_raw = torch.cat(batch_grasp_heatmap_raw, dim=0) #(\Sigma Ni)
    stage1_graspness_mask = torch.cat(stage1_graspness_mask, dim=0)  # (\Sigma Ni)
    stage2_graspness_mask = torch.stack(stage2_graspness_mask, dim=0)  # (B, Ns)
    stage2_grasp_view_mask = torch.stack(stage2_grasp_view_mask, dim=0)  # (B, Ns, V)

    end_points["batch_grasp_point"] = batch_grasp_points
    end_points["batch_grasp_heatmap"] = batch_grasp_heatmap
    end_points["batch_grasp_view"] = batch_grasp_views
    end_points["batch_grasp_view_rot"] = batch_grasp_views_rot
    end_points["batch_grasp_label"] = batch_grasp_labels
    end_points["batch_grasp_width"] = batch_grasp_widths
    end_points["batch_grasp_view_heatmap"] = batch_grasp_views_heatmap
    end_points["stage2_objectness_mask"] = batch_objectness_mask
    end_points["stage1_graspness_mask"] = stage1_graspness_mask
    end_points["stage2_graspness_mask"] = stage2_graspness_mask
    end_points["stage2_grasp_view_mask"] = stage2_grasp_view_mask

    return end_points


def match_grasp_view_and_label(end_points):
    template_views = end_points["batch_grasp_view"]  # (B, Ns, V, 3)
    template_views_rot = end_points["batch_grasp_view_rot"]  # (B, Ns, V, 3, 3)
    grasp_labels = end_points["batch_grasp_label"]  # (B, Ns, V, A, D)
    grasp_widths = end_points["batch_grasp_width"]  # (B, Ns, V, A, D)
    stage2_grasp_view_mask = end_points["stage2_grasp_view_mask"]  # (B, Ns, V)
    top_view_inds = end_points["stage2_view_inds"]  # (B, Ns)

    B, Ns, V, A, D = grasp_labels.size()
    top_view_inds_ = top_view_inds.view(B, Ns, 1, 1).expand(-1, -1, -1, 3)
    top_template_views = torch.gather(template_views, 2, top_view_inds_).squeeze(2)  # (B, Ns, 3)
    top_view_inds_ = top_view_inds.view(B, Ns, 1, 1, 1).expand(-1, -1, -1, 3, 3)
    top_template_views_rot = torch.gather(template_views_rot, 2, top_view_inds_).squeeze(2)  # (B, Ns, 3, 3)
    top_view_inds_ = top_view_inds.view(B, Ns, 1, 1, 1).expand(-1, -1, -1, A, D)
    top_view_grasp_labels = torch.gather(grasp_labels, 2, top_view_inds_).squeeze(2)  # (B, Ns, A, D)
    top_view_grasp_widths = torch.gather(grasp_widths, 2, top_view_inds_).squeeze(2)  # (B, Ns, A, D)
    top_view_inds_ = top_view_inds.view(B, Ns, 1)
    stage3_grasp_view_mask = torch.gather(stage2_grasp_view_mask, 2, top_view_inds_).squeeze(2)  # (B, Ns)

    end_points["batch_grasp_view"] = top_template_views
    end_points["batch_grasp_view_rot"] = top_template_views_rot
    end_points["batch_grasp_label"] = top_view_grasp_labels
    end_points["batch_grasp_width"] = top_view_grasp_widths
    end_points["stage3_grasp_view_mask"] = stage3_grasp_view_mask

    return top_template_views_rot, end_points


def compute_objectness_loss(end_points):
    criterion = nn.CrossEntropyLoss(reduction="mean")
    objectness_score = end_points["stage1_objectness_pred"]
    objectness_label = end_points["objectness_label"]
    loss = criterion(objectness_score, objectness_label)

    end_points["losses/stage1_objectness_loss"] = loss
    objectness_pred = torch.argmax(objectness_score, 1)
    end_points["stage1_objectness_acc"] = (objectness_pred == objectness_label.long()).float().mean()
    end_points["stage1_objectness_prec"] = (
        (objectness_pred == objectness_label.long())[objectness_pred == 1].float().mean()
    )
    end_points["stage1_objectness_recall"] = (
        (objectness_pred == objectness_label.long())[objectness_label == 1].float().mean()
    )

    return loss, end_points


def compute_heatmap_loss(end_points):
    # grasp heatmap
    heatmap_pred = end_points["stage1_heatmap_pred"]
    heatmap_label = end_points["batch_grasp_heatmap"]
    objectness_mask = end_points["objectness_label"]
    graspness_mask = end_points["stage1_graspness_mask"]
    training_mask = (objectness_mask & graspness_mask).float()
    error = heatmap_pred - heatmap_label
    heatmap_loss = huber_loss(error, delta=1.0)
    heatmap_loss = torch.sum(heatmap_loss * training_mask) / (training_mask.sum() + 1e-6)
    acc_001 = torch.sum((torch.abs(error) < 0.01) * training_mask) / (training_mask.sum() + 1e-6)
    acc_003 = torch.sum((torch.abs(error) < 0.03) * training_mask) / (training_mask.sum() + 1e-6)
    acc_005 = torch.sum((torch.abs(error) < 0.05) * training_mask) / (training_mask.sum() + 1e-6)
    acc_010 = torch.sum((torch.abs(error) < 0.10) * training_mask) / (training_mask.sum() + 1e-6)

    end_points["losses/stage1_heatmap_loss"] = heatmap_loss
    end_points["stage1_heatmap_acc/001"] = acc_001
    end_points["stage1_heatmap_acc/003"] = acc_003
    end_points["stage1_heatmap_acc/005"] = acc_005
    end_points["stage1_heatmap_acc/010"] = acc_010
    end_points["stage1_heatmap_dist/label"] = heatmap_label[training_mask.bool()]
    end_points["stage1_heatmap_dist/pred"] = heatmap_pred[training_mask.bool()]

    # view heatmap
    view_heatmap_pred = end_points["stage2_view_heatmap_pred"]
    view_heatmap_label = end_points["batch_grasp_view_heatmap"]
    objectness_mask = end_points["stage2_objectness_mask"]
    graspness_mask = end_points["stage2_graspness_mask"]
    view_mask = end_points["stage2_grasp_view_mask"]
    training_mask = objectness_mask & graspness_mask
    training_mask = training_mask.unsqueeze(-1) & view_mask
    V = view_heatmap_pred.size(2)
    # Upd. Add half view sampling
    view_heatmap_label = view_heatmap_label[:, :, :V]
    training_mask = training_mask[:, :, :V].float()
    error = view_heatmap_pred - view_heatmap_label
    view_loss = huber_loss(error, delta=1.0)
    view_loss = torch.sum(view_loss * training_mask) / (training_mask.sum() + 1e-6)
    acc_001 = torch.sum((torch.abs(error) < 0.01) * training_mask) / (training_mask.sum() + 1e-6)
    acc_003 = torch.sum((torch.abs(error) < 0.03) * training_mask) / (training_mask.sum() + 1e-6)
    acc_005 = torch.sum((torch.abs(error) < 0.05) * training_mask) / (training_mask.sum() + 1e-6)
    acc_010 = torch.sum((torch.abs(error) < 0.10) * training_mask) / (training_mask.sum() + 1e-6)

    end_points["losses/stage2_view_heatmap_loss"] = view_loss
    end_points["stage2_view_heatmap_acc/001"] = acc_001
    end_points["stage2_view_heatmap_acc/003"] = acc_003
    end_points["stage2_view_heatmap_acc/005"] = acc_005
    end_points["stage2_view_heatmap_acc/010"] = acc_010
    end_points["stage2_view_heatmap_dist/label"] = view_heatmap_label[training_mask.bool()]
    end_points["stage2_view_heatmap_dist/pred"] = view_heatmap_pred[training_mask.bool()]

    loss = heatmap_loss + view_loss * 10
    return loss, end_points


def compute_grasp_loss(end_points):
    # load preds and labels
    grasp_score_pred = end_points["stage3_grasp_scores"]
    grasp_width_pred = end_points["stage3_normalized_grasp_widths"]
    top_view_grasp_labels = end_points["batch_grasp_label"]  # (B, Ns, A, D)
    top_view_grasp_widths = end_points["batch_grasp_width"]  # (B, Ns, A, D)
    B, Ns, A, D = top_view_grasp_labels.size()

    # process grasp labels
    objectness_mask = end_points["stage2_objectness_mask"]  # (B, Ns)
    graspness_mask = end_points["stage2_graspness_mask"]  # (B, Ns)
    view_mask = end_points["stage3_grasp_view_mask"]  # (B, Ns)
    training_mask = objectness_mask & graspness_mask & view_mask  # (B, Ns)
    log_mask = torch.zeros_like(top_view_grasp_labels).view(B, Ns, A * D)
    gt_idxs = torch.argmax(top_view_grasp_labels.view(B, Ns, A * D), dim=2, keepdim=True)
    log_mask.scatter_(dim=2, index=gt_idxs, src=torch.ones_like(gt_idxs, dtype=log_mask.dtype))
    training_mask = training_mask.view(B, Ns, 1, 1).expand(-1, -1, A, D)
    grasp_mask = top_view_grasp_labels > 0.01
    grasp_mask = grasp_mask & training_mask
    top_view_grasp_labels[grasp_mask] = torch.log(MAX_MU / top_view_grasp_labels[grasp_mask])
    top_view_grasp_labels[~grasp_mask] = 0
    top_view_grasp_labels = top_view_grasp_labels / MAX_GRASP_SCORE

    # 1. grasp score loss
    grasp_score_error = grasp_score_pred - top_view_grasp_labels
    grasp_score_loss = huber_loss(grasp_score_error, delta=1.0)
    grasp_score_loss = torch.sum(grasp_score_loss * training_mask.float()) / (training_mask.float().sum() + 1e-6)
    end_points["losses/stage3_grasp_score_loss"] = grasp_score_loss
    end_points["stage3_grasp_score_dist/label"] = top_view_grasp_labels[training_mask]
    end_points["stage3_grasp_score_dist/pred"] = grasp_score_pred[training_mask]
    acc_001 = torch.sum((torch.abs(grasp_score_error) < 0.01) * training_mask.float()) / (
        training_mask.float().sum() + 1e-6
    )
    acc_003 = torch.sum((torch.abs(grasp_score_error) < 0.03) * training_mask.float()) / (
        training_mask.float().sum() + 1e-6
    )
    acc_005 = torch.sum((torch.abs(grasp_score_error) < 0.05) * training_mask.float()) / (
        training_mask.float().sum() + 1e-6
    )
    acc_010 = torch.sum((torch.abs(grasp_score_error) < 0.10) * training_mask.float()) / (
        training_mask.float().sum() + 1e-6
    )
    end_points["stage3_grasp_score_acc/001"] = acc_001
    end_points["stage3_grasp_score_acc/003"] = acc_003
    end_points["stage3_grasp_score_acc/005"] = acc_005
    end_points["stage3_grasp_score_acc/010"] = acc_010

    # 2. grasp width loss
    grasp_width_error = grasp_width_pred - top_view_grasp_widths / MAX_GRASP_WIDTH
    grasp_width_loss = huber_loss(grasp_width_error, delta=1.0)
    grasp_width_loss = torch.sum(grasp_width_loss * grasp_mask) / (grasp_mask.float().sum() + 1e-6)
    end_points["losses/stage3_grasp_width_loss"] = grasp_width_loss
    end_points["stage3_grasp_width_dist/label"] = top_view_grasp_widths[grasp_mask]
    end_points["stage3_grasp_width_dist/pred"] = grasp_width_pred[grasp_mask] * MAX_GRASP_WIDTH
    acc_001 = torch.sum((torch.abs(grasp_width_error) < 0.01) * grasp_mask.float()) / (grasp_mask.float().sum() + 1e-6)
    acc_003 = torch.sum((torch.abs(grasp_width_error) < 0.03) * grasp_mask.float()) / (grasp_mask.float().sum() + 1e-6)
    acc_005 = torch.sum((torch.abs(grasp_width_error) < 0.05) * grasp_mask.float()) / (grasp_mask.float().sum() + 1e-6)
    acc_010 = torch.sum((torch.abs(grasp_width_error) < 0.10) * grasp_mask.float()) / (grasp_mask.float().sum() + 1e-6)
    end_points["stage3_grasp_width_acc/001"] = acc_001
    end_points["stage3_grasp_width_acc/003"] = acc_003
    end_points["stage3_grasp_width_acc/005"] = acc_005
    end_points["stage3_grasp_width_acc/010"] = acc_010

    # overall loss
    grasp_loss = grasp_score_loss + grasp_width_loss
    end_points["losses/stage3_grasp_loss"] = grasp_loss

    return grasp_loss, end_points


def get_loss(end_points):
    objectness_loss, end_points = compute_objectness_loss(end_points)
    heatmap_loss, end_points = compute_heatmap_loss(end_points)
    grasp_loss, end_points = compute_grasp_loss(end_points)
    loss = objectness_loss + heatmap_loss * 10 + grasp_loss * 10
    end_points["losses/overall"] = loss
    return loss, end_points


class GraspLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, end_points):
        return get_loss(end_points)


class MultifingerType1Loss(nn.Module):
    def __init__(
        self,
        num_multifinger_type=1,
        num_multifinger_depth=4,
        num_two_finger_angle=12,
        num_two_finger_depth=5,
        train_type=0,
        add_collision=False,
    ):
        super().__init__()
        self.num_multifinger_type = num_multifinger_type
        self.num_multifinger_depth = num_multifinger_depth
        self.num_two_finger_angle = num_two_finger_angle
        self.num_two_finger_depth = num_two_finger_depth
        self.train_type = train_type
        self.add_collision = add_collision
        self.mseloss = nn.MSELoss()
        self.bceloss = nn.BCELoss(reduction="mean")
        self.softmax_bceloss = nn.CrossEntropyLoss()
        self.true_threshold = 0.5
        self.false_threshold = 0.5

    def calculate_acc(self, y_true, y_pred):
        tp = (y_true * y_pred).sum().to(torch.float32)
        tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum().to(torch.float32)

        epsilon = 1e-7

        precision = tp / (tp + fp + epsilon)
        recall = tp / (tp + fn + epsilon)

        f1 = 2 * (precision * recall) / (precision + recall + epsilon)
        return precision, recall, f1, tp

    def calculate_every_class_acc(self, y_true, y_pred, multifinger_hand_finger_type):
        every_class_acc = []
        for multifinger_type in range(self.num_multifinger_type):
            y_true_multifinger_type = y_true[torch.where(multifinger_hand_finger_type == self.train_type)[0]]
            y_pred_multifinger_type = y_pred[torch.where(multifinger_hand_finger_type == self.train_type)[0]]
            p, r, f1, tp = self.calculate_acc(y_true_multifinger_type, y_pred_multifinger_type)
            every_class_acc.append([p, r, f1, tp])
        return every_class_acc

    def forward(self, end_points):
        B = end_points["stage4_grasp_preds_five_hand"].shape[0]
        multifinger_hand_finger_type = end_points["multifinger_pose_finger_type"]
        multifinger_hand_depth_type = end_points["multifinger_pose_depth_type"]
        two_fingers_pose_angle_type = end_points["two_fingers_pose_angle_type"]
        two_fingers_pose_depth_type = end_points["two_fingers_pose_depth_type"]
        label = end_points["result"].float().view(-1)

        valid_place = (
            (
                (0 * self.num_two_finger_depth + two_fingers_pose_depth_type) * self.num_multifinger_depth
                + multifinger_hand_depth_type
            )
            .long()
            .view(-1)
        )

        batch_range = torch.tensor(range(B), dtype=valid_place.dtype)
        all_scores = end_points["stage4_grasp_preds_five_hand"].view(B, -1)  # (batch, 16)

        if not self.add_collision:
            selected_scores = all_scores[batch_range, valid_place]

            loss = self.mseloss(selected_scores, label)  # + 0.5*self.mseloss(selected_scores, selected_scores_label)

            selected_scores_acc = torch.where(
                selected_scores > self.true_threshold,
                torch.ones(selected_scores.size(), dtype=selected_scores.dtype, device=selected_scores.device),
                selected_scores,
            )
            selected_scores_acc = torch.where(
                selected_scores_acc < self.false_threshold,
                torch.zeros(
                    selected_scores_acc.size(), dtype=selected_scores_acc.dtype, device=selected_scores_acc.device
                ),
                selected_scores_acc,
            )

            selected_scores_acc_7 = torch.where(
                selected_scores > 0.7,
                torch.ones(selected_scores.size(), dtype=selected_scores.dtype, device=selected_scores.device),
                selected_scores,
            )
            selected_scores_acc_7 = torch.where(
                selected_scores_acc_7 < 0.7,
                torch.zeros(
                    selected_scores_acc_7.size(), dtype=selected_scores_acc_7.dtype, device=selected_scores_acc_7.device
                ),
                selected_scores_acc_7,
            )

            selected_scores_acc_9 = torch.where(
                selected_scores > 0.9,
                torch.ones(selected_scores.size(), dtype=selected_scores.dtype, device=selected_scores.device),
                selected_scores,
            )
            selected_scores_acc_9 = torch.where(
                selected_scores_acc_9 < 0.9,
                torch.zeros(
                    selected_scores_acc_9.size(), dtype=selected_scores_acc_9.dtype, device=selected_scores_acc_9.device
                ),
                selected_scores_acc_9,
            )

            pre_5, recall_5, f1_5, tp_5 = self.calculate_acc(label, selected_scores_acc)
            pre_7, recall_7, f1_7, tp_7 = self.calculate_acc(label, selected_scores_acc_7)
            pre_9, recall_9, f1_9, tp_9 = self.calculate_acc(label, selected_scores_acc_9)

            every_class_acc_5 = self.calculate_every_class_acc(label, selected_scores_acc, multifinger_hand_finger_type)
            every_class_acc_7 = self.calculate_every_class_acc(
                label, selected_scores_acc_7, multifinger_hand_finger_type
            )
            every_class_acc_9 = self.calculate_every_class_acc(
                label, selected_scores_acc_9, multifinger_hand_finger_type
            )
            return (
                loss,
                [pre_5, recall_5, f1_5, pre_7, recall_7, f1_7, pre_9, recall_9, f1_9],
                [every_class_acc_5, every_class_acc_7, every_class_acc_9],
            )
        else:
            if_collision = torch.logical_not(end_points["if_collision"]).int()
            not_collision = -torch.ones(if_collision.size(), dtype=if_collision.dtype, device=if_collision.device)
            acc_label = torch.where(if_collision == 1, not_collision, if_collision)
            acc_label_index = acc_label.index_put((batch_range, valid_place), label.int())
            acc_label = acc_label_index[
                (acc_label_index > -0.5) & ((all_scores > self.true_threshold) | (all_scores < self.false_threshold))
            ]

            acc_scores = torch.where(
                all_scores > self.true_threshold,
                torch.ones(all_scores.size(), dtype=all_scores.dtype, device=all_scores.device),
                all_scores,
            )
            acc_scores = torch.where(
                acc_scores < self.false_threshold,
                torch.zeros(acc_scores.size(), dtype=acc_scores.dtype, device=acc_scores.device),
                acc_scores,
            )

            acc_scores = acc_scores[
                (acc_label_index > -0.5) & ((acc_scores > self.true_threshold) | (acc_scores < self.false_threshold))
            ]

            pre, recall, f1 = self.calculate_acc(acc_label, acc_scores)
            selected_scores = all_scores[
                (acc_label_index > -0.5) & ((all_scores > self.true_threshold) | (all_scores < self.false_threshold))
            ]

            loss = torch.mean(focal_loss(selected_scores, acc_label, alpha=0.7, gamma=2)) * 10
            return loss, [pre, recall, f1]
