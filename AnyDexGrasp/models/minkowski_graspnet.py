import torch
import torch.nn as nn
import torch.nn.functional as F


class MinkowskiGraspNetMultifingerType1(nn.Module):
    def __init__(
        self, num_multifinger_type=12, num_multifinger_depth=4, num_two_finger_angle=12, num_two_finger_depth=5
    ):
        super().__init__()
        dropout_pro = 0.0
        self.dropout1 = nn.Dropout(p=dropout_pro)
        self.dropout2 = nn.Dropout(p=dropout_pro)
        self.dropout3 = nn.Dropout(p=dropout_pro)
        self.num_multifinger_type = num_multifinger_type
        self.num_multifinger_depth = num_multifinger_depth
        self.num_two_finger_angle = num_two_finger_angle
        self.num_two_finger_depth = num_two_finger_depth

        output_dim = num_multifinger_type * num_multifinger_depth * num_two_finger_angle * num_two_finger_depth  # 11

        self.conv1 = nn.Conv1d(480, 512, 1)
        self.conv2 = nn.Conv1d(512, 1024, 1)
        self.conv3 = nn.Conv1d(1024, 512, 1)
        self.conv4 = nn.Conv1d(512, 1024, 1)
        self.conv5 = nn.Conv1d(1024, 512, 1)
        self.conv6 = nn.Conv1d(512, 512, 1)
        self.conv7 = nn.Conv1d(512, output_dim, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(1024)
        self.bn3 = nn.BatchNorm1d(512)
        self.bn4 = nn.BatchNorm1d(1024)
        self.bn5 = nn.BatchNorm1d(512)
        self.bn6 = nn.BatchNorm1d(512)
        self.bn7 = nn.BatchNorm1d(output_dim)

    def forward(self, end_points):
        B = end_points["two_fingers_pose_angle_type"].size()[0]
        two_fingers_pose_angle_type = end_points["two_fingers_pose_angle_type"]
        two_fingers_pose_depth_type = end_points["two_fingers_pose_depth_type"]
        multifinger_pose_finger_type = end_points["multifinger_pose_finger_type"]
        multifinger_pose_depth_type = end_points["multifinger_pose_depth_type"]

        grasp_preds_features = end_points["grasp_preds_features"]

        if_flip = end_points["if_flip"]

        grasp_features_five_finger = torch.cat([grasp_preds_features], dim=1).view(B, -1, 1)

        grasp_preds = F.relu(self.bn1(self.conv1(grasp_features_five_finger)), inplace=True)
        grasp_preds_res = F.relu(self.bn2(self.conv2(grasp_preds)), inplace=True)
        grasp_preds = F.relu(self.bn3(self.conv3(grasp_preds_res)), inplace=True)
        grasp_preds = F.relu(self.bn4(self.conv4(grasp_preds)), inplace=True)
        grasp_preds = F.relu(self.bn5(self.conv5(grasp_preds + grasp_preds_res)), inplace=True)
        grasp_preds = F.relu(self.bn6(self.conv6(grasp_preds)), inplace=True)
        grasp_preds = (
            F.relu((self.dropout3(self.bn7(self.conv7(grasp_preds)))), inplace=True).transpose(1, 2).contiguous()
        )

        end_points["stage4_grasp_preds_five_hand"] = grasp_preds

        return grasp_preds, end_points
