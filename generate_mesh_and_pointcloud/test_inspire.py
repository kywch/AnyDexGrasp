import json
import math

import numpy as np
import open3d as o3d
from tqdm import tqdm

import pybullet as p
import pybullet_data

from transforms3d.euler import quat2mat

from generate_mesh_and_pointcloud.recover_inspire_hand_to_stl import (
    read_excel_6Dangle_to_12Dangle,
    save_stl_and_pointcloud,
)


# Below copied from: ur_toolbox.robot.InspireHandR_grasp import grasp_types
# What are facenet_thumb and facenet_index?
grasp_types = {
    "1": {
        "name": "Ring",
        "facenet_thumb": [[207598, 207599]],
        "facenet_index": [[146358, 146357], [53344, 53345]],
        "width": [0, 0.11],
    },
    "2": {
        "name": "Prismatic_2_Finger",
        "facenet_thumb": [[207598, 207599]],
        "facenet_index": [[146358, 146357], [53344, 53345]],
        "width": [0, 0.11],
    },
    "3": {
        "name": "Prismatic_3_Finger",
        "facenet_thumb": [[207598, 207599]],
        "facenet_index": [[146358, 146357], [53344, 53345]],
        "width": [0, 0.11],
    },
    "4": {
        "name": "Large_Diameter",
        "facenet_thumb": [[207598, 207599]],
        "facenet_index": [[146358, 146357], [53344, 53345]],
        "width": [0, 0.11],
    },
    "5": {
        "name": "Medium_Wrap",
        "facenet_thumb": [[207416]],
        "facenet_index": [[146358, 146357], [53344, 53345]],
        "width": [0, 0.11],
    },
    "6": {
        "name": "Tripod",
        "facenet_thumb": [[207416]],
        "facenet_index": [[146358, 146357], [52220]],
        "width": [0.025, 0.10],
    },
    "7": {
        "name": "Sphere_3_Finger",
        "facenet_thumb": [[207416]],
        "facenet_index": [[146358, 146357], [81881]],
        "width": [0.025, 0.10],
    },
    "8": {
        "name": "Distal_Type",
        "facenet_thumb": [[207606]],
        "facenet_index": [[53344, 53345], [82744]],
        "width": [0.025, 0.10],
    },
}

joint_mesh_mapping = {
    "base": "Link111.STL",
    "0": "Link1.STL",
    "1": "Link11.STL",
    "2": "Link2.STL",
    "3": "Link22.STL",
    "4": "Link3.STL",
    "5": "Link33.STL",
    "6": "Link4.STL",
    "7": "Link44.STL",
    "8": "Link5.STL",
    "9": "Link51.STL",
    "10": "Link52.STL",
    "11": "Link53.STL",
}


def four_meta_to_matrix(x):
    xyz = [x[0], x[1], x[2]]
    rpy = [x[6], x[3], x[4], x[5]]  # quat in wxyz?

    rot_mat = quat2mat(rpy)
    return np.array(
        [
            [rot_mat[0][0], rot_mat[0][1], rot_mat[0][2], xyz[0]],
            [rot_mat[1][0], rot_mat[1][1], rot_mat[1][2], xyz[1]],
            [rot_mat[2][0], rot_mat[2][1], rot_mat[2][2], xyz[2]],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )


def get_mesh(po, id, PATH):
    t_angle = four_meta_to_matrix(po)
    path_file = PATH + joint_mesh_mapping[str(id)]
    link = o3d.io.read_triangle_mesh(path_file)
    link = link.transform(t_angle)
    return link


class InspireAngles2STLs:
    def __init__(self, grasp_types):
        self.grasp_types = grasp_types

    def normalize(self, x):
        return np.array([x[0], x[1], x[2]]) / math.sqrt(np.power(x[0], 2) + np.power(x[1], 2) + np.power(x[2], 2))

    def get_normal_vector(self, p1, p2, p3):
        a = (p2[1] - p1[1]) * (p3[2] - p1[2]) - (p2[2] - p1[2]) * (p3[1] - p1[1])
        b = (p2[2] - p1[2]) * (p3[0] - p1[0]) - (p2[0] - p1[0]) * (p3[2] - p1[2])
        c = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
        return np.array([a, b, c])

    def di(self, x, y):
        return np.sqrt(np.power(x[0] - y[0], 2) + np.power(x[1] - y[1], 2) + np.power(x[2] - y[2], 2))

    def compute_center(self, meshes, facenet_thumb, facenet_index):
        distance1s = []
        triangle1 = meshes.triangles[facenet_thumb[0]]
        distance = [meshes.vertices[triangle1[0]], meshes.vertices[triangle1[1]], meshes.vertices[triangle1[2]]]
        distance1s.append(distance)

        triangle1 = meshes.triangles[facenet_thumb[1]]
        distance = [meshes.vertices[triangle1[0]], meshes.vertices[triangle1[1]], meshes.vertices[triangle1[2]]]
        distance1s.append(distance)

        distance2s = []
        triangle1 = meshes.triangles[facenet_index[0]]
        distance = [meshes.vertices[triangle1[0]], meshes.vertices[triangle1[1]], meshes.vertices[triangle1[2]]]
        distance2s.append(distance)
        triangle1 = meshes.triangles[facenet_index[1]]

        distance = [meshes.vertices[triangle1[0]], meshes.vertices[triangle1[1]], meshes.vertices[triangle1[2]]]
        distance2s.append(distance)

        midpoint = []
        center1 = []
        center2 = []
        for n in range(2):
            center1.append((distance1s[n][0] + distance1s[n][1] + distance1s[n][2]) / 3.0)
            center2.append((distance2s[n][0] + distance2s[n][1] + distance2s[n][2]) / 3.0)
        center = [(center1[0] + center1[1]) / 2.0, (center2[0] + center2[1]) / 2.0]
        midpoint = (center[0] + center[1]) / 2.0
        mesh_di = self.di(center[0], center[1])
        return midpoint, center, mesh_di

    def get_center_orientation(self, meshes, facnet):
        triangle1 = meshes.triangles[facnet]
        point = (meshes.vertices[triangle1[0]] + meshes.vertices[triangle1[1]] + meshes.vertices[triangle1[2]]) / 3
        return point

    def thumb_index_grasp_inforamtion(self, meshes, facenet_thumb, facenet_index):
        point0 = (
            self.get_center_orientation(meshes, facenet_thumb[0][0])
            + self.get_center_orientation(meshes, facenet_thumb[0][1])
        ) / 2
        point1 = (
            self.get_center_orientation(meshes, facenet_index[0][0])
            + self.get_center_orientation(meshes, facenet_index[0][1])
        ) / 2
        grasp_translation = (point0 + point1) / 2

        tool_x = self.normalize(
            self.get_normal_vector(point0, point1, self.get_center_orientation(meshes, facenet_index[0][1]))
        )
        tool_y = self.normalize(point0 - grasp_translation)
        tool_z = np.cross(tool_x, tool_y)
        grasp_rotation = np.c_[tool_x, tool_y, tool_z]
        return grasp_translation, grasp_rotation

    def common_grasp_information(self, grasp_type, meshes, facenet_thumb, facenet_index):
        midpoint1, centers1, mesh_di1 = self.compute_center(
            meshes, facenet_thumb=facenet_thumb[0], facenet_index=facenet_index[0]
        )
        midpoint2, centers2, mesh_di2 = self.compute_center(
            meshes, facenet_thumb=facenet_thumb[0], facenet_index=facenet_index[1]
        )
        midpoint = (midpoint1 + midpoint2) / 2.0
        centers = [(centers1[0] + centers2[0]) / 2.0, (centers1[1] + centers2[1]) / 2.0]

        triangle1 = meshes.triangles[145998]
        center1 = (meshes.vertices[triangle1[0]] + meshes.vertices[triangle1[1]] + meshes.vertices[triangle1[2]]) / 3.0
        triangle2 = meshes.triangles[145999]
        center2 = (meshes.vertices[triangle2[0]] + meshes.vertices[triangle2[1]] + meshes.vertices[triangle2[2]]) / 3.0
        x1 = (center1 + center2) / 2.0

        triangle1 = meshes.triangles[52980]
        center1 = (meshes.vertices[triangle1[0]] + meshes.vertices[triangle1[1]] + meshes.vertices[triangle1[2]]) / 3.0
        triangle2 = meshes.triangles[52981]
        center2 = (meshes.vertices[triangle2[0]] + meshes.vertices[triangle2[1]] + meshes.vertices[triangle2[2]]) / 3.0
        x2 = (center1 + center2) / 2.0

        x = (x1 + x2) / 2.0

        if grasp_type == 0:
            x = x1
            midpoint = midpoint1
            centers = centers1
        vector = self.get_normal_vector(centers[0], centers[1], x)
        vector = self.normalize(vector) / 40
        p4 = np.array(midpoint) + np.array(vector)
        normal_vector = self.get_normal_vector(centers[1], centers[0], p4)
        normal_vector = self.normalize(normal_vector) / 40
        p5 = np.array(midpoint) + np.array(normal_vector)

        rotation = [
            [
                list(self.normalize(p5 - midpoint))[0],
                list(self.normalize(centers[0] - midpoint))[0],
                list(self.normalize(p4 - midpoint))[0],
            ],
            [
                list(self.normalize(p5 - midpoint))[1],
                list(self.normalize(centers[0] - midpoint))[1],
                list(self.normalize(p4 - midpoint))[1],
            ],
            [
                list(self.normalize(p5 - midpoint))[2],
                list(self.normalize(centers[0] - midpoint))[2],
                list(self.normalize(p4 - midpoint))[2],
            ],
        ]

        translation = midpoint
        return np.array(translation), np.array(rotation)

    def special_grasp_information(self, grasp_type, meshes, facenet_thumb, facenet_index):
        point0 = self.get_center_orientation(meshes, facenet_thumb[0][0])
        point1 = (
            self.get_center_orientation(meshes, facenet_index[0][0])
            + self.get_center_orientation(meshes, facenet_index[0][1])
        ) / 2
        point2 = self.get_center_orientation(meshes, facenet_index[1][0])

        end_point0 = point0
        end_point1 = (point1 + point2) / 2
        grasp_translation = (end_point0 + end_point1) / 2

        # The finger is not vertical
        if grasp_type == 10:
            grasp_translation[1] = grasp_translation[1] - 0.003
        tool_x = self.normalize(self.get_normal_vector(point0, point1, point2))
        tool_y = self.normalize(point0 - grasp_translation)
        tool_z = np.cross(tool_x, tool_y)
        grasp_rotation = np.c_[tool_x, tool_y, tool_z]
        return grasp_translation, grasp_rotation

    def thumb_index_lateral_grasp_information(self, meshes, facenet_thumb, facenet_index):
        point0 = self.get_center_orientation(meshes, facenet_thumb[0][0])
        point1 = (
            self.get_center_orientation(meshes, facenet_index[0][0])
            + self.get_center_orientation(meshes, facenet_index[0][1])
        ) / 2
        grasp_translation = (point0 + point1) / 2

        tool_x = self.normalize(
            self.get_normal_vector(point1, point0, self.get_center_orientation(meshes, facenet_index[0][1]))
        )
        tool_y = self.normalize(point0 - grasp_translation)
        tool_z = np.cross(tool_x, tool_y)
        grasp_rotation = np.c_[tool_x, tool_y, tool_z]
        return grasp_translation, grasp_rotation

    def get_pose_information(self, meshes, grasp_type, vis=False):
        facenet_thumb = self.grasp_types[str(grasp_type + 1)]["facenet_thumb"]
        facenet_index = self.grasp_types[str(grasp_type + 1)]["facenet_index"]
        if grasp_type < 4:
            translation, rotation = self.common_grasp_information(grasp_type, meshes, facenet_thumb, facenet_index)
        elif grasp_type < 11:
            translation, rotation = self.special_grasp_information(grasp_type, meshes, facenet_thumb, facenet_index)
        elif grasp_type == 11:
            translation, rotation = self.thumb_index_lateral_grasp_information(meshes, facenet_thumb, facenet_index)

        if vis:
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
            grasp_matrix = np.vstack((np.hstack((rotation, translation.reshape((3, 1)))), np.array((0, 0, 0, 1))))
            frame_grasp = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1).transform(grasp_matrix)
            o3d.visualization.draw_geometries([meshes, frame, frame_grasp])
        return translation, rotation


def open_pybullet(urdf_path, vis=False):
    clid = p.connect(p.SHARED_MEMORY)
    if clid < 0:
        if vis:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setPhysicsEngineParameter(solverResidualThreshold=0, maxNumCmdPer1ms=1000)
    fps = 240
    timeStep = 1.0 / fps
    p.setTimeStep(timeStep)
    p.resetDebugVisualizerCamera(
        cameraDistance=1.3, cameraYaw=38, cameraPitch=-22, cameraTargetPosition=[0.35, -0.13, 0.5]
    )

    flags = p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES

    base_orn = [0, 0, 0]
    base_orn = p.getQuaternionFromEuler(base_orn)

    hand = p.loadURDF(urdf_path, [0.0, 0.0, 0.0], base_orn, useFixedBase=True)
    return p, hand


def get_meshes(angles, stl_path, output_path, width_12Dangle_6Dangel_json, urdf_path, if_source, vis):
    p, hand = open_pybullet(urdf_path)
    angles_to_stls = InspireAngles2STLs(grasp_types)
    width_12Dangle_6Dangel = dict()
    for grasp_type, angle8 in enumerate(angles):
        for id in tqdm([i for i in range(len(angle8))]):
            angle = angle8[id]
            width = np.round(angle[12], 1)

            path_file = stl_path + joint_mesh_mapping["base"]
            base = o3d.io.read_triangle_mesh(path_file)
            pose = []

            for joint_id in range(12):
                p.resetJointState(hand, joint_id, angle[joint_id])
            for joint_id in range(12):
                po = p.getLinkState(hand, joint_id)
                pose.append([po[4][0], po[4][1], po[4][2], po[5][0], po[5][1], po[5][2], po[5][3]])
            for joint_id in range(2, 8):
                mesh = get_mesh(pose[joint_id], joint_id, stl_path)
                base = base + mesh
            for joint_id in range(2):
                mesh = get_mesh(pose[joint_id], joint_id, stl_path)
                base = base + mesh
            for joint_id in range(8, 12):
                mesh = get_mesh(pose[joint_id], joint_id, stl_path)
                base = base + mesh
            # transform to the center of wrist
            base.transform([[1, 0, 0, 0.04123], [0, 1, 0, 0.00804], [0, 0, 1, -0.01796], [0, 0, 0, 1]])
            # add the ring of metal which is used to fix screw
            base.transform([[1, 0, 0, 0.0078], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
            base.compute_triangle_normals()
            name = grasp_types[str(grasp_type + 1)]["name"]
            save_stl_and_pointcloud(name, angle, base, output_path)
            if if_source:
                if grasp_type == 4:
                    translation = width_12Dangle_6Dangel["Ring"][str(width)]["translation"]
                    rotation = width_12Dangle_6Dangel["Ring"][str(width)]["rotation"]
                else:
                    translation, rotation = angles_to_stls.get_pose_information(base, grasp_type, vis=vis)
                    translation, rotation = translation.tolist(), rotation.tolist()
                if str(name) not in width_12Dangle_6Dangel.keys():
                    width_12Dangle_6Dangel[name] = dict()
                if width in width_12Dangle_6Dangel[grasp_types[str(grasp_type + 1)]["name"]].keys():
                    print(
                        "*******************************\n\nerror!!!!!!!!",
                        grasp_type,
                        grasp_type[str(grasp_type + 1)],
                        width,
                    )
                width_12Dangle_6Dangel[name][str(width)] = {
                    "12d": angles[grasp_type][id][:12],
                    "6d": angles[grasp_type][id][13:],
                    "translation": translation,
                    "rotation": rotation,
                }

    # NOTE: Re-creating width_12Dangle_6Dangel.json file, which is already included in the repo
    if if_source:
        json_str = json.dumps(width_12Dangle_6Dangel, indent=4)
        with open(width_12Dangle_6Dangel_json, "w") as json_file:
            json_file.write(json_str)


if __name__ == "__main__":
    path_6d = "./generate_mesh_and_pointcloud/inspire_urdf/inspire_hand_routine_to_angle-use.xlsx"
    path_12d = "./generate_mesh_and_pointcloud/inspire_urdf/driver_routine_to_angle.xls"

    source_stl_path = "./generate_mesh_and_pointcloud/inspire_urdf/urdf-five3/meshes/"
    output_path = "./generate_mesh_and_pointcloud/inspire_urdf/meshes/"
    json_path = "./generate_mesh_and_pointcloud/inspire_urdf/width_12Dangle_6Dangle.json"

    urdf_path = "./generate_mesh_and_pointcloud/inspire_urdf/urdf-five3/robots/urdf-five3.urdf"

    ### Processing excel files are isolated here
    angles = read_excel_6Dangle_to_12Dangle(path_6d, path_12d)
    # len(angles) = 8, for each grip type

    # get_meshes(angles, source_stl_path, output_path, json_path, urdf_path, if_source=False, vis=True)

    angles_to_stls = InspireAngles2STLs(grasp_types)

    ### Visualize the grasp pose and axes
    # Get one instance of for looptranslation
    grasp_type = 7  # 0 - 7
    grasp_name = grasp_types[str(grasp_type + 1)]["name"]
    angle8 = angles[grasp_type]
    id = 10

    angle = angle8[id]
    width = np.round(angle[12], 1)

    p, hand = open_pybullet(urdf_path, vis=False)

    # Set the joint angles, as defined in the 12d angles
    for joint_id in range(12):
        p.resetJointState(hand, joint_id, angle[joint_id])

    pose = []
    for joint_id in range(12):
        po = p.getLinkState(hand, joint_id)
        pose.append([po[4][0], po[4][1], po[4][2], po[5][0], po[5][1], po[5][2], po[5][3]])
        # Use four_meta_to_matrix() to convert pose into a homogeneous matrix (4x4)

    curr_pos, curr_ori = p.getBasePositionAndOrientation(hand)

    # Make a new mesh to export
    path_file = source_stl_path + joint_mesh_mapping["base"]
    base = o3d.io.read_triangle_mesh(path_file)

    # NOTE: does meshes need to be brought in a specific order?
    # get_mesh applies individual transformations to each mesh
    for joint_id in range(2, 8):
        mesh = get_mesh(pose[joint_id], joint_id, source_stl_path)
        base = base + mesh
    for joint_id in range(2):
        mesh = get_mesh(pose[joint_id], joint_id, source_stl_path)
        base = base + mesh
    for joint_id in range(8, 12):
        mesh = get_mesh(pose[joint_id], joint_id, source_stl_path)
        base = base + mesh

    # transform to the center of wrist
    base.transform([[1, 0, 0, 0.04123], [0, 1, 0, 0.00804], [0, 0, 1, -0.01796], [0, 0, 0, 1]])
    # add the ring of metal which is used to fix screw
    # base.transform([[1, 0, 0, 0.0078],
    #                 [0, 1, 0, 0],
    #                 [0, 0, 1, 0],
    #                 [0, 0, 0, 1]])
    base.compute_triangle_normals()

    translation, rotation = angles_to_stls.get_pose_information(base, grasp_type, vis=True)

    print()
