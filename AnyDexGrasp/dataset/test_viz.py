from graspnetAPI import GraspNet

graspnet_root = "logs/data/representation_model/graspnet_v1_newformat"

g = GraspNet(graspnet_root, camera="realsense", split="custom", sceneIds=list(range(5)))


# show 6d poses -- OK
# g.show6DPose(sceneIds = 0, show = True)

# show scene rectangle grasps
# Error: 'logs/data/representation_model/graspnet_v1_newformat/scenes/scene_0000/realsense/rect/0000.npy'
g.showSceneGrasp(sceneId=0, camera="realsense", annId=0, format="rect", numGrasp=20)

# show object grasps
# Error: logs/data/representation_model/graspnet_v1_newformat//grasp_label/000_labels.npz'
# g.showObjGrasp(objIds = 0, show=True)

print()
