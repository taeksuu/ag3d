import os
import cv2
import argparse
import glob
import ffmpeg
import shutil
import pickle
import numpy as np
import torch


from scipy.spatial.transform import Rotation as R
from training.fast_snarf.lib.model.smpl import SMPLServer

def mat2vec(R):
    # reference: https://courses.cs.duke.edu/fall13/compsci527/notes/rodrigues.pdf
    A = 0.5 * (R - R.transpose(1, 2))
    rho = A.reshape(-1, 9)[:, [7, 2, 3]]
    s = torch.norm(rho, dim=1, keepdim=True)
    c = 0.5 * (R.reshape(-1, 9)[:, [0, 4, 8]].sum(dim=1, keepdim=True) - 1)
    u = rho / s
    theta = torch.arctan2(s, c)

    rotvec = u * theta
    return rotvec

smpl_server = SMPLServer()

with open('/home/taeksoo/Desktop/VCLAB/github/EVA3D/datasets/DeepFashion/smpl.pkl', 'rb') as f:
    data = pickle.load(f)

data = data['WOMEN-Skirts-id_00006360-01_1_front']
smpl = np.concatenate([np.array([[1, 0, 0, 0]]), data['global_orient'], data['body_pose'], data['betas']], axis=1)

# with open('/media/taeksoo/SSD1/ag3d/deepfashion_curated/smpl/00007.pkl', 'rb') as f:
#     data = pickle.load(f)

# pose = data['theta'][:, 13:85].cpu().numpy()
# # smpl = np.concatenate([np.array([[1, 0, 0, 0]]), pose, data['theta'][:, 3:13].cpu().numpy()], axis=1)
# # smpl_output = smpl_server(torch.tensor(smpl, device='cuda').float(), absolute=True)
# # pelvis = smpl_output['smpl_jnts'][0][0].cpu().numpy()

# # pose[:, 0] += np.pi 
# global_orient = pose[:,:3]
# global_orient_mat = R.from_rotvec(global_orient).as_matrix()
# r_xpi = R.from_euler('x', 180, degrees=True).as_matrix()
# final = r_xpi @ global_orient_mat
# pose[:, :3] = R.from_matrix(final).as_rotvec()

# smpl = np.concatenate([np.array([[1, 0, 0, 0]]), pose, data['theta'][:, 3:13].cpu().numpy()], axis=1)


smpl_output = smpl_server(torch.tensor(smpl, device='cuda').float(), absolute=True)

pelvis = smpl_output['smpl_jnts'][0][0].cpu().numpy()

import open3d as o3d
# verts = smpl_output['smpl_verts'][0]
verts = smpl_server.verts_c
verts = verts.cpu().numpy()
faces = smpl_output['faces']
mesh = o3d.geometry.TriangleMesh()
mesh.vertices = o3d.utility.Vector3dVector(verts[0])
mesh.triangles = o3d.utility.Vector3iVector(faces)
mesh.compute_vertex_normals()
r = R.from_euler('x', 180, degrees=True).as_matrix()
# rotvec = mat2vec(r)

verts = smpl_output['smpl_verts']
verts = verts.cpu().numpy()
faces = smpl_output['faces']
mesh1 = o3d.geometry.TriangleMesh()
mesh1.vertices = o3d.utility.Vector3dVector(verts[0])
mesh1.triangles = o3d.utility.Vector3iVector(faces)
mesh1.compute_vertex_normals()


o3d.visualization.draw_geometries([mesh1])
o3d.io.write_triangle_mesh('mesh.obj', mesh1)


render = o3d.visualization.rendering.OffscreenRenderer(256, 256)

mtl = o3d.visualization.rendering.MaterialRecord()  # or MaterialRecord(), for later versions of Open3D
mtl.base_color = [1.0, 0.0, 0.0, 1.0]  # RGBA
mtl.shader = "defaultUnlit"

render.scene.add_geometry("rotated_model", mesh1, mtl)
# render.scene.set_background([0.1, 0.2, 0.3, 1.0])

intrinsics = o3d.camera.PinholeCameraIntrinsic(256, 256, 5000/16, 5000/16, 128, 128)
extrinsics = np.eye(4)

cam = data['theta'][:, :3].cpu().numpy()
sx, tx, ty = cam[0]
cam_t = np.array([tx, -ty, -2 * 5000/16 / (256 * sx + 1e-9)])
cam_t1 = r_xpi @ (cam_t - pelvis) + pelvis
# extrinsics[:3, :3] = r_xpi
cam_t1[1] -= 0.3
extrinsics[:3, 3] = cam_t1

# extrinsics[1:3, :] = -extrinsics[1:3, :]

# render.scene.camera.set_projection()
render.setup_camera(intrinsics, extrinsics)
img_o3d = render.render_to_image()

# we can now save the rendered image right at this point 
o3d.io.write_image("output.png", img_o3d, 9)

from PIL import Image
img1 = Image.open('/media/taeksoo/SSD1/ag3d/deepfashion_curated/images/00007.jpg')
img2 = Image.open('output.png')
result = Image.blend(img1, img2, alpha=0.5)
result.save('result.jpg')