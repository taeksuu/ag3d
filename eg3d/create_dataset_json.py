import os
import pickle
import numpy as np
import json
from tqdm import tqdm
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from training.fast_snarf.lib.model.smpl import SMPLServer
import torch

dataset_res = 256
neural_res = 64

focal = 5000

if neural_res == 64:
    focal = 5000 / 16
elif neural_res == 128:
    focal = 5000 / 8
elif neural_res == 256:
    focal = 5000 / 4
elif neural_res == 512:
    focal = 5000 / 2
elif neural_res == 1024:
    focal = 5000


intrinsics = np.array([
    [focal / dataset_res, 0, 0.5],
    [0, focal / dataset_res, 0.5],
    [0, 0, 1]
])

labels = []
smpls = []

smpl_server = SMPLServer()
r_xpi = R.from_euler('x', 180, degrees=True).as_matrix()

for i in tqdm(range(12000)):
    with open(f'/media/taeksoo/SSD1/ag3d/deepfashion_curated/smpl/{i:05d}.pkl', 'rb') as f:
        data = pickle.load(f)

    cam = data['theta'][:, :3].cpu().numpy()
    shape = data['theta'][:, 3:13].cpu().numpy()
    pose = data['theta'][:, 13:85].cpu().numpy()

    global_orient = pose[:,:3]
    global_orient_mat = R.from_rotvec(global_orient).as_matrix()
    final = r_xpi @ global_orient_mat
    pose[:, :3] = R.from_matrix(final).as_rotvec()

    smpl = np.concatenate([np.array([[1, 0, 0, 0]]), pose, shape], axis=1)
    smpl_output = smpl_server(torch.tensor(smpl, device='cuda').float(), absolute=True)

    pelvis = smpl_output['smpl_jnts'][0][0].cpu().numpy()


    extrinsics = np.eye(4)
    sx, tx, ty = cam[0]
    cam_t = [tx, -ty, -2 * focal / (dataset_res * sx + 1e-9)]
    cam_t1 = r_xpi @ (cam_t - pelvis) + pelvis
    cam_t1[1] -= 0.3
    extrinsics[:3, :3] = r_xpi
    extrinsics[:3, 3] = cam_t1


    labels.append([f'images/{i:05d}.jpg', extrinsics.flatten().tolist() + intrinsics.flatten().tolist()])
    smpls.append([f'images/{i:05d}.jpg', smpl.flatten().tolist()])
    


label_dict = {}
label_dict['labels'] = labels
label_dict['smpls'] = smpls

with open('/media/taeksoo/SSD1/ag3d/deepfashion_curated/dataset.json', 'w') as f:
    json.dump(label_dict, f)