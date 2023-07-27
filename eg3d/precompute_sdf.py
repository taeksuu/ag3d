from training.fast_snarf.lib.model.smpl import SMPLServer
import open3d as o3d

smpl_server = SMPLServer()

verts = smpl_server.verts_c.data.cpu().numpy()
faces = smpl_server.faces



o3d_mesh = o3d.geometry.TriangleMesh()
o3d_mesh.vertices = o3d.utility.Vector3dVector(verts[0])
o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)
o3d_mesh = o3d.t.geometry.TriangleMesh.from_legacy(o3d_mesh)

scene = o3d.t.geometry.RaycastingScene()
_ = scene.add_triangles(o3d_mesh)


import numpy as np
import torch

bmins=np.array([-1.0, -1.0, -1.0])
bmaxs=np.array([1.0, 1.0, 1.0])

bmins=torch.tensor(bmins).float().view(1,-1)
bmaxs=torch.tensor(bmaxs).float().view(1,-1)

W,H,D= [128, 128, 32]
resolutions = np.array( [128, 128, 32])

resolutions = torch.tensor(resolutions).float().view(-1)
arrangeX = torch.linspace(0, W-1, W).long()
arrangeY = torch.linspace(0, H-1, H).long()
arrangeZ = torch.linspace(0, D-1, D).long()
gridD, girdH, gridW = torch.meshgrid([arrangeZ, arrangeY, arrangeX])
coords = torch.stack([gridW, girdH, gridD]) # [3, steps[0], steps[1], steps[2]]
coords = coords.view(3, -1).t() # [N, 3]

coords2D = coords.float() / (resolutions[None,:] - 1)
coords2D = coords2D * (bmaxs - bmins) + bmins
dists=[]

for coords in torch.split(coords2D,50000):
    sdf = scene.compute_signed_distance(coords.numpy()).numpy().reshape(-1, 1)
    dists.append(torch.tensor(sdf))

dists = torch.cat(dists, dim=0)
dists=dists.transpose(0,1).reshape(1,-1,D,H,W)

np.save('precomputed_sdf.npy', dists.numpy())




