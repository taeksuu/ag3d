# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
The renderer is a module that takes in rays, decides where to sample along each
ray, and computes pixel colors using the volume rendering equation.
"""

import math
import torch
import torch.nn as nn
import torch.autograd as autograd
import pytorch3d.ops as ops

from training.volumetric_rendering_AG3D.ray_marcher import MipRayMarcher2
from training.volumetric_rendering_AG3D import math_utils

from training.fast_snarf.lib.model.fast_snarf import ForwardDeformer
from training.fast_snarf.lib.model.smpl import SMPLServer

def grid_sample(image, optical):
    N, C, IH, IW = image.shape
    _, H, W, _ = optical.shape

    ix = optical[..., 0]
    iy = optical[..., 1]

    ix = ((ix + 1) / 2) * (IW-1);
    iy = ((iy + 1) / 2) * (IH-1);
    with torch.no_grad():
        ix_nw = torch.floor(ix);
        iy_nw = torch.floor(iy);
        ix_ne = ix_nw + 1;
        iy_ne = iy_nw;
        ix_sw = ix_nw;
        iy_sw = iy_nw + 1;
        ix_se = ix_nw + 1;
        iy_se = iy_nw + 1;

    nw = (ix_se - ix)    * (iy_se - iy)
    ne = (ix    - ix_sw) * (iy_sw - iy)
    sw = (ix_ne - ix)    * (iy    - iy_ne)
    se = (ix    - ix_nw) * (iy    - iy_nw)
    
    with torch.no_grad():
        torch.clamp(ix_nw, 0, IW-1, out=ix_nw)
        torch.clamp(iy_nw, 0, IH-1, out=iy_nw)

        torch.clamp(ix_ne, 0, IW-1, out=ix_ne)
        torch.clamp(iy_ne, 0, IH-1, out=iy_ne)
 
        torch.clamp(ix_sw, 0, IW-1, out=ix_sw)
        torch.clamp(iy_sw, 0, IH-1, out=iy_sw)
 
        torch.clamp(ix_se, 0, IW-1, out=ix_se)
        torch.clamp(iy_se, 0, IH-1, out=iy_se)

    image = image.view(N, C, IH * IW)


    nw_val = torch.gather(image, 2, (iy_nw * IW + ix_nw).long().view(N, 1, H * W).repeat(1, C, 1))
    ne_val = torch.gather(image, 2, (iy_ne * IW + ix_ne).long().view(N, 1, H * W).repeat(1, C, 1))
    sw_val = torch.gather(image, 2, (iy_sw * IW + ix_sw).long().view(N, 1, H * W).repeat(1, C, 1))
    se_val = torch.gather(image, 2, (iy_se * IW + ix_se).long().view(N, 1, H * W).repeat(1, C, 1))

    out_val = (nw_val.view(N, C, H, W) * nw.view(N, 1, H, W) + 
               ne_val.view(N, C, H, W) * ne.view(N, 1, H, W) +
               sw_val.view(N, C, H, W) * sw.view(N, 1, H, W) +
               se_val.view(N, C, H, W) * se.view(N, 1, H, W))

    return out_val

def index_custom(feat, uv):
    '''
    args:
        feat: (B, C, H, W)
        uv: (B, 2, N)
    return:
        (B, C, N)
    '''
    device = feat.device
    B, C, H, W = feat.size()
    _, _, N = uv.size()
    
    x, y = uv[:,0], uv[:,1]
    x = (W-1.0) * (0.5 * x.contiguous().view(-1) + 0.5)
    y = (H-1.0) * (0.5 * y.contiguous().view(-1) + 0.5)

    x0 = torch.floor(x).int()
    x1 = x0 + 1
    y0 = torch.floor(y).int()
    y1 = y0 + 1

    max_x = W - 1
    max_y = H - 1

    x0_clamp = torch.clamp(x0, 0, max_x)
    x1_clamp = torch.clamp(x1, 0, max_x)
    y0_clamp = torch.clamp(y0, 0, max_y)
    y1_clamp = torch.clamp(y1, 0, max_y)

    dim2 = W
    dim1 = W * H

    base = (dim1 * torch.arange(B).int()).view(B, 1).expand(B, N).contiguous().view(-1).to(device)

    base_y0 = base + y0_clamp * dim2
    base_y1 = base + y1_clamp * dim2

    idx_y0_x0 = base_y0 + x0_clamp
    idx_y0_x1 = base_y0 + x1_clamp
    idx_y1_x0 = base_y1 + x0_clamp
    idx_y1_x1 = base_y1 + x1_clamp

    # (B,C,H,W) -> (B,H,W,C)
    im_flat = feat.permute(0,2,3,1).contiguous().view(-1, C)
    i_y0_x0 = torch.gather(im_flat, 0, idx_y0_x0.unsqueeze(1).expand(-1,C).long())
    i_y0_x1 = torch.gather(im_flat, 0, idx_y0_x1.unsqueeze(1).expand(-1,C).long())
    i_y1_x0 = torch.gather(im_flat, 0, idx_y1_x0.unsqueeze(1).expand(-1,C).long())
    i_y1_x1 = torch.gather(im_flat, 0, idx_y1_x1.unsqueeze(1).expand(-1,C).long())
    
    # Check the out-of-boundary case.
    x0_valid = (x0 <= max_x) & (x0 >= 0)
    x1_valid = (x1 <= max_x) & (x1 >= 0)
    y0_valid = (y0 <= max_y) & (y0 >= 0)
    y1_valid = (y1 <= max_y) & (y1 >= 0)

    x0 = x0.float()
    x1 = x1.float()
    y0 = y0.float()
    y1 = y1.float()

    w_y0_x0 = ((x1 - x) * (y1 - y) * (x1_valid * y1_valid).float()).unsqueeze(1)
    w_y0_x1 = ((x - x0) * (y1 - y) * (x0_valid * y1_valid).float()).unsqueeze(1)
    w_y1_x0 = ((x1 - x) * (y - y0) * (x1_valid * y0_valid).float()).unsqueeze(1)
    w_y1_x1 = ((x - x0) * (y - y0) * (x0_valid * y0_valid).float()).unsqueeze(1)

    output = w_y0_x0 * i_y0_x0 + w_y0_x1 * i_y0_x1 + w_y1_x0 * i_y1_x0 + w_y1_x1 * i_y1_x1 # (B, N, C)

    return output.view(B, N, C).permute(0,2,1).contiguous()


def generate_planes():
    """
    Defines planes by the three vectors that form the "axes" of the
    plane. Should work with arbitrary number of planes and planes of
    arbitrary orientation.
    """
    return torch.tensor([[[1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1]],
                            [[1, 0, 0],
                            [0, 0, 1],
                            [0, 1, 0]],
                            [[0, 0, 1],
                            [0, 1, 0],
                            [1, 0, 0]]], dtype=torch.float32)

def project_onto_planes(planes, coordinates):
    """
    Does a projection of a 3D point onto a batch of 2D planes,
    returning 2D plane coordinates.

    Takes plane axes of shape n_planes, 3, 3
    # Takes coordinates of shape N, M, 3
    # returns projections of shape N*n_planes, M, 2
    """
    N, M, C = coordinates.shape
    n_planes, _, _ = planes.shape
    coordinates = coordinates.unsqueeze(1).expand(-1, n_planes, -1, -1).reshape(N*n_planes, M, 3)
    inv_planes = torch.linalg.inv(planes).unsqueeze(0).expand(N, -1, -1, -1).reshape(N*n_planes, 3, 3)
    projections = torch.bmm(coordinates, inv_planes)
    return projections[..., :2]

def sample_from_planes(plane_axes, plane_features, coordinates, mode='bilinear', padding_mode='zeros', box_warp=None):
    assert padding_mode == 'zeros'
    N, n_planes, C, H, W = plane_features.shape
    _, M, _ = coordinates.shape
    plane_features = plane_features.view(N*n_planes, C, H, W)

    coordinates = (2/box_warp) * coordinates # TODO: add specific box bounds

    # projected_coordinates = project_onto_planes(plane_axes, coordinates).unsqueeze(1)
    projected_coordinates = project_onto_planes(plane_axes, coordinates).permute(0, 2, 1)
    output_features = index_custom(plane_features, projected_coordinates.float()).permute(0, 2, 1).reshape(N, n_planes, M, C)

    # projected_coordinates = project_onto_planes(plane_axes, coordinates).unsqueeze(1)
    # output_features = torch.nn.functional.grid_sample(plane_features, projected_coordinates.float(), mode=mode, padding_mode=padding_mode, align_corners=False).permute(0, 3, 2, 1).reshape(N, n_planes, M, C)

    return output_features

def sample_from_3dgrid(grid, coordinates):
    """
    Expects coordinates in shape (batch_size, num_points_per_batch, 3)
    Expects grid in shape (1, channels, H, W, D)
    (Also works if grid has batch size)
    Returns sampled features of shape (batch_size, num_points_per_batch, feature_channels)
    """
    batch_size, n_coords, n_dims = coordinates.shape
    sampled_features = torch.nn.functional.grid_sample(grid.expand(batch_size, -1, -1, -1, -1),
                                                       coordinates.reshape(batch_size, 1, 1, -1, n_dims),
                                                       mode='bilinear', padding_mode='zeros', align_corners=False)
    N, C, H, W, D = sampled_features.shape
    sampled_features = sampled_features.permute(0, 4, 3, 2, 1).reshape(N, H*W*D, C)
    return sampled_features

import numpy as np

class ImportanceRenderer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ray_marcher = MipRayMarcher2()
        self.plane_axes = generate_planes()

        self.smpl_server = SMPLServer()
        self.deformer = ForwardDeformer(smpl_server=self.smpl_server)

        self.sigmoid_beta = torch.nn.Parameter(0.1 * torch.ones(1))
        # self.sigmoid_beta = 0.01
        self.precomputed_sdf_grid = np.load('eg3d/training/precomputed_sdf.npy')


    def forward(self, planes, smpl, decoder, canonical, ray_origins, ray_directions, rendering_options):
        self.plane_axes = self.plane_axes.to(ray_origins.device)

        if rendering_options['ray_start'] == rendering_options['ray_end'] == 'auto':
            ray_start, ray_end = math_utils.get_ray_limits_box(ray_origins, ray_directions, box_side_length=rendering_options['box_warp'])
            is_ray_valid = ray_end > ray_start
            if torch.any(is_ray_valid).item():
                ray_start[~is_ray_valid] = ray_start[is_ray_valid].min()
                ray_end[~is_ray_valid] = ray_start[is_ray_valid].max()
            depths_coarse = self.sample_stratified(ray_origins, ray_start, ray_end, rendering_options['depth_resolution'], rendering_options['disparity_space_sampling'])
        else:
            # Create stratified depth samples
            depths_coarse = self.sample_stratified(ray_origins, rendering_options['ray_start'], rendering_options['ray_end'], rendering_options['depth_resolution'], rendering_options['disparity_space_sampling'])

        batch_size, num_rays, samples_per_ray, _ = depths_coarse.shape

        # Coarse Pass
        sample_coordinates = (ray_origins.unsqueeze(-2) + depths_coarse * ray_directions.unsqueeze(-2)).reshape(batch_size, -1, 3)
        sample_directions = ray_directions.unsqueeze(-2).expand(-1, -1, samples_per_ray, -1).reshape(batch_size, -1, 3)


        out = self.run_model(planes, smpl, decoder, canonical, sample_coordinates, sample_directions, rendering_options)
        colors_coarse = out['rgb']
        densities_coarse = out['sigma']
        defomer_weight_diffs = out['deformer_weight_diffs']
        zero_delta_sdf = out['zero_delta_sdf']
        normal = out['normal']
        bbox = out['bbox']
        colors_coarse = colors_coarse.reshape(batch_size, num_rays, samples_per_ray, colors_coarse.shape[-1])
        densities_coarse = densities_coarse.reshape(batch_size, num_rays, samples_per_ray, 1)

        # Fine Pass
        N_importance = rendering_options['depth_resolution_importance']
        if N_importance > 0:
            _, _, weights = self.ray_marcher(colors_coarse, densities_coarse, depths_coarse, rendering_options)

            depths_fine = self.sample_importance(depths_coarse, weights, N_importance)

            sample_directions = ray_directions.unsqueeze(-2).expand(-1, -1, N_importance, -1).reshape(batch_size, -1, 3)
            sample_coordinates = (ray_origins.unsqueeze(-2) + depths_fine * ray_directions.unsqueeze(-2)).reshape(batch_size, -1, 3)

            out = self.run_model(planes, smpl, decoder, canonical, sample_coordinates, sample_directions, rendering_options)
            colors_fine = out['rgb']
            densities_fine = out['sigma']
            defomer_weight_diffs += out['deformer_weight_diffs']
            zero_delta_sdf += out['zero_delta_sdf']
            normal += out['normal']
            bbox += out['bbox']
            colors_fine = colors_fine.reshape(batch_size, num_rays, N_importance, colors_fine.shape[-1])
            densities_fine = densities_fine.reshape(batch_size, num_rays, N_importance, 1)

            all_depths, all_colors, all_densities = self.unify_samples(depths_coarse, colors_coarse, densities_coarse,
                                                                  depths_fine, colors_fine, densities_fine)

            # Aggregate
            rgb_final, depth_final, weights = self.ray_marcher(all_colors, all_densities, all_depths, rendering_options)
        else:
            rgb_final, depth_final, weights = self.ray_marcher(colors_coarse, densities_coarse, depths_coarse, rendering_options)


        return rgb_final, depth_final, weights.sum(2), defomer_weight_diffs, normal, zero_delta_sdf, bbox

    def run_model(self, planes, smpl, decoder, canonical, sample_coordinates, sample_directions, options):

        num_batch_final, num_point_final, num_dim_final = sample_coordinates.shape
        
        smpl_output = self.smpl_server(smpl, absolute=True)
        smpl_tfs, smpl_verts = smpl_output['smpl_tfs'], smpl_output['smpl_verts']
        smpl_tfs = torch.einsum('bnij,njk->bnik', smpl_tfs.to(smpl.device), self.smpl_server.tfs_c_inv.to(smpl.device))

        # will set density=0 if points are far from predicted SMPL vertices
        distance, index, _  = ops.knn_points(sample_coordinates, smpl_verts[:,::10,:], K=1, return_nn=False)
        mask_smpl = torch.ones((num_batch_final, num_point_final), device=sample_coordinates.device, dtype=torch.bool)
        mask_smpl = mask_smpl & (distance < 0.15).squeeze(2)
        if mask_smpl.sum() == 0: mask_smpl = torch.ones((num_batch_final, num_point_final), device=sample_coordinates.device, dtype=torch.bool)
        # sample_coordinates = sample_coordinates[mask_smpl][None]

        # placeholder for final outputs
        sigma_final = torch.zeros((num_batch_final, num_point_final, 1),  device=sample_coordinates.device)
        rgb_final = torch.zeros((num_batch_final, num_point_final, 32),  device=sample_coordinates.device)
        pts_c_final = torch.zeros((num_batch_final, num_point_final, 3),  device=sample_coordinates.device)

        for b in range(num_batch_final):
            mask_smpl_ = mask_smpl[b][None]
            if not canonical:
                pts_c, intermediates = self.deformer(sample_coordinates[b][None], {}, smpl_tfs)
                num_batch, num_point, num_init, num_dim = pts_c.shape
                pts_c = pts_c.reshape(num_batch, num_point * num_init, num_dim)
                mask = intermediates['valid_ids'].reshape(num_batch, -1)
                if mask.sum() == 0: mask = torch.ones((num_batch, num_point * num_init), device=pts_c.device, dtype=torch.bool)
                mask = torch.repeat_interleave(mask_smpl_, num_init, dim=1) & mask

                pts_c = pts_c[mask]

            else:
                pts_c = sample_coordinates[b][None]
                num_batch, num_point, num_dim = pts_c.shape
                pts_c = pts_c[mask_smpl_]

            sampled_features = sample_from_planes(self.plane_axes, planes[b][None], pts_c[None], padding_mode='zeros', box_warp=options['box_warp'])
            sampled_features = sampled_features.mean(1)

            if not canonical:
                # sampled_features = sampled_features[mask]
            
                out = decoder(sampled_features, sample_directions)

                sdf_grid = torch.tensor(self.precomputed_sdf_grid).to(sampled_features.device)
                queried_sdf = torch.nn.functional.grid_sample(sdf_grid, pts_c[None][:,:,None,None,:3], padding_mode='border', align_corners=True)
                queried_sdf = queried_sdf.reshape(-1, 1)
                sdf = queried_sdf + out['delta_sdf']
            
                sigma = torch.zeros((num_batch, num_point * num_init, 1),  device=pts_c.device)
                rgb = torch.zeros((num_batch, num_point * num_init, 32),  device=pts_c.device)
                sigma[mask] = torch.sigmoid(-sdf / self.sigmoid_beta) / self.sigmoid_beta
                pts_c_ = torch.zeros((num_batch, num_point * num_init, 3), device=pts_c.device)
                pts_c_[mask] = pts_c
                pts_c = pts_c_
                rgb[mask] = out['rgb']

                sigma = sigma.reshape(num_batch, -1, num_init, 1)
                rgb = rgb.reshape(num_batch, -1, num_init, 32)
                pts_c = pts_c.reshape(num_batch, -1, num_init, 3)
                # sampled_features = sampled_features.reshape(num_batch, 3, -1, num_init, 32)

                out['sigma'], idx_c = sigma.max(dim=2)
                out['rgb'] = torch.gather(rgb, 2, idx_c.unsqueeze(-1).expand(-1,-1, 1, rgb.shape[-1])).squeeze(2)
                pts_c = torch.gather(pts_c, 2, idx_c.unsqueeze(-1).expand(-1,-1, 1, pts_c.shape[-1])).squeeze(2) 
                # sampled_features = torch.gather(sampled_features, 3, idx_c.unsqueeze(-1).unsqueeze(1).expand(-1,3, -1, 1, sampled_features.shape[-1])).squeeze(3)

                sigma_final[b] = out['sigma']
                rgb_final[b] = out['rgb']
                pts_c_final[b] = pts_c

            else:
                # sampled_features = sampled_features[mask_smpl_]
                # pts_c = pts_c[mask_smpl_]

                out = decoder(sampled_features, sample_directions)

                sdf_grid = torch.tensor(self.precomputed_sdf_grid).to(sampled_features.device)
                queried_sdf = torch.nn.functional.grid_sample(sdf_grid, pts_c[None][:,:,None,None,:3], padding_mode='border', align_corners=True)
                queried_sdf = queried_sdf.reshape(-1, 1)
                sdf = queried_sdf + out['delta_sdf']

                sigma = torch.zeros((num_batch, num_point, 1),  device=pts_c.device)
                rgb = torch.zeros((num_batch, num_point, 32),  device=pts_c.device)
                sigma[mask_smpl_] = torch.sigmoid(-sdf / self.sigmoid_beta) / self.sigmoid_beta
                rgb[mask_smpl_] = out['rgb']

                out['sigma'] = sigma
                out['rgb'] = rgb

                sigma_final[b] = out['sigma']
                rgb_final[b] = out['rgb']

        out['sigma'] = sigma_final
        out['rgb'] = rgb_final
            
        
        out['deformer_weight_diffs'] = 0

        # Eikonal loss
        loss_eik = 0
        if not canonical:
            with torch.enable_grad():
                pts_c_ = pts_c_final.clone().detach()
                pts_c_.requires_grad_()
                
                sampled_features = sample_from_planes(self.plane_axes, planes, pts_c_, padding_mode='zeros', box_warp=options['box_warp'])
                sampled_features = sampled_features.mean(1)

                # sampled_features = sampled_features[mask_smpl[0][None]]
                # pts_c_ = pts_c_[mask_smpl[0][None]]

                out_ = decoder(sampled_features, sample_directions)['delta_sdf']
                normal = autograd.grad(out_, pts_c_, grad_outputs=torch.ones_like(out_, device=out_.device),
                                            create_graph=True)[0].view(num_batch_final,-1, 3)
                
            # mask = torch.gather(mask.reshape(num_batch, -1, num_init, 1), 2, idx_c.unsqueeze(-1).expand(-1,-1, 1, 1)).squeeze(2)
            normal = normal[mask_smpl] 
        
            loss_eik = (torch.norm(normal, p=2, dim=1) - 1).pow(2).mean()
        out['normal'] = loss_eik
        
        # zero_delta_sdf = torch.nn.functional.mse_loss(out['delta_sdf'], torch.zeros_like(out['delta_sdf']))
        out['zero_delta_sdf'] = 0

        # pts_bbox = torch.rand([num_batch, 2000, 3], device=pts_c.device) * 2 - 1
        # sampled_features = sample_from_planes(self.plane_axes, planes, pts_bbox, padding_mode='zeros', box_warp=options['box_warp'])
        # out_ = decoder(pts_bbox, sampled_features, sample_directions)
        # loss_rand_bbox = torch.exp(-100 * torch.abs(out_['queried_sdf'] + out_['delta_sdf'])).mean()
        out['bbox'] = 0

        if options.get('density_noise', 0) > 0:
            out['sigma'] += torch.randn_like(out['sigma']) * options['density_noise']
        return out

    def sort_samples(self, all_depths, all_colors, all_densities):
        _, indices = torch.sort(all_depths, dim=-2)
        all_depths = torch.gather(all_depths, -2, indices)
        all_colors = torch.gather(all_colors, -2, indices.expand(-1, -1, -1, all_colors.shape[-1]))
        all_densities = torch.gather(all_densities, -2, indices.expand(-1, -1, -1, 1))
        return all_depths, all_colors, all_densities

    def unify_samples(self, depths1, colors1, densities1, depths2, colors2, densities2):
        all_depths = torch.cat([depths1, depths2], dim = -2)
        all_colors = torch.cat([colors1, colors2], dim = -2)
        all_densities = torch.cat([densities1, densities2], dim = -2)

        _, indices = torch.sort(all_depths, dim=-2)
        all_depths = torch.gather(all_depths, -2, indices)
        all_colors = torch.gather(all_colors, -2, indices.expand(-1, -1, -1, all_colors.shape[-1]))
        all_densities = torch.gather(all_densities, -2, indices.expand(-1, -1, -1, 1))

        return all_depths, all_colors, all_densities

    def sample_stratified(self, ray_origins, ray_start, ray_end, depth_resolution, disparity_space_sampling=False):
        """
        Return depths of approximately uniformly spaced samples along rays.
        """
        N, M, _ = ray_origins.shape
        if disparity_space_sampling:
            depths_coarse = torch.linspace(0,
                                    1,
                                    depth_resolution,
                                    device=ray_origins.device).reshape(1, 1, depth_resolution, 1).repeat(N, M, 1, 1)
            depth_delta = 1/(depth_resolution - 1)
            depths_coarse += torch.rand_like(depths_coarse) * depth_delta
            depths_coarse = 1./(1./ray_start * (1. - depths_coarse) + 1./ray_end * depths_coarse)
        else:
            if type(ray_start) == torch.Tensor:
                depths_coarse = math_utils.linspace(ray_start, ray_end, depth_resolution).permute(1,2,0,3)
                depth_delta = (ray_end - ray_start) / (depth_resolution - 1)
                depths_coarse += torch.rand_like(depths_coarse) * depth_delta[..., None]
            else:
                depths_coarse = torch.linspace(ray_start, ray_end, depth_resolution, device=ray_origins.device).reshape(1, 1, depth_resolution, 1).repeat(N, M, 1, 1)
                depth_delta = (ray_end - ray_start)/(depth_resolution - 1)
                depths_coarse += torch.rand_like(depths_coarse) * depth_delta

        return depths_coarse

    def sample_importance(self, z_vals, weights, N_importance):
        """
        Return depths of importance sampled points along rays. See NeRF importance sampling for more.
        """
        with torch.no_grad():
            batch_size, num_rays, samples_per_ray, _ = z_vals.shape

            z_vals = z_vals.reshape(batch_size * num_rays, samples_per_ray)
            weights = weights.reshape(batch_size * num_rays, -1) # -1 to account for loss of 1 sample in MipRayMarcher

            # smooth weights
            weights = torch.nn.functional.max_pool1d(weights.unsqueeze(1).float(), 2, 1, padding=1)
            weights = torch.nn.functional.avg_pool1d(weights, 2, 1).squeeze()
            weights = weights + 0.01

            z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:])
            importance_z_vals = self.sample_pdf(z_vals_mid, weights[:, 1:-1],
                                             N_importance).detach().reshape(batch_size, num_rays, N_importance, 1)
        return importance_z_vals

    def sample_pdf(self, bins, weights, N_importance, det=False, eps=1e-5):
        """
        Sample @N_importance samples from @bins with distribution defined by @weights.
        Inputs:
            bins: (N_rays, N_samples_+1) where N_samples_ is "the number of coarse samples per ray - 2"
            weights: (N_rays, N_samples_)
            N_importance: the number of samples to draw from the distribution
            det: deterministic or not
            eps: a small number to prevent division by zero
        Outputs:
            samples: the sampled samples
        """
        N_rays, N_samples_ = weights.shape
        weights = weights + eps # prevent division by zero (don't do inplace op!)
        pdf = weights / torch.sum(weights, -1, keepdim=True) # (N_rays, N_samples_)
        cdf = torch.cumsum(pdf, -1) # (N_rays, N_samples), cumulative distribution function
        cdf = torch.cat([torch.zeros_like(cdf[: ,:1]), cdf], -1)  # (N_rays, N_samples_+1)
                                                                   # padded to 0~1 inclusive

        if det:
            u = torch.linspace(0, 1, N_importance, device=bins.device)
            u = u.expand(N_rays, N_importance)
        else:
            u = torch.rand(N_rays, N_importance, device=bins.device)
        u = u.contiguous()

        inds = torch.searchsorted(cdf, u, right=True)
        below = torch.clamp_min(inds-1, 0)
        above = torch.clamp_max(inds, N_samples_)

        inds_sampled = torch.stack([below, above], -1).view(N_rays, 2*N_importance)
        cdf_g = torch.gather(cdf, 1, inds_sampled).view(N_rays, N_importance, 2)
        bins_g = torch.gather(bins, 1, inds_sampled).view(N_rays, N_importance, 2)

        denom = cdf_g[...,1]-cdf_g[...,0]
        denom[denom<eps] = 1 # denom equals 0 means a bin has weight 0, in which case it will not be sampled
                             # anyway, therefore any value for it is fine (set to 1 here)

        samples = bins_g[...,0] + (u-cdf_g[...,0])/denom * (bins_g[...,1]-bins_g[...,0])
        return samples