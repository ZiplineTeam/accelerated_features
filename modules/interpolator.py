"""
	"XFeat: Accelerated Features for Lightweight Image Matching, CVPR 2024."
	https://www.verlab.dcc.ufmg.br/descriptors/xfeat_cvpr24/
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InterpolateSparse2d(nn.Module):
    """ Efficiently interpolate tensor at given sparse 2D positions. """

    def __init__(self, mode='bicubic', align_corners=False):
        super().__init__()
        self.mode = mode
        self.align_corners = align_corners

    def normgrid(self, x, H, W):
        """ Normalize coords to [-1,1]. """
        # Use tensor operations to avoid creating tensors from Python scalars
        # This approach is more trace-friendly
        device, dtype = x.device, x.dtype

        # Create a tensor of the same shape as x for broadcasting
        # x has shape [..., 2], so we need to create [W-1, H-1] for the last dimension
        w_minus_1 = torch.full_like(
            x[..., :1], W-1, device=device, dtype=dtype)
        h_minus_1 = torch.full_like(
            x[..., :1], H-1, device=device, dtype=dtype)
        norm_tensor = torch.cat([w_minus_1, h_minus_1], dim=-1)

        return 2. * (x / norm_tensor) - 1.

    def forward(self, x, pos, H, W):
        """
        Input
            x: [B, C, H, W] feature tensor
            pos: [B, N, 2] tensor of positions
            H, W: int, original resolution of input 2d positions -- used in normalization [-1,1]

        Returns
            [B, N, C] sampled channels at 2d positions
        """
        grid = self.normgrid(pos, H, W).unsqueeze(-2).to(x.dtype)
        x = F.grid_sample(x, grid, mode=self.mode, align_corners=False)
        return x.permute(0, 2, 3, 1).squeeze(-2)
