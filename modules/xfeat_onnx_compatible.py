# This module was added by zipline
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.model import XFeatModel
from modules.interpolator import InterpolateSparse2d


def unfold2d_onnx_compatible(x, ws=2):
    """
    ONNX-compatible version of 2D unfold operation.
    Replaces torch.unfold which is not supported in ONNX.
    """
    B, C, H, W = x.shape

    # Calculate output dimensions
    H_out = H // ws
    W_out = W // ws

    # Use reshape and permute operations instead of unfold
    # This is equivalent to the original unfold operation but ONNX-compatible
    x_reshaped = x.view(B, C, H_out, ws, W_out, ws)
    x_permuted = x_reshaped.permute(0, 1, 3, 5, 2, 4)
    x_final = x_permuted.contiguous().view(B, C * ws * ws, H_out, W_out)

    return x_final


class XFeatModelONNX(XFeatModel):
    """
    ONNX-compatible version of XFeatModel that replaces unfold operations.
    """

    def _unfold2d(self, x, ws=2):
        """
        ONNX-compatible version of the unfold operation.
        """
        return unfold2d_onnx_compatible(x, ws)


class XFeat(nn.Module):
    def __init__(self, weights=os.path.abspath(os.path.dirname(__file__)) + '/../weights/xfeat.pt'):  # ! change to top_k=2096
        super().__init__()
        self.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = XFeatModelONNX().to(self.dev).eval()
        self.interpolator = InterpolateSparse2d('bicubic')

        if weights is not None:
            if isinstance(weights, str):
                print('loading weights from: ' + weights)
                self.net.load_state_dict(torch.load(
                    weights, map_location=self.dev, weights_only=True))
            else:
                self.net.load_state_dict(weights)

        # Try to import LightGlue from Kornia
        self.kornia_available = False
        self.lighterglue = None
        self.kornia_available = True
        print("xfeat match star initialized")

    @torch.inference_mode()
    def detectAndComputeDense(self, x, top_k=None, multiscale=True):
        """
            Compute dense *and coarse* descriptors. Supports batched mode.

            input:
                x -> torch.Tensor(B, C, H, W): grayscale or rgb image
                top_k -> int: keep best k features
            return: features sorted by their reliability score -- from most to least
                List[Dict]: 
                    'keypoints'    ->   torch.Tensor(top_k, 2): coarse keypoints
                    'scales'       ->   torch.Tensor(top_k,): extraction scale
                    'descriptors'  ->   torch.Tensor(top_k, 64): coarse local features
        """
        if top_k is None:
            top_k = self.top_k
        if multiscale:
            print("doing multiscale")
            mkpts, scale_tensor, feats = self.extract_dualscale(x, top_k)
        else:
            print("doing single scale")
            mkpts, scale_tensor, feats = self.extractDense(x, top_k)

        print("keypoint ranges: ", mkpts[:, :, 0].min(
        ), mkpts[:, :, 0].max(), mkpts[:, :, 1].min(), mkpts[:, :, 1].max())

        return {'keypoints': mkpts,
                'descriptors': feats,
                'scales': scale_tensor}

    @torch.inference_mode()
    def xfeat_detect(self, img, top_k):
        im_set = self.parse_input(img)
        return self.detectAndComputeDense(im_set, top_k)

    @torch.inference_mode()
    def xfeat_detect_sparse(self, img, top_k, detection_threshold, kernel_size):
        """ONNX-compatible sparse detector wrapper."""

        # Preprocess the input image
        img_preprocessed, rh, rw = self.preprocess_tensor(img)

        # Run detection on preprocessed image
        result = self.detectAndCompute(
            img_preprocessed, top_k, detection_threshold, kernel_size)

        # Scale keypoints back to original image coordinates
        rw = torch.full((), rw, device=result['keypoints'].device)
        rh = torch.full((), rh, device=result['keypoints'].device)
        scale_tensor = torch.stack([rw, rh], dim=0).view(1, 1, -1)
        scaled_keypoints = result['keypoints'] * scale_tensor

        return scaled_keypoints, result['descriptors'], result['scores']

    @torch.inference_mode()
    def detectAndCompute(self, x, top_k, detection_threshold, kernel_size):
        """
            Compute sparse keypoints & descriptors. Supports batched mode.

            input:
                x -> torch.Tensor(B, C, H, W): preprocessed grayscale image (divisible by 32)
                top_k -> int: keep best k features
                detection_threshold -> float: detection threshold for NMS
            return:
                Dict: 
                    'keypoints'    ->   torch.Tensor(B, N, 2): keypoints (x,y) in input image coordinates
                    'scores'       ->   torch.Tensor(B, N): keypoint scores
                    'descriptors'  ->   torch.Tensor(B, N, 64): local features
        """

        # Input is already preprocessed, so no need to call preprocess_tensor
        B, _, _H1, _W1 = x.shape
        print(f"DEBUG detectAndCompute: Input shape = {x.shape}")

        M1, K1, H1 = self.net(x)
        M1 = F.normalize(M1, dim=1)
        print(
            f"DEBUG detectAndCompute: Network output shapes M1={M1.shape}, K1={K1.shape}, H1={H1.shape}")

        # Convert logits to heatmap and extract kpts
        K1h = self.get_kpts_heatmap(K1)
        print(f"DEBUG detectAndCompute: Heatmap shape = {K1h.shape}")
        print(
            f"DEBUG detectAndCompute: Heatmap range = [{K1h.min()}, {K1h.max()}]")

        mkpts = self.NMS(K1h, detection_threshold, kernel_size)

        # Compute reliability scores
        _nearest = InterpolateSparse2d('nearest')
        _bilinear = InterpolateSparse2d('bilinear')
        scores = (_nearest(K1h, mkpts, _H1, _W1) *
                  _bilinear(H1, mkpts, _H1, _W1)).squeeze(-1)
        print(f"DEBUG detectAndCompute: Scores shape = {scores.shape}")

        # Mark invalid keypoints (where coordinates are 0,0)
        valid = ~torch.all(mkpts == 0, dim=-1)
        scores = scores * valid.float() - (1 - valid.float())  # Set invalid scores to -1
        print(f"DEBUG detectAndCompute: Valid keypoints count = {valid.sum()}")

        # Select top-k features by combined reliability and keypoint heatmap score.
        top_k_tensor = torch.full(
            (), top_k, device=mkpts.device, dtype=torch.long)
        mkpts_shape1_tensor = torch.full(
            (), mkpts.shape[1], device=mkpts.device, dtype=torch.long)
        actual_top_k = torch.min(top_k_tensor, mkpts_shape1_tensor)
        top_scores, idxs = torch.topk(
            scores, k=actual_top_k, dim=-1, largest=True, sorted=False)

        # Update scores to use the top-k selected scores
        scores = top_scores

        # Gather keypoints using indices
        mkpts = torch.gather(mkpts, 1, idxs.unsqueeze(-1).expand(-1, -1, 2))

        print(
            f"DEBUG detectAndCompute: After top-k selection, keypoints range = x:[{mkpts[:,:,0].min()}, {mkpts[:,:,0].max()}], y:[{mkpts[:,:,1].min()}, {mkpts[:,:,1].max()}]")

        # Interpolate descriptors at kpts positions
        feats = self.interpolator(M1, mkpts, H=_H1, W=_W1)

        # L2-Normalize
        feats = F.normalize(feats, dim=-1)

        # Filter valid keypoints for output
        valid_out = scores > 0

        mkpts_out = torch.zeros((B, actual_top_k, 2), device=mkpts.device)
        scores_out = torch.full((B, actual_top_k), -1.0, device=scores.device)
        feats_out = torch.zeros((B, actual_top_k, 64), device=feats.device)

        # Copy valid data
        mkpts_out = mkpts * valid_out.unsqueeze(-1).float()
        scores_out = scores * valid_out.float() - (1 - valid_out.float())
        feats_out = feats * valid_out.unsqueeze(-1).float()

        print(
            f"DEBUG detectAndCompute: Final output keypoints range = x:[{mkpts_out[:,:,0].min()}, {mkpts_out[:,:,0].max()}], y:[{mkpts_out[:,:,1].min()}, {mkpts_out[:,:,1].max()}]")

        return {'keypoints': mkpts_out,
                'scores': scores_out,
                'descriptors': feats_out}

    def get_kpts_heatmap(self, kpts, softmax_temp=1.0):
        scores = F.softmax(kpts*softmax_temp, 1)[:, :64]
        B, _, H, W = scores.shape
        heatmap = scores.permute(0, 2, 3, 1).reshape(B, H, W, 8, 8)
        heatmap = heatmap.permute(0, 1, 3, 2, 4).reshape(B, 1, H*8, W*8)
        return heatmap

    @torch.inference_mode()
    def NMS(self, x, threshold=0.05, kernel_size=5):
        B, C, H, W = x.shape
        pad = kernel_size//2
        # # Apply max pooling along width
        # x_horizontal = F.max_pool1d(
        #     x.view(B * C * H, W).unsqueeze(1),
        #     kernel_size=kernel_size,
        #     stride=1,
        #     padding=pad
        # ).squeeze(1).view(B, C, H, W)

        # # Apply max pooling along height
        # local_max = F.max_pool1d(
        #     x_horizontal.permute(0, 1, 3, 2).contiguous().view(
        #         B * C * W, H).unsqueeze(1),
        #     kernel_size=kernel_size,
        #     stride=1,
        #     padding=pad
        # ).squeeze(1).view(B, C, W, H).permute(0, 1, 3, 2)

        # use 2d maxpooling
        local_max = F.max_pool2d(
            x, kernel_size=kernel_size, stride=1, padding=pad)

        pos = (x == local_max) & (x >= threshold)

        y_coords, x_coords = torch.meshgrid(
            torch.arange(H, device=x.device, dtype=torch.long),
            torch.arange(W, device=x.device, dtype=torch.long),
            indexing='ij'
        )

        # Stack coordinates [x, y] and flatten spatial dimensions
        coords_grid = torch.stack([x_coords, y_coords], dim=-1)  # [H, W, 2]
        coords_flat = coords_grid.view(-1, 2)  # [H*W, 2]

        # Apply mask using broadcasting
        pos_mask = pos.view(B, -1, 1).float()  # [B, H*W, 1]

        # Broadcasting: [H*W, 2] * [B, H*W, 1] -> [B, H*W, 2]
        valid_coords = coords_flat * pos_mask

        return valid_coords

    @torch.inference_mode()
    def match_xfeat_star(self, dec1_ids, dec1_kps, dec1_desc, dec1_scales, dec2_ids, dec2_kps, dec2_desc, dec2_scales):

        dec1_scales = dec1_scales.squeeze(2)
        matched_idx_ref = []
        matched_idx_target = []

        dec1_ids = dec1_ids[0]
        dec2_ids = dec2_ids[0]

        # Match batches of pairs
        idxs_list = self.batch_match(dec1_desc, dec2_desc)

        # Refine coarse matches
        matches = []
        for b in range(1):
            match, idx0, idx1 = self.refine_matches(
                dec1_kps, dec1_desc, dec1_scales, dec2_kps, dec2_desc, dec2_scales, matches=idxs_list, batch_idx=b)
            match = match.float()
            matches.append(match)
            matched_idx_ref.append(idx0)
            matched_idx_target.append(idx1)

        ref_output_idx_vec = []
        # ids_vec = ids_vec.float()
        for i in matched_idx_ref:
            ref_output_idx_vec.append(dec1_ids[i])

        target_output_idx_vec = []
        for i in matched_idx_target:
            target_output_idx_vec.append(dec2_ids[i])

        ref_points = matches[0][:, :2].clone().detach()
        dst_points = matches[0][:, 2:].clone().detach()

        # return (ref_points, dst_points, ref_output_idx_vec, final_mask)
        return (ref_output_idx_vec, ref_points, target_output_idx_vec, dst_points)

    @torch.inference_mode()
    def match_xfeat_star_onnx_no_refinement(self, dec1_desc, dec2_desc, min_cossim=0.82):
        """
        Simplified ONNX-compatible matching without batched inference or subpixel refinement.
        Returns a single array of indices in the target image, indexed by the indices in the source image.
        For points in the source image without a match in the target image, the index is -1.
        """
        # Remove batch dimension and work with single batch
        dec1_desc = dec1_desc[0]
        dec2_desc = dec2_desc[0]

        # Perform descriptor matching without batching
        cossim = torch.mm(dec1_desc, dec2_desc.t())
        match12 = torch.argmax(cossim, dim=-1)
        match21 = torch.argmax(cossim.t(), dim=-1)

        # Find mutual matches
        idx0 = torch.arange(match12.shape[0], device=match12.device)
        mutual = match21[match12] == idx0

        # Filter by cosine similarity threshold
        cossim_max, _ = cossim.max(dim=1)
        good = cossim_max > min_cossim

        # Get valid match indices
        valid_matches = mutual & good

        # Create output array initialized with -1 (no match)
        target_indices = torch.full(
            (match12.shape[0],), -1, dtype=torch.long, device=match12.device)

        # Fill in valid matches
        target_indices[valid_matches] = match12[valid_matches]

        return target_indices

    def preprocess_tensor(self, x, scale_factor=1.0):
        """
        First resize the image to a maximum size if needed, then scale it by the optional scale factor.
        Finally, make the image dimensions divisible by 32.
        """
        if isinstance(x, np.ndarray) and len(x.shape) == 3:
            x = torch.tensor(x).permute(2, 0, 1)[None]
        x = x.float()

        # For ONNX compatibility, enforce maximum size constraints using tensor operations
        B, C, H, W = x.shape

        # Enforce maximum input dimensions (before the optional scale factor is applied)
        max_height = torch.full((), 1440, device=x.device, dtype=H.dtype)
        max_width = torch.full((), 2560, device=x.device, dtype=W.dtype)

        target_H = torch.minimum(H, max_height)
        target_W = torch.minimum(W, max_width)

        # Apply additional scale factor
        target_H = target_H * scale_factor
        target_W = target_W * scale_factor

        # Make divisible by 32
        target_H = (target_H // 32) * 32
        target_W = (target_W // 32) * 32

        # Calculate final scaling factors
        rh = H.float() / target_H.float()
        rw = W.float() / target_W.float()

        # Resize to target dimensions
        x = F.interpolate(x, size=(target_H.int(), target_W.int()),
                          mode='bilinear', align_corners=False)

        return x, rh, rw

    def batch_match(self, feats1, feats2, min_cossim=0.82):
        B = feats1.shape[0]
        cossim = torch.bmm(feats1, feats2.permute(0, 2, 1))
        match12 = torch.argmax(cossim, dim=-1)
        match21 = torch.argmax(cossim.permute(0, 2, 1), dim=-1)

        idx0 = torch.arange(match12[0].shape[0], device=match12.device)

        batched_matches = []

        for b in range(B):
            mutual = match21[b][match12[b]] == idx0
            cossim_max, _ = cossim[b].max(dim=1)
            good = cossim_max > min_cossim
            idx0_b = idx0[mutual & good]
            idx1_b = match12[b][mutual & good]

            batched_matches.append((idx0_b, idx1_b))

        return batched_matches

    def subpix_softmax2d(self, heatmaps, temp=3):
        N, H, W = heatmaps.shape
        heatmaps = torch.softmax(
            temp * heatmaps.view(-1, H*W), -1).view(-1, H, W)
        x, y = torch.meshgrid(torch.arange(W, device=heatmaps.device), torch.arange(
            H, device=heatmaps.device), indexing='xy')
        x = x - (W//2)
        y = y - (H//2)

        coords_x = (x[None, ...] * heatmaps)
        coords_y = (y[None, ...] * heatmaps)
        coords = torch.cat(
            [coords_x[..., None], coords_y[..., None]], -1).view(N, H*W, 2)
        coords = coords.sum(1)

        return coords

    def refine_matches(self, dec1_kps, dec1_desc, dec1_scales, dec2_kps, dec2_desc, dec2_scales, matches, batch_idx, fine_conf=0.25):
        idx0, idx1 = matches[batch_idx]
        feats1 = dec1_desc[batch_idx][idx0]
        feats2 = dec2_desc[batch_idx][idx1]
        mkpts_0 = dec1_kps[batch_idx][idx0]
        mkpts_1 = dec2_kps[batch_idx][idx1]
        sc0 = dec1_scales[batch_idx][idx0]
        sc1 = dec2_scales[batch_idx][idx1]

        # Compute fine offsets
        offsets = self.net.fine_matcher(torch.cat([feats1, feats2], dim=-1))
        conf = F.softmax(offsets*3, dim=-1).max(dim=-1)[0]
        offsets = self.subpix_softmax2d(offsets.view(-1, 8, 8))

        mkpts_0 += offsets * (sc0[:, None] / sc1[:, None])

        mask_good = conf > fine_conf
        mkpts_0 = mkpts_0[mask_good]
        mkpts_1 = mkpts_1[mask_good]

        return (torch.cat([mkpts_0, mkpts_1], dim=-1), idx0[mask_good], idx1[mask_good])

    @torch.inference_mode()
    def compute_pixel_offsets(self, dec1_desc, dec1_scales, dec2_desc, matches, fine_conf=0.25):
        """
        Compute pixel offsets for a single batch of matches.
        Returns offsets and a mask of valid matches.
        """
        idx1, idx2 = matches
        feats1 = dec1_desc[idx1]
        feats2 = dec2_desc[idx2]
        sc1 = dec1_scales[idx1]
        # sc2 = dec2_scales[idx2]

        relative_scale = sc1

        print("feats1.shape: ", feats1.shape)
        print("feats2.shape: ", feats2.shape)
        print("sc1.shape: ", sc1.shape)
        # print("sc2.shape: ", sc2.shape)
        print("relative_scale.shape: ", relative_scale.shape)

        # Compute fine offsets
        offsets = self.net.fine_matcher(torch.cat([feats1, feats2], dim=-1))
        conf = F.softmax(offsets*3, dim=-1).max(dim=-1)[0]
        offsets = self.subpix_softmax2d(offsets.view(-1, 8, 8))

        print("offsets.shape: ", offsets.shape)

        # Apply scale to offsets
        offsets = offsets * relative_scale
        mask_good = conf > fine_conf

        return (offsets, mask_good)

    def create_xy(self, h, w, dev):
        y, x = torch.meshgrid(torch.arange(h, device=dev),
                              torch.arange(w, device=dev), indexing='ij')
        xy = torch.cat([x[..., None], y[..., None]], -1).reshape(-1, 2)
        return xy

    def extractDense(self, x, top_k=8_000, scale_factor=1.0):
        if top_k < 1:
            top_k = 100_000_000

        x, rh, rw = self.preprocess_tensor(x, scale_factor)
        print("extractDense: x shape: ", x.shape)
        print("extractDense: (rh1, rw1): ", (rh, rw))

        M1, _K1, H1 = self.net(x)

        B, C, _H1, _W1 = M1.shape

        xy1 = (self.create_xy(_H1, _W1, M1.device) * 8).expand(B, -1, -1)

        M1 = M1.permute(0, 2, 3, 1).reshape(B, -1, C)
        H1 = H1.permute(0, 2, 3, 1).reshape(B, -1)

        # _, top_k = torch.topk(H1, k = min(H1[0].shape[0], top_k), dim=-1) # ! changes here. this is orig
        print("H1: ", H1.dtype)
        print("H1 shape: ", H1.shape)

        # ONNX-compatible version - replace torch.tensor with torch.full
        top_k_tensor = torch.full(
            (), top_k, device=H1.device, dtype=torch.int64)
        actual_top_k = torch.min(H1.shape[1], top_k_tensor)

        # Display the actual top_k used
        print(f"Using top_k value: {actual_top_k}")
        _, top_k_indices = torch.topk(H1, k=actual_top_k, dim=-1)

        feats = torch.gather(M1, 1, top_k_indices[..., None].expand(-1, -1, C))
        mkpts = torch.gather(
            xy1, 1, top_k_indices[..., None].expand(-1, -1, 2))

        if isinstance(rw, float):  # Ensure rw and rh are tensors
            rw = torch.full((), rw, device=mkpts.device)
        if isinstance(rh, float):
            rh = torch.full((), rh, device=mkpts.device)

        scale_w = torch.full((1, mkpts.shape[1], 1), rw, device=mkpts.device)
        scale_h = torch.full((1, mkpts.shape[1], 1), rh, device=mkpts.device)
        scale_tensor = torch.cat([scale_w, scale_h], dim=-1)
        mkpts = mkpts * scale_tensor

        return mkpts, scale_tensor, feats

    def extract_dualscale(self, x, top_k, s1=0.6, s2=1.3):
        B, _, _, _ = x.shape

        mkpts_1, scale_tensor_1, feats_1 = self.extractDense(
            x, int(top_k*0.20), scale_factor=s1)
        mkpts_2, scale_tensor_2, feats_2 = self.extractDense(
            x, int(top_k*0.80), scale_factor=s2)

        mkpts = torch.cat([mkpts_1, mkpts_2], dim=1)
        scale_tensor = torch.cat([scale_tensor_1, scale_tensor_2], dim=1)
        feats = torch.cat([feats_1, feats_2], dim=1)

        return mkpts, scale_tensor, feats

    def parse_input(self, x):
        if len(x.shape) == 3:
            x = x[None, ...]

        if isinstance(x, np.ndarray):
            x = torch.tensor(x).permute(0, 3, 1, 2)/255

        return x

    @torch.inference_mode()
    def detect_and_match_dense(self, img1, img2, top_k=2000, min_cossim=0.82, fine_conf=0.25):
        """
        ONNX-compatible version of detect_and_match_dense function with feature refinement.
        Detects dense features in two images, matches them with refinement, and returns all features plus correspondence information.

        Args:
            img1: First image tensor (B, C, H, W)
            img2: Second image tensor (B, C, H, W)
            top_k: Number of top features to extract per image
            min_cossim: Minimum cosine similarity for matching
            fine_conf: Confidence threshold for refined matches

        Returns:
            Tuple containing:
                keypoints1: torch.Tensor(N, 2) - All keypoint locations in image 1 (with refinement applied to matched ones)
                keypoints2: torch.Tensor(M, 2) - All keypoint locations in image 2  
                descriptors1: torch.Tensor(N, 64) - All descriptors for features in image 1
                descriptors2: torch.Tensor(M, 64) - All descriptors for features in image 2
                scales1: torch.Tensor(N,) - All scales for features in image 1
                scales2: torch.Tensor(M,) - All scales for features in image 2
                target_indices: torch.Tensor(N,) - For each feature in image 1, index in image 2 (-1 for no match)
        """
        # Detect dense features in both images
        kpts1, scales1, desc1 = self.extract_dualscale(img1, top_k)
        kpts2, scales2, desc2 = self.extract_dualscale(img2, top_k)

        # Perform initial matching using batch_match approach
        B = desc1.shape[0]
        cossim = torch.bmm(desc1, desc2.permute(0, 2, 1))
        match12 = torch.argmax(cossim, dim=-1)
        match21 = torch.argmax(cossim.permute(0, 2, 1), dim=-1)

        idx0 = torch.arange(match12[0].shape[0], device=match12.device)

        # Find mutual matches for first batch (assuming B=1)
        b = 0
        mutual = match21[b][match12[b]] == idx0
        cossim_max, _ = cossim[b].max(dim=1)
        good = cossim_max > min_cossim
        idx0_matched = idx0[mutual & good]
        idx1_matched = match12[b][mutual & good]

        # Extract matched features for refinement
        feats1_matched = desc1[b][idx0_matched]
        feats2_matched = desc2[b][idx1_matched]
        mkpts_0 = kpts1[b][idx0_matched].clone()
        mkpts_1 = kpts2[b][idx1_matched]
        sc0 = scales1[b][idx0_matched]

        # Compute fine offsets using the fine matcher
        offsets = self.net.fine_matcher(
            torch.cat([feats1_matched, feats2_matched], dim=-1))
        conf = F.softmax(offsets*3, dim=-1).max(dim=-1)[0]
        offsets = self.subpix_softmax2d(offsets.view(-1, 8, 8))

        # Apply refinement to keypoints in image 1
        mkpts_0 += offsets * (sc0[:, None])

        # Filter matches based on confidence
        mask_good = conf > fine_conf
        refined_idx0 = idx0_matched[mask_good]
        refined_idx1 = idx1_matched[mask_good]
        refined_mkpts_0 = mkpts_0[mask_good]

        # Create target indices array for all features in image 1
        target_indices_all = torch.full(
            (kpts1.shape[1],), -1, dtype=torch.long, device=kpts1.device)
        target_indices_all[refined_idx0] = refined_idx1

        # Apply refinements to all keypoints in image 1
        all_kpts1 = kpts1[b].clone()
        all_kpts1[refined_idx0] = refined_mkpts_0

        return all_kpts1, kpts2[b], desc1[b], desc2[b], scales1[b], scales2[b], target_indices_all
