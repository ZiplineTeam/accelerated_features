# This module was added by zipline
import numpy as np
import os
import torch
import torch.nn.functional as F

from kornia.geometry.ransac import RANSAC

import tqdm
import kornia

from modules.model import *
from modules.interpolator import InterpolateSparse2d

class XFeat(nn.Module):
    def __init__(self, weights = os.path.abspath(os.path.dirname(__file__)) + '/../weights/xfeat.pt', top_k = 6096, detection_threshold=0.05): # ! change to top_k=2096
        super().__init__()
        self.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = XFeatModel().to(self.dev).eval()
        self.top_k = top_k
        self.detection_threshold = detection_threshold

        self.interpolator = InterpolateSparse2d('bicubic')

        if weights is not None:
            if isinstance(weights, str):
                print('loading weights from: ' + weights)
                self.net.load_state_dict(torch.load(weights, map_location=self.dev, weights_only=True))
            else:
                self.net.load_state_dict(weights)

        #Try to import LightGlue from Kornia
        self.kornia_available = False
        self.lighterglue = None
        self.kornia_available=True
        print("xfeat match star initialized")


    @torch.inference_mode()
    def detectAndComputeDense(self, x, top_k = None, multiscale = True):
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
        if top_k is None: top_k = self.top_k
        if multiscale:
            print("doing multiscale")
            mkpts, sc, feats = self.extract_dualscale(x, top_k)
        else:
            print("doing single scale")
            mkpts, feats = self.extractDense(x, top_k)
            sc = torch.ones(mkpts.shape[:2], device=mkpts.device)

        print("mkpts: ", mkpts.dtype)
        print("feats: ", feats.dtype)
        print("sc: ", sc.dtype)

        return {'keypoints': mkpts,
                'descriptors': feats,
                'scales': sc }

    @torch.inference_mode()
    def xfeat_detect(self, img, top_k = None):
        if top_k is None: top_k = self.top_k
        im_set = self.parse_input(img)
        return self.detectAndComputeDense(im_set, top_k)
    
    # @torch.inference_mode()
    # # def match_xfeat_star(self, im_set1, im_set2, top_k = None):
    # def match_xfeat_star(self, dec1_kps, dec1_desc, dec1_scales, dec2_kps, dec2_desc, top_k = None):
    #     """
    #         Extracts coarse feats, then match pairs and finally refine matches, currently supports batched mode.
    #         input:
    #             im_set1 -> torch.Tensor(B, C, H, W) or np.ndarray (H,W,C): grayscale or rgb images.
    #             im_set2 -> torch.Tensor(B, C, H, W) or np.ndarray (H,W,C): grayscale or rgb images.
    #             top_k -> int: keep best k features
    #         returns:
    #             matches -> List[torch.Tensor(N, 4)]: List of size B containing tensor of pairwise matches (x1,y1,x2,y2)
    #     """
    #     if top_k is None: top_k = self.top_k

    #     #Match batches of pairs
    #     dec1_scales = dec1_scales.squeeze(-1)
    #     idxs_list = self.batch_match(dec1_desc, dec2_desc )
    #     B = 1

    #     #Refine coarse matches
    #     #this part is harder to batch, currently iterate
    #     matches = []
    #     for b in range(B):
    #         matches.append(self.refine_matches(dec1_kps, dec1_desc, dec1_scales, dec2_kps, dec2_desc, matches = idxs_list, batch_idx=b))

    #     ans = (matches[0][:, :2], matches[0][:, 2:])
    #     print("ans: ", ans)
    #     return ans
    
    @torch.inference_mode()
    # def match_xfeat_star_onnx(self):
    def match_xfeat_star_onnx(self, dec1_ids, dec1_kps, dec1_desc, dec1_scales, dec2_ids, dec2_kps, dec2_desc):

        dec1_scales = dec1_scales.squeeze(2)
        matched_idx_ref = []
        matched_idx_target = []

        dec1_ids = dec1_ids[0]
        dec2_ids = dec2_ids[0]

        # Match batches of pairs
        idxs_list = self.batch_match(dec1_desc, dec2_desc)

        #Refine coarse matches
        matches = []
        for b in range(1):
            match, idx0, idx1 = self.refine_matches(dec1_kps, dec1_desc, dec1_scales, dec2_kps, dec2_desc, matches = idxs_list, batch_idx=b)
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

        # ref_output_idx_vec = [ref_ids_vec[i].item() for i in matched_idx_ref]

        ref_points = matches[0][:, :2].clone().detach()
        dst_points = matches[0][:, 2:].clone().detach()

        # ransac = RANSAC('homography').cpu()
        # _, mask = ransac(ref_points, dst_points, weights=None)
        # mask = mask.flatten()

        # final_mask = list(range(len(mask)))

        # print("ref_points: ", ref_points.shape)
        # print("dst_points: ", dst_points.shape)
        # print("ref_output_idx_vec: ", ref_output_idx_vec.shape)
        # print("final_mask: ", final_mask.shape)

        # return (ref_points, dst_points, ref_output_idx_vec, final_mask)
        return (ref_output_idx_vec, ref_points, target_output_idx_vec, dst_points)
    
    
    def preprocess_tensor(self, x):
        """ Guarantee that image is divisible by 32 to avoid aliasing artifacts. """
        if isinstance(x, np.ndarray) and len(x.shape) == 3:
            x = torch.tensor(x).permute(2,0,1)[None]
        x = x.to(self.dev).float()

        H, W = x.shape[-2:]
        _H, _W = (H//32) * 32, (W//32) * 32
        rh, rw = H/_H, W/_W

        x = F.interpolate(x, (_H, _W), mode='bilinear', align_corners=False)
        return x, rh, rw

    @torch.inference_mode()
    def batch_match(self, feats1, feats2, min_cossim = -1):
        B = feats1.shape[0]
        cossim = torch.bmm(feats1, feats2.permute(0,2,1))
        match12 = torch.argmax(cossim, dim=-1)
        match21 = torch.argmax(cossim.permute(0,2,1), dim=-1)

        idx0 = torch.arange(match12[0].shape[0], device=match12.device)

        batched_matches = []

        for b in range(B):
            mutual = match21[b][match12[b]] == idx0

            if min_cossim > 0:
                cossim_max, _ = cossim[b].max(dim=1)
                good = cossim_max > min_cossim
                idx0_b = idx0[mutual & good]
                idx1_b = match12[b][mutual & good]
            else:
                idx0_b = idx0[mutual]
                idx1_b = match12[b][mutual]

            batched_matches.append((idx0_b, idx1_b))

        return batched_matches

    def subpix_softmax2d(self, heatmaps, temp = 3):
        N, H, W = heatmaps.shape
        heatmaps = torch.softmax(temp * heatmaps.view(-1, H*W), -1).view(-1, H, W)
        x, y = torch.meshgrid(torch.arange(W, device =  heatmaps.device ), torch.arange(H, device =  heatmaps.device ), indexing = 'xy')
        x = x - (W//2)
        y = y - (H//2)

        coords_x = (x[None, ...] * heatmaps)
        coords_y = (y[None, ...] * heatmaps)
        coords = torch.cat([coords_x[..., None], coords_y[..., None]], -1).view(N, H*W, 2)
        coords = coords.sum(1)

        return coords

    def refine_matches(self, dec1_kps, dec1_desc, dec1_scales, dec2_kps, dec2_desc, matches, batch_idx, fine_conf = 0.25):
        idx0, idx1 = matches[batch_idx]
        feats1 = dec1_desc[batch_idx][idx0]
        feats2 = dec2_desc[batch_idx][idx1]
        mkpts_0 = dec1_kps[batch_idx][idx0]
        mkpts_1 = dec2_kps[batch_idx][idx1]
        sc0 = dec1_scales[batch_idx][idx0]

        #Compute fine offsets
        offsets = self.net.fine_matcher(torch.cat([feats1, feats2],dim=-1))
        conf = F.softmax(offsets*3, dim=-1).max(dim=-1)[0]
        offsets = self.subpix_softmax2d(offsets.view(-1,8,8))

        mkpts_0 += offsets* (sc0[:,None]) # ! change this back to to mkpts_1

        mask_good = conf > fine_conf
        # print("mask_good: ", mask_good)
        mkpts_0 = mkpts_0[mask_good]
        mkpts_1 = mkpts_1[mask_good]

        # print("idx0: ", idx0)
        # print("idx1: ", idx1)

        # print("mkpts_0: ", mkpts_0)
        # print("mkpts_1: ", mkpts_1)
        # if mkpts_0.shape[0] == 0:
        #     return torch.tensor([[-1, -1, -1, -1]])

        return (torch.cat([mkpts_0, mkpts_1], dim=-1), idx0[mask_good], idx1[mask_good])

    def create_xy(self, h, w, dev):
        y, x = torch.meshgrid(torch.arange(h, device = dev), 
                                torch.arange(w, device = dev), indexing='ij')
        xy = torch.cat([x[..., None],y[..., None]], -1).reshape(-1,2)
        return xy

    def extractDense(self, x, top_k = 8_000):
        if top_k < 1:
            top_k = 100_000_000

        x, rh1, rw1 = self.preprocess_tensor(x)

        M1, K1, H1 = self.net(x)
        
        B, C, _H1, _W1 = M1.shape
        
        xy1 = (self.create_xy(_H1, _W1, M1.device) * 8).expand(B,-1,-1)

        M1 = M1.permute(0,2,3,1).reshape(B, -1, C)
        H1 = H1.permute(0,2,3,1).reshape(B, -1)

        # _, top_k = torch.topk(H1, k = min(H1[0].shape[0], top_k), dim=-1) # ! changes here. this is orig
        print("H1: ", H1.dtype)
        print("H1 shape: ", H1.shape)

        top_k_tensor = torch.tensor(top_k, device=H1.device, dtype=torch.int64)
        actual_top_k = torch.min(H1.shape[1], top_k_tensor)

        print(f"Using top_k value: {actual_top_k}")  # Display the actual top_k used
        _, top_k_indices = torch.topk(H1, k=actual_top_k, dim=-1)


        feats = torch.gather(M1, 1, top_k_indices[..., None].expand(-1, -1, C))
        mkpts = torch.gather(xy1, 1, top_k_indices[..., None].expand(-1, -1, 2))

        if isinstance(rw1, float):  # Ensure rw1 and rh1 are tensors
            rw1 = torch.tensor(rw1, device=mkpts.device)
        if isinstance(rh1, float):
            rh1 = torch.tensor(rh1, device=mkpts.device)
        scale_tensor = torch.stack([rw1, rh1], dim=0).view(1, -1)
        mkpts = mkpts * scale_tensor

        return mkpts, feats


        # feats = torch.gather( M1, 1, top_k[...,None].expand(-1, -1, 64))
        # mkpts = torch.gather(xy1, 1, top_k[...,None].expand(-1, -1, 2))
        # mkpts = mkpts * torch.tensor([rw1, rh1], device=mkpts.device).view(1,-1)

        # return mkpts, feats

    def extract_dualscale(self, x, top_k, s1 = 0.6, s2 = 1.3):
        x1 = F.interpolate(x, scale_factor=s1, align_corners=False, mode='bilinear')
        x2 = F.interpolate(x, scale_factor=s2, align_corners=False, mode='bilinear')

        B, _, _, _ = x.shape

        mkpts_1, feats_1 = self.extractDense(x1, int(top_k*0.20))
        mkpts_2, feats_2 = self.extractDense(x2, int(top_k*0.80))

        mkpts = torch.cat([mkpts_1/s1, mkpts_2/s2], dim=1)
        sc1 = torch.ones(mkpts_1.shape[:2], device=mkpts_1.device) * (1/s1)
        sc2 = torch.ones(mkpts_2.shape[:2], device=mkpts_2.device) * (1/s2)
        sc = torch.cat([sc1, sc2],dim=1)
        feats = torch.cat([feats_1, feats_2], dim=1)

        return mkpts, sc, feats

    def parse_input(self, x):
        if len(x.shape) == 3:
            x = x[None, ...]

        if isinstance(x, np.ndarray):
            x = torch.tensor(x).permute(0,3,1,2)/255

        return x
