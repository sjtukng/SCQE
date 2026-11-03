"""Support encoding: object/context representatives p_c, t_c from K instances."""
from collections import defaultdict
from typing import Dict, List, Tuple

import torch

from .config import BETA, GRID, EPS_NORM
from .geometry import crop_to_valid, expand_box
from .pooling import phi, grid_centers, nu


class SupportEncoder:
    """Encodes support episodes with the PF-RPN encoder; caches per image."""

    def __init__(self, runtime):
        self.rt = runtime

    @torch.no_grad()
    def _encode_image(self, img_path: str) -> Dict:
        st = self.rt.native_pass(img_path)   # full pass; we keep encoder side
        keep = dict(memory_feats=st.features,
                    valid_feature_hw=st.valid_feature_hw,
                    img_W=st.img_W, img_H=st.img_H,
                    ori_W=st.ori_W, ori_H=st.ori_H,
                    scale=st.scale_factor)
        return keep

    def encode_episode(self, instances: List[Dict], n_classes: int
                       ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """instances: dicts with class_id, image_path, box_xyxy (original).

        Returns p [C,D], t [C,D], c_valid [C] bool.
        """
        by_img = defaultdict(list)
        for it in instances:
            by_img[it["image_path"]].append(it)
        u, v = defaultdict(list), defaultdict(list)
        for path, group in by_img.items():
            enc = self._encode_image(path)
            s = enc["scale"][0]  # uniform keep-ratio scale
            W, H = enc["img_W"], enc["img_H"]
            # all supplied boxes of this image (resized coords), for exclusion
            all_boxes = []
            for it in group:
                b = torch.tensor(it["box_xyxy"], dtype=torch.float32,
                                 device=enc["memory_feats"][0].device) * s
                all_boxes.append(b)
            for it, box_r in zip(group, all_boxes):
                box_c = crop_to_valid(box_r.clone(), W, H)
                if not bool(((box_c[2] > box_c[0]) & (box_c[3] > box_c[1]))):
                    continue  # degenerate supplied box: invalid instance
                u_vec, u_ok = phi(enc["memory_feats"], box_c[None], W, H,
                                  enc["valid_feature_hw"])
                expanded = expand_box(box_c[None], BETA, W, H)
                centers = grid_centers(expanded)
                excl = []
                for ob in all_boxes:
                    inside = (centers[..., 0] >= ob[0]) & (centers[..., 0] <= ob[2]) & \
                             (centers[..., 1] >= ob[1]) & (centers[..., 1] <= ob[3])
                    excl.append(inside)
                v_vec, v_ok = phi(enc["memory_feats"], expanded, W, H,
                                  enc["valid_feature_hw"], exclude_masks=excl)
                c = int(it["class_id"])
                if bool(u_ok[0]):
                    u[c].append(u_vec[0])
                if bool(v_ok[0]):
                    v[c].append(v_vec[0])
        dev = self.rt.device
        D = None
        # infer D from any encoded vector
        for lst in list(u.values()) + list(v.values()):
            D = lst[0].shape[0]
            break
        if D is None:
            raise ValueError("no valid support instances")
        p = torch.zeros(n_classes, D, device=dev)
        t = torch.zeros(n_classes, D, device=dev)
        c_valid = torch.zeros(n_classes, dtype=torch.bool, device=dev)
        for c in range(n_classes):
            if u.get(c) and v.get(c):
                pu = torch.stack(u[c]).mean(dim=0)
                pt = torch.stack(v[c]).mean(dim=0)
                if float(pu.norm()) > EPS_NORM and float(pt.norm()) > EPS_NORM:
                    p[c] = nu(pu[None])[0]
                    t[c] = nu(pt[None])[0]
                    c_valid[c] = True
        return p, t, c_valid
