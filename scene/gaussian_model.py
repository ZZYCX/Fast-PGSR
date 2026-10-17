#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation, build_scaling
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from pytorch3d.transforms import quaternion_to_matrix

# SSP: version of the persisted semantic state (checkpoint fields / semantic_state.npz).
# v2 adds _semantic_candidate_id / _semantic_candidate_stable_count (all-class revision state).
from utils.semantic_io import SEMANTIC_STATE_VERSION, _xyz_checksum, _semantic_checksum
SEMANTIC_FIELDS = ["_object_id", "_object_prob", "_semantic_confidence", "_source_point3D_id",
                   "_semantic_evidence_sum", "_semantic_obs_count", "_semantic_stable_count",
                   "_semantic_candidate_id", "_semantic_candidate_stable_count"]

# v2 sidecar array names (core 7 + revision candidate 2).
SEMANTIC_SIDEAR_KEYS = SEMANTIC_FIELDS
CANDIDATE_SEMANTIC_KEYS = ["_semantic_candidate_id", "_semantic_candidate_stable_count"]


def dilate(bin_img, ksize=5):
    pad = (ksize - 1) // 2
    bin_img = torch.nn.functional.pad(bin_img, pad=[pad, pad, pad, pad], mode='reflect')
    out = torch.nn.functional.max_pool2d(bin_img, kernel_size=ksize, stride=1, padding=0)
    return out

def erode(bin_img, ksize=5):
    out = 1 - dilate(1 - bin_img, ksize)
    return out

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, sh_degree : int, sls_debug : bool = False):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self._xyz = torch.empty(0)
        self._knn_f = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.max_weight = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.xyz_gradient_accum_abs = torch.empty(0)
        self.denom = torch.empty(0)
        self.denom_abs = torch.empty(0)
        self.optimizer = None
        self.shoptimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.knn_dists = None
        self.knn_idx = None
        # Non-learnable semantic attributes, empty until create_from_pcd(semantics=...) attaches them.
        self._object_id = torch.empty(0)
        self._object_prob = torch.empty(0)
        self._semantic_confidence = torch.empty(0)
        self._source_point3D_id = torch.empty(0)
        # SLS dynamic semantic state (established now, not used for label updates yet).
        self._semantic_evidence_sum = torch.empty(0)
        self._semantic_obs_count = torch.empty(0)
        self._semantic_stable_count = torch.empty(0)
        # VGER+: all-class revision candidate state (which class is most likely to
        # replace the current label, and for how many consecutive windows).
        self._semantic_candidate_id = torch.empty(0)
        self._semantic_candidate_stable_count = torch.empty(0)
        self._semantics_attached = False
        self._sls_debug = sls_debug
        self._iteration_hint = None
        self.setup_functions()
        self.use_app = False

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._knn_f,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.max_weight,
            self.xyz_gradient_accum,
            self.xyz_gradient_accum_abs,
            self.denom,
            self.denom_abs,
            self.optimizer.state_dict(),
            self.shoptimizer.state_dict(),
            self.spatial_lr_scale,
            # SSP semantic state (appended at the end so old non-semantic
            # checkpoints stay loadable via the 17-field branch of restore()).
            self._semantics_attached,
            SEMANTIC_STATE_VERSION,
            self._object_id.detach(),
            self._object_prob.detach(),
            self._semantic_confidence.detach(),
            self._source_point3D_id.detach(),
            self._semantic_evidence_sum.detach(),
            self._semantic_obs_count.detach(),
            self._semantic_stable_count.detach(),
            self._semantic_candidate_id.detach(),
            self._semantic_candidate_stable_count.detach(),
        )

    def restore(self, model_args, training_args):
        # Semantic checkpoint v2 (28 fields, with revision candidate state),
        # semantic checkpoint v1 (26 fields, without candidate state), or legacy
        # non-semantic (17 fields).
        candidate_id = candidate_stable = None
        if len(model_args) >= 28:
            (self.active_sh_degree,
             self._xyz, self._knn_f, self._features_dc, self._features_rest,
             self._scaling, self._rotation, self._opacity, self.max_radii2D, self.max_weight,
             xyz_gradient_accum, xyz_gradient_accum_abs, denom, denom_abs,
             opt_dict, shopt_dict, self.spatial_lr_scale,
             semantics_attached, semantic_version,
             object_id, object_prob, semantic_confidence, source_id,
             evidence_sum, obs_count, stable_count,
             candidate_id, candidate_stable) = model_args
        elif len(model_args) >= 26:
            (self.active_sh_degree,
             self._xyz, self._knn_f, self._features_dc, self._features_rest,
             self._scaling, self._rotation, self._opacity, self.max_radii2D, self.max_weight,
             xyz_gradient_accum, xyz_gradient_accum_abs, denom, denom_abs,
             opt_dict, shopt_dict, self.spatial_lr_scale,
             semantics_attached, semantic_version,
             object_id, object_prob, semantic_confidence, source_id,
             evidence_sum, obs_count, stable_count) = model_args
        else:
            (self.active_sh_degree,
             self._xyz, self._knn_f, self._features_dc, self._features_rest,
             self._scaling, self._rotation, self._opacity, self.max_radii2D, self.max_weight,
             xyz_gradient_accum, xyz_gradient_accum_abs, denom, denom_abs,
             opt_dict, shopt_dict, self.spatial_lr_scale) = model_args
            semantics_attached = False
            semantic_version = None
            object_id = object_prob = semantic_confidence = source_id = None
            evidence_sum = obs_count = stable_count = None

        if semantics_attached:
            if int(semantic_version) not in (SEMANTIC_STATE_VERSION, 1):
                raise RuntimeError(
                    f"[SSP] 未知语义状态版本 {semantic_version}，当前支持 {SEMANTIC_STATE_VERSION}")
            device = self._xyz.device
            self._object_id = object_id.to(device=device, dtype=torch.int64)
            self._object_prob = object_prob.to(device=device, dtype=torch.float32)
            self._semantic_confidence = semantic_confidence.to(device=device, dtype=torch.float32)
            self._source_point3D_id = source_id.to(device=device, dtype=torch.int64)
            self._semantic_evidence_sum = evidence_sum.to(device=device, dtype=torch.float32)
            self._semantic_obs_count = obs_count.to(device=device, dtype=torch.int64)
            self._semantic_stable_count = stable_count.to(device=device, dtype=torch.int64)
            n = self._xyz.shape[0]
            if candidate_id is not None:
                self._semantic_candidate_id = candidate_id.to(device=device, dtype=torch.int64)
                self._semantic_candidate_stable_count = candidate_stable.to(device=device, dtype=torch.int64)
            else:
                # legacy v1 checkpoint: fresh all-class revision state (no candidate).
                self._semantic_candidate_id = torch.full((n,), -1, dtype=torch.int64, device=device)
                self._semantic_candidate_stable_count = torch.zeros((n,), dtype=torch.int64, device=device)
            self._semantics_attached = True
            self._assert_semantic_alignment("restore")
        else:
            self._reset_semantic_tensors()

        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.xyz_gradient_accum_abs = xyz_gradient_accum_abs
        self.denom = denom
        self.denom_abs = denom_abs
        self.optimizer.load_state_dict(opt_dict)
        self.shoptimizer.load_state_dict(shopt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
        
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_features_dc(self):
        return self._features_dc
    
    @property
    def get_features_rest(self):
        return self._features_rest
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_smallest_axis(self, return_idx=False):
        rotation_matrices = self.get_rotation_matrix()
        smallest_axis_idx = self.get_scaling.min(dim=-1)[1][..., None, None].expand(-1, 3, -1)
        smallest_axis = rotation_matrices.gather(2, smallest_axis_idx)
        if return_idx:
            return smallest_axis.squeeze(dim=2), smallest_axis_idx[..., 0, 0]
        return smallest_axis.squeeze(dim=2)
    
    def get_normal(self, view_cam):
        normal_global = self.get_smallest_axis()
        gaussian_to_cam_global = view_cam.camera_center - self._xyz
        neg_mask = (normal_global * gaussian_to_cam_global).sum(-1) < 0.0
        normal_global[neg_mask] = -normal_global[neg_mask]
        return normal_global
    
    def get_rotation_matrix(self):
        return quaternion_to_matrix(self.get_rotation)

    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float, semantics=None, export_debug_path=None):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist = torch.sqrt(torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001))
        # print(f"new scale {torch.quantile(dist, 0.1)}")
        scales = torch.log(dist)[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        knn_f = torch.randn((fused_point_cloud.shape[0], 6)).float().cuda()
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._knn_f = nn.Parameter(knn_f.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.max_weight = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        if semantics is not None:
            self._attach_semantics(semantics, export_debug_path=export_debug_path)

    def _attach_semantics(self, semantics, export_debug_path=None):
        """Attach non-learnable semantic tensors aligned 1:1 with the initial Gaussians.

        Strictly by the Fast-PGSR-filtered POINT3D_ID order. These tensors are NOT
        nn.Parameter, have no requires_grad, and are NOT part of any optimizer, so the
        original scaling/rotation/opacity/SH/knn_f/covariance init is untouched.

        Runs the full acceptance verification and prints the summary.
        """
        n = self._xyz.shape[0]
        ids = torch.from_numpy(np.asarray(semantics["point3D_ids"])).long().cuda()
        obj_id = torch.from_numpy(np.asarray(semantics["object_id"])).long().cuda()
        probs = torch.from_numpy(np.asarray(semantics["object_probs"])).float().cuda()
        conf = torch.from_numpy(np.asarray(semantics["confidence"])).float().cuda()

        # ---- (1) duplicate / matching checks (filtered & npz dups done in dataset_readers;
        # re-check filtered IDs here so the acceptance is self-contained) ----
        assert torch.unique(ids).numel() == ids.numel(), \
            "[Semantic Init] 过滤后 POINT3D_ID 存在重复"

        # ---- (2) every semantic tensor length == initial Gaussian count ----
        assert n == ids.shape[0] == obj_id.shape[0] == probs.shape[0] == conf.shape[0], (
            f"[Semantic Init] Gaussian 数量({n}) != 语义数量"
            f"(ids={ids.shape[0]}, object_id={obj_id.shape[0]}, "
            f"object_probs={probs.shape[0]}, confidence={conf.shape[0]})")

        # ---- (3) object_probs validity ----
        assert probs.dim() == 2 and probs.shape[1] == 4, \
            f"[Semantic Init] object_prob 形状应为 [N,4]，实际 {tuple(probs.shape)}"
        assert bool(torch.isfinite(probs).all()), \
            "[Semantic Init] object_probs 含 NaN 或 Inf"
        assert bool((probs >= 0).all()), \
            "[Semantic Init] object_probs 存在负值"
        row_sum = probs.sum(dim=-1)
        assert bool(torch.allclose(row_sum, torch.ones_like(row_sum), atol=1e-4)), \
            "[Semantic Init] 每行 object_prob 概率和与 1 的误差应小于 1e-4"

        # ---- (4) object_id only in {-1,0,1,2,3} ----
        valid = (obj_id == -1) | (obj_id == 0) | (obj_id == 1) | (obj_id == 2) | (obj_id == 3)
        assert bool(valid.all()), "[Semantic Init] object_id 仅允许包含 {{-1,0,1,2,3}}"

        # ---- (6) non-learnable: not nn.Parameter, requires_grad False ----
        for name, t in [("_object_id", obj_id), ("_object_prob", probs),
                        ("_semantic_confidence", conf), ("_source_point3D_id", ids)]:
            assert not isinstance(t, nn.Parameter), f"[Semantic Init] {name} 不应为 nn.Parameter"
            assert t.requires_grad is False, f"[Semantic Init] {name} requires_grad 应为 False"

        # ---- attach (in initial-Gaussian order, non-learnable) ----
        self._object_id = obj_id
        self._object_prob = probs
        self._semantic_confidence = conf
        self._source_point3D_id = ids
        # SLS dynamic state starts zeroed for every initial Gaussian.
        self._semantics_attached = True
        self._semantic_evidence_sum = torch.zeros((n, 4), dtype=torch.float32, device="cuda")
        self._semantic_obs_count = torch.zeros((n,), dtype=torch.int64, device="cuda")
        self._semantic_stable_count = torch.zeros((n,), dtype=torch.int64, device="cuda")
        # VGER+: candidate state starts neutral (-1 = no replacement candidate).
        self._semantic_candidate_id = torch.full((n,), -1, dtype=torch.int64, device="cuda")
        self._semantic_candidate_stable_count = torch.zeros((n,), dtype=torch.int64, device="cuda")
        self._assert_semantic_alignment("init")

        # ---- acceptance summary ----
        stats = semantics.get("stats", {})
        obj_np = obj_id.detach().cpu().numpy()
        probs_np = probs.detach().cpu().numpy()
        ids_np = ids.detach().cpu().numpy()
        row_sum_np = row_sum.detach().cpu().numpy()
        counts = {
            "unknown": int((obj_np == -1).sum()),
            "background": int((obj_np == 0).sum()),
            "plant": int((obj_np == 1).sum()),
            "rubik_1": int((obj_np == 2).sum()),
            "rubik_2": int((obj_np == 3).sum()),
        }
        print("[Semantic Init] ================= 语义绑定验收摘要 =================")
        print(f"[Semantic Init] raw_colmap_points       : {stats.get('raw_num_points', '?')}")
        print(f"[Semantic Init] filtered_colmap_points  : {n}")
        print(f"[Semantic Init] npz_points              : {stats.get('npz_total', '?')}")
        print(f"[Semantic Init] matched_points          : {stats.get('matched', '?')}")
        print(f"[Semantic Init] missing_ids             : {stats.get('missing_id_count', '?')}")
        print(f"[Semantic Init] unused_npz_points       : {stats.get('unused_npz', '?')}")
        _max_err = stats.get('max_xyz_alignment_error', None)
        _err_str = f"{_max_err:.3e}" if isinstance(_max_err, (int, float)) else "?"
        print(f"[Semantic Init] max_xyz_alignment_error : {_err_str}")
        print(f"[Semantic Init] 类别统计                 : unknown={counts['unknown']}, "
              f"background={counts['background']}, plant={counts['plant']}, "
              f"rubik_1={counts['rubik_1']}, rubik_2={counts['rubik_2']}")
        print(f"[Semantic Init] probability_sum         : min={row_sum_np.min():.8f}, max={row_sum_np.max():.8f}")
        print("[Semantic Init] 前5个 POINT3D_ID / object_id / object_probs:")
        for k in range(min(5, n)):
            print(f"    id={ids_np[k]}  object_id={obj_np[k]}  probs={probs_np[k]}")
        print("[Semantic Init] ================= 验收通过 =========================")

        # ---- (7) optional debug export (after init, before densify) ----
        if export_debug_path:
            self.export_semantic_binding_debug(export_debug_path)

    def export_semantic_binding_debug(self, path):
        """Export a debug PLY of the initial Gaussian positions colored by object_id."""
        xyz = self._xyz.detach().cpu().numpy()
        obj = self._object_id.detach().cpu().numpy()
        color_map = {
            -1: (128, 128, 128),    # unknown   grey
            0:  (70, 130, 180),     # background steel blue
            1:  (0, 200, 0),        # plant     green
            2:  (220, 20, 60),      # rubik_1   crimson
            3:  (255, 215, 0),      # rubik_2   gold
        }
        rgb = np.array([color_map.get(int(o), (255, 255, 255)) for o in obj], dtype=np.uint8)
        dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                 ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
        elements = np.empty(xyz.shape[0], dtype=dtype)
        elements['x'] = xyz[:, 0]
        elements['y'] = xyz[:, 1]
        elements['z'] = xyz[:, 2]
        elements['red'] = rgb[:, 0]
        elements['green'] = rgb[:, 1]
        elements['blue'] = rgb[:, 2]
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
        print(f"[Semantic Init] 已导出调试 PLY: {path} ({xyz.shape[0]} 点)")

    # ------------------------------------------------------------------ #
    # SLS: Semantic Lifecycle Synchronization helpers
    # ------------------------------------------------------------------ #
    def _reset_semantic_tensors(self):
        """Clear all semantic tensors (used when restoring/loading without semantics)."""
        self._semantics_attached = False
        self._object_id = torch.empty(0)
        self._object_prob = torch.empty(0)
        self._semantic_confidence = torch.empty(0)
        self._source_point3D_id = torch.empty(0)
        self._semantic_evidence_sum = torch.empty(0)
        self._semantic_obs_count = torch.empty(0)
        self._semantic_stable_count = torch.empty(0)
        self._semantic_candidate_id = torch.empty(0)
        self._semantic_candidate_stable_count = torch.empty(0)

    def _semantic_class_counts(self):
        """Per-class object_id counts (labels: -1/0/1/2/3)."""
        obj = self._object_id.detach().cpu().numpy()
        return {"unknown": int((obj == -1).sum()),
                "background": int((obj == 0).sum()),
                "plant": int((obj == 1).sum()),
                "rubik_1": int((obj == 2).sum()),
                "rubik_2": int((obj == 3).sum())}

    def _assert_semantic_alignment(self, context):
        """Every semantic tensor must stay row-aligned with _xyz (no-op when no
        semantic binding is active, so the original Fast-PGSR path is untouched)."""
        if not self._semantics_attached:
            return True
        n = self._xyz.shape[0]
        lengths = {
            "_object_id": self._object_id.shape[0],
            "_object_prob": self._object_prob.shape[0],
            "_semantic_confidence": self._semantic_confidence.shape[0],
            "_source_point3D_id": self._source_point3D_id.shape[0],
            "_semantic_evidence_sum": self._semantic_evidence_sum.shape[0],
            "_semantic_obs_count": self._semantic_obs_count.shape[0],
            "_semantic_stable_count": self._semantic_stable_count.shape[0],
            "_semantic_candidate_id": self._semantic_candidate_id.shape[0],
            "_semantic_candidate_stable_count": self._semantic_candidate_stable_count.shape[0],
        }
        mism = {k: v for k, v in lengths.items() if v != n}
        assert not mism, f"[SLS] 语义对齐失败({context}): N={n}, 不一致={mism}"
        return True

    def _append_semantics_from_parents(self, parent_indices):
        """Append semantic rows for clone/split children.

        Long-term semantics (_object_id / _object_prob / _semantic_confidence /
        _source_point3D_id) are inherited from the parent Gaussian. Dynamic evidence
        state (_semantic_evidence_sum / _semantic_obs_count / _semantic_stable_count)
        is reset to zero. _source_point3D_id may be shared by many children.
        """
        if parent_indices is None or parent_indices.shape[0] == 0:
            return
        device = self._object_id.device
        m = parent_indices.shape[0]
        self._object_id = torch.cat([self._object_id, self._object_id[parent_indices]])
        self._object_prob = torch.cat([self._object_prob, self._object_prob[parent_indices]])
        self._semantic_confidence = torch.cat([self._semantic_confidence, self._semantic_confidence[parent_indices]])
        self._source_point3D_id = torch.cat([self._source_point3D_id, self._source_point3D_id[parent_indices]])
        self._semantic_evidence_sum = torch.cat([self._semantic_evidence_sum,
                                                 torch.zeros((m, 4), dtype=torch.float32, device=device)])
        self._semantic_obs_count = torch.cat([self._semantic_obs_count,
                                              torch.zeros((m,), dtype=torch.int64, device=device)])
        self._semantic_stable_count = torch.cat([self._semantic_stable_count,
                                                 torch.zeros((m,), dtype=torch.int64, device=device)])
        # VGER+: clone/split children reset candidate state (no replacement candidate yet).
        self._semantic_candidate_id = torch.cat([self._semantic_candidate_id,
                                                 torch.full((m,), -1, dtype=torch.int64, device=device)])
        self._semantic_candidate_stable_count = torch.cat([self._semantic_candidate_stable_count,
                                                           torch.zeros((m,), dtype=torch.int64, device=device)])

    def _prune_semantics(self, valid_mask):
        """Apply the exact same keep mask as _xyz to every semantic tensor.
        No reordering, no independently constructed prune mask."""
        self._object_id = self._object_id[valid_mask]
        self._object_prob = self._object_prob[valid_mask]
        self._semantic_confidence = self._semantic_confidence[valid_mask]
        self._source_point3D_id = self._source_point3D_id[valid_mask]
        self._semantic_evidence_sum = self._semantic_evidence_sum[valid_mask]
        self._semantic_obs_count = self._semantic_obs_count[valid_mask]
        self._semantic_stable_count = self._semantic_stable_count[valid_mask]
        self._semantic_candidate_id = self._semantic_candidate_id[valid_mask]
        self._semantic_candidate_stable_count = self._semantic_candidate_stable_count[valid_mask]

    def _log_semantic_op(self, op_type, before_n, after_n, before_counts,
                         parent_count=None, removed_count=None):
        """Log one clone / split / prune count-changing operation.
        Gated by _sls_debug so normal training stays quiet by default.
        `parent_count` = number of selected parents (clone/split), `removed_count` =
        number of pruned Gaussians (prune)."""
        if not self._semantics_attached or not self._sls_debug:
            return
        after_counts = self._semantic_class_counts()
        delta = after_n - before_n
        if op_type == "prune":
            if removed_count is None:
                removed_count = before_n - after_n
            print(f"[SLS] prune: before={before_n} removed={removed_count} after={after_n} | "
                  f"classes {before_counts}->{after_counts} | alignment=OK")
        else:
            if parent_count is None:
                parent_count = delta            # clone: each parent -> 1 child
            print(f"[SLS] {op_type}: before={before_n} parent={parent_count} new={delta} after={after_n} | "
                  f"classes {before_counts}->{after_counts} | alignment=OK")

    def accumulate_semantic_evidence(self, indices, evidence, enable_all_class=False):
        """VEA: accumulate 4-class evidence for the given Gaussian rows.

        By default only unknown (object_id == -1) rows may accumulate (original
        behaviour). With enable_all_class=True the known target classes
        (plant/rubik_1/rubik_2) may also accumulate evidence so already-labelled
        targets stay verifiable; background (object_id == 0) is a TERMINAL state and
        must never be passed in (a hard assert guards against it).

        Vectorised, inside torch.no_grad(), never touches learnable parameters and
        never enters the optimizer. Only _semantic_evidence_sum and _semantic_obs_count
        change; labels / probabilities / confidence / stable_count stay untouched.
        """
        if not self._semantics_attached:
            return
        indices = indices.to(torch.long)
        assert indices.dim() == 1, "[VEA] indices 必须是一维"
        assert evidence.dim() == 2 and evidence.shape[1] == 4, "[VEA] evidence 形状应为 [K,4]"
        assert indices.numel() == evidence.shape[0], "[VEA] indices 与 evidence 行数不一致"
        assert bool(torch.isfinite(evidence).all()), "[VEA] evidence 含 NaN/Inf"
        assert bool((evidence >= 0).all()), "[VEA] evidence 含负值"
        row_sum = evidence.sum(dim=-1)
        assert bool(torch.allclose(row_sum, torch.ones_like(row_sum), atol=1e-3)), \
            "[VEA] evidence 每行概率和应≈1"
        if enable_all_class:
            assert bool((self._object_id[indices] != 0).all()), \
                "[VEA] background 是终止语义状态，不得累计语义证据：" \
                "Background Gaussians are terminal semantic states and must not accumulate semantic evidence."
        else:
            assert bool((self._object_id[indices] == -1).all()), \
                "[VEA] 默认只能为 unknown 行累计证据（启用 enable_all_class 可包含目标类别）"
        with torch.no_grad():
            self._semantic_evidence_sum[indices] += evidence
            self._semantic_obs_count[indices] += 1
        self._assert_semantic_alignment("accumulate_semantic_evidence")

    def resolve_unknown_semantics(self, opt, iteration=None):
        """VGER: resolve semantics from the accumulated multi-view evidence.

        With enable_all_class_semantic_revision=True the unknown and known target
        classes (plant/rubik_1/rubik_2) are re-verified each window (a conflicted
        target first falls back to unknown and can later be re-confirmed; unknown may
        become a target or background). background (object_id == 0) is a TERMINAL
        state and is never revised. With the switch OFF the original behaviour is
        kept exactly: only unknown rows are resolved, only unknown -> target/background.

        Only the semantic tensors change; learnable parameters / loss / geometry are
        untouched. Runs in torch.no_grad().
        """
        if not self._semantics_attached:
            return None
        self._assert_semantic_alignment("resolve_unknown_semantics")
        if bool(getattr(opt, "enable_all_class_semantic_revision", False)):
            return self._resolve_all_class(opt, iteration)
        return self._resolve_unknown_legacy(opt, iteration)

    def _resolve_unknown_legacy(self, opt, iteration=None):
        """VGER (original path): resolve unknown Gaussians only. Kept verbatim so the
        feature switches being OFF reproduces the pre-existing behaviour exactly."""
        prior_strength = float(getattr(opt, "semantic_prior_strength", 5.0))
        target_min_obs = int(getattr(opt, "semantic_target_min_observations", 5))
        target_conf = float(getattr(opt, "semantic_target_confidence", 0.75))
        target_margin = float(getattr(opt, "semantic_target_margin", 0.20))
        bg_min_obs = int(getattr(opt, "semantic_background_min_observations", 8))
        bg_conf = float(getattr(opt, "semantic_background_confidence", 0.95))
        bg_max_tgt = float(getattr(opt, "semantic_background_max_target_prob", 0.05))
        bg_stable = int(getattr(opt, "semantic_background_stable_windows", 2))
        debug = bool(getattr(opt, "semantic_resolution_debug", False))

        with torch.no_grad():
            unknown = self._object_id == -1
            obs = self._semantic_obs_count
            ev = self._semantic_evidence_sum
            old_prob = self._object_prob
            cand = unknown & (obs > 0)
            cand_idx = cand.nonzero().flatten()

            stats = {"iteration": iteration, "unknown_before": int(unknown.sum()),
                     "observed_unknown": int(cand_idx.numel()),
                     "target_resolved": 0, "plant_resolved": 0, "rubik_1_resolved": 0,
                     "rubik_2_resolved": 0, "background_candidates": 0,
                     "background_resolved": 0, "unknown_after": int(unknown.sum()),
                     "insufficient_observation": 0, "low_confidence": 0, "low_margin": 0,
                     "evidence_window_reset": "OK"}

            if cand_idx.numel() > 0:
                obs_k = obs[cand_idx].float()
                # window probability (checked, not used directly by the fusion below)
                window_prob = ev[cand_idx] / obs_k.unsqueeze(1)
                assert window_prob.shape[1] == 4, "[VGER] window_prob 形状应为 [K,4]"
                assert bool(torch.isfinite(window_prob).all()), "[VGER] window_prob 含 NaN/Inf"
                assert bool((window_prob >= 0).all()), "[VGER] window_prob 含负值"
                assert bool(torch.allclose(window_prob.sum(-1), torch.ones_like(window_prob.sum(-1)), atol=1e-3)), \
                    "[VGER] window_prob 每行和应≈1"

                # prior fusion: new_prob = (prior*old_prob + evidence_sum) / (prior + obs_count)
                new_prob = (prior_strength * old_prob[cand_idx] + ev[cand_idx]) \
                           / (prior_strength + obs_k.unsqueeze(1))
                self._object_prob[cand_idx] = new_prob
                self._semantic_confidence[cand_idx] = new_prob.max(dim=-1).values

                top1_prob, top1_class = new_prob.max(dim=-1)
                top2_prob = new_prob.scatter(-1, top1_class.unsqueeze(-1), 0.0).max(dim=-1).values
                margin = top1_prob - top2_prob

                obs_int = obs[cand_idx]
                stats["insufficient_observation"] = int((obs_int < target_min_obs).sum())
                stats["low_confidence"] = int(((obs_int >= target_min_obs) & (top1_prob < target_conf)).sum())
                stats["low_margin"] = int(((obs_int >= target_min_obs) & (top1_prob >= target_conf)
                                           & (margin < target_margin)).sum())

                # ---- target conversion (current window only) ----
                target_mask = (obs_int >= target_min_obs) & (top1_class >= 1) & (top1_class <= 3) \
                              & (top1_prob >= target_conf) & (margin >= target_margin)
                tgt_idx = cand_idx[target_mask]
                if tgt_idx.numel() > 0:
                    self._object_id[tgt_idx] = top1_class[target_mask]
                    self._semantic_stable_count[tgt_idx] = 0
                    stats["target_resolved"] = int(tgt_idx.numel())
                    stats["plant_resolved"] = int((top1_class[target_mask] == 1).sum())
                    stats["rubik_1_resolved"] = int((top1_class[target_mask] == 2).sum())
                    stats["rubik_2_resolved"] = int((top1_class[target_mask] == 3).sum())

                # ---- background (still-unknown candidates; more conservative) ----
                remain = cand_idx[~target_mask]
                if remain.numel() > 0:
                    new_prob_r = new_prob[~target_mask]
                    obs_r = obs[remain]
                    bg_cond = (obs_r >= bg_min_obs) & (new_prob_r[:, 0] >= bg_conf) \
                              & (new_prob_r[:, 1:].sum(-1) <= bg_max_tgt)
                    stats["background_candidates"] = int(bg_cond.sum())
                    new_stable = torch.where(bg_cond,
                                             self._semantic_stable_count[remain] + 1,
                                             torch.zeros_like(self._semantic_stable_count[remain]))
                    self._semantic_stable_count[remain] = new_stable
                    bg_conv = bg_cond & (new_stable >= bg_stable)
                    bg_idx = remain[bg_conv]
                    if bg_idx.numel() > 0:
                        self._object_id[bg_idx] = 0
                        self._semantic_stable_count[bg_idx] = 0
                        stats["background_resolved"] = int(bg_idx.numel())

            # ---- window cleanup: reset evidence for the next window ----
            self._semantic_evidence_sum.zero_()
            self._semantic_obs_count.zero_()
            stats["unknown_after"] = int((self._object_id == -1).sum())
            self._assert_semantic_alignment("resolve_unknown_semantics")

        if debug:
            self._log_resolution_debug(stats)
        return stats

    def _resolve_all_class(self, opt, iteration=None):
        """VGER+: non-background dynamic semantic revision.

        Every unknown / known-target row with window observations is re-verified each
        refresh window (background object_id == 0 is a TERMINAL state and never enters):
          * long-term object_prob is still fused with the prior_strength formula;
          * keep / reject decisions use the WINDOW probability, so a stale high-
            confidence prior can never permanently lock a label;
          * a strongly-conflicted known target first falls back to unknown (2 stable
            windows) and can later be re-confirmed; it is never directly relabelled
            to a different known target;
          * unknown -> target is single-window, unknown -> background needs 2 stable
            windows; once background, the row is frozen forever;
          * background candidate state is cleaned to neutral (-1/0) but its object_id /
            object_prob / semantic_confidence are never touched;
          * evidence window resets for every row (targets keep re-verifying).
        """
        prior_strength = float(getattr(opt, "semantic_prior_strength", 5.0))
        target_min_obs = int(getattr(opt, "semantic_target_min_observations", 5))
        target_conf = float(getattr(opt, "semantic_target_confidence", 0.75))
        target_margin = float(getattr(opt, "semantic_target_margin", 0.20))
        known_keep_thr = float(getattr(opt, "semantic_known_keep_threshold", 0.55))
        known_reject_thr = float(getattr(opt, "semantic_known_reject_threshold", 0.35))
        cand_conf = float(getattr(opt, "semantic_revision_candidate_confidence", 0.70))
        cand_margin = float(getattr(opt, "semantic_revision_margin", 0.20))
        rev_min_obs = int(getattr(opt, "semantic_revision_min_observations", 6))
        rev_stable = int(getattr(opt, "semantic_revision_stable_windows", 2))
        bg_conf = float(getattr(opt, "semantic_background_candidate_confidence", 0.75))
        bg_max_tgt = float(getattr(opt, "semantic_background_candidate_max_target_prob", 0.25))
        known_keep_min_obs = 5   # spec: 保持当前类别条件 obs_count >= 5
        debug = bool(getattr(opt, "semantic_resolution_debug", False))

        with torch.no_grad():
            oid = self._object_id
            obs = self._semantic_obs_count
            ev = self._semantic_evidence_sum
            # background (object_id == 0) is a TERMINAL semantic state: it never takes
            # part in any revision decision, label update or candidate tracking. Its
            # candidate state is always cleaned back to neutral (object_prob and
            # semantic_confidence stay untouched).
            background = oid == 0
            if bool(background.any()):
                self._semantic_candidate_id[background] = -1
                self._semantic_candidate_stable_count[background] = 0
            cand = (obs > 0) & (~background)
            cand_idx = cand.nonzero().flatten()

            stats = {"iteration": iteration, "unknown_before": int((oid == -1).sum()),
                     "background_count": int(background.sum()),
                     "observed_gaussians": int(cand_idx.numel()),
                     "target_resolved": 0, "plant_resolved": 0, "rubik_1_resolved": 0,
                     "rubik_2_resolved": 0, "background_candidates": 0,
                     "background_resolved": 0, "known_kept": 0,
                     "known_candidate_updated": 0, "known_reverted": 0,
                     "unknown_after": int((oid == -1).sum()),
                     "insufficient_observation": 0, "evidence_window_reset": "OK"}

            if cand_idx.numel() > 0:
                obs_k = obs[cand_idx].float()
                # window probability (decisions use window_prob, not the fused prior)
                window_prob = ev[cand_idx] / obs_k.unsqueeze(1)
                assert window_prob.shape[1] == 4, "[VGER+] window_prob 形状应为 [K,4]"
                assert bool(torch.isfinite(window_prob).all()), "[VGER+] window_prob 含 NaN/Inf"
                assert bool((window_prob >= 0).all()), "[VGER+] window_prob 含负值"
                assert bool(torch.allclose(window_prob.sum(-1), torch.ones_like(window_prob.sum(-1)), atol=1e-3)), \
                    "[VGER+] window_prob 每行和应≈1"

                # long-term object_prob prior fusion (ALL observed rows, known included)
                old_prob = self._object_prob
                new_prob = (prior_strength * old_prob[cand_idx] + ev[cand_idx]) \
                           / (prior_strength + obs_k.unsqueeze(1))
                self._object_prob[cand_idx] = new_prob
                self._semantic_confidence[cand_idx] = new_prob.max(dim=-1).values

                # window statistics
                wp = window_prob
                wp_top1_prob, wp_top1_class = wp.max(dim=-1)
                wp_top2_prob = wp.scatter(-1, wp_top1_class.unsqueeze(-1), 0.0).max(dim=-1).values
                wp_margin = wp_top1_prob - wp_top2_prob
                cur = oid[cand_idx]                                            # [K]
                wp_cur = wp.gather(-1, cur.clamp(0, 3).unsqueeze(-1)).squeeze(-1)  # [K]

                # background is excluded from cand_idx, so only known targets and unknown
                known = (cur == 1) | (cur == 2) | (cur == 3)
                unk = cur == -1

                # ================= known rows: keep / conflict / revert =================
                known_idx = cand_idx[known]
                if known_idx.numel() > 0:
                    obs_kk = obs_k[known]
                    wp_cur_k = wp_cur[known]
                    top1_k = wp_top1_class[known]
                    top1_pk = wp_top1_prob[known]
                    margin_k = wp_margin[known]
                    cur_k = cur[known]

                    keep = (obs_kk >= known_keep_min_obs) & (wp_cur_k >= known_keep_thr)
                    conflict = (obs_kk >= rev_min_obs) & (wp_cur_k <= known_reject_thr) \
                               & (top1_k != cur_k) & (top1_pk >= cand_conf) \
                               & (margin_k >= cand_margin)

                    old_cid = self._semantic_candidate_id[known_idx]
                    old_cst = self._semantic_candidate_stable_count[known_idx]
                    same = old_cid == top1_k
                    # conflict -> track top1 candidate; keep -> reset candidate to none
                    new_cid = torch.where(conflict, top1_k, old_cid)
                    new_cid = torch.where(keep, torch.full_like(old_cid, -1), new_cid)
                    new_cst = torch.where(conflict,
                                          torch.where(same, old_cst + 1, torch.ones_like(old_cst)),
                                          torch.where(keep, torch.zeros_like(old_cst), old_cst))
                    self._semantic_candidate_id[known_idx] = new_cid
                    self._semantic_candidate_stable_count[known_idx] = new_cst

                    stats["known_kept"] = int(keep.sum())
                    stats["known_candidate_updated"] = int(conflict.sum())

                    revert = conflict & (new_cst >= rev_stable)
                    rev_idx = known_idx[revert]
                    if rev_idx.numel() > 0:
                        wp_rev = wp[known][revert]
                        self._object_id[rev_idx] = -1
                        self._object_prob[rev_idx] = wp_rev
                        self._semantic_confidence[rev_idx] = wp_rev.max(dim=-1).values
                        self._semantic_candidate_id[rev_idx] = -1
                        self._semantic_candidate_stable_count[rev_idx] = 0
                    stats["known_reverted"] = int(rev_idx.numel())

                # ================= unknown rows: target / background =================
                unk_idx = cand_idx[unk]
                if unk_idx.numel() > 0:
                    obs_ku = obs_k[unk]
                    top1_u = wp_top1_class[unk]
                    top1_pu = wp_top1_prob[unk]
                    margin_u = wp_margin[unk]
                    wp_u = wp[unk]

                    # unknown -> target (single window)
                    target = (obs_ku >= target_min_obs) & (top1_u >= 1) & (top1_u <= 3) \
                             & (top1_pu >= target_conf) & (margin_u >= target_margin)
                    tgt_idx = unk_idx[target]
                    if tgt_idx.numel() > 0:
                        self._object_id[tgt_idx] = top1_u[target]
                        self._semantic_candidate_id[tgt_idx] = -1
                        self._semantic_candidate_stable_count[tgt_idx] = 0
                        stats["target_resolved"] = int(tgt_idx.numel())
                        stats["plant_resolved"] = int((top1_u[target] == 1).sum())
                        stats["rubik_1_resolved"] = int((top1_u[target] == 2).sum())
                        stats["rubik_2_resolved"] = int((top1_u[target] == 3).sum())

                    # unknown -> background (needs rev_stable consecutive windows)
                    bg_cond = (obs_ku >= rev_min_obs) & (wp_u[:, 0] >= bg_conf) \
                              & (wp_u[:, 1:].sum(-1) <= bg_max_tgt) & (margin_u >= cand_margin)
                    stats["background_candidates"] = int(bg_cond.sum())
                    old_cid_u = self._semantic_candidate_id[unk_idx]
                    old_cst_u = self._semantic_candidate_stable_count[unk_idx]
                    same_bg = old_cid_u == 0
                    new_cid_u = torch.where(bg_cond, torch.zeros_like(old_cid_u),
                                            torch.full_like(old_cid_u, -1))
                    new_cst_u = torch.where(bg_cond,
                                            torch.where(same_bg, old_cst_u + 1, torch.ones_like(old_cst_u)),
                                            torch.zeros_like(old_cst_u))
                    self._semantic_candidate_id[unk_idx] = new_cid_u
                    self._semantic_candidate_stable_count[unk_idx] = new_cst_u
                    bg_conv = bg_cond & (new_cst_u >= rev_stable)
                    bg_idx = unk_idx[bg_conv]
                    if bg_idx.numel() > 0:
                        self._object_id[bg_idx] = 0
                        self._semantic_candidate_id[bg_idx] = -1
                        self._semantic_candidate_stable_count[bg_idx] = 0
                        stats["background_resolved"] = int(bg_idx.numel())

                stats["insufficient_observation"] = int((obs_k < target_min_obs).sum())

            # ---- window cleanup: reset evidence for the next window ----
            self._semantic_evidence_sum.zero_()
            self._semantic_obs_count.zero_()
            stats["unknown_after"] = int((self._object_id == -1).sum())
            self._assert_semantic_alignment("resolve_unknown_semantics")

        if debug:
            self._log_resolution_debug(stats)
        return stats

    def _log_resolution_debug(self, stats):
        prob = self._object_prob.detach().cpu().numpy()
        conf = self._semantic_confidence.detach().cpu().numpy()
        observed = stats.get("observed_gaussians", stats.get("observed_unknown", 0))
        print("[VGER] iteration={} unknown_before={} observed={} target_resolved={} "
              "plant_resolved={} rubik_1_resolved={} rubik_2_resolved={} background_candidates={} "
              "background_resolved={} known_kept={} known_candidate_updated={} known_reverted={} "
              "unknown_after={} insufficient_observation={} low_confidence={} low_margin={} "
              "evidence_window_reset=OK".format(
                  stats.get("iteration"), stats.get("unknown_before", 0), observed,
                  stats.get("target_resolved", 0), stats.get("plant_resolved", 0),
                  stats.get("rubik_1_resolved", 0), stats.get("rubik_2_resolved", 0),
                  stats.get("background_candidates", 0), stats.get("background_resolved", 0),
                  stats.get("known_kept", 0), stats.get("known_candidate_updated", 0),
                  stats.get("known_reverted", 0), stats.get("unknown_after", 0),
                  stats.get("insufficient_observation", 0), stats.get("low_confidence", 0),
                  stats.get("low_margin", 0)))
        print("[VGER]   object_prob mean={} | confidence min/mean/max={:.4f}/{:.4f}/{:.4f}".format(
            np.round(prob.mean(0), 4).tolist(), conf.min(), conf.mean(), conf.max()))

    # ------------------------------------------------------------------ #
    # TADC: target-aware density control (semantic grow gating)
    # ------------------------------------------------------------------ #
    def get_semantic_grow_mask(self, unknown_target_threshold):
        """Unified semantic growth mask [N] bool.

        known_target (id 1/2/3) -> True, unknown (id -1) -> True iff p_target >=
        threshold, background (id 0) -> always False. When semantics are not attached
        returns all-True so the original Fast-PGSR behaviour is preserved. No mutation.
        """
        assert 0.0 <= float(unknown_target_threshold) <= 1.0, \
            "[TADC] semantic_unknown_grow_threshold 必须在 [0, 1]"
        if not self._semantics_attached:
            return torch.ones(self._xyz.shape[0], dtype=torch.bool, device=self._xyz.device)
        with torch.no_grad():
            prob = self._object_prob
            assert bool(torch.isfinite(prob).all()), "[TADC] object_prob 含 NaN/Inf"
            assert bool((prob >= 0).all()), "[TADC] object_prob 含负值"
            assert bool(torch.allclose(prob.sum(-1), torch.ones_like(prob.sum(-1)), atol=1e-3)), \
                "[TADC] object_prob 每行和应≈1"
            oid = self._object_id
            known_target = (oid == 1) | (oid == 2) | (oid == 3)
            unknown = (oid == -1)
            p_target = prob[:, 1] + prob[:, 2] + prob[:, 3]
            grow = known_target | (unknown & (p_target >= float(unknown_target_threshold)))
            assert grow.shape[0] == self._xyz.shape[0], "[TADC] grow mask 长度与 Gaussian 不一致"
            return grow

    def _apply_semantic_grow_gate(self, mask, args, op_type):
        """AND the original densify mask with the semantic grow mask (TADC)."""
        if not getattr(args, "enable_semantic_density_control", False) or args is None:
            return mask
        grow = self.get_semantic_grow_mask(getattr(args, "semantic_unknown_grow_threshold", 0.4))
        final = mask & grow
        self._log_tadc(op_type, mask, final, args)
        return final

    def _log_tadc(self, op_type, orig_mask, final_mask, args):
        if not self._semantics_attached:
            return
        if not getattr(args, "semantic_density_debug", False):
            return
        it = getattr(self, "_iteration_hint", None)
        interval = getattr(args, "semantic_density_debug_interval", 100)
        if it is not None and interval and it % interval != 0:
            return
        with torch.no_grad():
            oid = self._object_id

            def per(m):
                return {"unknown": int((m & (oid == -1)).sum()),
                        "background": int((m & (oid == 0)).sum()),
                        "plant": int((m & (oid == 1)).sum()),
                        "rubik_1": int((m & (oid == 2)).sum()),
                        "rubik_2": int((m & (oid == 3)).sum())}
            o, f = per(orig_mask), per(final_mask)
            orig_n, final_n = int(orig_mask.sum()), int(final_mask.sum())
            known_target = o["plant"] + o["rubik_1"] + o["rubik_2"]
            suppression = (orig_n - final_n) / max(orig_n, 1)
            print(f"[TADC][{op_type}] iteration={it} gaussian_total={self._xyz.shape[0]} "
                  f"original_candidates={orig_n} known_target_candidates={known_target} "
                  f"unknown_candidates={o['unknown']} unknown_allowed={f['unknown']} "
                  f"background_candidates_blocked={o['background']} final_candidates={final_n} "
                  f"suppression_ratio={suppression:.3f}")
            print(f"[TADC][{op_type}]   类别 original/final: "
                  f"unknown={o['unknown']}/{f['unknown']} background={o['background']}/{f['background']} "
                  f"plant={o['plant']}/{f['plant']} rubik_1={o['rubik_1']}/{f['rubik_1']} "
                  f"rubik_2={o['rubik_2']}/{f['rubik_2']}")

    def _verify_semantics_not_in_optimizer(self):
        """Acceptance: the non-learnable semantic tensors must not appear in any
        optimizer param group (checked after optimizer creation)."""
        for opt in [self.optimizer, self.shoptimizer]:
            for name in ["_object_id", "_object_prob", "_semantic_confidence", "_source_point3D_id",
                         "_semantic_evidence_sum", "_semantic_obs_count", "_semantic_stable_count",
                         "_semantic_candidate_id", "_semantic_candidate_stable_count"]:
                t = getattr(self, name)
                if t.numel() == 0:
                    continue
                for group in opt.param_groups:
                    assert not any(t is p for p in group["params"]), \
                        f"[Semantic Init] {name} 不应出现在 optimizer 参数组"
        print("[Semantic Init] 验收通过: 语义 Tensor 不存在于任何 optimizer 参数组")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.abs_split_radii2D_threshold = training_args.abs_split_radii2D_threshold
        self.max_abs_split_points = training_args.max_abs_split_points
        self.max_all_points = training_args.max_all_points
        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._knn_f], 'lr': 0.01, "name": "knn_f"},
            {'params': [self._features_dc], 'lr': training_args.lowfeature_lr, "name": "f_dc"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        sh_l = [{'params': [self._features_rest], 'lr': training_args.highfeature_lr / 20.0, "name": "f_rest"}]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.shoptimizer = torch.optim.Adam(sh_l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

        # Acceptance: semantic tensors must not join any optimizer param group.
        if self._object_id.numel() > 0:
            self._verify_semantics_not_in_optimizer()
    
    def clip_grad(self, norm=1.0):
        optimizers = [self.optimizer]
        if self.shoptimizer: optimizers.append(self.shoptimizer)
        for opt in optimizers:
            for group in opt.param_groups:
                torch.nn.utils.clip_grad_norm_(group["params"][0], norm)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path, mask=None):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

        # SSP: persist semantic state next to the PLY (keeps point_cloud.ply format intact).
        if self._semantics_attached:
            self._save_semantic_sidecar(os.path.join(os.path.dirname(path), "semantic_state.npz"))

    def _save_semantic_sidecar(self, path):
        xyz = np.ascontiguousarray(self._xyz.detach().cpu().numpy(), dtype=np.float32)
        obj = self._object_id.detach().cpu().numpy()
        prob = self._object_prob.detach().cpu().numpy()
        conf = self._semantic_confidence.detach().cpu().numpy()
        src = self._source_point3D_id.detach().cpu().numpy()
        ev = self._semantic_evidence_sum.detach().cpu().numpy()
        oc = self._semantic_obs_count.detach().cpu().numpy()
        sc = self._semantic_stable_count.detach().cpu().numpy()
        cid = self._semantic_candidate_id.detach().cpu().numpy()
        cst = self._semantic_candidate_stable_count.detach().cpu().numpy()
        np.savez(path,
                 object_id=obj,
                 object_prob=prob,
                 semantic_confidence=conf,
                 source_point3D_id=src,
                 semantic_evidence_sum=ev,
                 semantic_obs_count=oc,
                 semantic_stable_count=sc,
                 semantic_candidate_id=cid,
                 semantic_candidate_stable_count=cst,
                 semantics_attached=self._semantics_attached,
                 semantic_state_version=SEMANTIC_STATE_VERSION,
                 gaussian_count=int(self._xyz.shape[0]),
                 class_names=np.array(["background", "plant", "rubik_1", "rubik_2"]),
                 xyz_order_checksum=_xyz_checksum(xyz),
                 semantic_order_checksum=_semantic_checksum(obj, prob, conf, src, ev, oc, sc, cid, cst))
        print(f"[SSP] saved semantic state: path={path} gaussians={self._xyz.shape[0]} version={SEMANTIC_STATE_VERSION}")

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path, load_semantics=False):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

        # SSP: semantic state persistence.
        #   - sidecar present  -> auto-detect and restore (natural for render.py --load_iteration)
        #   - sidecar absent + load_semantics=True -> explicit request, hard error
        #   - sidecar absent + load_semantics=False -> original non-semantic behaviour
        sidecar = os.path.join(os.path.dirname(path), "semantic_state.npz")
        if os.path.exists(sidecar):
            self._load_semantic_sidecar(sidecar, xyz)
        elif load_semantics:
            raise RuntimeError(f"[SSP] --load_semantics 已启用但 semantic_state.npz 缺失: {sidecar}")
        else:
            self._reset_semantic_tensors()

    def _load_semantic_sidecar(self, sidecar, xyz_np):
        data = np.load(sidecar)
        version = int(data["semantic_state_version"])
        if version not in (SEMANTIC_STATE_VERSION, 1):
            raise RuntimeError(f"[SSP] 未知语义状态版本 {version}，当前支持 {SEMANTIC_STATE_VERSION}")
        if not bool(data["semantics_attached"]):
            raise RuntimeError("[SSP] semantic_state.npz 标记 semantics_attached=False 但语义模式已启用")
        n = xyz_np.shape[0]
        if int(data["gaussian_count"]) != n:
            raise RuntimeError(f"[SSP] gaussian_count {int(data['gaussian_count'])} != PLY 点数 {n}")
        if str(data["xyz_order_checksum"]) != _xyz_checksum(xyz_np):
            raise RuntimeError("[SSP] xyz_order_checksum 校验失败：PLY 顺序与语义状态不一致，"
                               "请检查 PLY 是否被重新排序或来自不同模型")

        obj = np.asarray(data["object_id"], dtype=np.int64)
        prob = np.asarray(data["object_prob"], dtype=np.float32)
        conf = np.asarray(data["semantic_confidence"], dtype=np.float32)
        src = np.asarray(data["source_point3D_id"], dtype=np.int64)
        ev = np.asarray(data["semantic_evidence_sum"], dtype=np.float32)
        oc = np.asarray(data["semantic_obs_count"], dtype=np.int64)
        sc = np.asarray(data["semantic_stable_count"], dtype=np.int64)
        # v1 sidecars have no candidate arrays -> neutral defaults.
        cid = np.asarray(data["semantic_candidate_id"], dtype=np.int64) if "semantic_candidate_id" in data else None
        cst = np.asarray(data["semantic_candidate_stable_count"], dtype=np.int64) if "semantic_candidate_stable_count" in data else None
        if not (obj.shape == (n,) and prob.shape == (n, 4) and conf.shape == (n,)
                and src.shape == (n,) and ev.shape == (n, 4) and oc.shape == (n,) and sc.shape == (n,)):
            raise RuntimeError(f"[SSP] 语义张量形状错误: obj={obj.shape} prob={prob.shape} "
                               f"conf={conf.shape} src={src.shape} ev={ev.shape} oc={oc.shape} sc={sc.shape}")
        if cid is not None and (cid.shape != (n,) or cst.shape != (n,)):
            raise RuntimeError(f"[SSP] 候选张量形状错误: cid={cid.shape} cst={cst.shape}")
        if not np.isin(obj, [-1, 0, 1, 2, 3]).all():
            raise RuntimeError("[SSP] object_id 含非法值（仅允许 -1/0/1/2/3）")
        if not np.isfinite(prob).all() or (prob < 0).any():
            raise RuntimeError("[SSP] object_prob 含 NaN/Inf 或负值")
        if not np.allclose(prob.sum(-1), 1.0, atol=1e-4):
            raise RuntimeError("[SSP] object_prob 每行概率和不为 1")
        if "semantic_order_checksum" in data:
            if str(data["semantic_order_checksum"]) != _semantic_checksum(obj, prob, conf, src, ev, oc, sc, cid, cst):
                raise RuntimeError("[SSP] semantic_order_checksum 校验失败：semantic_state.npz 内部行序被改动")

        device = self._xyz.device
        self._object_id = torch.from_numpy(obj).to(device=device, dtype=torch.int64)
        self._object_prob = torch.from_numpy(prob).to(device=device, dtype=torch.float32)
        self._semantic_confidence = torch.from_numpy(conf).to(device=device, dtype=torch.float32)
        self._source_point3D_id = torch.from_numpy(src).to(device=device, dtype=torch.int64)
        self._semantic_evidence_sum = torch.from_numpy(ev).to(device=device, dtype=torch.float32)
        self._semantic_obs_count = torch.from_numpy(oc).to(device=device, dtype=torch.int64)
        self._semantic_stable_count = torch.from_numpy(sc).to(device=device, dtype=torch.int64)
        if cid is not None:
            self._semantic_candidate_id = torch.from_numpy(cid).to(device=device, dtype=torch.int64)
            self._semantic_candidate_stable_count = torch.from_numpy(cst).to(device=device, dtype=torch.int64)
        else:
            self._semantic_candidate_id = torch.full((n,), -1, dtype=torch.int64, device=device)
            self._semantic_candidate_stable_count = torch.zeros((n,), dtype=torch.int64, device=device)
        self._semantics_attached = True
        self._assert_semantic_alignment("load_ply_semantics")
        print(f"[SSP] restored semantic state: path={sidecar} gaussians={n} alignment=OK xyz_checksum=OK")

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        optimizers = [self.optimizer]
        if self.shoptimizer: optimizers.append(self.shoptimizer)

        for opt in optimizers:
            for group in opt.param_groups:
                stored_state = opt.state.get(group['params'][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                    del opt.state[group['params'][0]]
                    group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                    opt.state[group['params'][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def optimizer_step(self, iteration, app_model):
        ''' An optimization schdeuler. The goal is similar to the sparse Adam of taming 3dgs.'''
        if iteration <= 15000:
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none = True)
            app_model.optimizer.step()
            app_model.optimizer.zero_grad(set_to_none = True)
            if iteration % 16 == 0:
                self.shoptimizer.step()
                self.shoptimizer.zero_grad(set_to_none = True)
        elif iteration <= 20000:
            if iteration % 24 ==0:
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none = True)
                self.shoptimizer.step()
                self.shoptimizer.zero_grad(set_to_none = True)
                app_model.optimizer.step()
                app_model.optimizer.zero_grad(set_to_none = True)
        else:
            if iteration % 64 ==0:
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none = True)
                self.shoptimizer.step()
                self.shoptimizer.zero_grad(set_to_none = True)
                app_model.optimizer.step()
                app_model.optimizer.zero_grad(set_to_none = True)

    def prune_points(self, mask):
        valid_points_mask = ~mask
        # SLS: snapshot pre-prune state for logging.
        if self._semantics_attached:
            sls_before_n = self._object_id.shape[0]
            sls_before_counts = self._semantic_class_counts()

        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._knn_f = optimizable_tensors["knn_f"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.xyz_gradient_accum_abs = self.xyz_gradient_accum_abs[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.denom_abs = self.denom_abs[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.max_weight = self.max_weight[valid_points_mask]

        # SLS: prune every semantic tensor with the exact same keep mask, assert, log.
        if self._semantics_attached:
            self._prune_semantics(valid_points_mask)
            self._assert_semantic_alignment("prune")
            self._log_semantic_op("prune", sls_before_n, self._object_id.shape[0],
                                  sls_before_counts, removed_count=sls_before_n - self._object_id.shape[0])

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        optimizers = [self.optimizer]
        if self.shoptimizer: optimizers.append(self.shoptimizer)

        for opt in optimizers:
            for group in opt.param_groups:
                assert len(group["params"]) == 1
                extension_tensor = tensors_dict[group["name"]]
                stored_state = opt.state.get(group['params'][0], None)
                if stored_state is not None:

                    stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                    stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                    del opt.state[group['params'][0]]
                    group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                    opt.state[group['params'][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_knn_f, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_parent_ids=None, op_type="clone"):
        # SLS: snapshot pre-append state for logging.
        if self._semantics_attached:
            sls_before_n = self._object_id.shape[0]
            sls_before_counts = self._semantic_class_counts()

        d = {"xyz": new_xyz,
        "knn_f": new_knn_f,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}
        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._knn_f = optimizable_tensors["knn_f"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        # SLS: append semantic rows (inheritance for clone/split children), assert, log.
        if self._semantics_attached:
            self._append_semantics_from_parents(new_parent_ids)
            self._assert_semantic_alignment(op_type)
            parent_count = None
            if new_parent_ids is not None and new_parent_ids.numel() > 0:
                parent_count = int(new_parent_ids.unique().numel())  # clone: k ; split: k (each repeated N)
            self._log_semantic_op(op_type, sls_before_n, self._object_id.shape[0],
                                  sls_before_counts, parent_count=parent_count)

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.max_weight = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, grads_abs, grad_abs_threshold, scene_extent, max_radii2D, N=2, args=None):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        padded_grads_abs = torch.zeros((n_init_points), device="cuda")
        padded_grads_abs[:grads_abs.shape[0]] = grads_abs.squeeze()
        padded_max_radii2D = torch.zeros((n_init_points), device="cuda")
        padded_max_radii2D[:max_radii2D.shape[0]] = max_radii2D.squeeze()

        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)
        if selected_pts_mask.sum() + n_init_points > self.max_all_points:
            limited_num = self.max_all_points - n_init_points
            padded_grad[~selected_pts_mask] = 0
            ratio = limited_num / float(n_init_points)
            threshold = torch.quantile(padded_grad, (1.0-ratio))
            selected_pts_mask = torch.where(padded_grad > threshold, True, False)
            # print(f"split {selected_pts_mask.sum()}, raddi2D {padded_max_radii2D.max()} ,{padded_max_radii2D.median()}")
        else:
            padded_grads_abs[selected_pts_mask] = 0
            mask = (torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent) & (padded_max_radii2D > self.abs_split_radii2D_threshold)
            padded_grads_abs[~mask] = 0
            selected_pts_mask_abs = torch.where(padded_grads_abs >= grad_abs_threshold, True, False)
            limited_num = min(self.max_all_points - n_init_points - selected_pts_mask.sum(), self.max_abs_split_points)
            if selected_pts_mask_abs.sum() > limited_num:
                ratio = limited_num / float(n_init_points)
                threshold = torch.quantile(padded_grads_abs, (1.0-ratio))
                selected_pts_mask_abs = torch.where(padded_grads_abs > threshold, True, False)
            selected_pts_mask = torch.logical_or(selected_pts_mask, selected_pts_mask_abs)
            # print(f"split {selected_pts_mask.sum()}, abs {selected_pts_mask_abs.sum()}, raddi2D {padded_max_radii2D.max()} ,{padded_max_radii2D.median()}")

        # TADC: gate the final original split mask (no replacement of Fast-PGSR criteria)
        selected_pts_mask = self._apply_semantic_grow_gate(selected_pts_mask, args, "split")

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_knn_f = self._knn_f[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_knn_f, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation,
                                   new_parent_ids=selected_pts_mask.nonzero().flatten().repeat(N), op_type="split")

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent, args=None):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        if selected_pts_mask.sum() + n_init_points > self.max_all_points:
            limited_num = self.max_all_points - n_init_points
            grads_tmp = grads.squeeze().clone()
            grads_tmp[~selected_pts_mask] = 0
            ratio = limited_num / float(n_init_points)
            threshold = torch.quantile(grads_tmp, (1.0-ratio))
            selected_pts_mask = torch.where(grads_tmp > threshold, True, False)

        # TADC: gate the final original clone mask (no replacement of Fast-PGSR criteria)
        selected_pts_mask = self._apply_semantic_grow_gate(selected_pts_mask, args, "clone")

        if selected_pts_mask.sum() > 0:
            # print(f"clone {selected_pts_mask.sum()}")
            new_xyz = self._xyz[selected_pts_mask]

            stds = self.get_scaling[selected_pts_mask]
            means =torch.zeros((stds.size(0), 3),device="cuda")
            samples = torch.normal(mean=means, std=stds)
            rots = build_rotation(self._rotation[selected_pts_mask])
            new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask]
            
            new_features_dc = self._features_dc[selected_pts_mask]
            new_features_rest = self._features_rest[selected_pts_mask]
            new_opacities = self._opacity[selected_pts_mask]
            new_scaling = self._scaling[selected_pts_mask]
            new_rotation = self._rotation[selected_pts_mask]
            new_knn_f = self._knn_f[selected_pts_mask]

            self.densification_postfix(new_xyz, new_knn_f, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation,
                                       new_parent_ids=selected_pts_mask.nonzero().flatten(), op_type="clone")

    def densify_and_prune(self, max_grad, abs_max_grad, min_opacity, extent, max_screen_size, args=None):
        grads = self.xyz_gradient_accum / self.denom
        grads_abs = self.xyz_gradient_accum_abs / self.denom_abs
        grads[grads.isnan()] = 0.0
        grads_abs[grads_abs.isnan()] = 0.0
        max_radii2D = self.max_radii2D.clone()

        self.densify_and_clone(grads, max_grad, extent, args)
        self.densify_and_split(grads, max_grad, grads_abs, abs_max_grad, extent, max_radii2D, args=args)

        prune_mask = (self.get_opacity < min_opacity).squeeze()

        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        # print(f"all points {self._xyz.shape[0]}")
        torch.cuda.empty_cache()

    def densify_and_split_fastgs(self, metric_mask, filter, N=2, args=None):
        n_init_points = self.get_xyz.shape[0]

        selected_pts_mask = torch.zeros((n_init_points), dtype=bool, device="cuda")
        mask = torch.logical_and(metric_mask, filter)
        selected_pts_mask[:mask.shape[0]] = mask

        # TADC: gate the final fastgs split mask
        selected_pts_mask = self._apply_semantic_grow_gate(selected_pts_mask, args, "split")

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_knn_f = self._knn_f[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_knn_f, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation,
                                   new_parent_ids=selected_pts_mask.nonzero().flatten().repeat(N), op_type="split")

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone_fastgs(self, metric_mask, filter, args=None):
        selected_pts_mask = torch.logical_and(metric_mask, filter)

        # TADC: gate the final fastgs clone mask
        selected_pts_mask = self._apply_semantic_grow_gate(selected_pts_mask, args, "clone")

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_knn_f = self._knn_f[selected_pts_mask]

        self.densification_postfix(new_xyz, new_knn_f, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation,
                                   new_parent_ids=selected_pts_mask.nonzero().flatten(), op_type="clone")

    def densify_and_prune_fastgs(self, max_screen_size, min_opacity, extent, radii, args, importance_score = None, pruning_score = None):
        
        ''' 
            Densification and Pruning based on FastGS criteria:
            1.  The gaussians candidate for densification are selected based on the gradient of their position first.
            2.  Then, based on their average metric score (computed over multiple sampled views), they are either densified (cloned) or split.
                This is our main contribution compared to the vanilla 3DGS.
            3.  Finally, gaussians with low opacity or very large size are pruned.
        '''
        grad_vars = self.xyz_gradient_accum / self.denom
        grad_vars[grad_vars.isnan()] = 0.0

        grads_abs = self.xyz_gradient_accum_abs / self.denom
        grads_abs[grads_abs.isnan()] = 0.0

        grad_qualifiers = torch.where(torch.norm(grad_vars, dim=-1) >= args.grad_thresh, True, False)
        grad_qualifiers_abs = torch.where(torch.norm(grads_abs, dim=-1) >= args.grad_abs_thresh, True, False)
        clone_qualifiers = torch.max(self.get_scaling, dim=1).values <= args.dense*extent
        split_qualifiers = torch.max(self.get_scaling, dim=1).values > args.dense*extent

        all_clones = torch.logical_and(clone_qualifiers, grad_qualifiers)
        all_splits = torch.logical_and(split_qualifiers, grad_qualifiers_abs)

        # This is our multi-view consisent metric for densification
        # We use this metric to further filter the candidates for densification, which is similar to taming 3dgs.
        metric_mask = importance_score > 5

        self.densify_and_clone_fastgs(metric_mask, all_clones, args=args)
        self.densify_and_split_fastgs(metric_mask, all_splits, args=args)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)

        scores = 1 - pruning_score 
        to_remove = torch.sum(prune_mask)
        remove_budget = int(0.5 * to_remove)

        # The budget is not necessary for our method.
        if remove_budget:
            n_init_points = self.get_xyz.shape[0]
            padded_importance = torch.zeros((n_init_points), dtype=torch.float32)
            padded_importance[:scores.shape[0]] = 1 / (1e-6 + scores.squeeze())
            selected_pts_mask = torch.zeros_like(padded_importance, dtype=bool, device="cuda")
            sampled_indices = torch.multinomial(padded_importance, remove_budget, replacement=False)
            selected_pts_mask[sampled_indices] = True
            final_prune = torch.logical_and(prune_mask, selected_pts_mask)
            self.prune_points(final_prune)
        
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.8))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

        torch.cuda.empty_cache()

    def final_prune_fastgs(self, min_opacity, pruning_score = None):
        """Final-stage pruning: remove Gaussians based on opacity and multi-view consistency.
        In the final stage we remove Gaussians that have low opacity or that are flagged by
        our multi-view reconstruction consistency metric (provided as `pruning_score`)."""
        prune_mask = (self.get_opacity < min_opacity).squeeze() 
        scores_mask = pruning_score > 0.9
        final_prune = torch.logical_or(prune_mask, scores_mask)
        self.prune_points(final_prune)

    def add_densification_stats(self, viewspace_point_tensor, viewspace_point_tensor_abs, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.xyz_gradient_accum_abs[update_filter] += torch.norm(viewspace_point_tensor_abs.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
        self.denom_abs[update_filter] += 1

    def get_points_depth_in_depth_map(self, fov_camera, depth, points_in_camera_space, scale=1):
        st = max(int(scale/2)-1,0)
        depth_view = depth[None,:,st::scale,st::scale]
        W, H = int(fov_camera.image_width/scale), int(fov_camera.image_height/scale)
        depth_view = depth_view[:H, :W]
        pts_projections = torch.stack(
                        [points_in_camera_space[:,0] * fov_camera.Fx / points_in_camera_space[:,2] + fov_camera.Cx,
                         points_in_camera_space[:,1] * fov_camera.Fy / points_in_camera_space[:,2] + fov_camera.Cy], -1).float()/scale
        mask = (pts_projections[:, 0] > 0) & (pts_projections[:, 0] < W) &\
               (pts_projections[:, 1] > 0) & (pts_projections[:, 1] < H) & (points_in_camera_space[:,2] > 0.1)

        pts_projections[..., 0] /= ((W - 1) / 2)
        pts_projections[..., 1] /= ((H - 1) / 2)
        pts_projections -= 1
        pts_projections = pts_projections.view(1, -1, 1, 2)
        map_z = torch.nn.functional.grid_sample(input=depth_view,
                                                grid=pts_projections,
                                                mode='bilinear',
                                                padding_mode='border',
                                                align_corners=True
                                                )[0, :, :, 0]
        return map_z, mask
    
    def get_points_from_depth(self, fov_camera, depth, scale=1):
        st = int(max(int(scale/2)-1,0))
        depth_view = depth.squeeze()[st::scale,st::scale]
        rays_d = fov_camera.get_rays(scale=scale)
        depth_view = depth_view[:rays_d.shape[0], :rays_d.shape[1]]
        pts = (rays_d * depth_view[..., None]).reshape(-1,3)
        R = fov_camera.rotation_tensor
        T = fov_camera.translation_tensor
        pts = (pts-T)@R.transpose(-1,-2)
        return pts
    