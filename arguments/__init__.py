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

from argparse import ArgumentParser, Namespace
import sys
import os
import glob

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None
            if isinstance(value, (list, tuple)):
                # multi-value options (e.g. multiscale ratios/weights) -> nargs="+"
                item_t = type(value[0]) if len(value) else float
                kw = {"default": list(value), "type": item_t, "nargs": "+"}
                if shorthand:
                    group.add_argument("--" + key, "-" + key[0:1], **kw)
                else:
                    group.add_argument("--" + key, **kw)
                continue
            if t == bool:
                if not shorthand and value is True:
                    # default-ON boolean: --<key> stays an (explicit) enable flag, while
                    # --disable_<key> turns it OFF. Both share dest=key; argparse keeps the
                    # last default (True) so the feature is ON unless --disable_<key> is given.
                    group.add_argument("--" + key, default=True, action="store_true", dest=key)
                    group.add_argument("--disable_" + key, default=True, action="store_false", dest=key)
                elif shorthand:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, action="store_true")
            elif shorthand:
                group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.semantic_init_path = ""
        self.export_semantic_binding_debug = False
        self.sls_debug = False
        self.load_semantics = False
        self.semantic_mask_dir = ""
        self.export_mask_alignment_debug = False
        self.semantic_pipeline = False
        self.data_device = "cuda"
        self.eval = False
        self.preload_img = True
        self.ncc_scale = 1.0
        self.multi_view_num = 8
        self.multi_view_max_angle = 30
        self.multi_view_min_dis = 0.01
        self.multi_view_max_dis = 2.5
        self.semantic_init_path = ""

        self.semantic_mask_dir = ""
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):



        g = super().extract(args)



        # 兼容旧版本 cfg_args，补充新版参数

        for key, value in vars(self).items():

            if not hasattr(g, key):

                setattr(g, key, value)



        g.source_path = os.path.abspath(g.source_path)



        if g.semantic_init_path:

            g.semantic_init_path = os.path.abspath(g.semantic_init_path)



        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.separate_sh = True
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.001
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        self.scale_loss_weight = 100.0
        
        self.wo_image_weight = False
        self.single_view_weight = 0.015
        self.single_view_weight_from_iter = 7000

        self.use_virtul_cam = False
        self.virtul_cam_prob = 0.5
        self.use_multi_view_trim = True
        self.multi_view_ncc_weight = 0.15
        self.multi_view_geo_weight = 0.03
        self.multi_view_weight_from_iter = 7000
        self.multi_view_weight_until_iter = 10000
        self.multi_view_patch_size = 3
        self.multi_view_sample_num = 102400
        self.multi_view_pixel_noise_th = 1.0
        self.wo_use_geo_occ_aware = False

        self.opacity_cull_threshold = 0.005
        self.densify_abs_grad_threshold = 0.0008
        self.abs_split_radii2D_threshold = 20
        self.max_abs_split_points = 50_000
        self.max_all_points = 6000_000
        self.exposure_compensation = False
        self.random_background = False

        # fastgs parameters
        self.loss_thresh = 0.1
        self.grad_abs_thresh = 0.0012  
        self.highfeature_lr = 0.005
        self.lowfeature_lr = 0.0025
        self.grad_thresh = 0.0002
        self.dense = 0.001
        self.mult = 0.5      # multiplier for the compact box to control the tile number of each splat

        # VEA: visibility-filtered semantic evidence accumulation
        self.enable_semantic_evidence = False
        self.semantic_warmup_iterations = 1000
        self.semantic_patch_size = 3
        self.semantic_depth_abs_tolerance = 0.1
        self.semantic_depth_rel_tolerance = 0.02
        self.semantic_min_render_alpha = 0.5
        self.semantic_evidence_debug = False
        self.semantic_evidence_debug_interval = 100
        self.semantic_evidence_debug_export = False

        # VEA+: dynamic projection-scale multi-scale center-weighted Mask voting.
        # ON by default; pass --no-enable_dynamic_multiscale_semantic_patch-style
        # store-false if the original fixed 5x5 equal-weight kernel is required
        # (the code fallback when the key is absent is still False).
        self.enable_dynamic_multiscale_semantic_patch = False
        self.semantic_multiscale_ratios = (0.333333, 0.666667, 1.0)
        self.semantic_multiscale_weights = (0.55, 0.30, 0.15)
        self.semantic_multiscale_sigma_ratio = 0.25
        # 0 = no artificial cap; a positive value is only a safety upper bound on the
        # dynamic large_size (never a fixed scale rule).
        self.semantic_dynamic_patch_max_size = 0

        # VGER: visibility-guided evidence resolution (unknown -> target/background)
        self.semantic_refresh_interval = 500
        self.semantic_prior_strength = 5.0
        self.semantic_target_min_observations = 5
        self.semantic_target_confidence = 0.75
        self.semantic_target_margin = 0.20
        self.semantic_background_min_observations = 8
        self.semantic_background_confidence = 0.95
        self.semantic_background_max_target_prob = 0.05
        self.semantic_background_stable_windows = 2
        self.semantic_resolution_debug = False

        # VGER+: all-class dynamic semantic revision.
        # ON by default; use --disable_enable_all_class_semantic_revision to fall back
        # to only-unknown resolution (original behaviour).
        self.enable_all_class_semantic_revision = True
        self.semantic_known_keep_threshold = 0.55
        self.semantic_known_reject_threshold = 0.35
        self.semantic_revision_candidate_confidence = 0.70
        self.semantic_revision_margin = 0.20
        self.semantic_revision_min_observations = 6
        self.semantic_revision_stable_windows = 2
        self.semantic_background_candidate_confidence = 0.75
        self.semantic_background_candidate_max_target_prob = 0.25

        # TADC: target-aware density control (semantic grow gating on clone/split)
        self.enable_semantic_density_control = False
        self.semantic_unknown_grow_threshold = 0.40
        self.semantic_density_debug = False
        self.semantic_density_debug_interval = 500
        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)


def validate_semantic_params(opt):
    """Validate every semantic parameter range. Raises ValueError before training starts."""
    checks = [
        ("semantic_warmup_iterations", opt.semantic_warmup_iterations >= 0, "必须 >= 0"),
        ("semantic_patch_size", opt.semantic_patch_size > 0 and opt.semantic_patch_size % 2 == 1, "必须为正奇数"),
        ("semantic_depth_abs_tolerance", opt.semantic_depth_abs_tolerance >= 0, "必须 >= 0"),
        ("semantic_depth_rel_tolerance", opt.semantic_depth_rel_tolerance >= 0, "必须 >= 0"),
        ("semantic_min_render_alpha", 0.0 <= opt.semantic_min_render_alpha <= 1.0, "必须在 [0,1]"),
        ("semantic_refresh_interval", opt.semantic_refresh_interval > 0, "必须 > 0"),
        ("semantic_prior_strength", opt.semantic_prior_strength >= 0, "必须 >= 0"),
        ("semantic_target_min_observations", opt.semantic_target_min_observations >= 1, "必须 >= 1"),
        ("semantic_target_confidence", 0.0 <= opt.semantic_target_confidence <= 1.0, "必须在 [0,1]"),
        ("semantic_target_margin", 0.0 <= opt.semantic_target_margin <= 1.0, "必须在 [0,1]"),
        ("semantic_background_min_observations", opt.semantic_background_min_observations >= 1, "必须 >= 1"),
        ("semantic_background_confidence", 0.0 <= opt.semantic_background_confidence <= 1.0, "必须在 [0,1]"),
        ("semantic_background_max_target_prob", 0.0 <= opt.semantic_background_max_target_prob <= 1.0, "必须在 [0,1]"),
        ("semantic_background_stable_windows", opt.semantic_background_stable_windows >= 1, "必须 >= 1"),
        ("semantic_unknown_grow_threshold", 0.0 <= opt.semantic_unknown_grow_threshold <= 1.0, "必须在 [0,1]"),
        ("semantic_evidence_debug_interval", opt.semantic_evidence_debug_interval > 0, "必须 > 0"),
        ("semantic_density_debug_interval", opt.semantic_density_debug_interval > 0, "必须 > 0"),
        ("semantic_multiscale_ratios", len(getattr(opt, "semantic_multiscale_ratios", [])) >= 1
         and all(r > 0 for r in opt.semantic_multiscale_ratios), "必须为非空正数列表"),
        ("semantic_multiscale_weights", len(getattr(opt, "semantic_multiscale_weights", [])) >= 1
         and len(opt.semantic_multiscale_weights) == len(opt.semantic_multiscale_ratios)
         and all(w > 0 for w in opt.semantic_multiscale_weights), "必须与ratios同长度且全为正"),
        ("semantic_multiscale_sigma_ratio", getattr(opt, "semantic_multiscale_sigma_ratio", 0.25) > 0, "必须 > 0"),
        ("semantic_dynamic_patch_max_size", getattr(opt, "semantic_dynamic_patch_max_size", 0) >= 0, "必须 >= 0"),
        ("semantic_known_keep_threshold", 0.0 <= getattr(opt, "semantic_known_keep_threshold", 0.55) <= 1.0, "必须在 [0,1]"),
        ("semantic_known_reject_threshold", 0.0 <= getattr(opt, "semantic_known_reject_threshold", 0.35) <= 1.0, "必须在 [0,1]"),
        ("semantic_revision_candidate_confidence", 0.0 <= getattr(opt, "semantic_revision_candidate_confidence", 0.70) <= 1.0, "必须在 [0,1]"),
        ("semantic_revision_margin", 0.0 <= getattr(opt, "semantic_revision_margin", 0.20) <= 1.0, "必须在 [0,1]"),
        ("semantic_revision_min_observations", getattr(opt, "semantic_revision_min_observations", 6) >= 1, "必须 >= 1"),
        ("semantic_revision_stable_windows", getattr(opt, "semantic_revision_stable_windows", 2) >= 1, "必须 >= 1"),
        ("semantic_background_candidate_confidence", 0.0 <= getattr(opt, "semantic_background_candidate_confidence", 0.75) <= 1.0, "必须在 [0,1]"),
        ("semantic_background_candidate_max_target_prob", 0.0 <= getattr(opt, "semantic_background_candidate_max_target_prob", 0.25) <= 1.0, "必须在 [0,1]"),
    ]
    for name, ok, msg in checks:
        if not ok:
            raise ValueError(f"[Semantic Pipeline] 参数 {name} 非法：{msg}，当前值 {getattr(opt, name)}")


def _print_semantic_config(dataset, opt):
    print("[Semantic Pipeline Configuration]")
    print("enabled: True")
    print("source_path:")
    print(f"  {dataset.source_path}")
    print("semantic_init_path:")
    print(f"  {dataset.semantic_init_path}")
    print("semantic_mask_dir:")
    print(f"  {dataset.semantic_mask_dir}")
    print("VEA:")
    print(f"  enabled: {opt.enable_semantic_evidence}")
    print(f"  warmup_iterations: {opt.semantic_warmup_iterations}")
    print(f"  patch_size: {opt.semantic_patch_size}")
    print(f"  depth_abs_tolerance: {opt.semantic_depth_abs_tolerance}")
    print(f"  depth_rel_tolerance: {opt.semantic_depth_rel_tolerance}")
    print(f"  min_render_alpha: {opt.semantic_min_render_alpha}")
    print(f"  dynamic_multiscale_patch: {getattr(opt, 'enable_dynamic_multiscale_semantic_patch', False)}")
    print(f"  multiscale_ratios: {getattr(opt, 'semantic_multiscale_ratios', (0.333333, 0.666667, 1.0))}")
    print(f"  multiscale_weights: {getattr(opt, 'semantic_multiscale_weights', (0.55, 0.30, 0.15))}")
    print(f"  multiscale_sigma_ratio: {getattr(opt, 'semantic_multiscale_sigma_ratio', 0.25)}")
    print(f"  dynamic_patch_max_size: {getattr(opt, 'semantic_dynamic_patch_max_size', 0)}")
    print("VGER:")
    print(f"  refresh_interval: {opt.semantic_refresh_interval}")
    print(f"  prior_strength: {opt.semantic_prior_strength}")
    print(f"  all_class_revision: {getattr(opt, 'enable_all_class_semantic_revision', False)}")
    print(f"  target_min_observations: {opt.semantic_target_min_observations}")
    print(f"  target_confidence: {opt.semantic_target_confidence}")
    print(f"  target_margin: {opt.semantic_target_margin}")
    print(f"  background_min_observations: {opt.semantic_background_min_observations}")
    print(f"  background_confidence: {opt.semantic_background_confidence}")
    print(f"  background_max_target_prob: {opt.semantic_background_max_target_prob}")
    print(f"  background_stable_windows: {opt.semantic_background_stable_windows}")
    print(f"  known_keep_threshold: {getattr(opt, 'semantic_known_keep_threshold', 0.55)}")
    print(f"  known_reject_threshold: {getattr(opt, 'semantic_known_reject_threshold', 0.35)}")
    print(f"  revision_candidate_confidence: {getattr(opt, 'semantic_revision_candidate_confidence', 0.70)}")
    print(f"  revision_margin: {getattr(opt, 'semantic_revision_margin', 0.20)}")
    print(f"  revision_min_observations: {getattr(opt, 'semantic_revision_min_observations', 6)}")
    print(f"  revision_stable_windows: {getattr(opt, 'semantic_revision_stable_windows', 2)}")
    print("TADC:")
    print(f"  enabled: {opt.enable_semantic_density_control}")
    print(f"  unknown_grow_threshold: {opt.semantic_unknown_grow_threshold}")
    print("Debug:")
    print(f"  evidence: {opt.semantic_evidence_debug}")
    print(f"  resolution: {opt.semantic_resolution_debug}")
    print(f"  density: {opt.semantic_density_debug}")
    print(f"  sls: {dataset.sls_debug}")


def resolve_semantic_pipeline_config(dataset, opt):
    """Consolidated semantic-pipeline resolution. Must run after parse_args/extract and
    BEFORE Scene / Camera creation.

    With --semantic_pipeline:
      * enables VEA + TADC,
      * derives semantic_init_path / semantic_mask_dir from the real -s source_path
        (explicit command-line values always win),
      * validates parameter ranges,
      * verifies the semantic NPZ and a non-empty object_mask directory exist,
      * prints the final resolved configuration once.

    Without --semantic_pipeline the original Fast-PGSR behaviour is untouched.
    """
    semantic_active = (dataset.semantic_pipeline or opt.enable_semantic_evidence
                       or opt.enable_semantic_density_control)
    if semantic_active:
        validate_semantic_params(opt)
    if not dataset.semantic_pipeline:
        return
    opt.enable_semantic_evidence = True
    opt.enable_semantic_density_control = True
    src = dataset.source_path
    if not dataset.semantic_init_path:
        dataset.semantic_init_path = os.path.abspath(
            os.path.join(src, "semantic_initialization", "point_semantics.npz"))
    if not dataset.semantic_mask_dir:
        dataset.semantic_mask_dir = os.path.abspath(os.path.join(src, "object_mask"))
    if not os.path.isfile(dataset.semantic_init_path):
        raise RuntimeError(f"[Semantic Pipeline] 语义NPZ不存在: {dataset.semantic_init_path}")
    if not os.path.isdir(dataset.semantic_mask_dir):
        raise RuntimeError(f"[Semantic Pipeline] Mask目录不存在: {dataset.semantic_mask_dir}")
    if not glob.glob(os.path.join(dataset.semantic_mask_dir, "*.png")):
        raise RuntimeError(f"[Semantic Pipeline] Mask目录为空（无png）: {dataset.semantic_mask_dir}")
    _print_semantic_config(dataset, opt)
