#
# Semantic Training Configuration Consolidation acceptance test.
#
# Verifies the --semantic_pipeline master switch, auto path derivation from the real
# -s source_path, explicit CLI override priority, parameter range validation, the
# startup config summary, and that plain / semantic short trainings both run.
#
# Run: python scripts/test_semantic_pipeline_config.py  (needs CUDA for the 2 short trainings)
#

import argparse
import contextlib
import io
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from collections import Counter
from argparse import ArgumentParser
from arguments import (ModelParams, PipelineParams, OptimizationParams,
                       resolve_semantic_pipeline_config, validate_semantic_params)

RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print(f"[CFG-T] {'PASS' if cond else 'FAIL'}  {name}{(' | ' + detail) if detail else ''}")
    assert cond, f"CFG-T FAIL: {name}"


def make_parser():
    parser = ArgumentParser()
    lp = ModelParams(parser); op = OptimizationParams(parser); pp = PipelineParams(parser)
    return parser, lp, op, pp


def parse(argv):
    parser, lp, op, pp = make_parser()
    a = parser.parse_args(argv)
    return lp.extract(a), op.extract(a)


# --------------------------------------------------------------------------- #
print("=" * 78)
print("T1/T12 默认值")
print("=" * 78)
ds0, op0 = parse(["-s", "data/mungbean13_0730", "-m", "/tmp/cfg"])
check("T1 semantic_pipeline 默认 False", ds0.semantic_pipeline is False)
defaults_opt = {
    "semantic_warmup_iterations": 1000, "semantic_patch_size": 5,
    "semantic_depth_abs_tolerance": 0.1, "semantic_depth_rel_tolerance": 0.02,
    "semantic_min_render_alpha": 0.5, "semantic_evidence_debug": False,
    "semantic_evidence_debug_interval": 100,
    "semantic_refresh_interval": 500, "semantic_prior_strength": 5.0,
    "semantic_target_min_observations": 5, "semantic_target_confidence": 0.75,
    "semantic_target_margin": 0.20, "semantic_background_min_observations": 8,
    "semantic_background_confidence": 0.95, "semantic_background_max_target_prob": 0.05,
    "semantic_background_stable_windows": 2, "semantic_resolution_debug": False,
    "enable_semantic_density_control": False, "semantic_unknown_grow_threshold": 0.40,
    "semantic_density_debug": False, "semantic_density_debug_interval": 500,
    "enable_semantic_evidence": False,
}
defaults_ds = {"sls_debug": False, "export_mask_alignment_debug": False}
for k, v in defaults_opt.items():
    check(f"T12 默认值 {k} == {v}", getattr(op0, k) == v)
for k, v in defaults_ds.items():
    check(f"T12 默认值 {k} == {v}", getattr(ds0, k) == v)

# --------------------------------------------------------------------------- #
print("=" * 78)
print("T2  不启用 semantic_pipeline 时普通场景正常解析")
print("=" * 78)
ds, opt = parse(["-s", "data/mungbean13_0730", "-m", "/tmp/cfg"])
resolve_semantic_pipeline_config(ds, opt)
check("T2 不要求语义文件、不启用VEA/TADC", opt.enable_semantic_evidence is False
      and opt.enable_semantic_density_control is False and ds.semantic_init_path == "")

# --------------------------------------------------------------------------- #
print("=" * 78)
print("T3/T4/T24 启用 semantic_pipeline -> 自动启用 VEA + TADC")
print("=" * 78)
ds, opt = parse(["-s", "data/mungbean13_0730", "-m", "/tmp/cfg", "--semantic_pipeline"])
resolve_semantic_pipeline_config(ds, opt)
check("T3 enable_semantic_evidence == True", opt.enable_semantic_evidence is True)
check("T4 enable_semantic_density_control == True", opt.enable_semantic_density_control is True)
check("T24 VGER 由 refresh 驱动（enable_sem_ev=True 即可激活）", opt.semantic_refresh_interval > 0)

# --------------------------------------------------------------------------- #
print("=" * 78)
print("T5/T6/T7/T8 相对/绝对路径推导 与 显式覆盖优先")
print("=" * 78)
ds, _ = parse(["-s", "data/mungbean13_0730", "-m", "/tmp/cfg", "--semantic_pipeline"])
resolve_semantic_pipeline_config(ds, opt)
check("T5 相对 -s 推导 init_path 以正确后缀结尾",
      ds.semantic_init_path.endswith("mungbean13_0730/semantic_initialization/point_semantics.npz"))
check("T5 推导 mask_dir 以正确后缀结尾", ds.semantic_mask_dir.endswith("mungbean13_0730/object_mask"))
check("T5 推导路径为绝对路径", os.path.isabs(ds.semantic_init_path) and os.path.isabs(ds.semantic_mask_dir))

abs_src = os.path.abspath("data/mungbean13_0730")
ds2, _ = parse(["-s", abs_src, "-m", "/tmp/cfg", "--semantic_pipeline"])
resolve_semantic_pipeline_config(ds2, opt)
check("T6 绝对 -s 推导绝对路径", ds2.semantic_init_path == os.path.join(abs_src, "semantic_initialization", "point_semantics.npz"))

# explicit override wins (use a real temp npz so existence check passes)
with tempfile.TemporaryDirectory() as td:
    npz = os.path.join(td, "custom.npz")
    open(npz, "w").close()
    maskd = os.path.join(td, "masks"); os.makedirs(maskd)
    open(os.path.join(maskd, "m.png"), "w").close()
    ds3, _ = parse(["-s", "data/mungbean13_0730", "-m", "/tmp/cfg", "--semantic_pipeline",
                    "--semantic_init_path", npz, "--semantic_mask_dir", maskd])
    resolve_semantic_pipeline_config(ds3, opt)
    check("T7 显式 semantic_init_path 优先", os.path.abspath(ds3.semantic_init_path) == os.path.abspath(npz))
    check("T8 显式 semantic_mask_dir 优先", os.path.abspath(ds3.semantic_mask_dir) == os.path.abspath(maskd))

# --------------------------------------------------------------------------- #
print("=" * 78)
print("T9/T10/T11 缺失NPZ / 缺失Mask目录 / 空Mask目录 报错")
print("=" * 78)
with tempfile.TemporaryDirectory() as td:
    src_empty = os.path.join(td, "empty_scene"); os.makedirs(src_empty)
    # T9: no npz, no mask -> error on missing NPZ
    dsx, optx = parse(["-s", src_empty, "-m", "/tmp/x", "--semantic_pipeline"])
    try:
        resolve_semantic_pipeline_config(dsx, optx)
        check("T9 缺失NPZ报错", False)
    except RuntimeError:
        check("T9 缺失NPZ报错", True)
    # T10: create npz, no mask -> error on missing mask dir
    npz = os.path.join(src_empty, "semantic_initialization", "point_semantics.npz")
    os.makedirs(os.path.dirname(npz)); open(npz, "w").close()
    dsx, optx = parse(["-s", src_empty, "-m", "/tmp/x", "--semantic_pipeline"])
    try:
        resolve_semantic_pipeline_config(dsx, optx)
        check("T10 缺失Mask目录报错", False)
    except RuntimeError:
        check("T10 缺失Mask目录报错", True)
    # T11: create empty mask dir -> error on empty
    os.makedirs(os.path.join(src_empty, "object_mask"))
    dsx, optx = parse(["-s", src_empty, "-m", "/tmp/x", "--semantic_pipeline"])
    try:
        resolve_semantic_pipeline_config(dsx, optx)
        check("T11 空Mask目录报错", False)
    except RuntimeError:
        check("T11 空Mask目录报错", True)

# --------------------------------------------------------------------------- #
print("=" * 78)
print("T13/T14/T15 命令行覆盖单个参数生效")
print("=" * 78)
_, op = parse(["-s", "data/mungbean13_0730", "-m", "/tmp/x", "--semantic_patch_size", "7"])
check("T13 覆盖 patch_size=7", op.semantic_patch_size == 7)
_, op = parse(["-s", "data/mungbean13_0730", "-m", "/tmp/x", "--semantic_target_confidence", "0.90"])
check("T14 覆盖 target_confidence=0.90", abs(op.semantic_target_confidence - 0.90) < 1e-9)
_, op = parse(["-s", "data/mungbean13_0730", "-m", "/tmp/x", "--semantic_unknown_grow_threshold", "0.60"])
check("T15 覆盖 unknown_grow_threshold=0.60", abs(op.semantic_unknown_grow_threshold - 0.60) < 1e-9)

# --------------------------------------------------------------------------- #
print("=" * 78)
print("T16/T17/T18 非法参数报错")
print("=" * 78)
for argv, name in [
        (["-s", "data/mungbean13_0730", "-m", "/tmp/x", "--enable_semantic_evidence", "--semantic_patch_size", "6"], "T16 偶数patch_size"),
        (["-s", "data/mungbean13_0730", "-m", "/tmp/x", "--enable_semantic_evidence", "--semantic_target_confidence", "1.5"], "T17 非法confidence"),
        (["-s", "data/mungbean13_0730", "-m", "/tmp/x", "--enable_semantic_evidence", "--semantic_refresh_interval", "0"], "T18 非法refresh_interval"),
        (["-s", "data/mungbean13_0730", "-m", "/tmp/x", "--enable_semantic_density_control", "--semantic_unknown_grow_threshold", "2.0"], "T17 非法grow_threshold")]:
    dsx, optx = parse(argv)
    try:
        resolve_semantic_pipeline_config(dsx, optx)
        check(f"{name} 报错", False)
    except (ValueError, RuntimeError):
        check(f"{name} 报错", True)

# --------------------------------------------------------------------------- #
print("=" * 78)
print("T19/T20 parser 无重复参数；--help 正常")
print("=" * 78)
parser, _, _, _ = make_parser()
option_strings = []
for act in parser._actions:
    option_strings += act.option_strings
dups = [s for s, c in Counter(option_strings).items() if c > 1]
check("T19 无重复参数", not dups, str(dups))
try:
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        parser.parse_args(["--help"])
    check("T20 --help 正常", False)
except SystemExit as e:
    check("T20 --help 正常（exit=0）", e.code == 0)

# --------------------------------------------------------------------------- #
print("=" * 78)
print("T21 配置摘要显示真实路径")
print("=" * 78)
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    dsx, optx = parse(["-s", "data/mungbean13_0730", "-m", "/tmp/cfg", "--semantic_pipeline"])
    resolve_semantic_pipeline_config(dsx, optx)
out = buf.getvalue()
check("T21 摘要包含真实 source_path", "data/mungbean13_0730" in out and "/tmp/cfg" not in out)
check("T21 摘要包含真实 init_path", "point_semantics.npz" in out)
check("T21 摘要包含真实 mask_dir", "object_mask" in out)
check("T21 摘要包含 VEA/TADC enabled: True", "enabled: True" in out)

# --------------------------------------------------------------------------- #
print("=" * 78)
print("T22 普通 Fast-PGSR 短训练通过（无 semantic_pipeline）")
print("=" * 78)
r = subprocess.run([sys.executable, "train.py", "-s", "data/mungbean13_0730",
                    "-m", "/tmp/cfg_train_normal", "--iterations", "3", "--save_iterations", "3",
                    "--test_iterations", "3", "--data_device", "cuda"],
                   cwd=REPO, capture_output=True, text=True)
check("T22 普通短训练 exit=0", r.returncode == 0, r.stderr[-200:] if r.returncode else "")

# --------------------------------------------------------------------------- #
print("=" * 78)
print("T23 semantic_pipeline 短训练通过")
print("=" * 78)
r = subprocess.run([sys.executable, "train.py", "-s", "data/mungbean13_0730",
                    "-m", "/tmp/cfg_train_sem", "--iterations", "3", "--save_iterations", "3",
                    "--test_iterations", "3", "--data_device", "cuda",
                    "--semantic_pipeline", "--semantic_warmup_iterations", "1"],
                   cwd=REPO, capture_output=True, text=True)
check("T23 semantic_pipeline 短训练 exit=0", r.returncode == 0, r.stderr[-200:] if r.returncode else "")
check("T23 训练日志含配置摘要", "[Semantic Pipeline Configuration]" in r.stdout)

# --------------------------------------------------------------------------- #
print("=" * 78)
print("T25 既有 SLS/SSP/VMA/VEA/VGER/TADC 测试回归")
print("=" * 78)
for suite in ["test_sls_sync.py", "test_semantic_persistence.py", "test_vma_alignment.py",
              "test_semantic_evidence_accumulation.py", "test_semantic_resolution.py",
              "test_target_aware_density_control.py"]:
    r = subprocess.run([sys.executable, os.path.join("scripts", suite)],
                       cwd=REPO, capture_output=True, text=True)
    check(f"T25 {suite} 通过", r.returncode == 0)

# --------------------------------------------------------------------------- #
print("=" * 78)
fails = [n for n, ok in RESULTS if not ok]
print(f"配置整合测试总计 {len(RESULTS)} 项检查, 失败 {len(fails)} 项")
if fails:
    print("失败项:", fails[:20])
    raise SystemExit(1)
print("语义训练配置整合测试全部通过 ✔")
print("=" * 78)
