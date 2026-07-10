#!/usr/bin/env python3
"""
离线监控训练输出目录下的 checkpoint-*：权重就绪后依次调用
run_vllm_infer_mutil_turn.py → cal_meta_action_acc.py + cal_ade_rfs.py，
同时统计 meta action 准确率、3s/5s ADE 与 mean RFS（推理 jsonl 与
cal_ade.py 一致；RFS 需额外提供 --meta_jsonl，含 preference_trajectories）。
**同时选 best 5s ADE（越小越好）与 best RFS（越大越好）**，full_metrics 内保留 meta_action / ade / rfs。

典型用法（训练每 100 step 存一次，持续往同一 checkpoint_root 写）：

  python eval_watch_meta_action_ade_rfs_checkpoints.py \\
    --checkpoint_root /path/to/v1-xxxxxx \\
    --input_jsonl /path/val_samples_479.jsonl \\
    --label_jsonl /path/val_labels.jsonl \\
    --meta_jsonl /path/val_samples_434.jsonl \\
    --output_dir /path/to/eval_runs/run1

说明：
- 就绪条件：存在 config.json，且至少有一份权重（*.safetensors 或 pytorch_model.bin 等）。
- 已完成的 checkpoint 记录在 output_dir/eval_state.json，避免重复跑。
- best 写在 output_dir/best_model.json（分别记录 ade_5s 与 mean_rfs 的 best，含 meta_action + ade + rfs）；历史追加在 output_dir/history.jsonl。
- 默认在评测完成后删除非 best 的 checkpoint-* 目录（最多保留 best 5s ADE 与 best RFS 两个 checkpoint；若相同则只保留一个）。可用 --no_delete_non_best 关闭。
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INFER = "/cephfs/zhengwc/FluidDrive/ms-swift-3.5/tools/infer_scripts/run_vllm_infer_mutil_turn_w_cot.py"
DEFAULT_CAL_META = "/cephfs/zhengwc/FluidDrive/ms-swift-3.5/tools/eval-one-step/cal_meta_action_acc.py"
DEFAULT_CAL_ADE_RFS = "/cephfs/zhengwc/FluidDrive/ms-swift-3.5/tools/eval-one-step/cal_ade_rfs.py"
ROOT = "/cephfs/zhengwc/FluidDrive/ms-swift-3.5/output/CoT-exp/Qwen3.5-2B/cot-meta-traj/v1-20260509-200516"
INPUT_JSONL = "/cephfs/zhengwc/FluidDrive/ms-swift-3.5/data/train_data/meta_data/val_samples_434.jsonl"
LABEL_JSONL = "/cephfs/zhengwc/LongTail_synthesizer_V2/v2/data/new-labels/val_samples_479_meta_action_label.jsonl"
META_JSONL = INPUT_JSONL

_CKPT_RE = re.compile(r"^checkpoint-(\d+)$", re.IGNORECASE)
BEST_METRIC_KEYS = ("ade_5s", "mean_rfs")


@dataclass
class CheckpointRecord:
    checkpoint_name: str
    checkpoint_path: str
    infer_jsonl: str
    metrics_meta_json: str
    metrics_ade_json: str
    joint_acc: Optional[float]
    longitudinal_acc: Optional[float]
    lateral_acc: Optional[float]
    evaluated_pairs: int
    ade_3s: Optional[float]
    ade_5s: Optional[float]
    mean_rfs: Optional[float]
    finished_at: str


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def parse_checkpoint_step(name: str) -> Optional[int]:
    m = _CKPT_RE.match(name.strip())
    if not m:
        return None
    return int(m.group(1))


def list_checkpoint_dirs(root: Path) -> List[Path]:
    """返回 checkpoint-* 目录列表，按步数从大到小（优先评测较新的 checkpoint）。"""
    if not root.is_dir():
        return []
    out: List[Path] = []
    for p in root.iterdir():
        if not p.is_dir():
            continue
        if parse_checkpoint_step(p.name) is None:
            continue
        out.append(p)
    out.sort(key=lambda p: parse_checkpoint_step(p.name) or 0, reverse=True)
    return out


def is_checkpoint_ready(path: Path) -> bool:
    """避免 trainer 尚未写完时误启动：需 config + 至少一份权重文件。"""
    if not path.is_dir():
        return False
    if not (path / "config.json").is_file():
        return False
    if any(path.glob("*.safetensors")):
        return True
    if any(path.glob("model-*-of-*.safetensors")):
        return True
    if (path / "pytorch_model.bin").is_file():
        return True
    return False


def load_state(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {"checkpoints": {}}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_state(path: Path, state: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


def append_history(history_path: Path, row: Dict[str, Any]) -> None:
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_infer(
    *,
    python_exe: str,
    infer_script: Path,
    model_path: str,
    output_jsonl: str,
    input_jsonl: str,
    batch_size: int,
    temperature: float,
    max_tokens: int,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    max_samples: int,
    repeat_num: int,
    repeat_seed_base: int,
    include_jsonl: Optional[str],
    exclude_jsonl: Optional[str],
) -> None:
    cmd: List[str] = [
        python_exe,
        str(infer_script),
        "--model_path",
        model_path,
        "--input_jsonl",
        input_jsonl,
        "--output_jsonl",
        output_jsonl,
        "--batch_size",
        str(batch_size),
        "--temperature",
        str(temperature),
        "--max_tokens",
        str(max_tokens),
        "--tensor_parallel_size",
        str(tensor_parallel_size),
        "--gpu_memory_utilization",
        str(gpu_memory_utilization),
        "--repeat_num",
        str(repeat_num),
        "--repeat_seed_base",
        str(repeat_seed_base),
    ]
    if max_samples > 0:
        cmd.extend(["--max_samples", str(max_samples)])
    if include_jsonl:
        cmd.extend(["--include_jsonl", include_jsonl])
    if exclude_jsonl:
        cmd.extend(["--exclude_jsonl", exclude_jsonl])

    print(f"[eval_watch] 启动推理: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def run_cal_meta_action(
    *,
    python_exe: str,
    cal_script: Path,
    infer_jsonl: str,
    label_jsonl: str,
    metrics_out: str,
) -> Dict[str, Any]:
    cmd = [
        python_exe,
        str(cal_script),
        "--infer_jsonl",
        infer_jsonl,
        "--label_jsonl",
        label_jsonl,
        "--output_file",
        metrics_out,
    ]
    print(f"[eval_watch] 启动 meta action 准确率: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    with open(metrics_out, "r", encoding="utf-8") as f:
        return json.load(f)


def run_cal_ade_rfs(
    *,
    python_exe: str,
    cal_script: Path,
    infer_jsonl: str,
    meta_jsonl: str,
    summary_out: str,
    scenario_json: Optional[str],
    frequency: int,
    length_seconds: int,
    score_th: float,
) -> Dict[str, Any]:
    """
    调用 cal_ade_rfs.py：与 cal_ade.py 相同地以推理 jsonl 为输入（此处为 --infer-jsonl），
    同时计算 ADE 与 RFS；汇总写入 --output-summary-json。
    """
    summary_out_path = Path(summary_out)
    summary_out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd: List[str] = [
        python_exe,
        str(cal_script),
        "--infer-jsonl",
        infer_jsonl,
        "--meta-jsonl",
        meta_jsonl,
        "--output-summary-json",
        str(summary_out_path),
        "--frequency",
        str(frequency),
        "--length-seconds",
        str(length_seconds),
        "--score-th",
        str(score_th),
    ]
    if scenario_json:
        cmd.extend(["--scenario-json", scenario_json])
    print(f"[eval_watch] 启动 ADE+RFS (cal_ade_rfs): {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    with open(summary_out_path, "r", encoding="utf-8") as f:
        return json.load(f)


def pick_best_metric(stats: Dict[str, Any], key: str) -> Optional[float]:
    v = stats.get(key)
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def pick_metric_from_merged(merged: Dict[str, Any], key: str) -> Optional[float]:
    """从合并后的 metrics 中取标量：先查 meta_action，再查 ade，再查 rfs。"""
    ma = merged.get("meta_action")
    if isinstance(ma, dict) and key in ma:
        return pick_best_metric(ma, key)
    ad = merged.get("ade")
    if isinstance(ad, dict) and key in ad:
        return pick_best_metric(ad, key)
    rf = merged.get("rfs")
    if isinstance(rf, dict) and key in rf:
        return pick_best_metric(rf, key)
    return None


def is_lower_better_metric(key: str) -> bool:
    """ADE 类指标越小越好；准确率类与 mean_rfs 越大越好。"""
    return key in ("ade_3s", "ade_5s")


def metric_is_better(key: str, new_val: float, old_val: float) -> bool:
    if is_lower_better_metric(key):
        return new_val < old_val
    return new_val > old_val


def delete_non_best_checkpoint_dir(
    checkpoint_root: Path,
    checkpoint_name: str,
    *,
    tag: str = "",
) -> None:
    """删除 checkpoint_root 下名为 checkpoint-* 的目录（用于释放磁盘）；异常仅打印警告。"""
    if parse_checkpoint_step(checkpoint_name) is None:
        return
    target = (checkpoint_root / checkpoint_name).resolve()
    root = checkpoint_root.resolve()
    try:
        target.relative_to(root)
    except ValueError:
        print(
            f"[eval_watch] [WARN] 跳过删除（路径不在 checkpoint_root 下）: {target}",
            flush=True,
        )
        return
    if not target.is_dir():
        return
    prefix = f"[eval_watch] 已删除{tag}checkpoint 目录" if tag else "[eval_watch] 已删除 checkpoint 目录"
    try:
        shutil.rmtree(target)
        print(f"{prefix}: {target}", flush=True)
    except OSError as e:
        print(f"[eval_watch] [WARN] 删除失败 {target}: {e}", flush=True)


def load_best_payload(best_path: Path) -> Dict[str, Any]:
    if not best_path.is_file():
        return {"best": {}}
    try:
        with open(best_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {"best": {}}
    if not isinstance(payload, dict):
        return {"best": {}}
    best = payload.get("best")
    if not isinstance(best, dict):
        payload["best"] = {}
    return payload


def best_checkpoint_names(best_payload: Dict[str, Any]) -> List[str]:
    best = best_payload.get("best")
    if not isinstance(best, dict):
        return []
    names: List[str] = []
    for metric_key in BEST_METRIC_KEYS:
        item = best.get(metric_key)
        if not isinstance(item, dict):
            continue
        name = item.get("best_checkpoint")
        if isinstance(name, str) and name not in names:
            names.append(name)
    return names


def save_best_payload(best_path: Path, payload: Dict[str, Any]) -> None:
    names = best_checkpoint_names(payload)
    payload["best_checkpoints"] = names
    payload["updated_at"] = _now_iso()
    best_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = best_path.with_suffix(best_path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    tmp.replace(best_path)


def prune_evaluated_non_best_checkpoints(
    *,
    checkpoint_root: Path,
    best_path: Path,
    state: Dict[str, Any],
) -> None:
    """
    删除已在 eval_state 中记录、且非当前 best 的 checkpoint-* 目录。
    用于修复「已在 state 中则提前 return」导致从未走到评测后删除逻辑的情况。
    """
    best_names = set(best_checkpoint_names(load_best_payload(best_path)))
    if not best_names:
        return
    evaluated = state.get("checkpoints")
    if not isinstance(evaluated, dict):
        return
    for name in sorted(
        evaluated.keys(),
        key=lambda n: parse_checkpoint_step(n) or 0,
    ):
        if name in best_names:
            continue
        if parse_checkpoint_step(name) is None:
            continue
        delete_non_best_checkpoint_dir(
            checkpoint_root,
            name,
            tag="已评测非 best ",
        )


def update_best_file(
    best_path: Path,
    *,
    stats_merged: Dict[str, Any],
    record: CheckpointRecord,
) -> Dict[str, Dict[str, Any]]:
    """分别维护 ade_5s 与 mean_rfs 的 best。"""
    payload = load_best_payload(best_path)
    best = payload.setdefault("best", {})
    updates: Dict[str, Dict[str, Any]] = {}

    for metric_key in BEST_METRIC_KEYS:
        new_val = pick_metric_from_merged(stats_merged, metric_key)
        old_entry = best.get(metric_key)
        old_best = None
        if isinstance(old_entry, dict):
            old_best = pick_best_metric(old_entry, "best_value")
        is_new_best = new_val is not None and (
            old_best is None or metric_is_better(metric_key, new_val, old_best)
        )
        updates[metric_key] = {
            "is_new_best": is_new_best,
            "previous_best": old_best,
            "current_metric": new_val,
            "previous_best_checkpoint": (
                old_entry.get("best_checkpoint") if isinstance(old_entry, dict) else None
            ),
        }
        if not is_new_best:
            continue

        best[metric_key] = {
            "best_metric": metric_key,
            "best_value": new_val,
            "best_is_lower_better": is_lower_better_metric(metric_key),
            "best_checkpoint": record.checkpoint_name,
            "best_checkpoint_path": record.checkpoint_path,
            "best_infer_jsonl": record.infer_jsonl,
            "best_metrics_meta_json": record.metrics_meta_json,
            "best_metrics_ade_json": record.metrics_ade_json,
            "full_metrics": stats_merged,
            "updated_at": _now_iso(),
            "previous_best_value": old_best,
        }

    if any(item["is_new_best"] for item in updates.values()):
        save_best_payload(best_path, payload)
    return updates


def process_one_checkpoint(
    *,
    ckpt_path: Path,
    checkpoint_root: Path,
    output_dir: Path,
    state: Dict[str, Any],
    state_path: Path,
    best_path: Path,
    history_path: Path,
    delete_non_best: bool,
    python_exe: str,
    infer_script: Path,
    cal_meta_script: Path,
    cal_ade_rfs_script: Path,
    input_jsonl: str,
    label_jsonl: str,
    meta_jsonl: str,
    ade_scenario_json: Optional[str],
    rfs_frequency: int,
    rfs_length_seconds: int,
    rfs_score_th: float,
    infer_prefix: str,
    batch_size: int,
    temperature: float,
    max_tokens: int,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    max_samples: int,
    repeat_num: int,
    repeat_seed_base: int,
    include_jsonl: Optional[str],
    exclude_jsonl: Optional[str],
) -> None:
    name = ckpt_path.name
    infer_out = output_dir / f"{infer_prefix}{name}.jsonl"
    metrics_meta_out = output_dir / "metrics" / f"{infer_prefix}{name}_meta_action.json"
    metrics_ade_rfs_out = output_dir / "metrics" / f"{infer_prefix}{name}_ade_rfs.json"

    checkpoints: Dict[str, Any] = state.setdefault("checkpoints", {})
    if name in checkpoints:
        print(f"[eval_watch] 跳过（已在 state 中）: {name}", flush=True)
        return

    if not is_checkpoint_ready(ckpt_path):
        print(f"[eval_watch] 跳过（未就绪）: {ckpt_path}", flush=True)
        return

    print(f"[eval_watch] >>> 评测 checkpoint: {ckpt_path}", flush=True)

    run_infer(
        python_exe=python_exe,
        infer_script=infer_script,
        model_path=str(ckpt_path),
        output_jsonl=str(infer_out),
        input_jsonl=input_jsonl,
        batch_size=batch_size,
        temperature=temperature,
        max_tokens=max_tokens,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_samples=max_samples,
        repeat_num=repeat_num,
        repeat_seed_base=repeat_seed_base,
        include_jsonl=include_jsonl,
        exclude_jsonl=exclude_jsonl,
    )

    stats_meta = run_cal_meta_action(
        python_exe=python_exe,
        cal_script=cal_meta_script,
        infer_jsonl=str(infer_out),
        label_jsonl=label_jsonl,
        metrics_out=str(metrics_meta_out),
    )
    summary_full = run_cal_ade_rfs(
        python_exe=python_exe,
        cal_script=cal_ade_rfs_script,
        infer_jsonl=str(infer_out),
        meta_jsonl=meta_jsonl,
        summary_out=str(metrics_ade_rfs_out),
        scenario_json=ade_scenario_json,
        frequency=rfs_frequency,
        length_seconds=rfs_length_seconds,
        score_th=rfs_score_th,
    )
    stats_ade = summary_full.get("ade") if isinstance(summary_full.get("ade"), dict) else {}
    stats_rfs = summary_full.get("rfs") if isinstance(summary_full.get("rfs"), dict) else None

    stats_merged = {"meta_action": stats_meta, "ade": stats_ade, "rfs": stats_rfs}

    mean_rfs: Optional[float] = None
    if isinstance(stats_rfs, dict):
        mean_rfs = pick_best_metric(stats_rfs, "mean_rfs")

    record = CheckpointRecord(
        checkpoint_name=name,
        checkpoint_path=str(ckpt_path.resolve()),
        infer_jsonl=str(infer_out.resolve()),
        metrics_meta_json=str(metrics_meta_out.resolve()),
        metrics_ade_json=str(metrics_ade_rfs_out.resolve()),
        joint_acc=pick_best_metric(stats_meta, "joint_acc"),
        longitudinal_acc=pick_best_metric(stats_meta, "longitudinal_acc"),
        lateral_acc=pick_best_metric(stats_meta, "lateral_acc"),
        evaluated_pairs=int(stats_meta.get("evaluated_pairs") or 0),
        ade_3s=pick_best_metric(stats_ade, "ade_3s"),
        ade_5s=pick_best_metric(stats_ade, "ade_5s"),
        mean_rfs=mean_rfs,
        finished_at=_now_iso(),
    )

    checkpoints[name] = asdict(record)
    save_state(state_path, state)

    updates = update_best_file(
        best_path,
        stats_merged=stats_merged,
        record=record,
    )
    updated = any(item["is_new_best"] for item in updates.values())

    if delete_non_best:
        keep_names = set(best_checkpoint_names(load_best_payload(best_path)))
        if updated:
            previous_names = {
                item.get("previous_best_checkpoint")
                for item in updates.values()
                if item.get("is_new_best") and isinstance(item.get("previous_best_checkpoint"), str)
            }
            for previous_name in sorted(previous_names):
                if previous_name in keep_names:
                    continue
                delete_non_best_checkpoint_dir(
                    checkpoint_root,
                    previous_name,
                    tag="旧 best ",
                )
        if name not in keep_names:
            delete_non_best_checkpoint_dir(checkpoint_root, name)
    hist_row = {
        "event": "eval_done",
        "record": asdict(record),
        "best_metrics": {
            metric_key: {
                "best_is_lower_better": is_lower_better_metric(metric_key),
                **update,
            }
            for metric_key, update in updates.items()
        },
        "meta_action": stats_meta,
        "ade": stats_ade,
        "rfs": stats_rfs,
    }
    append_history(history_path, hist_row)

    new_best_msgs = []
    for metric_key, update in updates.items():
        if not update["is_new_best"]:
            continue
        cmp_word = "低于" if is_lower_better_metric(metric_key) else "高于"
        new_best_msgs.append(
            f"{metric_key}={update['current_metric']} "
            f"({cmp_word}上次 best={update['previous_best']})"
        )
    if new_best_msgs:
        print(
            f"[eval_watch] ★ 新 best: {', '.join(new_best_msgs)} "
            f"(checkpoint={name})",
            flush=True,
        )
    print(
        f"[eval_watch] 完成 {name}: ade_5s={record.ade_5s} "
        f"mean_rfs={record.mean_rfs} joint={record.joint_acc}",
        flush=True,
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "监控 checkpoint-*：mutil_turn 推理 + meta action acc + cal_ade_rfs（ADE+ RFS）；"
            "同时按 ade_5s 与 mean_rfs 选 best"
        )
    )
    p.add_argument(
        "--checkpoint_root",
        type=str,
        default=ROOT,
        help="含 checkpoint-100, checkpoint-200, ... 的训练输出目录",
    )
    p.add_argument(
        "--input_jsonl",
        type=str,
        default=INPUT_JSONL,
        help="传给推理脚本的 meta jsonl",
    )
    p.add_argument(
        "--label_jsonl",
        type=str,
        default=LABEL_JSONL,
        help="传给 cal_meta_action_acc 的标注 jsonl",
    )
    p.add_argument(
        "--meta_jsonl",
        type=str,
        default=META_JSONL,
        help="传给 cal_ade_rfs 的原始数据 jsonl（含 preference_trajectories，与 cal_ade_rfs --meta-jsonl 一致）",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default=ROOT,
        help="推理输出、metrics、state、best 的根目录",
    )
    p.add_argument(
        "--infer_prefix",
        type=str,
        default="infer_",
        help="输出文件名前缀，例如 infer_ -> infer_checkpoint-1100.jsonl",
    )
    p.add_argument(
        "--infer_script",
        type=str,
        default=str(DEFAULT_INFER),
        help="推理脚本路径（默认 run_vllm_infer_mutil_turn.py）",
    )
    p.add_argument(
        "--cal_meta_script",
        type=str,
        default=str(DEFAULT_CAL_META),
        help="cal_meta_action_acc.py 路径",
    )
    p.add_argument(
        "--cal_ade_rfs_script",
        type=str,
        default=str(DEFAULT_CAL_ADE_RFS),
        help="cal_ade_rfs.py 路径（也可用 --cal_ade_script 指定同一脚本，兼容旧参数名）",
    )
    p.add_argument(
        "--cal_ade_script",
        type=str,
        default=None,
        help="已弃用：兼容别名；若设置则覆盖 --cal_ade_rfs_script",
    )
    p.add_argument(
        "--ade_scenario_json",
        type=str,
        default=None,
        help="可选：传给 cal_ade_rfs --scenario-json（与 cal_ade.py --scenario_json 语义一致，JSON 含 scenario_ids）",
    )
    p.add_argument(
        "--rfs_frequency",
        type=int,
        default=4,
        help="RFS：--frequency（Hz），与 cal_ade_rfs 一致",
    )
    p.add_argument(
        "--rfs_length_seconds",
        type=int,
        default=5,
        help="RFS：--length-seconds，与 cal_ade_rfs 一致",
    )
    p.add_argument(
        "--rfs_score_th",
        type=float,
        default=4.5,
        help="RFS：--score-th，与 cal_ade_rfs 一致",
    )
    p.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="解释器路径（默认当前 python）",
    )
    p.add_argument(
        "--poll_interval_sec",
        type=float,
        default=5,
        help="轮询间隔（秒）",
    )
    p.add_argument(
        "--once",
        action="store_true",
        help="只扫描一轮后退出（不持续监控）",
    )
    p.add_argument(
        "--no_delete_non_best",
        action="store_true",
        help="评测完成后不删除非 best 5s ADE / best RFS 的 checkpoint-* 目录（默认会删以省空间）",
    )

    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.01)
    p.add_argument("--max_tokens", type=int, default=2048)
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    p.add_argument("--max_samples", type=int, default=-1)
    p.add_argument(
        "--repeat_num",
        type=int,
        default=3,
        help="与 run_vllm_infer_mutil_turn 一致时每样本重复次数（默认 10）",
    )
    p.add_argument("--repeat_seed_base", type=int, default=42)
    p.add_argument("--include_jsonl", type=str, default=None)
    p.add_argument("--exclude_jsonl", type=str, default=None)

    args = p.parse_args()

    checkpoint_root = Path(args.checkpoint_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    state_path = output_dir / "eval_state.json"
    best_path = output_dir / "best_model.json"
    history_path = output_dir / "history.jsonl"

    infer_script = Path(args.infer_script).resolve()
    cal_meta_script = Path(args.cal_meta_script).resolve()
    cal_ade_rfs_path = (
        Path(args.cal_ade_script).resolve()
        if args.cal_ade_script
        else Path(args.cal_ade_rfs_script).resolve()
    )
    if not infer_script.is_file():
        raise SystemExit(f"推理脚本不存在: {infer_script}")
    if not cal_meta_script.is_file():
        raise SystemExit(f"meta action 脚本不存在: {cal_meta_script}")
    if not cal_ade_rfs_path.is_file():
        raise SystemExit(f"cal_ade_rfs 脚本不存在: {cal_ade_rfs_path}")

    def scan_loop() -> None:
        state = load_state(state_path)
        ckpts = list_checkpoint_dirs(checkpoint_root)
        if not ckpts:
            print(f"[eval_watch] 未发现 checkpoint-* 子目录: {checkpoint_root}", flush=True)
            return
        for ckpt_path in ckpts:
            try:
                process_one_checkpoint(
                    ckpt_path=ckpt_path,
                    checkpoint_root=checkpoint_root,
                    output_dir=output_dir,
                    state=state,
                    state_path=state_path,
                    best_path=best_path,
                    history_path=history_path,
                    delete_non_best=not args.no_delete_non_best,
                    python_exe=args.python,
                    infer_script=infer_script,
                    cal_meta_script=cal_meta_script,
                    cal_ade_rfs_script=cal_ade_rfs_path,
                    input_jsonl=args.input_jsonl,
                    label_jsonl=args.label_jsonl,
                    meta_jsonl=args.meta_jsonl,
                    ade_scenario_json=args.ade_scenario_json,
                    rfs_frequency=args.rfs_frequency,
                    rfs_length_seconds=args.rfs_length_seconds,
                    rfs_score_th=args.rfs_score_th,
                    infer_prefix=args.infer_prefix,
                    batch_size=args.batch_size,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    tensor_parallel_size=args.tensor_parallel_size,
                    gpu_memory_utilization=args.gpu_memory_utilization,
                    max_samples=args.max_samples,
                    repeat_num=args.repeat_num,
                    repeat_seed_base=args.repeat_seed_base,
                    include_jsonl=args.include_jsonl,
                    exclude_jsonl=args.exclude_jsonl,
                )
            except subprocess.CalledProcessError as e:
                print(
                    f"[eval_watch] [ERROR] {ckpt_path.name} 子进程失败 "
                    f"(exit={e.returncode})，不写入 state，稍后重试。",
                    flush=True,
                )
            except Exception as e:
                print(f"[eval_watch] [ERROR] {ckpt_path.name}: {e}", flush=True)
                traceback.print_exc()

        if not args.no_delete_non_best:
            prune_evaluated_non_best_checkpoints(
                checkpoint_root=checkpoint_root,
                best_path=best_path,
                state=load_state(state_path),
            )

    if args.once:
        scan_loop()
        return

    print(
        f"[eval_watch] 监控 {checkpoint_root} -> 输出 {output_dir} "
        f"（轮询 {args.poll_interval_sec}s，Ctrl+C 退出）",
        flush=True,
    )
    while True:
        scan_loop()
        time.sleep(args.poll_interval_sec)


if __name__ == "__main__":
    main()
