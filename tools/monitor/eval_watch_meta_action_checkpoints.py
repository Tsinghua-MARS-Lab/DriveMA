#!/usr/bin/env python3
"""
离线监控训练输出目录下的 checkpoint-*：权重就绪后依次调用
run_vllm_infer_meta_action_wo_cot.py 与 cal_meta_action_acc.py，
将指标落盘并维护 best（默认按 joint_acc）。

典型用法（训练每 100 step 存一次，持续往同一 checkpoint_root 写）：

  python eval_watch_meta_action_checkpoints.py \\
    --checkpoint_root /path/to/v1-xxxxxx \\
    --input_jsonl /path/val_samples_479.jsonl \\
    --longitudinal_label_jsonl /path/val_lon_labels.jsonl \\
    --lateral_label_jsonl /path/val_lat_labels.jsonl \\
    --output_dir /path/to/eval_runs/run1

说明：
- 就绪条件：存在 config.json，且至少有一份权重（*.safetensors 或 pytorch_model.bin 等）。
- 已完成的 checkpoint 记录在 output_dir/eval_state.json，避免重复跑。
- best 写在 output_dir/best_meta_action.json；历史追加在 output_dir/history.jsonl。
- 默认在评测完成后删除非 best 的 checkpoint-* 目录（出现新 best 时删除旧 best）；每轮扫描末也会按 state+best 清理已评测非 best（含「已在 state 中」早退的情形）。可用 --no_delete_non_best 关闭。
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
DEFAULT_INFER = "/cephfs/zhengwc/FluidDrive/ms-swift-3.5/tools/infer_scripts/run_vllm_infer_meta_action_w_cot.py"
DEFAULT_CAL = "/cephfs/zhengwc/FluidDrive/ms-swift-3.5/tools/eval-one-step/cal_meta_action_acc.py"
ROOT = "/cephfs/zhengwc/FluidDrive/ms-swift-3.5/output/Qwen3.5-4B/Pretrained-new-cot/v1-20260502-092939"
INPUT_JSONL = "/cephfs/zhengwc/FluidDrive/ms-swift-3.5/data/train_data/meta_data/val_samples_434.jsonl"
LABEL_JSONL = "/cephfs/zhengwc/LongTail_synthesizer_V2/v2/data/new-labels/val_samples_479_meta_action_label.jsonl"
_CKPT_RE = re.compile(r"^checkpoint-(\d+)$", re.IGNORECASE)


@dataclass
class CheckpointRecord:
    checkpoint_name: str
    checkpoint_path: str
    infer_jsonl: str
    metrics_json: str
    joint_acc: Optional[float]
    lon1_acc: Optional[float]
    lon2_acc: Optional[float]
    longitudinal_acc: Optional[float]
    lateral_acc: Optional[float]
    evaluated_pairs: int
    finished_at: str


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def parse_checkpoint_step(name: str) -> Optional[int]:
    m = _CKPT_RE.match(name.strip())
    if not m:
        return None
    return int(m.group(1))


def list_checkpoint_dirs(root: Path) -> List[Path]:
    if not root.is_dir():
        return []
    out: List[Path] = []
    for p in root.iterdir():
        if not p.is_dir():
            continue
        if parse_checkpoint_step(p.name) is None:
            continue
        out.append(p)
    out.sort(key=lambda p: parse_checkpoint_step(p.name) or 0)
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


def run_cal_acc(
    *,
    python_exe: str,
    cal_script: Path,
    infer_jsonl: str,
    longitudinal_label_jsonl: str,
    lateral_label_jsonl: str,
    label_jsonl: str,
    metrics_out: str,
    no_consistency_jsonl: Optional[str],
) -> Dict[str, Any]:
    if longitudinal_label_jsonl is None:
            cmd = [
            python_exe,
            str(cal_script),
            "--infer_jsonl",
            infer_jsonl,
            "--output_file",
            metrics_out,
            "--label_jsonl",
            label_jsonl
        ]
    
    else:

        cmd = [
            python_exe,
            str(cal_script),
            "--infer_jsonl",
            infer_jsonl,
            "--longitudinal_label_jsonl",
            longitudinal_label_jsonl,
            "--lateral_label_jsonl",
            lateral_label_jsonl,
            "--output_file",
            metrics_out,
        ]

        
    if no_consistency_jsonl:
        cmd.extend(["--no_consistency_jsonl", no_consistency_jsonl])
    print(f"[eval_watch] 启动准确率: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    with open(metrics_out, "r", encoding="utf-8") as f:
        return json.load(f)


def pick_best_metric(stats: Dict[str, Any], key: str) -> Optional[float]:
    v = stats.get(key)
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


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
    best_name: Optional[str] = None
    if best_path.is_file():
        try:
            with open(best_path, "r", encoding="utf-8") as f:
                best_data = json.load(f)
            raw = best_data.get("best_checkpoint")
            if isinstance(raw, str):
                best_name = raw
        except (json.JSONDecodeError, OSError):
            pass
    if not best_name:
        return
    evaluated = state.get("checkpoints")
    if not isinstance(evaluated, dict):
        return
    for name in sorted(
        evaluated.keys(),
        key=lambda n: parse_checkpoint_step(n) or 0,
    ):
        if name == best_name:
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
    best_metric_key: str,
    stats: Dict[str, Any],
    record: CheckpointRecord,
) -> Tuple[bool, Optional[float], Optional[float]]:
    """若当前 joint_acc（或指定指标）优于历史 best，则写 best 文件。返回 (updated, old_best, new_best)。"""
    new_val = pick_best_metric(stats, best_metric_key)
    if new_val is None:
        return False, None, None

    old_best: Optional[float] = None
    if best_path.is_file():
        try:
            with open(best_path, "r", encoding="utf-8") as f:
                prev = json.load(f)
            old_best = pick_best_metric(prev, "best_value")
        except (json.JSONDecodeError, OSError):
            old_best = None

    if old_best is not None and new_val <= old_best:
        return False, old_best, new_val

    payload = {
        "best_metric": best_metric_key,
        "best_value": new_val,
        "best_checkpoint": record.checkpoint_name,
        "best_checkpoint_path": record.checkpoint_path,
        "best_infer_jsonl": record.infer_jsonl,
        "best_metrics_json": record.metrics_json,
        "full_metrics": stats,
        "updated_at": _now_iso(),
        "previous_best_value": old_best,
    }
    best_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = best_path.with_suffix(best_path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    tmp.replace(best_path)
    return True, old_best, new_val


def process_one_checkpoint(
    *,
    ckpt_path: Path,
    checkpoint_root: Path,
    output_dir: Path,
    state: Dict[str, Any],
    state_path: Path,
    best_path: Path,
    history_path: Path,
    best_metric_key: str,
    delete_non_best: bool,
    python_exe: str,
    infer_script: Path,
    cal_script: Path,
    input_jsonl: str,
    longitudinal_label_jsonl: str,
    lateral_label_jsonl: str,
    label_jsonl: str,
    write_no_consistency: bool,
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
    metrics_out = output_dir / "metrics" / f"{infer_prefix}{name}.json"
    no_consistency_out = (
        output_dir / "no_consistency" / f"{infer_prefix}{name}.jsonl"
        if write_no_consistency
        else None
    )

    checkpoints: Dict[str, Any] = state.setdefault("checkpoints", {})
    if name in checkpoints:
        print(f"[eval_watch] 跳过（已在 state 中）: {name}", flush=True)
        return

    # if not is_checkpoint_ready(ckpt_path):
    #     print(f"[eval_watch] 跳过（未就绪）: {ckpt_path}", flush=True)
    #     return

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


    stats = run_cal_acc(
        python_exe=python_exe,
        cal_script=cal_script,
        infer_jsonl=str(infer_out),
        longitudinal_label_jsonl=longitudinal_label_jsonl,
        lateral_label_jsonl=lateral_label_jsonl,
        label_jsonl = label_jsonl,
        metrics_out=str(metrics_out),
        no_consistency_jsonl=str(no_consistency_out) if no_consistency_out else None,
    )




    record = CheckpointRecord(
        checkpoint_name=name,
        checkpoint_path=str(ckpt_path.resolve()),
        infer_jsonl=str(infer_out.resolve()),
        metrics_json=str(metrics_out.resolve()),
        joint_acc=pick_best_metric(stats, "joint_acc"),
        lon1_acc=pick_best_metric(stats, "lon1_acc"),
        lon2_acc=pick_best_metric(stats, "lon2_acc"),
        longitudinal_acc=pick_best_metric(stats, "longitudinal_acc"),
        lateral_acc=pick_best_metric(stats, "lateral_acc"),
        evaluated_pairs=int(stats.get("evaluated_pairs") or 0),
        finished_at=_now_iso(),
    )

    checkpoints[name] = asdict(record)
    save_state(state_path, state)

    prev_best_ckpt_name: Optional[str] = None
    if best_path.is_file():
        try:
            with open(best_path, "r", encoding="utf-8") as f:
                prev_best = json.load(f)
            prev_best_ckpt_name = prev_best.get("best_checkpoint")
            if not isinstance(prev_best_ckpt_name, str):
                prev_best_ckpt_name = None
        except (json.JSONDecodeError, OSError):
            prev_best_ckpt_name = None

    updated, old_b, new_b = update_best_file(
        best_path, best_metric_key=best_metric_key, stats=stats, record=record
    )

    if delete_non_best:
        if updated:
            if (
                prev_best_ckpt_name
                and prev_best_ckpt_name != name
            ):
                delete_non_best_checkpoint_dir(
                    checkpoint_root,
                    prev_best_ckpt_name,
                    tag="旧 best ",
                )
        else:
            delete_non_best_checkpoint_dir(checkpoint_root, name)
    hist_row = {
        "event": "eval_done",
        "record": asdict(record),
        "best_metric": best_metric_key,
        "is_new_best": updated,
        "previous_best": old_b,
        "current_metric": new_b,
    }
    append_history(history_path, hist_row)

    if updated:
        print(
            f"[eval_watch] ★ 新 best ({best_metric_key}): {new_b} "
            f"(checkpoint={name}, 上次 best={old_b})",
            flush=True,
        )
    else:
        print(
            f"[eval_watch] 完成 {name}: {best_metric_key}={new_b} "
            f"(当前历史 best={old_b})",
            flush=True,
        )


def main() -> None:
    p = argparse.ArgumentParser(
        description="监控 checkpoint-* 目录，就绪后推理 + cal_meta_action_acc_two_stage，并记录 best"
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
        "--longitudinal_label_jsonl",
        type=str,
        default=None,
        help="纵向 two-steps 标注 jsonl",
    )
    p.add_argument(
        "--lateral_label_jsonl",
        type=str,
        default=None,
        help="two-steps 横向标注 jsonl",
    )
    p.add_argument(
        "--label_jsonl",
        type=str,
        default=LABEL_JSONL,
        help="one step时作为 longitudinal/lateral 标注 jsonl",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default=ROOT,
        help="推理输出、metrics、state、best 的根目录, 建议和checkpoint_root一个路径",
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
        help="推理脚本路径",
    )
    p.add_argument(
        "--cal_script",
        type=str,
        default=str(DEFAULT_CAL),
        help="准确率脚本路径",
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
        default=10,
        help="轮询间隔（秒）",
    )
    p.add_argument(
        "--once",
        action="store_true",
        help="只扫描一轮后退出（不持续监控）",
    )
    p.add_argument(
        "--best_metric",
        type=str,
        default="joint_acc",
        choices=("joint_acc", "longitudinal_acc", "lateral_acc", "lon1_acc", "lon2_acc"),
        help="用于比较 best 的指标字段（来自 cal_meta_action_acc_two_stage 的 JSON）",
    )
    p.add_argument(
        "--no_delete_non_best",
        action="store_true",
        help="评测完成后不删除非 best 的 checkpoint-* 目录（默认会删以省空间）",
    )

    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.01)
    p.add_argument("--max_tokens", type=int, default=2048)
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    p.add_argument("--max_samples", type=int, default=-1)
    p.add_argument("--repeat_num", type=int, default=5)
    p.add_argument("--repeat_seed_base", type=int, default=42)
    p.add_argument("--include_jsonl", type=str, default=None)
    p.add_argument("--exclude_jsonl", type=str, default=None)
    p.add_argument(
        "--write_no_consistency",
        action="store_true",
        help="额外写出每个 checkpoint 的不一致样本 JSONL",
    )

    args = p.parse_args()

    checkpoint_root = Path(args.checkpoint_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    state_path = output_dir / "eval_state.json"
    best_path = output_dir / "best_meta_action.json"
    history_path = output_dir / "history.jsonl"

    infer_script = Path(args.infer_script).resolve()
    cal_script = Path(args.cal_script).resolve()
    if not infer_script.is_file():
        raise SystemExit(f"推理脚本不存在: {infer_script}")
    if not cal_script.is_file():
        raise SystemExit(f"准确率脚本不存在: {cal_script}")

    longitudinal_label_jsonl = args.longitudinal_label_jsonl
    lateral_label_jsonl =  args.lateral_label_jsonl
    label_jsonl = args.label_jsonl

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
                    best_metric_key=args.best_metric,
                    delete_non_best=not args.no_delete_non_best,
                    python_exe=args.python,
                    infer_script=infer_script,
                    cal_script=cal_script,
                    input_jsonl=args.input_jsonl,
                    longitudinal_label_jsonl=longitudinal_label_jsonl,
                    lateral_label_jsonl=lateral_label_jsonl,
                    label_jsonl=label_jsonl,
                    write_no_consistency=args.write_no_consistency,
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
