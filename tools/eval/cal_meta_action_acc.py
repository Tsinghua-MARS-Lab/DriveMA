#!/usr/bin/env python3
"""
评估 meta action 准确率：推理 jsonl + 标注 jsonl。
纵向：stop 与 wait 等价（含标注里的 "stop, wait"），其余须字符串一致（忽略大小写与首尾空白）。
横向：与标注脚本一致的三组候选，若二者同属于某一组则判对（并集规则）。
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# 与 LongTail_synthesizer_V2/v2/labeler/run_vllm_infer_label_meta_action.py 一致
_LAT_STRAIGHT = frozenset(
    {"straight", "lane_follow", "left_shift_slightly", "right_shift_slightly"}
)
_LAT_LEFT = frozenset(
    {"left_turn", "left_lane_change", "left_shift_slightly", "lane_follow", "turn_around"}
)
_LAT_RIGHT = frozenset(
    {"right_turn", "right_lane_change", "right_shift_slightly", "lane_follow"}
)
_LAT_GROUPS: Tuple[frozenset, ...] = (_LAT_STRAIGHT, _LAT_LEFT, _LAT_RIGHT)

_THINK_END_TAG = "</redacted_thinking>"

# 纵向与横向之间可能是分号或逗号（推理侧常见「..., lateral action:」）
_DECISION_RE = re.compile(
    r"<decision>\s*longitudinal\s+action:\s*(.+?)(?:[,;])\s*lateral\s+action:\s*([^<]+?)\s*</decision>",
    re.IGNORECASE | re.DOTALL,
)

# 无 <decision> 包裹：longitudinal action: ...; lateral action: ...（在 think 后缀中匹配，避免 CoT 内误命中）
_PLAIN_ACTION_RE = re.compile(
    r"longitudinal\s+action:\s*(.+?)\s*[,;]\s*lateral\s+action:\s*([^\n<]+)",
    re.IGNORECASE | re.DOTALL,
)


def _norm_longitudinal(s: str) -> str:
    """逗号视作分隔符，再压成单一下划线 token 序列，便于非 stop/wait 的严格相等。"""
    t = (s or "").strip().lower().replace(",", " ")
    t = " ".join(t.split())
    return t.replace(" ", "_")


def _parse_lateral_name(raw: str) -> str:
    """模型输出可能是 lane follow / lane_follow，统一为下划线形式。"""
    t = (raw or "").strip().lower()
    t = re.sub(r"\s+", "_", t)
    return t


def longitudinal_match(pred: str, ref: str) -> bool:
    if not (pred or "").strip() or not (ref or "").strip():
        return False

    def _is_stop_wait(x: str) -> bool:
        t = (x or "").strip().lower().replace(",", " ")
        parts = [a for a in t.split() if a]
        if not parts:
            return False
        sw = {"stop", "wait"}
        return all(a in sw for a in parts)

    if _is_stop_wait(pred) and _is_stop_wait(ref):
        return True
    return _norm_longitudinal(pred) == _norm_longitudinal(ref)


def lateral_match(pred: str, ref: str) -> bool:
    p = _parse_lateral_name(pred)
    q = _parse_lateral_name(ref)
    if not p or not q:
        return False
    if p == q:
        return True
    return any(p in g and q in g for g in _LAT_GROUPS)


def _strip_markdown_json_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def _json_after_think(text: str) -> str:
    if _THINK_END_TAG in text:
        return text.split(_THINK_END_TAG, 1)[1].strip()
    return text.strip()


def parse_actions_from_generation(generation: str) -> Tuple[str, str]:
    """
    从 generation 解析 longitudinal_action / lateral_action。
    优先 <decision>...</decision>，其次 think 后缀里的
    「longitudinal action: ...; lateral action: ...」，否则尝试 JSON（含 think 后缀）。
    """
    if not generation or not isinstance(generation, str):
        return "", ""

    m = _DECISION_RE.search(generation)
    if m:
        long_a = m.group(1).strip()
        lat_a = m.group(2).strip()
        return long_a, lat_a

    tail = _json_after_think(generation)
    m = _PLAIN_ACTION_RE.search(tail)
    if m:
        return m.group(1).strip(), m.group(2).strip()

    text = _strip_markdown_json_fence(tail)
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            lo = str(obj.get("longitudinal_action", "") or "").strip()
            la = str(obj.get("lateral_action", "") or "").strip()
            return lo, la
    except json.JSONDecodeError:
        pass

    # Fallback：纯「纵向, 横向」两词，如 "decelerate, straight"（取第一个逗号前后整段，strip 后作两路动作）
    s = tail.strip()
    if (
        "," in s
        and not s.startswith("{")
        and "longitudinal action" not in s.lower()
    ):
        left, right = s.split(",", 1)
        left, right = left.strip(), right.strip()
        if left and right:
            return left, right

    return "", ""


def load_label_jsonl(path: Path) -> Dict[str, Dict[str, str]]:
    """sample_id -> {longitudinal_action, lateral_action}"""
    out: Dict[str, Dict[str, str]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no} JSON 解析失败: {e}") from e
            sid = str(row.get("sample_id", "") or "").strip()
            if not sid:
                continue
            out[sid] = {
                "longitudinal_action": str(row.get("longitudinal_action", "") or "").strip(),
                "lateral_action": str(row.get("lateral_action", "") or "").strip(),
            }
    return out


def evaluate(
    infer_path: Path,
    label_path: Path,
    no_consistency_jsonl: Optional[Path] = None,
) -> Dict[str, Any]:
    labels = load_label_jsonl(label_path)

    total = 0
    missing_label = 0
    bad_parse = 0
    long_correct = 0
    lat_correct = 0
    both_correct = 0
    no_consistency_rows = 0

    out_fp = None
    if no_consistency_jsonl is not None:
        no_consistency_jsonl.parent.mkdir(parents=True, exist_ok=True)
        out_fp = open(no_consistency_jsonl, "w", encoding="utf-8")

    try:
        with open(infer_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"{infer_path}:{line_no} JSON 解析失败: {e}") from e

                sid = str(row.get("sample_id", "") or "").strip()
                if not sid:
                    continue

                ref = labels.get(sid)
                if ref is None:
                    missing_label += 1
                    continue

                p_long = str(row.get("longitudinal_action", "") or "").strip()
                p_lat = str(row.get("lateral_action", "") or "").strip()
                if not p_long and not p_lat:
                    p_long, p_lat = parse_actions_from_generation(
                        str(row.get("generation", "") or "")
                    )

                if not p_long or not p_lat:
                    bad_parse += 1
                    continue

                r_long = ref["longitudinal_action"]
                r_lat = ref["lateral_action"]
                if not r_long or not r_lat:
                    bad_parse += 1
                    continue

                total += 1
                ok_l = longitudinal_match(p_long, r_long)
                ok_t = lateral_match(p_lat, r_lat)
                if ok_l:
                    long_correct += 1
                if ok_t:
                    lat_correct += 1
                if ok_l and ok_t:
                    both_correct += 1
                else:
                    no_consistency_rows += 1
                    if out_fp is not None:
                        out_fp.write(
                            json.dumps(
                                {
                                    "sample_id": sid,
                                    "pred_longitudinal_action": p_long,
                                    "gt_longitudinal_action": r_long,
                                    "pred_lateral_action": p_lat,
                                    "gt_lateral_action": r_lat,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
    finally:
        if out_fp is not None:
            out_fp.close()

    def _rate(num: int, den: int) -> Optional[float]:
        if den <= 0:
            return None
        return num / den

    return {
        "infer_jsonl": str(infer_path),
        "label_jsonl": str(label_path),
        "labeled_ids": len(labels),
        "evaluated_pairs": total,
        "missing_label_rows": missing_label,
        "parse_or_empty_ref_skipped": bad_parse,
        "longitudinal_acc": _rate(long_correct, total),
        "lateral_acc": _rate(lat_correct, total),
        "joint_acc": _rate(both_correct, total),
        "longitudinal_correct": long_correct,
        "lateral_correct": lat_correct,
        "joint_correct": both_correct,
        "no_consistency_rows": no_consistency_rows,
        "no_consistency_jsonl": str(no_consistency_jsonl) if no_consistency_jsonl else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="计算 meta action 准确率（推理 jsonl + 标注 jsonl）")
    parser.add_argument(
        "--infer_jsonl",
        type=str,
        default="/cephfs/zhengwc/FluidDrive/ms-swift-3.5/output/Qwen3.5-2B/Mutil-Turn-w-mask-subsample/Second-Turn/v0-20260413-205852/checkpoint-950-5000_joint_random-validation.jsonl",
        help="推理结果 JSONL（含 generation 或 longitudinal_action/lateral_action）",
    )
    parser.add_argument(
        "--label_jsonl",
        type=str,
        default="/cephfs/zhengwc/LongTail_synthesizer_V2/v2/data/new-labels/5000_val_samples_lat_action_labels.jsonl",
        help="标注 JSONL（每行含 sample_id, longitudinal_action, lateral_action）",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="可选：将指标写入 JSON 文件",
    )
    parser.add_argument(
        "--no_consistency_jsonl",
        type=str,
        default=None,
        help="可选：将不一致样本写入 JSONL（sample_id + pred/gt 纵横向动作）",
    )
    args = parser.parse_args()

    infer_path = Path(args.infer_jsonl)
    label_path = Path(args.label_jsonl)
    if not infer_path.is_file():
        print(f"[ERROR] 推理文件不存在: {infer_path}")
        return
    if not label_path.is_file():
        print(f"[ERROR] 标注文件不存在: {label_path}")
        return

    no_consistency_path = Path(args.no_consistency_jsonl) if args.no_consistency_jsonl else None
    stats = evaluate(infer_path, label_path, no_consistency_path)

    print("=" * 60)
    print("Meta action 准确率")
    print("=" * 60)
    print(f"标注条数 (sample_id): {stats['labeled_ids']}")
    print(f"参与评估的推理行数: {stats['evaluated_pairs']}")
    print(f"无标注匹配的推理行: {stats['missing_label_rows']}")
    print(f"解析失败或标注动作为空跳过: {stats['parse_or_empty_ref_skipped']}")
    print(f"不一致样本数: {stats['no_consistency_rows']}")
    print()
    for name, key in [
        ("纵向", "longitudinal_acc"),
        ("横向", "lateral_acc"),
        ("联合", "joint_acc"),
    ]:
        v = stats[key]
        if v is None:
            print(f"{name}准确率: N/A（无有效样本）")
        else:
            print(f"{name}准确率: {v * 100:.2f}%")
    print("=" * 60)
    if stats.get("no_consistency_jsonl"):
        print(f"不一致样本已写入: {stats['no_consistency_jsonl']}")

    if args.output_file:
        outp = Path(args.output_file)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with open(outp, "w", encoding="utf-8") as wf:
            json.dump(stats, wf, indent=2, ensure_ascii=False)
        print(f"\n[INFO] 已写入: {outp}")


if __name__ == "__main__":
    main()
