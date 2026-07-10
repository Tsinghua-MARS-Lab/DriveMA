#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert trajectory jsonl into LLaMA-Factory multimodal image dataset (sharegpt style).

Input: jsonl, each line is a JSON object (or a Python dict literal) containing fields like:
  - images: dict view_name -> image_path
  - intent: str
  - past_traj: list[[x, y], ...]
  - past_accel: list[[x, y], ...]
  - past_vel: list[[x, y], ...]
  - future_traj: list[[x, y, ...], ...]  (we will use x,y only)

Output: jsonl, each line is an object:
  {
    "conversations": [
      {"from":"human","value":...},   # turn 1: context + <image> x3, task: output K-stage longitudinal decision
      {"from":"gpt","value":...},     # decision only
      {"from":"human","value":...},   # turn 2: predict trajectory
      {"from":"gpt","value":...},     # <answer>...</answer> only
    ],
    "images": ["path1", "path2", "path3"],
    "sample_id": ...
  }
where the number of "<image>" tokens in conversations equals len(images) (only in the first human turn).
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from typing import Any, Iterable, List, Sequence


DEFAULT_VIEW_KEYS = ("front_left", "front", "front_right")
DEFAULT_FUTURE_INDICES = (3, 7, 11, 15, 19)
MASK_NULL = "null"
MASK_NORMAL_THRESHOLD = 0.7
MASK_INTENT_THRESHOLD = 0.82
MASK_PAST_MOTION_THRESHOLD = 0.92


def _mask_group_for_sample(sample_id: Any, seed: int) -> str:
    """Stable 70/12/10/8 split keyed by sample_id."""
    key = f"{seed}:{sample_id}".encode("utf-8")
    digest = hashlib.blake2s(key, digest_size=8).digest()
    bucket = int.from_bytes(digest, "big") / float(1 << 64)
    if bucket < MASK_NORMAL_THRESHOLD:
        return "normal"
    if bucket < MASK_INTENT_THRESHOLD:
        return "mask_intent"
    if bucket < MASK_PAST_MOTION_THRESHOLD:
        return "mask_past_vel_accel"
    return "mask_core_motion"


def _parse_json_or_py_dict(line: str) -> dict[str, Any]:
    line = line.strip()
    if not line:
        raise ValueError("empty line")
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        # Some datasets are dumped as python dict literal with single quotes.
        obj = ast.literal_eval(line)
    if not isinstance(obj, dict):
        raise TypeError(f"expected dict per line, got {type(obj)}")
    return obj  # type: ignore[return-value]


def _fmt_num(x: Any) -> str:
    value = float(x)
    if round(value, 2) == 0:
        value = 0.0
    return f"{value:.2f}"


def _fmt_xy(pt: Sequence[Any]) -> str:
    if not isinstance(pt, (list, tuple)) or len(pt) < 2:
        raise ValueError(f"invalid point (need at least 2 dims): {pt!r}")
    return f"[{_fmt_num(pt[0])}, {_fmt_num(pt[1])}]"


def _fmt_traj(points: Iterable[Sequence[Any]]) -> str:
    return ", ".join(_fmt_xy(p) for p in points)


def _build_first_turn_prompt(
    intent: str,
    past_traj_text: str,
    past_accel_text: str,
    past_vel_text: str,
    current_vel_text: str,
    current_accel_text: str,
    longitudinal_stage_desc: str,
    decision_output_format: str,
) -> str:
    # Keep this prompt in raw text (no markdown).
    return (
        "You are an expert driver.\n"
        "Input:\n"
        "- 1 frame of multi-view images collected from the ego-vehicle at the present timestep: "
        "front_left_view: <image>; front_view:<image>; front_right_view:<image>\n"
        f"- Current high-level intent:{intent}\n"
        f"- Current acceleration is {current_accel_text}\n"
        f"- Current velocity is {current_vel_text}\n"
        f"- 4-second past trajectory (16 steps at 4 Hz):{past_traj_text}\n"
        f"- 4-second past acceleration (16 steps at 4 Hz):{past_accel_text}\n"
        f"- 4-second past velocity (16 steps at 4 Hz):{past_vel_text}\n"
        "Coordinate System Definition: X-axis: positive forward, negative backward; Y-axis: positive left, negative right.\n"
        f"Task: Inspect the input, make the driving decision. Predict longitudinal actions for {longitudinal_stage_desc}. Predict one lateral action for the whole 5-second horizon.\n"
        "Output format:\n"
        f"{decision_output_format}"
    )


def _build_second_turn_user_prompt() -> str:
    return (
        "Task: Given the above information, predict the optimal 5-second future trajectory (5 steps at 1 Hz) of the ego vehicle.\n"
        "Output format:\n"
        "[x_1, y_1], [x_2, y_2], [x_3, y_3], [x_4, y_4], [x_5, y_5]"
    )


def _build_decision_only_answer(decision: str | None) -> str:
    """Turn 1 assistant: decision text from meta-action jsonl."""
    decision_content = decision if decision is not None else ""
    return f"{decision_content}"


def _build_answer_only(
    future_traj: Sequence[Sequence[Any]],
    indices: Sequence[int],
) -> str:
    """Turn 2 assistant: only <answer>...</answer> (future trajectory)."""
    pts = []
    for idx in indices:
        if idx < 0 or idx >= len(future_traj):
            raise IndexError(
                f"future_traj length {len(future_traj)} is too short for index {idx}"
            )
        pts.append(_fmt_xy(future_traj[idx]))
    return ", ".join(pts)


def _split_csv(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _field_to_str(obj: dict[str, Any], key: str) -> str:
    v = obj.get(key)
    if v is None:
        return ""
    if isinstance(v, str):
        return v.strip()
    return str(v).strip()


def _first_non_empty_field(obj: dict[str, Any], keys: Sequence[str]) -> str:
    for key in keys:
        value = _field_to_str(obj, key)
        if value:
            return value
    return ""


def _format_time_seconds(value: Any) -> str:
    x = float(value)
    if abs(x - round(x)) < 1e-9:
        return f"{int(round(x))}s"
    return f"{x:g}s"


def _parse_phase_ranges(raw: str) -> list[list[float]]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if len(values) < 2:
        raise ValueError("--phase-ranges-seconds needs at least two comma-separated times")
    if abs(values[0]) > 1e-9:
        raise ValueError("--phase-ranges-seconds must start at 0")
    for prev, curr in zip(values[:-1], values[1:]):
        if curr <= prev:
            raise ValueError("--phase-ranges-seconds must be strictly increasing")
    return [[values[i], values[i + 1]] for i in range(len(values) - 1)]


def _phase_ranges_from_obj(obj: dict[str, Any]) -> list[list[float]]:
    value = obj.get("rule_temporal_longitudinal_phase_ranges_seconds")
    if not isinstance(value, list):
        return []
    ranges: list[list[float]] = []
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            return []
        ranges.append([float(item[0]), float(item[1])])
    return ranges


def _stage_description(phase_ranges: Sequence[Sequence[float]], num_stages: int) -> str:
    if phase_ranges:
        parts = [
            f"stage {idx + 1} [{_format_time_seconds(start)}, {_format_time_seconds(end)}]"
            for idx, (start, end) in enumerate(phase_ranges)
        ]
        return f"{len(parts)} future temporal stages: " + ", ".join(parts)
    return f"{num_stages} future temporal stages"


def _decision_output_format(num_stages: int) -> str:
    lon_parts = [f"lon{idx}: xx" for idx in range(1, num_stages + 1)]
    return ", ".join(lon_parts + ["lat: xx"])


def _output_decision(actions: Sequence[str], lat: str) -> str:
    lon_parts = [f"lon{idx}: {action}" for idx, action in enumerate(actions, start=1)]
    return ", ".join(lon_parts + [f"lat: {lat}"])


def _split_temporal_action(text: str) -> list[str]:
    if not text:
        return []
    if "->" in text:
        return [item.strip() for item in text.split("->") if item.strip()]
    if "," in text:
        return [item.strip() for item in text.split(",") if item.strip()]
    return []


def _longitudinal_actions_from_meta_action_obj(
    obj: dict[str, Any],
    num_stages: int | None,
) -> list[str]:
    actions = obj.get("rule_temporal_longitudinal_actions")
    if isinstance(actions, list):
        parsed = [str(action).strip() for action in actions if str(action).strip()]
        if parsed and (num_stages is None or len(parsed) == num_stages):
            return parsed

    parsed = _split_temporal_action(_field_to_str(obj, "rule_temporal_longitudinal_action"))
    if parsed and (num_stages is None or len(parsed) == num_stages):
        return parsed

    if num_stages is None:
        phase_ranges = _phase_ranges_from_obj(obj)
        if phase_ranges:
            num_stages = len(phase_ranges)
        else:
            num_stages = 0
            for phase_idx in range(1, 9):
                if _field_to_str(obj, f"rule_longitudinal_action_phase{phase_idx}"):
                    num_stages = phase_idx
                    continue
                granularity = _field_to_str(obj, "rule_temporal_longitudinal_granularity")
                if granularity and _field_to_str(
                    obj, f"rule_{granularity}_longitudinal_action_phase{phase_idx}"
                ):
                    num_stages = phase_idx

    fields: list[str] = []
    granularity = _field_to_str(obj, "rule_temporal_longitudinal_granularity")
    for phase_idx in range(1, (num_stages or 0) + 1):
        fields.append(
            _first_non_empty_field(
                obj,
                (
                    f"rule_longitudinal_action_phase{phase_idx}",
                    f"rule_{granularity}_longitudinal_action_phase{phase_idx}",
                    f"longitudinal_action_phase{phase_idx}",
                    f"phase{phase_idx}_longitudinal_action",
                ),
            )
        )
    if fields and all(fields):
        return fields
    return []


def _decision_from_meta_action_obj(obj: dict[str, Any], num_stages: int | None) -> str:
    """Build decision text from K-stage longitudinal fields and one-stage lateral field."""
    lon_actions = _longitudinal_actions_from_meta_action_obj(obj, num_stages)
    lat = _field_to_str(obj, "lateral_action")

    if lon_actions and lat:
        return _output_decision(lon_actions, lat)

    decision = _field_to_str(obj, "decision")
    if decision:
        return decision

    lon = _field_to_str(obj, "longitudinal_action")
    if lon and lat:
        repeat = num_stages or 2
        return _output_decision([lon] * repeat, lat)
    if not lon_actions and not lon and not lat:
        return ""
    if lon_actions:
        return _output_decision(lon_actions, lat)
    return _output_decision([lon] * (num_stages or 2), lat)


def _build_decision_content_from_meta_action_obj(
    obj: dict[str, Any],
    num_stages: int | None,
) -> str | None:
    """Decision text from K-stage longitudinal fields plus one-stage lateral field."""
    perception = _field_to_str(obj, "perception")
    reasoning = _field_to_str(obj, "reasoning")
    decision = _decision_from_meta_action_obj(obj, num_stages)
    if not perception and not reasoning and not decision:
        return None
    return f"{decision}"


def _load_sample_ids_txt(path: str) -> set[str]:
    allowed: set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                allowed.add(s)
    return allowed


def _load_lateral_action_map(path: str) -> dict[str, str]:
    lateral_map: dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                sys.stderr.write(
                    f"[lateral_jsonl] failed to parse line {line_no}: {e}, content={line[:200]}\n"
                )
                continue
            sid = obj.get("sample_id", None)
            if sid is None:
                sys.stderr.write(
                    f"[lateral_jsonl] missing sample_id at line {line_no}, obj={obj}\n"
                )
                continue
            lat = _field_to_str(obj, "lateral_action")
            if lat:
                lateral_map[str(sid)] = lat
    return lateral_map


def _infer_num_stages_from_obj(obj: dict[str, Any]) -> int:
    actions = _longitudinal_actions_from_meta_action_obj(obj, num_stages=None)
    if actions:
        return len(actions)
    ranges = _phase_ranges_from_obj(obj)
    if ranges:
        return len(ranges)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert trajectory jsonl to 2-turn multimodal jsonl: turn1 K-stage longitudinal decision, turn2 trajectory answer."
    )
    parser.add_argument("--input", default='/cephfs/zhengwc/FluidDrive/ms-swift-3.5/data/train_data/meta_data/train_samples_filtered_20pc_downsample.jsonl', help="Input jsonl path.")
    parser.add_argument(
        "--output",
        "-o",
        default="/cephfs/zhengwc/FluidDrive/ms-swift-3.5/data/train_data/chunk_size_1s/SFT_last.jsonl",
        help="Output jsonl path.",
    )
    parser.add_argument(
        "--meta-action-jsonl",
        "--cot-jsonl",
        dest="meta_action_jsonl",
        type=str,
        default='/cephfs/zhengwc/LongTail_synthesizer_V2/v2/data/step-ablation/train_samples_temporal_lon_chunk1s_rule.jsonl',
        help=(
            "Optional meta-action JSONL: sample_id; perception/reasoning (optional); "
            "rule_temporal_longitudinal_actions or rule_longitudinal_action_phase* "
            "for the decision block. --cot-jsonl is kept as a backward-compatible alias."
        ),
    )
    parser.add_argument(
        "--lateral-jsonl",
        type=str,
        default='/cephfs/zhengwc/LongTail_synthesizer_V2/v2/data/new-labels/train_samples_action_label_new.jsonl',
        help=(
            "Optional jsonl: sample_id+lateral_action. Used to fill lateral_action "
            "when --meta-action-jsonl only contains temporal longitudinal rule labels."
        ),
    )
    parser.add_argument(
        "--sample-ids",
        type=str,
        default=None,
        help="Optional txt file: one sample_id per line; only those samples are written to output.",
    )
    parser.add_argument(
        "--view-keys",
        default=",".join(DEFAULT_VIEW_KEYS),
        help="Comma-separated image view keys to use, in order. Default: front_left,front,front_right",
    )
    parser.add_argument(
        "--future-indices",
        default=",".join(map(str, DEFAULT_FUTURE_INDICES)),
        help="Comma-separated indices used to downsample future_traj. Default: 3,7,11,15,19",
    )
    parser.add_argument(
        "--conversations-key",
        default="conversations",
        help="Output conversations field name. Default: conversations",
    )
    parser.add_argument(
        "--skip-bad",
        action="store_true",
        help="Skip malformed lines instead of exiting.",
    )
    parser.add_argument(
        "--mask-seed",
        type=int,
        default=42,
        help=(
            "Seed for stable sample_id-based mask assignment. "
            "Default split: 70%% normal, 12%% mask intent, "
            "10%% mask past_vel+past_accel, 8%% mask past_traj+current_vel+current_accel."
        ),
    )
    parser.add_argument(
        "--num-stages",
        type=int,
        default=0,
        help="Number of longitudinal temporal stages. 0 means infer from --meta-action-jsonl.",
    )
    parser.add_argument(
        "--phase-ranges-seconds",
        default="",
        help=(
            "Optional comma-separated boundaries used in the prompt, e.g. '0,2,5' "
            "or '0,1.5,3.5,5'. Empty means infer from --meta-action-jsonl when available."
        ),
    )
    args = parser.parse_args(argv)
    if args.num_stages < 0:
        raise ValueError("--num-stages must be >= 0")

    phase_ranges: list[list[float]] = []
    if args.phase_ranges_seconds.strip():
        phase_ranges = _parse_phase_ranges(args.phase_ranges_seconds)
        if args.num_stages and len(phase_ranges) != args.num_stages:
            raise ValueError(
                f"--phase-ranges-seconds defines {len(phase_ranges)} stages, "
                f"but --num-stages={args.num_stages}"
            )
        if not args.num_stages:
            args.num_stages = len(phase_ranges)

    lateral_map: dict[str, str] = {}
    if args.lateral_jsonl:
        lateral_map = _load_lateral_action_map(args.lateral_jsonl)
        sys.stderr.write(
            f"loaded lateral_action map: {len(lateral_map)} samples from {args.lateral_jsonl}\n"
        )

    # Load meta-action mapping if provided.
    meta_action_map: dict[str, str] = {}
    inferred_phase_ranges: list[list[float]] = []
    inferred_num_stages = 0
    if args.meta_action_jsonl:
        with open(args.meta_action_jsonl, "r", encoding="utf-8") as f_meta_action:
            for line_no, line in enumerate(f_meta_action, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception as e:
                    sys.stderr.write(
                        f"[meta_action_jsonl] failed to parse line {line_no}: {e}, content={line[:200]}\n"
                    )
                    continue
                sid = obj.get("sample_id", None)
                if sid is None:
                    sys.stderr.write(
                        f"[meta_action_jsonl] missing sample_id at line {line_no}, obj={obj}\n"
                    )
                    continue
                if not inferred_phase_ranges:
                    inferred_phase_ranges = _phase_ranges_from_obj(obj)
                if not inferred_num_stages:
                    inferred_num_stages = _infer_num_stages_from_obj(obj)
                if not _field_to_str(obj, "lateral_action"):
                    lat = lateral_map.get(str(sid), "")
                    if lat:
                        obj["lateral_action"] = lat
                effective_num_stages = args.num_stages or inferred_num_stages or None
                decision = _build_decision_content_from_meta_action_obj(
                    obj, effective_num_stages
                )
                if decision is None:
                    sys.stderr.write(
                        f"[meta_action_jsonl] empty perception/reasoning/decision/k_stage_lon_lat at line {line_no}, obj={obj}\n"
                    )
                    continue
                meta_action_map[str(sid)] = decision

    if not phase_ranges and inferred_phase_ranges:
        phase_ranges = inferred_phase_ranges
    if not args.num_stages and inferred_num_stages:
        args.num_stages = inferred_num_stages
    if not args.num_stages:
        args.num_stages = 2
    if phase_ranges and len(phase_ranges) != args.num_stages:
        raise ValueError(
            f"inferred/provided phase ranges define {len(phase_ranges)} stages, "
            f"but num_stages={args.num_stages}"
        )
    longitudinal_stage_desc = _stage_description(phase_ranges, args.num_stages)
    decision_output_format = _decision_output_format(args.num_stages)
    sys.stderr.write(
        "decision schema: "
        f"num_stages={args.num_stages}, phase_ranges={phase_ranges or 'N/A'}, "
        f"output_format={decision_output_format}\n"
    )

    allowed_ids: set[str] | None = None
    if args.sample_ids:
        allowed_ids = _load_sample_ids_txt(args.sample_ids)
        if not allowed_ids:
            raise ValueError(f"--sample-ids file is empty or has no valid lines: {args.sample_ids}")

    view_keys = _split_csv(args.view_keys)
    future_indices = [int(x) for x in _split_csv(args.future_indices)]
    if not view_keys:
        raise ValueError("--view-keys cannot be empty")
    if not future_indices:
        raise ValueError("--future-indices cannot be empty")

    bad = 0
    total = 0
    written = 0
    mask_counts = {
        "normal": 0,
        "mask_intent": 0,
        "mask_past_vel_accel": 0,
        "mask_core_motion": 0,
    }

    with open(args.input, "r", encoding="utf-8") as fin, open(args.output, "w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, start=1):
            if not line.strip():
                continue
            total += 1
            try:
                sample = _parse_json_or_py_dict(line)
                sample_id_preview = sample.get("sample_id", line_no)
                if allowed_ids is not None and str(sample_id_preview) not in allowed_ids:
                    continue
                images_dict = sample.get("images", {})
                if not isinstance(images_dict, dict):
                    raise TypeError(f"images must be a dict, got {type(images_dict)}")
                image_paths = []
                for k in view_keys:
                    p = images_dict.get(k, None)
                    if not isinstance(p, str) or not p:
                        raise KeyError(f"missing images[{k!r}]")
                    image_paths.append(p)

                intent = str(sample.get("intent", ""))
                past_traj = sample.get("past_traj", None)
                past_accel = sample.get("past_accel", None)
                past_vel = sample.get("past_vel", None)
                current_vel = sample.get("current_vel", None)
                current_accel = sample.get("current_accel", None)
                future_traj = sample.get("future_traj", None)
                if not isinstance(past_traj, list):
                    raise TypeError(f"past_traj must be a list, got {type(past_traj)}")
                if not isinstance(past_accel, list):
                    raise TypeError(f"past_accel must be a list, got {type(past_accel)}")
                if not isinstance(past_vel, list):
                    raise TypeError(f"past_vel must be a list, got {type(past_vel)}")
                if not isinstance(current_vel, (list, tuple)) or len(current_vel) < 2:
                    raise TypeError(f"current_vel must be a list/tuple with >=2 dims, got {current_vel!r}")
                if not isinstance(current_accel, (list, tuple)) or len(current_accel) < 2:
                    raise TypeError(f"current_accel must be a list/tuple with >=2 dims, got {current_accel!r}")
                if not isinstance(future_traj, list):
                    raise TypeError(f"future_traj must be a list, got {type(future_traj)}")

                past_traj_text = _fmt_traj(past_traj)
                past_accel_text = _fmt_traj(past_accel)
                past_vel_text = _fmt_traj(past_vel)
                current_vel_text = _fmt_xy(current_vel)
                current_accel_text = _fmt_xy(current_accel)
                sample_id = sample.get("sample_id", line_no)
                mask_group = _mask_group_for_sample(sample_id, args.mask_seed)
                mask_counts[mask_group] += 1
                if mask_group == "mask_intent":
                    intent = MASK_NULL
                elif mask_group == "mask_past_vel_accel":
                    past_vel_text = MASK_NULL
                    past_accel_text = MASK_NULL
                elif mask_group == "mask_core_motion":
                    past_traj_text = MASK_NULL
                    current_vel_text = MASK_NULL
                    current_accel_text = MASK_NULL

                first_turn_user = _build_first_turn_prompt(
                    intent=intent,
                    past_traj_text=past_traj_text,
                    past_accel_text=past_accel_text,
                    past_vel_text=past_vel_text,
                    current_vel_text=current_vel_text,
                    current_accel_text=current_accel_text,
                    longitudinal_stage_desc=longitudinal_stage_desc,
                    decision_output_format=decision_output_format,
                )
                second_turn_user = _build_second_turn_user_prompt()

                # 查找对应 sample_id 的 meta-action decision
                meta_action_decision = None
                key_str = str(sample_id)
                if meta_action_map:
                    meta_action_decision = meta_action_map.get(key_str, None)
                else:
                    meta_action_decision = None

                first_turn_gpt = _build_decision_only_answer(meta_action_decision)
                if future_traj is not None and len(future_traj) > 0:
                    second_turn_gpt = _build_answer_only(
                        future_traj=future_traj,
                        indices=future_indices,
                    )
                else:
                    second_turn_gpt = ""
                out_obj: dict[str, Any] = {
                    args.conversations_key: [
                        {"from": "human", "value": first_turn_user},
                        {"from": "gpt", "value": first_turn_gpt},
                        {"from": "human", "value": second_turn_user},
                        {"from": "gpt", "value": second_turn_gpt},
                    ],
                    "images": image_paths,
                    "sample_id": sample_id,
                }

                fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
                written += 1
            except Exception as e:
                bad += 1
                if args.skip_bad:
                    sys.stderr.write(f"[skip] line {line_no}: {e}\n")
                    continue
                raise RuntimeError(f"failed at line {line_no}: {e}") from e

    sys.stderr.write(
        f"done. total_lines={total}, written={written}, bad_lines={bad}, output={args.output}\n"
    )
    sys.stderr.write(f"mask_counts={json.dumps(mask_counts, ensure_ascii=False)}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
