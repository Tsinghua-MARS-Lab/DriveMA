#!/usr/bin/env python3
"""
计算 meta action 一致性（两种输入）：

1) 默认：<decision> 与 <answer> 轨迹；或 infer 多行 JSONL（generation / generation_turn1+2），
   decision 可为「longitudinal … ; lateral …」或「longitudinal … , lateral …」无标签格式
   1Hz 五点轨迹在 t=0 补零后线性插值为 4Hz（0.25s..5s）
   → 按 labeler 规则推断纵向 / 横向动作组
   → 纵向：keep/accelerate/decelerate 等与轨迹推断原子词集合须一致；轨迹为静止类
     （推断为 stop 与 wait 的组合）时，decision 只写 stop 或只写 wait 亦可
   → 横向：decision 中的 lateral 落在轨迹推断的逗号分隔行为组内即正确

2) --label_jsonl：仅含 <answer> 轨迹时，用 label 中每行的 sample_id 对应
   longitudinal_action / lateral_action，与轨迹推断结果按上述同一套 match 规则比较。
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# 以下与 examples/train/grpo/plugin/trajectory_reward.py 同源，避免 import swift
# ---------------------------------------------------------------------------

_FUTURE_TRAJ_KEY_RE = re.compile(
    r'["\']?\s*future_trajectory\s*["\']?\s*:\s*',
    re.IGNORECASE,
)


def _extract_first_future_trajectory_from_str(gen: str) -> Optional[np.ndarray]:
    m = _FUTURE_TRAJ_KEY_RE.search(gen)
    if not m:
        return None
    start_val = m.end()
    if start_val >= len(gen):
        return None
    while start_val < len(gen) and gen[start_val] in " \t\n\r":
        start_val += 1
    if start_val >= len(gen) or gen[start_val] != "[":
        return None
    depth = 0
    end_val = start_val
    for i in range(start_val, len(gen)):
        if gen[i] == "[":
            depth += 1
        elif gen[i] == "]":
            depth -= 1
            if depth == 0:
                end_val = i + 1
                break
    if depth != 0:
        return None
    try:
        arr = json.loads(gen[start_val:end_val])
    except Exception:
        return None
    if not isinstance(arr, list) or not arr:
        return None
    coords: List[List[float]] = []
    for pt in arr:
        if isinstance(pt, (list, tuple)) and len(pt) >= 2:
            try:
                coords.append([float(pt[0]), float(pt[1])])
            except (TypeError, ValueError):
                continue
    if not coords:
        return None
    return np.asarray(coords, dtype=np.float32)


def _parse_trajectory_to_xy(traj_input: Any) -> Optional[np.ndarray]:
    if traj_input is None:
        return None
    if isinstance(traj_input, dict):
        traj = traj_input.get("future_trajectory", None)
        if traj is None or not isinstance(traj, list) or not traj:
            return None
        coords: List[List[float]] = []
        for pt in traj:
            if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                try:
                    coords.append([float(pt[0]), float(pt[1])])
                except (TypeError, ValueError):
                    continue
        if not coords:
            return None
        return np.asarray(coords, dtype=np.float32)
    if isinstance(traj_input, str):
        xy = _extract_first_future_trajectory_from_str(traj_input)
        if xy is not None:
            return xy
        pattern = r"\[([-\d.eE+]+),\s*([-\d.eE+]+)\]"
        matches = re.findall(pattern, traj_input)
        coords = []
        for match in matches:
            try:
                coords.append([float(match[0]), float(match[1])])
            except Exception:
                continue
        if not coords:
            return None
        return np.asarray(coords, dtype=np.float32)
    return None


_META_FUTURE_TRAJ_DT = 0.25
_META_LONG_SPEED_THRESH_MPS = 0.5
_META_TRAJ_VALID_DISP_THRESH_M = 0.5
_META_LONG_ACCEL_THRESH_MS2 = 0.3
_META_TRAJ_SMOOTH_WINDOW_SIZE = 3
_META_LAT_HEADING_TOTAL_THRESH_DEG = 15.0
_META_LAT_LANE_CHANGE_DISP_THRESH_M = 1.5
_META_HEADING_SMOOTH_WINDOW = 3
_META_TRAJ_HEADING_MIN_SEGMENT_LEN_M = 1e-6
_META_TRAJ_NET_WRAP_REPLACE_MIN_ABS_DEG = 270.0
_META_TRAJ_NET_CHORD_REPLACE_MAX_ABS_DEG = 45.0
_META_LAT_LIST_STRAIGHT = (
    "straight, lane_follow, left_shift_slightly, right_shift_slightly"
)
_META_LAT_LIST_LEFT = (
    "left_turn, left_lane_change, left_shift_slightly, lane_follow, turn_around"
)
_META_LAT_LIST_RIGHT = "right_turn, right_lane_change, right_shift_slightly, lane_follow"
_META_LONG_ACTIONS_DEFAULT = "keep, accelerate, decelerate, stop"
_META_LAT_ACTIONS_DEFAULT = (
    "straight, left_turn, right_turn, lane_follow, lane_change_left, lane_change_right, reverse"
)


def _meta_traj_segment_speeds(traj: List[List[float]], dt: float) -> List[float]:
    speeds: List[float] = []
    for i in range(len(traj) - 1):
        x0, y0 = float(traj[i][0]), float(traj[i][1])
        x1, y1 = float(traj[i + 1][0]), float(traj[i + 1][1])
        dist = math.hypot(x1 - x0, y1 - y0)
        speeds.append(dist / dt)
    return speeds


def _meta_smooth_trajectory(traj: List[List[float]], window_size: int = 3) -> List[List[float]]:
    if window_size < 2:
        return traj
    if window_size % 2 == 0:
        window_size -= 1
    if len(traj) < window_size:
        return traj
    traj_arr = np.asarray(traj, dtype=np.float64)
    if traj_arr.ndim != 2 or traj_arr.shape[1] < 2:
        return traj
    kernel = np.ones(window_size, dtype=np.float64) / float(window_size)
    pad = window_size // 2
    x_pad = np.pad(traj_arr[:, 0], (pad, pad), mode="edge")
    y_pad = np.pad(traj_arr[:, 1], (pad, pad), mode="edge")
    x_smooth = np.convolve(x_pad, kernel, mode="valid")
    y_smooth = np.convolve(y_pad, kernel, mode="valid")
    if len(x_smooth) != len(traj_arr):
        return traj
    return [[float(x), float(y)] for x, y in zip(x_smooth, y_smooth)]


def _meta_infer_longitudinal_meta_str(future_traj: List[List[float]]) -> str:
    speeds = _meta_traj_segment_speeds(future_traj, _META_FUTURE_TRAJ_DT)
    if not speeds:
        return "keep"

    x0, y0 = float(future_traj[0][0]), float(future_traj[0][1])
    x1, y1 = float(future_traj[-1][0]), float(future_traj[-1][1])
    total_displacement = math.hypot(x1 - x0, y1 - y0)

    if (
        max(speeds) < _META_LONG_SPEED_THRESH_MPS
        and total_displacement < _META_TRAJ_VALID_DISP_THRESH_M
    ):
        return "stop, wait"

    if len(speeds) < 2:
        return "keep"

    t = np.arange(len(speeds), dtype=np.float64) * _META_FUTURE_TRAJ_DT
    a_b = np.linalg.lstsq(
        np.column_stack([t, np.ones(len(t))]),
        np.asarray(speeds, dtype=np.float64),
        rcond=None,
    )[0]
    slope = float(a_b[0])

    if slope > _META_LONG_ACCEL_THRESH_MS2:
        return "accelerate"
    if slope < -_META_LONG_ACCEL_THRESH_MS2:
        return "decelerate"
    return "keep"


def _meta_calc_trajectory_metrics(traj: List[List[float]]) -> Dict[str, Any]:
    traj_arr = np.asarray(traj, dtype=np.float64)
    if traj_arr.ndim != 2 or traj_arr.shape[0] < 2 or traj_arr.shape[1] < 2:
        return {"is_valid": False}
    traj_arr = traj_arr[:, :2]
    n_points = len(traj_arr)

    dx = traj_arr[1:, 0] - traj_arr[:-1, 0]
    dy = traj_arr[1:, 1] - traj_arr[:-1, 1]
    segment_lengths = np.hypot(dx, dy)

    total_travel_distance = float(np.sum(segment_lengths))
    total_displacement = float(
        np.hypot(traj_arr[-1, 0] - traj_arr[0, 0], traj_arr[-1, 1] - traj_arr[0, 1])
    )
    is_valid = (total_displacement >= _META_TRAJ_VALID_DISP_THRESH_M) or (
        total_travel_distance >= _META_TRAJ_VALID_DISP_THRESH_M
    )

    if not is_valid:
        return {
            "is_valid": False,
            "total_displacement": total_displacement,
            "total_travel_distance": total_travel_distance,
        }

    def _segment_heading_rad(i: int, prev: float) -> float:
        if segment_lengths[i] < _META_TRAJ_HEADING_MIN_SEGMENT_LEN_M:
            return prev
        if dx[i] <= 0.0:
            return prev
        return float(np.arctan2(dy[i], dx[i]))

    headings = np.zeros(n_points, dtype=np.float64)
    h0 = 0.0
    for j in range(len(dx)):
        if segment_lengths[j] >= _META_TRAJ_HEADING_MIN_SEGMENT_LEN_M and dx[j] > 0.0:
            h0 = float(np.arctan2(dy[j], dx[j]))
            break
    headings[0] = h0
    for i in range(1, n_points - 1):
        headings[i] = _segment_heading_rad(i, headings[i - 1])
    headings[-1] = headings[-2]

    unwrapped = np.unwrap(headings)
    hw = _META_HEADING_SMOOTH_WINDOW
    if hw % 2 == 0:
        hw -= 1
    if hw > 1 and n_points >= hw:
        kernel = np.ones(hw, dtype=np.float64) / float(hw)
        pad_h = hw // 2
        u_pad = np.pad(unwrapped, (pad_h, pad_h), mode="edge")
        smoothed_unwrapped = np.convolve(u_pad, kernel, mode="valid")
        if len(smoothed_unwrapped) != n_points:
            smoothed_unwrapped = unwrapped
    else:
        smoothed_unwrapped = unwrapped

    net_heading_change_rad = float(smoothed_unwrapped[-1] - smoothed_unwrapped[0])
    net_heading_change_deg = float(np.degrees(net_heading_change_rad))

    chord_dx = float(traj_arr[-1, 0] - traj_arr[0, 0])
    chord_dy = float(traj_arr[-1, 1] - traj_arr[0, 1])
    chord_len = math.hypot(chord_dx, chord_dy)
    if chord_len > 1e-12 and abs(net_heading_change_deg) > _META_TRAJ_NET_WRAP_REPLACE_MIN_ABS_DEG:
        chord_bearing_deg = float(np.degrees(np.arctan2(chord_dy, chord_dx)))
        if abs(chord_bearing_deg) < _META_TRAJ_NET_CHORD_REPLACE_MAX_ABS_DEG:
            net_heading_change_deg = chord_bearing_deg

    end_lateral_disp_m = float(traj_arr[-1, 1] - traj_arr[0, 1])
    max_lateral_disp_m = float(np.max(np.abs(traj_arr[:, 1] - traj_arr[0, 1])))

    return {
        "is_valid": True,
        "total_displacement": total_displacement,
        "total_travel_distance": total_travel_distance,
        "net_heading_change_deg": net_heading_change_deg,
        "end_lateral_disp_m": end_lateral_disp_m,
        "max_lateral_disp_m": max_lateral_disp_m,
    }


def _meta_infer_lateral_action_from_trajectory(
    future_traj: List[List[float]], is_stopped: bool
) -> str:
    if is_stopped:
        return _META_LAT_LIST_STRAIGHT

    traj_metrics = _meta_calc_trajectory_metrics(future_traj)

    if not traj_metrics["is_valid"]:
        return _META_LAT_LIST_STRAIGHT

    net_heading_deg = float(traj_metrics["net_heading_change_deg"])
    end_lat_disp = float(traj_metrics["end_lateral_disp_m"])
    max_lat = float(traj_metrics["max_lateral_disp_m"])
    abs_net_heading = abs(net_heading_deg)

    if abs_net_heading <= _META_LAT_HEADING_TOTAL_THRESH_DEG and max_lat < _META_LAT_LANE_CHANGE_DISP_THRESH_M:
        return _META_LAT_LIST_STRAIGHT

    if net_heading_deg > _META_LAT_HEADING_TOTAL_THRESH_DEG and end_lat_disp > 0:
        return _META_LAT_LIST_LEFT

    if net_heading_deg < -_META_LAT_HEADING_TOTAL_THRESH_DEG and end_lat_disp < 0:
        return _META_LAT_LIST_RIGHT

    return _META_LAT_LIST_STRAIGHT


def infer_meta_action_lists_from_future_traj_reward(
    future_traj: List[List[float]],
) -> Tuple[str, str]:
    if not future_traj or len(future_traj) < 2:
        return _META_LONG_ACTIONS_DEFAULT, _META_LAT_ACTIONS_DEFAULT

    try:
        future_traj = _meta_smooth_trajectory(
            future_traj, window_size=_META_TRAJ_SMOOTH_WINDOW_SIZE
        )
        long_str = _meta_infer_longitudinal_meta_str(future_traj)
        is_stopped = long_str == "stop, wait"
        lat_str = _meta_infer_lateral_action_from_trajectory(future_traj, is_stopped)
    except (TypeError, ValueError, IndexError):
        return _META_LONG_ACTIONS_DEFAULT, _META_LAT_ACTIONS_DEFAULT

    return long_str, lat_str


def _interp_1hz_traj_to_4hz_5s(traj_1hz_xy: np.ndarray) -> List[List[float]]:
    traj_1hz_xy = np.asarray(traj_1hz_xy, dtype=np.float64)
    if traj_1hz_xy.ndim != 2 or traj_1hz_xy.shape[1] != 2 or traj_1hz_xy.shape[0] < 5:
        return []
    pts = traj_1hz_xy[:5]
    t_knots = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    origin = np.zeros((1, 2), dtype=np.float64)
    knots_xy = np.vstack([origin, pts])
    t_target = np.arange(0.25, 5.0 + 1e-9, 0.25, dtype=np.float64)
    x = np.interp(t_target, t_knots, knots_xy[:, 0])
    y = np.interp(t_target, t_knots, knots_xy[:, 1])
    return [[float(a), float(b)] for a, b in zip(x, y)]


_DECISION_BLOCK_RE = re.compile(
    r"<decision>\s*(.*?)\s*</decision>",
    re.IGNORECASE | re.DOTALL,
)

_ANSWER_BLOCK_RE = re.compile(
    r"<answer>\s*(.*?)\s*</answer>",
    re.IGNORECASE | re.DOTALL,
)


def _generation_text_for_traj_parse(gen: str) -> str:
    """优先用 <answer>...</answer> 内文本解析轨迹，否则用全文。"""
    if not isinstance(gen, str):
        return ""
    m = _ANSWER_BLOCK_RE.search(gen)
    if m:
        return m.group(1).strip()
    return gen


def _clean_decision_action_value(raw: str) -> str:
    """清理单个 action 字段，避免把下一轮轨迹文本一并吞进来。"""
    text = str(raw or "").strip()
    text = re.split(r"<\s*/?\s*(?:answer|decision)\b", text, maxsplit=1, flags=re.IGNORECASE)[0]
    text = text.split("<", 1)[0].strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        text = lines[0]
    return text.strip()


def _parse_decision_long_lat(gen: str) -> Tuple[Optional[str], Optional[str]]:
    """
    从 generation 解析纵向/横向 meta 文案。支持：
    - <decision>...</decision> 内：longitudinal ... ; lateral ...（分号）
    - 无标签全文：同上分号格式
    - infer 多行：longitudinal action: keep, lateral action: lane_follow（逗号分隔；
      若纵向含逗号如 stop, wait，以最后一个「, lateral action:」为界）
    """
    if not isinstance(gen, str) or not gen.strip():
        return None, None
    m = _DECISION_BLOCK_RE.search(gen)
    if m:
        block = m.group(1).strip()
    else:
        block = gen.strip()

    lm = re.search(
        r"longitudinal\s+action:\s*([^;]+?)\s*;\s*lateral\s+action:\s*(.+)",
        block,
        re.IGNORECASE | re.DOTALL,
    )
    if lm:
        long_a = _clean_decision_action_value(lm.group(1))
        lat_a = _clean_decision_action_value(lm.group(2))
        return long_a, lat_a

    # 逗号分隔：… longitudinal action: X, lateral action: Y（取最后一个 , lateral action:）
    lateral_split = re.compile(r",\s*lateral\s+action:\s*", re.IGNORECASE)
    matches = list(lateral_split.finditer(block))
    if matches:
        last = matches[-1]
        left = block[: last.start()]
        lat_a = _clean_decision_action_value(block[last.end() :])
        lm_left = re.search(
            r"longitudinal\s+action:\s*(.+)",
            left,
            re.IGNORECASE | re.DOTALL,
        )
        if lm_left:
            long_a = _clean_decision_action_value(lm_left.group(1))
            return long_a, lat_a

    return None, None


def _row_generation_text(row: Dict[str, Any]) -> str:
    """优先使用 generation；否则用 generation_turn1 + generation_turn2 拼成与训练一致的整段文本。"""
    gen = row.get("generation")
    if gen is not None:
        g = gen if isinstance(gen, str) else str(gen)
        if g.strip():
            return g
    t1 = row.get("generation_turn1", "")
    t2 = row.get("generation_turn2", "")
    if not isinstance(t1, str):
        t1 = str(t1 or "")
    if not isinstance(t2, str):
        t2 = str(t2 or "")
    parts = [p for p in (t1.strip(), t2.strip()) if p]
    if not parts:
        return ""
    return "\n\n".join(parts)


def _norm_action_str(s: str) -> str:
    return " ".join(s.lower().strip().split())


_STOP_WAIT_ATOMS = frozenset({"stop", "wait"})


def _longitudinal_atoms(s: str) -> frozenset:
    """逗号、空白切分为原子词，如 stop, wait -> {stop, wait}。"""
    t = (s or "").lower().replace(",", " ").split()
    return frozenset(w.strip() for w in t if w.strip())


def longitudinal_actions_match(pred_long: str, traj_long: str) -> bool:
    """
    纵向一致：非 stop/wait 时两边原子集合须完全相同。
    轨迹推断在静止时为 \"stop, wait\" 两个词；decision 里只写 stop 或只写 wait 也算对。
    """
    pa = _longitudinal_atoms(pred_long)
    ta = _longitudinal_atoms(traj_long)
    if not pa or not ta:
        return False
    if ta <= _STOP_WAIT_ATOMS:
        return pa <= _STOP_WAIT_ATOMS
    if pa <= _STOP_WAIT_ATOMS:
        return False
    return pa == ta


def consistency_from_label_and_trajectory(
    generation: str, pred_long: str, pred_lat: str
) -> Optional[Dict[str, Any]]:
    """
    用外部 label（纵向/横向字符串）与从 generation 解析的轨迹推断 meta 做一致性判定；
    匹配规则与 consistency_from_generation 相同。
    """
    if not isinstance(pred_long, str) or not isinstance(pred_lat, str):
        return None
    if not str(pred_long).strip() or not str(pred_lat).strip():
        return None
    gen = generation if isinstance(generation, str) else ""
    pred_xy = _parse_trajectory_to_xy(_generation_text_for_traj_parse(gen))
    if pred_xy is None:
        return None
    if pred_xy.shape[0] < 5:
        return None
    traj_4hz = _interp_1hz_traj_to_4hz_5s(pred_xy[:5])
    if not traj_4hz:
        return None
    traj_long, traj_lat_group = infer_meta_action_lists_from_future_traj_reward(traj_4hz)
    n_long_pred = _norm_action_str(pred_long)
    n_long_traj = _norm_action_str(traj_long)
    long_ok = longitudinal_actions_match(pred_long, traj_long)

    lat_group_tokens = {
        _norm_action_str(x)
        for x in traj_lat_group.split(",")
        if x.strip()
    }
    lat_ok = _norm_action_str(pred_lat) in lat_group_tokens

    return {
        "long_ok": long_ok,
        "lat_ok": lat_ok,
        "joint_ok": long_ok and lat_ok,
        "pred_long": pred_long,
        "pred_lat": pred_lat,
        "traj_long": traj_long,
        "traj_lat_group": traj_lat_group,
        "n_long_pred": n_long_pred,
        "n_long_traj": n_long_traj,
        "n_lat_pred": _norm_action_str(pred_lat),
    }


def consistency_from_generation(generation: str) -> Optional[Dict[str, Any]]:
    """
    单条 generation 的一致性判定；无法解析或轨迹不足时返回 None。
    返回字段含 long_ok, lat_ok, joint_ok 及 pred/traj 字符串便于排查。
    """
    pred_long, pred_lat = _parse_decision_long_lat(generation if isinstance(generation, str) else "")
    pred_xy = _parse_trajectory_to_xy(_generation_text_for_traj_parse(generation))
    if pred_long is None or pred_lat is None or pred_xy is None:
        return None
    if pred_xy.shape[0] < 5:
        return None
    traj_4hz = _interp_1hz_traj_to_4hz_5s(pred_xy[:5])
    if not traj_4hz:
        return None
    traj_long, traj_lat_group = infer_meta_action_lists_from_future_traj_reward(traj_4hz)
    n_long_pred = _norm_action_str(pred_long)
    n_long_traj = _norm_action_str(traj_long)
    long_ok = longitudinal_actions_match(pred_long, traj_long)

    lat_group_tokens = {
        _norm_action_str(x)
        for x in traj_lat_group.split(",")
        if x.strip()
    }
    lat_ok = _norm_action_str(pred_lat) in lat_group_tokens

    return {
        "long_ok": long_ok,
        "lat_ok": lat_ok,
        "joint_ok": long_ok and lat_ok,
        "pred_long": pred_long,
        "pred_lat": pred_lat,
        "traj_long": traj_long,
        "traj_lat_group": traj_lat_group,
        "n_long_pred": n_long_pred,
        "n_long_traj": n_long_traj,
        "n_lat_pred": _norm_action_str(pred_lat),
    }


def load_label_jsonl(path: Path) -> Dict[str, Tuple[str, str]]:
    """sample_id -> (longitudinal_action, lateral_action)。"""
    m: Dict[str, Tuple[str, str]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = row.get("sample_id")
            if not isinstance(sid, str) or not sid:
                continue
            pl = row.get("longitudinal_action")
            pa = row.get("lateral_action")
            if pl is None and isinstance(row.get("gt_meta_action"), dict):
                g = row["gt_meta_action"]
                pl = g.get("longitudinal_action")
                pa = g.get("lateral_action")
            if not isinstance(pl, str) or not isinstance(pa, str):
                continue
            if not pl.strip() or not pa.strip():
                continue
            m[sid] = (pl.strip(), pa.strip())
    return m


def evaluate_jsonl(
    infer_path: Path,
    no_consistency_out: Optional[Path] = None,
    label_map: Optional[Dict[str, Tuple[str, str]]] = None,
    label_jsonl_path: Optional[Path] = None,
) -> Dict[str, Any]:
    total_rows = 0
    skipped = 0
    long_ok_n = 0
    lat_ok_n = 0
    joint_ok_n = 0
    evaluated = 0
    no_consistency_count = 0

    out_fp = None
    if no_consistency_out is not None:
        no_consistency_out.parent.mkdir(parents=True, exist_ok=True)
        out_fp = open(no_consistency_out, "w", encoding="utf-8")

    try:
        with open(infer_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                total_rows += 1
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    skipped += 1
                    continue
                gen = _row_generation_text(row)

                if label_map is not None:
                    sid = row.get("sample_id")
                    if not isinstance(sid, str) or not sid:
                        skipped += 1
                        continue
                    lab = label_map.get(sid)
                    if lab is None:
                        skipped += 1
                        continue
                    pred_long, pred_lat = lab
                    r = consistency_from_label_and_trajectory(gen, pred_long, pred_lat)
                else:
                    r = consistency_from_generation(gen)

                if r is None:
                    skipped += 1
                    continue
                evaluated += 1
                if r["long_ok"]:
                    long_ok_n += 1
                if r["lat_ok"]:
                    lat_ok_n += 1
                if r["joint_ok"]:
                    joint_ok_n += 1

                if out_fp is not None and not r["joint_ok"]:
                    no_consistency_count += 1
                    out_row = dict(row)
                    out_row["_consistency"] = {
                        "long_ok": r["long_ok"],
                        "lat_ok": r["lat_ok"],
                        "pred_long": r["pred_long"],
                        "pred_lat": r["pred_lat"],
                        "traj_long": r["traj_long"],
                        "traj_lat_group": r["traj_lat_group"],
                        "n_long_pred": r["n_long_pred"],
                        "n_long_traj": r["n_long_traj"],
                        "n_lat_pred": r["n_lat_pred"],
                    }
                    out_fp.write(json.dumps(out_row, ensure_ascii=False) + "\n")
    finally:
        if out_fp is not None:
            out_fp.close()

    def _rate(a: int, b: int):
        return None if b <= 0 else a / b

    return {
        "infer_jsonl": str(infer_path),
        "label_jsonl": str(label_jsonl_path) if label_jsonl_path is not None else None,
        "label_entries": len(label_map) if label_map is not None else None,
        "no_consistency_jsonl": str(no_consistency_out) if no_consistency_out else None,
        "no_consistency_exported_rows": no_consistency_count,
        "total_jsonl_rows": total_rows,
        "skipped_rows": skipped,
        "evaluated_rows": evaluated,
        "longitudinal_consistency_acc": _rate(long_ok_n, evaluated),
        "lateral_consistency_acc": _rate(lat_ok_n, evaluated),
        "joint_consistency_acc": _rate(joint_ok_n, evaluated),
        "longitudinal_match_count": long_ok_n,
        "lateral_match_count": lat_ok_n,
        "joint_match_count": joint_ok_n,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="decision 与轨迹推断 meta action 一致性（对齐 TrajectoryConsistencyReward）"
    )
    parser.add_argument(
        "--infer_jsonl",
        type=str,
        default='/cephfs/zhengwc/FluidDrive/ms-swift-3.5/output/Qwen3.5-2B/Mutil-Turn-w-mask-subsample/RL-rfs-20ep/v0-20260415-171333/checkpoint-620/val_samples_479_minus_434_gt.jsonl',
        help="推理结果 JSONL，每行含 generation（或 generation_turn1/turn2）；"
        "默认模式需可解析的 longitudinal/lateral 文案 + 五点轨迹；"
        "若提供 --label_jsonl 则只需轨迹，并与 label 比较",
    )
    parser.add_argument(
        "--label_jsonl",
        type=str,
        default=None,
        help="可选：每行含 sample_id, longitudinal_action, lateral_action；"
        "与推理行按 sample_id 对齐，用 label 与轨迹推断 meta 算一致性（规则同默认）",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="可选：将指标写入 JSON",
    )
    parser.add_argument(
        "--no_consistency_jsonl",
        type=str,
        default=None,
        help="不一致样本导出路径；默认：<infer_jsonl 主名>.no_consistency.jsonl（与输入同目录）",
    )
    args = parser.parse_args()
    infer_path = Path(args.infer_jsonl)
    if not infer_path.is_file():
        print(f"[ERROR] 文件不存在: {infer_path}")
        return

    label_path: Optional[Path] = None
    label_map: Optional[Dict[str, Tuple[str, str]]] = None
    if args.label_jsonl:
        label_path = Path(args.label_jsonl)
        if not label_path.is_file():
            print(f"[ERROR] label 文件不存在: {label_path}")
            return
        label_map = load_label_jsonl(label_path)
        if not label_map:
            print(f"[ERROR] label_jsonl 未读到有效行: {label_path}")
            return

    if args.no_consistency_jsonl:
        nc_path = Path(args.no_consistency_jsonl)
    else:
        nc_path = infer_path.parent / f"{infer_path.stem}.no_consistency.jsonl"

    stats = evaluate_jsonl(
        infer_path,
        no_consistency_out=nc_path,
        label_map=label_map,
        label_jsonl_path=label_path,
    )
    print("=" * 60)
    if label_map is not None:
        print("Label–轨迹推断 meta 一致性（--label_jsonl）")
    else:
        print("Decision–轨迹 meta 一致性")
    print("=" * 60)
    print(f"JSONL 行数: {stats['total_jsonl_rows']}")
    if stats.get("label_jsonl"):
        print(f"label_jsonl: {stats['label_jsonl']}（条目数 {stats.get('label_entries')}）")
    print(f"跳过（解析失败/轨迹不足/无 sample_id 或无对应 label）: {stats['skipped_rows']}")
    print(f"有效评估行数: {stats['evaluated_rows']}")
    print()
    lat_desc = (
        "横向一致率（label lateral ∈ 轨迹行为组）"
        if label_map is not None
        else "横向一致率（decision lateral ∈ 轨迹行为组）"
    )
    for label, key in [
        ("纵向一致率", "longitudinal_consistency_acc"),
        (lat_desc, "lateral_consistency_acc"),
        ("联合一致率", "joint_consistency_acc"),
    ]:
        v = stats[key]
        if v is None:
            print(f"{label}: N/A")
        else:
            print(f"{label}: {v * 100:.2f}%")
    print("=" * 60)
    if stats.get("no_consistency_jsonl"):
        print(
            f"不一致样本已导出: {stats['no_consistency_jsonl']} "
            f"（{stats['no_consistency_exported_rows']} 行）"
        )

    if args.output_file:
        outp = Path(args.output_file)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with open(outp, "w", encoding="utf-8") as wf:
            json.dump(stats, wf, indent=2, ensure_ascii=False)
        print(f"\n[INFO] 已写入: {outp}")


if __name__ == "__main__":
    main()
