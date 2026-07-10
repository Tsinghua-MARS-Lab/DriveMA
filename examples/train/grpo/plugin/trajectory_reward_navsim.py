#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自定义轨迹与 meta 相关奖励：ADE、meta action 标注一致、轨迹-decision 一致性、格式等。
- ade_5s / GT：从文本中解析连续 ``[x, y]`` 点对（至少 5 个）作为轨迹；兼容旧版 ``future_trajectory`` JSON。
- meta_action_acc：从 ``longitudinal action:`` 与 ``lateral action:`` 解析（逗号或分号分隔）；兼容旧版 ``<decision>`` 标签。
- traj_consistency：用模型输出的 decision 文本 + 轨迹文本做一致性（不依赖 GT）。
"""

import math
import os
import re
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from swift.rewards import ORM, orms
from swift.utils import get_logger

logger = get_logger()

_LON_LABELER_DIR = Path("/cephfs/zhengwc/LongTail_synthesizer_V2/v2/labeler")
_LON_THRESHOLD_REPORT_PATH = _LON_LABELER_DIR / "navsim/train_thresholds_report_4s.json"
if str(_LON_LABELER_DIR) not in sys.path:
    sys.path.insert(0, str(_LON_LABELER_DIR))

from cluster_longitudinal_thresholds import (  # noqa: E402
    DEFAULT_DT_SECONDS as _LON_DEFAULT_DT_SECONDS,
    DEFAULT_SMOOTH_WINDOW as _LON_DEFAULT_SMOOTH_WINDOW,
    apply_suggested_rule as _lon_apply_suggested_rule,
    smooth_trajectory as _lon_smooth_trajectory,
    trajectory_features as _lon_trajectory_features,
)

# ============================================================================
# 辅助函数：轨迹解析
# ============================================================================

# 匹配字符串中第一个 "future_trajectory": [[...], ...] 的键名（允许单/双引号）
_FUTURE_TRAJ_KEY_RE = re.compile(
    r'["\']?\s*future_trajectory\s*["\']?\s*:\s*',
    re.IGNORECASE,
)


def _extract_first_future_trajectory_from_str(gen: str) -> Optional[np.ndarray]:
    """
    在字符串中找第一个 "future_trajectory": [[x,y], [x,y], ...]，解析为 np.ndarray [N,2]。
    """
    m = _FUTURE_TRAJ_KEY_RE.search(gen)
    if not m:
        return None
    start_val = m.end()
    if start_val >= len(gen):
        return None
    # 跳过空白，找到第一个 '['
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
        else:
            continue
    if not coords:
        return None
    return np.asarray(coords, dtype=np.float32)


def _parse_xy_bracket_pairs(text: str, min_points: int = 5) -> Optional[np.ndarray]:
    """
    从纯文本中按顺序匹配 ``[x, y]`` 点对（与 ``[8.30, 0.06], [17.01, 0.22], ...`` 一致）。
    至少 ``min_points`` 个点才认为解析成功。
    """
    if not isinstance(text, str) or not text.strip():
        return None
    pattern = re.compile(r'\[\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\]')
    matches = pattern.findall(text)
    if len(matches) < min_points:
        return None
    coords: List[List[float]] = []
    for a, b in matches:
        try:
            coords.append([float(a), float(b)])
        except (TypeError, ValueError):
            return None
    return np.asarray(coords, dtype=np.float32)


def _parse_trajectory_to_xy(traj_input: Any) -> Optional[np.ndarray]:
    """
    将轨迹解析成 [[x,y], ...]，返回 np.ndarray [N,2]。支持：
    1) 字符串：优先顺序匹配至少 5 个 ``[x, y]`` 点对（当前训练数据常用）；
    2) 字符串中第一个 ``"future_trajectory"``: [[x,y], ...]（旧版）；
    3) dict 且含 ``future_trajectory`` 键；
    4) 其余 ``[x,y]`` 成对回退。
    """
    if traj_input is None:
        return None
    
    # 3) dict 且含 future_trajectory
    if isinstance(traj_input, dict):
        traj = traj_input.get("future_trajectory", None)
        if traj is None:
            return None
        if not isinstance(traj, list) or not traj:
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
        # 1) 优先：直接匹配 ``[x, y], [x, y], ...``（至少 5 点）
        xy = _parse_xy_bracket_pairs(traj_input, min_points=5)
        if xy is not None:
            return xy
        # 2) 旧版：JSON 里 "future_trajectory": [...]
        xy = _extract_first_future_trajectory_from_str(traj_input)
        if xy is not None:
            return xy
        # 3) 回退：任意数量的 [x,y] 对（兼容短文本）
        pattern = r'\[([-\d.]+),\s*([-\d.]+)\]'
        matches = re.findall(pattern, traj_input)
        coords = []
        for match in matches:
            try:
                x = float(match[0])
                y = float(match[1])
                coords.append([x, y])
            except Exception:
                continue
        if not coords:
            return None
        return np.asarray(coords, dtype=np.float32)
    return None


# ============================================================================
# 轨迹 -> meta action（横向规则与 LongTail_synthesizer_V2/v2/labeler/run_vllm_infer_label_meta_action_lat_only.py
# 中规则保持一致；此处内联以避免导入该脚本时拉起 vllm/torch）
# ============================================================================

_META_TRAJ_VALID_DISP_THRESH_M = 1.44
_META_TRAJ_SMOOTH_WINDOW_SIZE = 1
_META_LAT_HEADING_TOTAL_THRESH_DEG = 12.0
_META_LAT_LANE_CHANGE_DISP_THRESH_M = 1.2
_META_LAT_STRAIGHT_SHIFT_THRESH_M = 0.60
_META_HEADING_SMOOTH_WINDOW = 1
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
_META_LAT_LIST_STRAIGHT_REFINED = "straight, lane_follow"


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


_LON_CLUSTER_REPORT_CACHE: Optional[Dict[str, Any]] = None


def _load_lon_cluster_report() -> Dict[str, Any]:
    global _LON_CLUSTER_REPORT_CACHE
    if _LON_CLUSTER_REPORT_CACHE is None:
        with open(_LON_THRESHOLD_REPORT_PATH, "r", encoding="utf-8") as f:
            report = json.load(f)
        rule = report.get("suggested_rule")
        if not isinstance(rule, dict):
            raise KeyError(f"{_LON_THRESHOLD_REPORT_PATH} 缺少 suggested_rule")
        for key in ("stop_wait", "decelerate", "keep", "accelerate"):
            if key not in rule:
                raise KeyError(f"{_LON_THRESHOLD_REPORT_PATH} suggested_rule 缺少键: {key}")
        _LON_CLUSTER_REPORT_CACHE = report
    return _LON_CLUSTER_REPORT_CACHE


def _cluster_apply_longitudinal_meta_str(future_traj: List[List[float]]) -> str:
    """
    与 apply_lon_thresholds_from_report.py 对齐：读取聚类 report 的 suggested_rule，
    对当前轨迹提取同一套特征后直接 apply 得到纵向动作。
    """
    report = _load_lon_cluster_report()
    dt = float(report.get("dt_seconds", _LON_DEFAULT_DT_SECONDS))
    smooth_window = int(report.get("smooth_window", _LON_DEFAULT_SMOOTH_WINDOW))

    traj = np.asarray(future_traj, dtype=np.float64)
    if traj.ndim != 2 or traj.shape[0] < 2 or traj.shape[1] < 2:
        return _META_LONG_ACTIONS_DEFAULT
    traj = traj[:, :2]
    if not np.all(np.isfinite(traj)):
        return _META_LONG_ACTIONS_DEFAULT

    if smooth_window > 1:
        traj = _lon_smooth_trajectory(traj, smooth_window)
    row: Dict[str, Any] = {"features": _lon_trajectory_features(traj, dt)}
    _lon_apply_suggested_rule([row], report["suggested_rule"])
    action = str(row.get("rule_longitudinal_action", "") or "").strip()
    if action == "stop":
        return "stop, wait"
    if action in {"keep", "accelerate", "decelerate"}:
        return action
    return _META_LONG_ACTIONS_DEFAULT


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
    total_heading_change_deg = float(
        np.sum(np.abs(np.degrees(np.diff(smoothed_unwrapped))))
    )

    end_lateral_disp_m = float(traj_arr[-1, 1] - traj_arr[0, 1])
    max_lateral_disp_m = float(np.max(np.abs(traj_arr[:, 1] - traj_arr[0, 1])))

    return {
        "is_valid": True,
        "total_displacement": total_displacement,
        "total_travel_distance": total_travel_distance,
        "net_heading_change_deg": net_heading_change_deg,
        "total_heading_change_deg": total_heading_change_deg,
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


def _meta_refine_straight_lateral_action_list(
    lat_str: str,
    future_traj: List[List[float]],
) -> str:
    """
    同步 lat-only labeler 的直行组规则：只有末端相对首点横向位移超过阈值时，
    才允许 left/right_shift_slightly；否则直行组只保留 straight/lane_follow。
    """
    if lat_str != _META_LAT_LIST_STRAIGHT:
        return lat_str
    if not future_traj or len(future_traj) < 2:
        return _META_LAT_LIST_STRAIGHT_REFINED

    try:
        dy = float(future_traj[-1][1]) - float(future_traj[0][1])
    except (TypeError, ValueError, IndexError):
        return _META_LAT_LIST_STRAIGHT_REFINED

    if dy > _META_LAT_STRAIGHT_SHIFT_THRESH_M:
        return "left_shift_slightly"
    if dy < -_META_LAT_STRAIGHT_SHIFT_THRESH_M:
        return "right_shift_slightly"
    return _META_LAT_LIST_STRAIGHT_REFINED


def infer_meta_action_lists_from_future_traj_reward(
    future_traj: List[List[float]],
) -> Tuple[str, str]:
    """
    纵向与 apply_lon_thresholds_from_report.py 对齐，直接 apply 聚类 report 的 suggested_rule；
    横向沿用 labeler 中 infer_meta_action_lists_from_future_traj 的规则。
    输入为 4Hz、0.25s 步长的未来轨迹点序列（例如由 1Hz 插值得到）。
    """
    if not future_traj or len(future_traj) < 2:
        return _META_LONG_ACTIONS_DEFAULT, _META_LAT_ACTIONS_DEFAULT

    try:
        smoothed_future_traj = _meta_smooth_trajectory(
            future_traj, window_size=_META_TRAJ_SMOOTH_WINDOW_SIZE
        )
        long_str = _cluster_apply_longitudinal_meta_str(future_traj)
        is_stopped = long_str == "stop, wait"
        lat_str = _meta_infer_lateral_action_from_trajectory(smoothed_future_traj, is_stopped)
        lat_str = _meta_refine_straight_lateral_action_list(lat_str, smoothed_future_traj)
    except (TypeError, ValueError, IndexError):
        return _META_LONG_ACTIONS_DEFAULT, _META_LAT_ACTIONS_DEFAULT

    return long_str, lat_str


def _interp_1hz_traj_to_4hz_5s(traj_1hz_xy: np.ndarray) -> List[List[float]]:
    """
    模型输出为 1Hz、对应 1s..5s 的 5 个未来点（自车当前位置为 t=0）。
    在 t=0 处补 (0,0)，线性插值得到 t=0.25,0.5,...,5.0 共 20 个 4Hz 点。
    """
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


def _interp_1hz_4s_traj_to_2hz_4s(traj_1hz_xy: np.ndarray) -> List[List[float]]:
    """
    模型输出为 1Hz、对应 1s..4s 的 4 个未来点；PDMS v1 server 需要
    0.5s..4.0s 的 8 个点。与 resample_inference_jsonl.py 一致，在 t=0
    处补 (0,0) 后做线性插值。
    """
    traj_1hz_xy = np.asarray(traj_1hz_xy, dtype=np.float64)
    if traj_1hz_xy.ndim != 2 or traj_1hz_xy.shape[1] != 2 or traj_1hz_xy.shape[0] < 4:
        return []
    pts = traj_1hz_xy[:4]
    t_knots = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    origin = np.zeros((1, 2), dtype=np.float64)
    knots_xy = np.vstack([origin, pts])
    t_target = np.arange(0.5, 4.0 + 1e-9, 0.5, dtype=np.float64)
    x = np.interp(t_target, t_knots, knots_xy[:, 0])
    y = np.interp(t_target, t_knots, knots_xy[:, 1])
    return [[float(a), float(b)] for a, b in zip(x, y)]


_DECISION_BLOCK_RE = re.compile(
    r"<decision>\s*(.*?)\s*</decision>",
    re.IGNORECASE | re.DOTALL,
)


def _parse_long_lat_plain_only(text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    无 XML 标签时，从正文解析 meta：
    - ``longitudinal action: xx, lateral action: yy``（逗号分隔，当前数据常用）
    - ``longitudinal action: xx; lateral action: yy``（分号分隔）
    """
    if not isinstance(text, str) or not text.strip():
        return None, None
    m = re.search(
        r"longitudinal\s+action:\s*([^,\n]+?)\s*,\s*lateral\s+action:\s*([^\n]+)",
        text,
        re.IGNORECASE,
    )
    if m:
        long_a = m.group(1).strip()
        lat_a = m.group(2).strip()
        return long_a, lat_a
    m = re.search(
        r"longitudinal\s+action:\s*(.+?)\s*;\s*lateral\s+action:\s*(.+)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if m:
        long_a = m.group(1).strip()
        lat_a = m.group(2).strip()
        lat_a = lat_a.split("<")[0].split("\n")[0].strip()
        return long_a, lat_a
    return None, None


def _extract_long_lat_from_text(text: str) -> Tuple[Optional[str], Optional[str]]:
    """先正文直接解析；失败则尝试 ``<decision>...</decision>`` 内再解析（兼容旧数据）。"""
    lo, la = _parse_long_lat_plain_only(text)
    if lo is not None and la is not None:
        return lo, la
    m = _DECISION_BLOCK_RE.search(text)
    if not m:
        return None, None
    block = m.group(1)
    lo, la = _parse_long_lat_plain_only(block)
    if lo is not None and la is not None:
        return lo, la
    lm = re.search(
        r"longitudinal\s+action:\s*([^;]+?)\s*;\s*lateral\s+action:\s*(.+)",
        block,
        re.IGNORECASE | re.DOTALL,
    )
    if not lm:
        return None, None
    long_a = lm.group(1).strip()
    lat_a = lm.group(2).strip()
    lat_a = lat_a.split("<")[0].strip()
    return long_a, lat_a


def _parse_decision_long_lat(gen: str) -> Tuple[Optional[str], Optional[str]]:
    """解析 longitudinal / lateral，用于 traj_consistency。"""
    return _extract_long_lat_from_text(gen)


def _norm_action_str(s: str) -> str:
    return " ".join(s.lower().strip().split())


_STOP_WAIT_ATOMS = frozenset({"stop", "wait"})


def _longitudinal_atoms(s: str) -> frozenset:
    """逗号、空白切分为原子词，如 stop, wait -> {stop, wait}。"""
    t = (s or "").lower().replace(",", " ").split()
    return frozenset(w.strip() for w in t if w.strip())


def _longitudinal_actions_match(pred_long: str, traj_long: str) -> bool:
    """
    纵向一致：非 stop/wait 时两边原子集合须完全相同。
    轨迹静止推断为 \"stop, wait\"；decision 只写 stop 或只写 wait 亦算对。
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


def _length_reward_scalar(length: int, l_max: float, l_cache: float) -> float:
    """
    分段长度奖励 R_length(y)，|y| 为 completion 长度（此处为字符数）：

    - |y| <= L_max - L_cache  -> 0
    - L_max - L_cache < |y| <= L_max  -> ((L_max - L_cache) - |y|) / L_cache  （由 0 线性降至 -1）
    - |y| > L_max  -> -1

    当 L_cache == 0 时，中间区间为空，等价于 |y| <= L_max 得 0，否则 -1。
    """
    ly = float(length)
    threshold = float(l_max) - float(l_cache)
    if ly <= threshold:
        return 0.0
    if ly > float(l_max):
        return -1.0
    lc = float(l_cache)
    if lc <= 0.0:
        return -1.0
    return (threshold - ly) / lc


# ============================================================================
# 奖励函数 1: 5s ADE (Average Displacement Error)
# ============================================================================

class TrajectoryADE5sReward(ORM):
    """
    计算5s轨迹奖励函数
    - 直接从数据集样本中的 GT assistant/solution 提取 GT 轨迹
    - 返回归一化后的奖励值（越接近GT越大）
    """
    
    def __init__(
        self,
        args=None,
        max_ade: float = 50.0,
        gt_file: Optional[str] = None,
        **kwargs,
    ):
        """
        Args:
            max_ade: 误差裁剪上限，避免极端值影响reward数值稳定性
            gt_file: 已废弃。GT 会直接从数据集样本中的 solution 提取
        """
        super().__init__(args, **kwargs)
        self.max_ade = max_ade
      


        if gt_file:
            logger.warning(
                "[TrajectoryADE5sReward] `gt_file`/`ADE_GT_FILE` 已废弃，"
                "当前会直接从数据集样本中的 solution 读取 GT 轨迹。"
            )
        
        logger.info(
            "[TrajectoryADE5sReward] 初始化, "
            f"max_ade={max_ade}"

        )
    
    def __call__(
        self,
        completions: List[str],
        solution: Optional[List[str]] = None,
        **kwargs
    ) -> List[float]:
        """
        计算5s ADE奖励
        
        Args:
            completions: 模型生成的轨迹文本列表
            solution: 数据集样本中的 GT assistant/solution 文本列表
            **kwargs: 其他参数
        
        Returns:
            奖励分数列表，范围 [0, 1]
        """
        if solution is None:
            logger.error(
                "[TrajectoryADE5sReward] 缺少 solution，无法从数据集样本中提取 GT 轨迹。"
            )
            return [0.0] * len(completions)
        
        rewards = []
        
        # 计算每个样本的ADE
        for idx, gen in enumerate(completions):
            try:
                gt = solution[idx] if idx < len(solution) else None
                if gt is None:
                    # GT轨迹未找到，给予0奖励
                    rewards.append(0.0)
                    continue
                
                # 解析预测轨迹和GT轨迹
                pred_xy = _parse_trajectory_to_xy(gen)
                gt_xy = _parse_trajectory_to_xy(gt)
                
                if pred_xy is None or gt_xy is None:
                    # 解析失败，给予0奖励
                    rewards.append(0.0)
                    continue
                
                # 确保至少有5个点
                if pred_xy.shape[0] < 5 or gt_xy.shape[0] < 5:
                    rewards.append(0.0)
                    continue
                
                # 取前5个时间步
                pred_5s = pred_xy[:5]
                gt_5s = gt_xy[:5]
                
                # 计算每个时间步的欧氏距离
                distances = np.sqrt(np.sum((pred_5s - gt_5s) ** 2, axis=1))
                
                # 计算平均位移误差
                ade_5s = float(np.mean(distances))

                # waypoint reward
                waypoint_reward = 2.0/(np.exp(ade_5s)+1)
                
                rewards.append(waypoint_reward)
                
                # 前几个样本打印调试信息
                if idx < 3 and len(rewards) <= 3:
                    logger.info(
                        f"[TrajectoryADE5sReward] 样本 {idx}: ADE={ade_5s:.4f}, "
                        f"wp_reward={waypoint_reward:.4f}, "
                    )
                
            except Exception as e:
                logger.warning(f"[TrajectoryADE5sReward] 计算失败: {e}")
                rewards.append(0.0)
        
        return rewards


class TrajectoryADE4sReward(ORM):
    """
    计算4s轨迹奖励函数。
    当前 NAVSIM 数据为 4 个 1Hz 未来点，直接从第二轮 GT assistant/solution 中取前 4 点。
    """

    def __init__(
        self,
        args=None,
        max_ade: float = 50.0,
        gt_file: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(args, **kwargs)
        self.max_ade = max_ade
        if gt_file:
            logger.warning(
                "[TrajectoryADE4sReward] `gt_file`/`ADE_GT_FILE` 已废弃，"
                "当前会直接从数据集样本中的 solution 读取 GT 轨迹。"
            )
        logger.info(f"[TrajectoryADE4sReward] 初始化, max_ade={max_ade}")

    def __call__(
        self,
        completions: List[str],
        solution: Optional[List[str]] = None,
        **kwargs,
    ) -> List[float]:
        if solution is None:
            logger.error(
                "[TrajectoryADE4sReward] 缺少 solution，无法从数据集样本中提取 GT 轨迹。"
            )
            return [0.0] * len(completions)

        rewards: List[float] = []
        for idx, gen in enumerate(completions):
            try:
                gt = solution[idx] if idx < len(solution) else None
                if gt is None:
                    rewards.append(0.0)
                    continue

                pred_xy = _parse_trajectory_to_xy(gen)
                gt_xy = _parse_trajectory_to_xy(gt)
                if pred_xy is None or gt_xy is None:
                    rewards.append(0.0)
                    continue
                if pred_xy.shape[0] < 4 or gt_xy.shape[0] < 4:
                    rewards.append(0.0)
                    continue

                pred_4s = pred_xy[:4]
                gt_4s = gt_xy[:4]
                distances = np.sqrt(np.sum((pred_4s - gt_4s) ** 2, axis=1))
                ade_4s = float(np.mean(distances))
                ade_4s = min(ade_4s, float(self.max_ade))
                waypoint_reward = 2.0 / (np.exp(ade_4s/2) + 1)
                rewards.append(float(waypoint_reward))

                if idx < 3 and len(rewards) <= 3:
                    logger.info(
                        f"[TrajectoryADE4sReward] 样本 {idx}: ADE={ade_4s:.4f}, "
                        f"wp_reward={waypoint_reward:.4f}, "
                    )
            except Exception as e:
                logger.warning(f"[TrajectoryADE4sReward] 计算失败: {e}")
                rewards.append(0.0)

        return rewards


# ============================================================================
# 奖励函数 2: RFS (Rater Feedback Score)
# ============================================================================

class TrajectoryRFSReward(ORM):
    """
    计算RFS（Rater Feedback Score）奖励函数
    - 从外部文件加载 preference_trajectories 数据
    - 使用 rater_feedback_utils 计算RFS分数
    """
    
    def __init__(
        self,
        args=None,
        frequency: int = 4,
        length_seconds: int = 5,
        preference_file: Optional[str] = None,
        **kwargs,
    ):
        """
        Args:
            frequency: 目标频率 Hz，默认 4
            length_seconds: 轨迹长度（秒），默认 5
            preference_file: 包含 preference_trajectories 的 JSONL 文件路径
                如果为 None，则尝试从环境变量 RFS_PREFERENCE_FILE 读取
        """
        super().__init__(args, **kwargs)
        self.frequency = frequency
        self.length_seconds = length_seconds
        self.target_T = frequency * length_seconds
        self.dt = 1.0 / float(frequency)
        
        # 尝试导入 rater_feedback_utils
        try:
            import rater_feedback_utils
            self.rater_feedback_utils = rater_feedback_utils
            logger.info(f"[TrajectoryRFSReward] 成功导入 rater_feedback_utils")
        except ImportError as e:
            logger.error(
                f"[TrajectoryRFSReward] 无法导入 rater_feedback_utils: {e}\n"
                "请确保 rater_feedback_utils.py 在 PYTHONPATH 中"
            )
            self.rater_feedback_utils = None
        
        # 加载 preference_trajectories 映射
        if preference_file is None:
            import os
            preference_file = os.environ.get('RFS_PREFERENCE_FILE', None)
        
        self.preference_map = {}
        if preference_file:
            self.preference_map = self._load_preference_map(preference_file)
            logger.info(
                f"[TrajectoryRFSReward] 从 {preference_file} 加载了 "
                f"{len(self.preference_map)} 个样本的 preference_trajectories"
            )
        else:
            logger.warning(
                "[TrajectoryRFSReward] 未指定 preference_file，将从数据集字段读取\n"
                "建议通过环境变量 RFS_PREFERENCE_FILE 或初始化参数指定外部文件"
            )
        
        logger.info(
            f"[TrajectoryRFSReward] 初始化，frequency={frequency}Hz, "
            f"length_seconds={length_seconds}s"
        )
    
    def _load_preference_map(self, jsonl_path: str) -> Dict[str, Any]:
        """从 JSONL 文件加载 sample_id -> preference_trajectories 映射"""
        pref_map: Dict[str, Any] = {}
        try:
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        if not isinstance(obj, dict):
                            continue
                        
                        sid = obj.get("sample_id", None)
                        pref = obj.get("preference_trajectories", None)
                        
                        if sid and pref:
                            pref_map[str(sid)] = pref
                    except json.JSONDecodeError as e:
                        logger.warning(
                            f"[TrajectoryRFSReward] 第 {line_num} 行 JSON 解析失败: {e}"
                        )
                        continue
        except FileNotFoundError:
            logger.error(f"[TrajectoryRFSReward] 文件不存在: {jsonl_path}")
        except Exception as e:
            logger.error(f"[TrajectoryRFSReward] 加载文件失败: {e}")
        
        return pref_map
    
    def _resample_traj_xy_np(
        self,
        traj_xy: Any,
        target_len: int,
        dst_t_start: float = 0.0,
        dst_t_end: float = 1.0,
    ) -> Optional[np.ndarray]:
        """
        将任意长度的 [N,2] 轨迹线性重采样到 [target_len,2]。
        """
        if traj_xy is None:
            return None
        traj_xy = np.asarray(traj_xy, dtype=np.float32)
        if traj_xy.ndim != 2 or traj_xy.shape[1] != 2:
            return None
        n = traj_xy.shape[0]
        if target_len <= 0 or n == 0:
            return None
        if n == target_len:
            return traj_xy
        if n == 1:
            return np.repeat(traj_xy, repeats=target_len, axis=0)
        
        x = traj_xy[:, 0]
        y = traj_xy[:, 1]
        src_t = np.linspace(0.0, 1.0, num=n, dtype=np.float32)
        dst_t = np.linspace(
            float(dst_t_start),
            float(dst_t_end),
            num=target_len,
            dtype=np.float32,
        )
        x2 = np.interp(dst_t, src_t, x).astype(np.float32)
        y2 = np.interp(dst_t, src_t, y).astype(np.float32)
        return np.stack([x2, y2], axis=-1)
    
    def _extract_pref(self, pref: Any) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        preference_trajectories 结构：
          [{"score": float, "traj_pos": [[x,y], ...]}, ...]
        返回 (traj_list, score_array)；如果无法解析会返回空。
        """
        traj_list: List[np.ndarray] = []
        score_list: List[float] = []
        if not pref:
            return traj_list, np.asarray(score_list, dtype=np.float32)
        if not isinstance(pref, list):
            return traj_list, np.asarray(score_list, dtype=np.float32)
        for item in pref:
            if not isinstance(item, dict):
                continue
            score = item.get("score", None)
            traj_pos = item.get("traj_pos", None)
            if score is None or traj_pos is None:
                continue
            score_list.append(float(score))
            traj_list.append(np.asarray(traj_pos, dtype=np.float32))
        return traj_list, np.asarray(score_list, dtype=np.float32)
    
    def _estimate_init_speeds_for_each_rater(
        self,
        traj_list: List[np.ndarray],
        dt: float,
    ) -> List[float]:
        """
        用重采样后的 rater 轨迹估计初速度：
          - 为每个 rater 轨迹单独计算其初速度；
          - speed = ||p[1] - p[0]|| / dt。
        """
        if dt <= 0.0:
            return [0.0] * len(traj_list)
        if not traj_list:
            return []
        
        init_speeds = []
        for traj in traj_list:
            try:
                traj_arr = np.asarray(traj, dtype=np.float32)
                if traj_arr.ndim != 2 or traj_arr.shape[0] < 2 or traj_arr.shape[1] != 2:
                    init_speeds.append(0.0)
                else:
                    v = (traj_arr[1] - traj_arr[0]) / float(dt)
                    init_speeds.append(float(np.linalg.norm(v)))
            except Exception:
                init_speeds.append(0.0)
        
        return init_speeds
    
    def __call__(
        self,
        completions: List[str],
        preference_trajectories: Optional[List[Any]] = None,
        sample_id: Optional[List[str]] = None,
        **kwargs
    ) -> List[float]:
        """
        计算RFS奖励
        
        Args:
            completions: 模型生成的轨迹文本列表
            preference_trajectories: 偏好轨迹列表（可选，如果数据集包含此字段）
                每个元素为 [{"score": float, "traj_pos": [[x,y], ...]}, ...]
            sample_id: 样本ID列表（可选，用于从外部文件查找）
            **kwargs: 其他参数
        
        Returns:
            奖励分数列表
        """
        if self.rater_feedback_utils is None:
            logger.warning(
                "[TrajectoryRFSReward] rater_feedback_utils 未导入，返回0奖励"
            )
            return [0.0] * len(completions)
        
        # 如果数据集没有提供 preference_trajectories，尝试从外部文件查找
        if preference_trajectories is None:
            if sample_id is None:
                logger.warning(
                    "[TrajectoryRFSReward] 既没有 preference_trajectories "
                    "也没有 sample_id，返回0奖励"
                )
                return [0.0] * len(completions)
            
            # 根据 sample_id 从外部文件查找
            preference_trajectories = []
            for sid in sample_id:
                pref = self.preference_map.get(str(sid), None)
                preference_trajectories.append(pref)
            
            # 检查是否有有效的 preference_trajectories
            valid_count = sum(1 for p in preference_trajectories if p is not None)
            if valid_count == 0:
                logger.warning(
                    f"[TrajectoryRFSReward] 所有样本的 preference_trajectories "
                    f"都未找到（总共 {len(sample_id)} 个样本）"
                )
                return [0.0] * len(completions)
            elif valid_count < len(completions):
                logger.warning(
                    f"[TrajectoryRFSReward] 只找到 {valid_count}/{len(completions)} "
                    f"个样本的 preference_trajectories"
                )
        
        dst_t_start_frac = 1.0 / float(self.target_T)
        
        # 准备批量数据
        valid_indices = []
        pred_xy_list = []
        rater_trajs_list = []
        rater_scores_list = []
        initial_speed_list = []
        
        for idx, (gen, pref) in enumerate(zip(completions, preference_trajectories)):
            # 跳过没有 preference_trajectories 的样本
            if pref is None:
                continue
            
            try:
                # 解析生成的轨迹
                gen_xy = _parse_trajectory_to_xy(gen)
                if gen_xy is None or gen_xy.shape[0] == 0:
                    continue
                
                # 解析偏好轨迹
                rater_trajs_raw, rater_scores = self._extract_pref(pref)
                if len(rater_trajs_raw) == 0:
                    continue
                
                # 在序列前面补一个 (0,0) 再做插值
                origin = np.zeros((1, 2), dtype=np.float32)
                gen_with_origin = np.concatenate([origin, gen_xy], axis=0)
                
                # 插值到目标频率和时长
                pred_xy = self._resample_traj_xy_np(
                    gen_with_origin,
                    self.target_T,
                    dst_t_start=dst_t_start_frac,
                    dst_t_end=1.0,
                )
                if pred_xy is None:
                    continue
                
                # 对每条 rater 轨迹重采样（直接取前20个点，与cal_rfs.py一致）
                # 同时过滤对应的分数，确保数量一致
                rater_trajs_resampled: List[np.ndarray] = []
                rater_scores_filtered: List[float] = []
                
                for i, txy in enumerate(rater_trajs_raw):
                    txy2 = txy[:self.target_T]
                    if txy2 is not None and len(txy2) == self.target_T:
                        rater_trajs_resampled.append(txy2)
                        # 同时保留对应的分数
                        if i < len(rater_scores):
                            rater_scores_filtered.append(rater_scores[i])
                
                if not rater_trajs_resampled:
                    continue
                
                # 确保轨迹和分数数量一致
                if len(rater_trajs_resampled) != len(rater_scores_filtered):
                    logger.warning(
                        f"[TrajectoryRFSReward] 样本 {idx}: rater 轨迹数量 "
                        f"({len(rater_trajs_resampled)}) 与分数数量 "
                        f"({len(rater_scores_filtered)}) 不一致，跳过"
                    )
                    continue
                
                # 估计初速度
                init_speeds = self._estimate_init_speeds_for_each_rater(
                    rater_trajs_resampled, self.dt
                )
                
                valid_indices.append(idx)
                pred_xy_list.append(pred_xy)
                rater_trajs_list.append(rater_trajs_resampled)
                rater_scores_list.append(np.asarray(rater_scores_filtered, dtype=np.float32))
                initial_speed_list.append(init_speeds)
                
            except Exception as e:
                logger.warning(f"[TrajectoryRFSReward] 样本 {idx} 处理失败: {e}")
                continue
        
        # 如果没有有效样本，返回全0
        if not valid_indices:
            logger.warning("[TrajectoryRFSReward] 没有有效样本，返回全0奖励")
            return [0.0] * len(completions)
        
        # 调用 rater_feedback_utils 计算 RFS
        try:
            B = len(valid_indices)
            prediction_trajectories = (
                np.stack(pred_xy_list, axis=0)[:, None, :, :].astype(np.float32)
            )  # [B, 1, T, 2]
            prediction_probabilities = np.ones((B, 1), dtype=np.float32)
            
            metrics = self.rater_feedback_utils.get_rater_feedback_score(
                prediction_trajectories,
                prediction_probabilities,
                rater_trajs_list,
                rater_scores_list,
                initial_speed_list,
                frequency=self.frequency,
                length_seconds=self.length_seconds,
                output_trust_region_visualization=False,
            )
            
            rfs_scores = metrics.get("rater_feedback_score", None)
            if rfs_scores is None:
                logger.error("[TrajectoryRFSReward] RFS计算返回None")
                return [0.0] * len(completions)
            
            rfs_arr = np.asarray(rfs_scores, dtype=np.float32).reshape(-1)
            
            # 构建最终奖励列表
            result_rewards = [0.0] * len(completions)
            for i, idx in enumerate(valid_indices):
                # RFS 分数归一化到 [0, 1]，假设RFS范围是[0, 10]
                reward = float(np.clip(rfs_arr[i] / 10.0, 0.0, 1.0))
                result_rewards[idx] = reward
            
            return result_rewards
            
        except Exception as e:
            logger.error(f"[TrajectoryRFSReward] RFS计算失败: {e}")
            return [0.0] * len(completions)


# ============================================================================
# 奖励函数: NAVSIM v1 PDMS
# ============================================================================

class PDMSReward(ORM):
    """
    将模型输出的 4 点 1Hz 轨迹插值为 NAVSIM v1 PDMS server 需要的 8 点 2Hz，
    使用 sample_id 作为 token 请求 server，并用 DAC/EP 构建轨迹 reward：
      reward = 0.5 * DAC + 0.5 * DAC * EP
    """

    def __init__(
        self,
        args=None,
        pdms_server: Optional[str] = None,
        pdms_timeout: Optional[float] = None,
        pdms_client_workers: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(args, **kwargs)
        servers_raw = (
            os.environ.get("PDMS_SERVERS")
            or pdms_server
            or os.environ.get("PDMS_SERVER")
            or "http://127.0.0.1:9001"
        )
        self.endpoints = [
            x.strip().rstrip("/")
            for x in str(servers_raw).split(",")
            if x.strip()
        ]
        if not self.endpoints:
            self.endpoints = ["http://127.0.0.1:9001"]

        timeout_raw = pdms_timeout
        if timeout_raw is None:
            timeout_raw = os.environ.get("PDMS_TIMEOUT", os.environ.get("REQUEST_TIMEOUT", 30.0))
        self.timeout = float(timeout_raw)

        workers_raw = pdms_client_workers
        if workers_raw is None:
            workers_raw = os.environ.get("PDMS_CLIENT_WORKERS", os.environ.get("CLIENT_WORKERS", ""))
        self.client_workers = int(workers_raw) if str(workers_raw).strip() else len(self.endpoints)
        self.client_workers = max(1, self.client_workers)

        self.time_horizon = float(os.environ.get("PDMS_TIME_HORIZON", 4.0))
        self.interval_length = float(os.environ.get("PDMS_INTERVAL_LENGTH", 0.5))
        self.log_first_n = max(0, int(os.environ.get("PDMS_LOG_FIRST_N", 3)))
        self.invalid_retry_times = 5
        self.invalid_retry_interval = 5.0
        logger.info(
            "[PDMSReward] 初始化, "
            f"endpoints={self.endpoints}, timeout={self.timeout}, "
            f"client_workers={self.client_workers}, sampling={self.time_horizon}s/{self.interval_length}s, "
            f"log_first_n={self.log_first_n}, "
            f"invalid_retry={self.invalid_retry_times}x/{self.invalid_retry_interval:.1f}s"
        )

    def _post_json(self, endpoint: str, path: str, body: Dict[str, Any]) -> Dict[str, Any]:
        req = urllib.request.Request(
            endpoint + path,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            try:
                payload = json.loads(e.read().decode("utf-8"))
            except Exception:
                payload = {"error": str(e)}
            raise RuntimeError(f"PDMS server HTTP {e.code}: {payload}") from e

    def _score_batch_once(self, endpoint: str, batch: List[Dict[str, Any]]) -> List[Tuple[int, Dict[str, Any]]]:
        out = self._post_json(
            endpoint,
            "/score_batch",
            {
                "items": [
                    {
                        "token": item["sample_id"],
                        "future_traj": item["future_traj"],
                        "request_id": f"{item['index']}:{item['sample_id']}",
                    }
                    for item in batch
                ],
                "time_horizon": self.time_horizon,
                "interval_length": self.interval_length,
            },
        )
        results = out.get("results", [])
        if len(results) != len(batch):
            raise RuntimeError(
                f"server returned {len(results)} results for {len(batch)} requests"
            )
        return [(item["index"], result) for item, result in zip(batch, results)]

    @staticmethod
    def _invalid_pdms_results(
        scored: List[Tuple[int, Dict[str, Any]]]
    ) -> List[Tuple[int, Dict[str, Any]]]:
        return [
            (idx, result)
            for idx, result in scored
            if not (
                result.get("valid")
                and result.get("score") is not None
                and result.get("drivable_area_compliance") is not None
                and result.get("ego_progress") is not None
            )
        ]

    @staticmethod
    def _dac_ep_reward(result: Dict[str, Any]) -> float:
        dac = float(result["drivable_area_compliance"])
        nc = float(result["no_at_fault_collisions"])

        ep = float(result["ego_progress"])
        ttc = float(result["time_to_collision_within_bound"])
        c = float(result["comfort"])
        # return 0.5 * dac + 0.5 * dac * ep
        re = dac * nc * (10*ep + 5* ttc + 2* c)/17
        return re

    def _score_batch(self, endpoint: str, batch: List[Dict[str, Any]]) -> List[Tuple[int, Dict[str, Any]]]:
        results_by_index: Dict[int, Dict[str, Any]] = {}
        pending = list(batch)
        last_invalid: List[Tuple[int, Dict[str, Any]]] = []

        for attempt in range(self.invalid_retry_times + 1):
            pending_count = len(pending)
            scored = self._score_batch_once(endpoint, pending)
            invalid = self._invalid_pdms_results(scored)
            invalid_indices = {idx for idx, _ in invalid}
            for idx, result in scored:
                if idx not in invalid_indices:
                    results_by_index[idx] = result

            if not invalid:
                return [(item["index"], results_by_index[item["index"]]) for item in batch]

            last_invalid = invalid
            examples = [
                {
                    "index": idx,
                    "error": result.get("error"),
                    "request_id": result.get("request_id"),
                }
                for idx, result in invalid[:3]
            ]
            if attempt >= self.invalid_retry_times:
                break

            pending = [item for item in pending if item["index"] in invalid_indices]
            logger.warning(
                "[PDMSReward] PDMS server 返回 invalid, "
                f"endpoint={endpoint}, invalid={len(invalid)}/{pending_count}, "
                f"retry={attempt + 1}/{self.invalid_retry_times}, "
                f"sleep={self.invalid_retry_interval:.1f}s, examples={examples}"
            )
            time.sleep(self.invalid_retry_interval)

        final_examples = [
            {
                "index": idx,
                "error": result.get("error"),
                "request_id": result.get("request_id"),
            }
            for idx, result in last_invalid[:5]
        ]
        raise RuntimeError(
            "PDMS server returned invalid results after "
            f"{self.invalid_retry_times} retries on {endpoint}: {final_examples}"
        )

    def _endpoint_index_for_sample_id(self, sample_id: str) -> int:
        digest = hashlib.md5(sample_id.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], byteorder="big", signed=False) % len(self.endpoints)

    def __call__(
        self,
        completions: List[str],
        sample_id: Optional[List[str]] = None,
        **kwargs,
    ) -> List[float]:
        if sample_id is None:
            logger.warning("[PDMSReward] 缺少 sample_id，返回 0")
            return [0.0] * len(completions)

        batches: List[List[Dict[str, Any]]] = [[] for _ in self.endpoints]
        rewards = [0.0] * len(completions)
        n_parse_bad = 0

        for idx, gen in enumerate(completions):
            sid = str(sample_id[idx]).strip() if idx < len(sample_id) and sample_id[idx] is not None else ""
            if not sid:
                n_parse_bad += 1
                continue
            pred_xy = _parse_trajectory_to_xy(gen)
            if pred_xy is None or pred_xy.shape[0] < 4:
                n_parse_bad += 1
                continue
            future_traj = _interp_1hz_4s_traj_to_2hz_4s(pred_xy[:4])
            if len(future_traj) != 8:
                n_parse_bad += 1
                continue
            endpoint_idx = self._endpoint_index_for_sample_id(sid)
            batches[endpoint_idx].append({
                "index": idx,
                "sample_id": sid,
                "future_traj": future_traj,
            })

        flat_batches = [
            (self.endpoints[i], batch)
            for i, batch in enumerate(batches)
            if batch
        ]
        if not flat_batches:
            logger.warning(
                f"[PDMSReward] 没有可请求的有效样本，parse_bad={n_parse_bad}/{len(completions)}"
            )
            return rewards

        t0 = time.time()
        n_valid = 0
        n_invalid = 0
        debug_rows: List[Tuple[int, str, Dict[str, Any]]] = []
        try:
            max_workers = min(self.client_workers, len(flat_batches))
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = [
                    pool.submit(self._score_batch, endpoint, batch)
                    for endpoint, batch in flat_batches
                ]
                for fut in as_completed(futures):
                    for idx, result in fut.result():
                        if result.get("valid") and result.get("score") is not None:
                            rewards[idx] = self._dac_ep_reward(result)
                            n_valid += 1
                            if self.log_first_n > 0:
                                sid = str(sample_id[idx]).strip() if idx < len(sample_id) else ""
                                debug_rows.append((idx, sid, result))
                        else:
                            n_invalid += 1
                            raise RuntimeError(
                                f"PDMS server returned invalid result after retry, "
                                f"idx={idx}, error={result.get('error')}, "
                                f"request_id={result.get('request_id')}"
                            )
        except Exception as e:
            logger.error(f"[PDMSReward] 请求 PDMS server 失败: {e}")
            raise

        logger.info(
            f"[PDMSReward] scored={n_valid}, invalid={n_invalid}, "
            f"parse_bad={n_parse_bad}, elapsed={time.time() - t0:.2f}s"
        )
        for idx, sid, result in sorted(debug_rows, key=lambda x: x[0])[:self.log_first_n]:
            logger.info(
                "[PDMSReward] sample "
                f"idx={idx} sample_id={sid} reward={self._dac_ep_reward(result):.4f} "
                f"pdms={float(result['score']):.4f} "
                f"NC={result.get('no_at_fault_collisions')} "
                f"DAC={result.get('drivable_area_compliance')} "
                f"EP={result.get('ego_progress')} "
                f"TTC={result.get('time_to_collision_within_bound')} "
                f"C={result.get('comfort')}"
            )
        return rewards


# ============================================================================
# Meta action 与标注一致
# ============================================================================

_MA_LAT_STRAIGHT = frozenset(
    {"straight", "lane_follow", "left_shift_slightly", "right_shift_slightly", "reverse"}
)
_MA_LAT_LEFT = frozenset(
    {"left_turn", "left_lane_change", "left_shift_slightly", "lane_follow", "turn_around"}
)
_MA_LAT_RIGHT = frozenset(
    {"right_turn", "right_lane_change", "right_shift_slightly", "lane_follow"}
)
_MA_LAT_GROUPS: Tuple[frozenset, ...] = (_MA_LAT_STRAIGHT, _MA_LAT_LEFT, _MA_LAT_RIGHT)

_MA_THINK_END_TAG = "</redacted_thinking>"

_MA_DECISION_RE = re.compile(
    r"<decision>\s*longitudinal\s+action:\s*([^;]+?);\s*lateral\s+action:\s*([^<]+?)\s*</decision>",
    re.IGNORECASE | re.DOTALL,
)


def _ma_norm_longitudinal(s: str) -> str:
    t = (s or "").strip().lower().replace(",", " ")
    t = " ".join(t.split())
    return t.replace(" ", "_")


def _ma_parse_lateral_name(raw: str) -> str:
    t = (raw or "").strip().lower()
    t = re.sub(r"\s+", "_", t)
    return t


def _ma_longitudinal_match(pred: str, ref: str) -> bool:
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
    return _ma_norm_longitudinal(pred) == _ma_norm_longitudinal(ref)


def _ma_lateral_match(pred: str, ref: str) -> bool:
    p = _ma_parse_lateral_name(pred)
    q = _ma_parse_lateral_name(ref)
    if not p or not q:
        return False
    if p == q:
        return True
    return any(p in g and q in g for g in _MA_LAT_GROUPS)


def _ma_longitudinal_exact(pred: str, ref: str) -> bool:
    """与 GT 字符串一致（经 `_ma_norm_longitudinal`），不含 stop/wait 等价。"""
    if not (pred or "").strip() or not (ref or "").strip():
        return False
    return _ma_norm_longitudinal(pred) == _ma_norm_longitudinal(ref)


def _ma_lateral_exact(pred: str, ref: str) -> bool:
    """与 GT 一致（经 `_ma_parse_lateral_name`），不含横向三组并集规则。"""
    p = _ma_parse_lateral_name(pred)
    q = _ma_parse_lateral_name(ref)
    if not p or not q:
        return False
    return p == q


def _ma_strip_markdown_json_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def _ma_json_after_think(text: str) -> str:
    if _MA_THINK_END_TAG in text:
        return text.split(_MA_THINK_END_TAG, 1)[1].strip()
    return text.strip()


def parse_actions_from_generation_ma(generation: str) -> Tuple[str, str]:
    if not generation or not isinstance(generation, str):
        return "", ""

    lo, la = _extract_long_lat_from_text(generation)
    if lo and la:
        return lo, la

    m = _MA_DECISION_RE.search(generation)
    if m:
        long_a = m.group(1).strip()
        lat_a = m.group(2).strip()
        return long_a, lat_a

    text = _ma_strip_markdown_json_fence(_ma_json_after_think(generation))
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            lo = str(obj.get("longitudinal_action", "") or "").strip()
            la = str(obj.get("lateral_action", "") or "").strip()
            return lo, la
    except json.JSONDecodeError:
        pass
    return "", ""


def _load_meta_action_label_map(jsonl_path: str) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning(
                        f"[_load_meta_action_label_map] 第 {line_num} 行 JSON 解析失败: {e}"
                    )
                    continue
                if not isinstance(row, dict):
                    continue
                sid = str(row.get("sample_id", "") or "").strip()
                if not sid:
                    continue
                out[sid] = {
                    "longitudinal_action": str(row.get("longitudinal_action", "") or "").strip(),
                    "lateral_action": str(row.get("lateral_action", "") or "").strip(),
                }
    except FileNotFoundError:
        logger.error(f"[_load_meta_action_label_map] 文件不存在: {jsonl_path}")
    except Exception as e:
        logger.error(f"[_load_meta_action_label_map] 加载失败: {e}")
    return out


class MetaActionAccReward(ORM):
    """
    与数据集样本中的 GT assistant/solution 对比 meta action（严格一致）：
    纵向、横向分别与 GT 规范化后完全相同才算对；奖励 = 0.5 * long_ok + 0.5 * lat_ok。
    """

    def __init__(self, args=None, label_file: Optional[str] = None, **kwargs):
        super().__init__(args, **kwargs)
        if label_file:
            logger.warning(
                "[MetaActionAccReward] `label_file`/`META_ACTION_LABEL_FILE` 已废弃，"
                "当前会直接从数据集样本中的 solution 读取 GT meta action。"
            )

    def __call__(
        self,
        completions: List[str],
        solution: Optional[List[str]] = None,
        **kwargs,
    ) -> List[float]:
        rewards: List[float] = []
        if solution is None:
            logger.warning("[MetaActionAccReward] 缺少 solution，返回 0")
            return [0.0] * len(completions)

        for idx, gen in enumerate(completions):
            if idx >= len(solution):
                rewards.append(0.0)
                continue
            gt = solution[idx]
            r_long, r_lat = parse_actions_from_generation_ma(gt if isinstance(gt, str) else "")
            if not r_long or not r_lat:
                rewards.append(0.0)
                continue

            g = gen if isinstance(gen, str) else ""
            p_long, p_lat = parse_actions_from_generation_ma(g)
            if not p_long or not p_lat:
                rewards.append(0.0)
                continue

            ok_l = _ma_longitudinal_exact(p_long, r_long)
            ok_t = _ma_lateral_exact(p_lat, r_lat)
            r = (0.5 if ok_l else 0.0) + (0.5 if ok_t else 0.0)
            rewards.append(float(r))
            if idx < 2:
                logger.info(
                    f"[MetaActionAccReward] idx={idx} long_ok={ok_l} "
                    f"lat_ok={ok_t} r={r:.2f}"
                )
        return rewards


# ============================================================================
# 奖励函数: decision 与轨迹 meta action 一致性
# ============================================================================

class TrajectoryConsistencyReward(ORM):
    """
    将模型输出的 4 点 1Hz 轨迹（1s..4s）在 t=0 补零后插值为 2Hz（0.5s..4s），
    再推断纵向/横向 meta；与模型给出的 longitudinal/lateral 文本对比（无 ``<decision>`` 亦可）：
    - 纵向：使用 apply_lon_thresholds_from_report.py 同源的聚类 report suggested_rule 直接 apply；
      keep/accelerate/decelerate 等与轨迹推断原子词集合须一致；轨迹为静止类
      （stop、wait）时 decision 只写 stop 或只写 wait 亦可
    - 横向：推断结果为动作组（逗号分隔），预测 lateral 落在该组内即算对
    纵向、横向各占 0.5，合计最高 1.0。

    单轮：decision 与 trajectory 均从同一 ``completions`` 文本解析。
    按轮 credit（``per_turn_credit_mode=True``）：从 ``messages`` 中取第一轮 assistant 作 decision、
    第二轮 assistant 作 trajectory（需 GRPO 传入该 kwargs）。
    """

    def _score_decision_vs_traj_text(self, decision_src: str, traj_src: str) -> float:
        pred_long, pred_lat = _parse_decision_long_lat(decision_src if isinstance(decision_src, str) else "")
        pred_xy = _parse_trajectory_to_xy(traj_src if isinstance(traj_src, str) else "")
        if pred_long is None or pred_lat is None or pred_xy is None:
            return 0.0
        if pred_xy.shape[0] < 4:
            return 0.0
        traj_2hz = _interp_1hz_4s_traj_to_2hz_4s(pred_xy[:4])
        if not traj_2hz:
            return 0.0
        traj_long, traj_lat_group = infer_meta_action_lists_from_future_traj_reward(traj_2hz)
        long_ok = _longitudinal_actions_match(pred_long, traj_long)
        lat_group_tokens = {
            _norm_action_str(x)
            for x in traj_lat_group.split(",")
            if x.strip()
        }
        lat_ok = _norm_action_str(pred_lat) in lat_group_tokens
        return float((0.5 if long_ok else 0.0) + (0.5 if lat_ok else 0.0))

    def __call__(
        self,
        completions: List[str],
        **kwargs,
    ) -> List[float]:
        if kwargs.get('per_turn_credit_mode'):
            from swift.rlhf_trainers.turn_credit import extract_two_assistant_messages
            messages_list = kwargs.get('messages')
            if not messages_list:
                logger.warning('[TrajectoryConsistencyReward] per_turn_credit_mode 但缺少 messages，返回 0')
                return [0.0] * len(completions)
            rewards: List[float] = []
            for idx, m in enumerate(messages_list):
                try:
                    g1, g2 = extract_two_assistant_messages(m)
                    r = self._score_decision_vs_traj_text(g1, g2)
                    rewards.append(r)
                    if idx < 2:
                        logger.info(f'[TrajectoryConsistencyReward] per_turn idx={idx} r={r:.2f}')
                except Exception as e:
                    logger.warning(f'[TrajectoryConsistencyReward] per_turn 样本 {idx} 失败: {e}')
                    rewards.append(0.0)
            return rewards

        rewards: List[float] = []
        for idx, gen in enumerate(completions):
            try:
                g = gen if isinstance(gen, str) else ''
                r = self._score_decision_vs_traj_text(g, g)
                rewards.append(r)
                if idx < 2:
                    pred_long, pred_lat = _parse_decision_long_lat(g)
                    pred_xy = _parse_trajectory_to_xy(g)
                    if pred_xy is not None and pred_xy.shape[0] >= 4:
                        traj_2hz = _interp_1hz_4s_traj_to_2hz_4s(pred_xy[:4])
                        if traj_2hz:
                            traj_long, _ = infer_meta_action_lists_from_future_traj_reward(traj_2hz)
                            n_long_pred = _norm_action_str(pred_long) if pred_long else ''
                            n_long_traj = _norm_action_str(traj_long)
                            logger.info(
                                f"[TrajectoryConsistencyReward] 样本 {idx}: pred L='{n_long_pred}' "
                                f"traj L='{n_long_traj}' r={r:.2f}"
                            )
            except Exception as e:
                logger.warning(f"[TrajectoryConsistencyReward] 样本 {idx} 失败: {e}")
                rewards.append(0.0)
        return rewards


# ============================================================================
# 奖励函数 3: 格式奖励 (thinking tag + Future trajectory 标题)
# ============================================================================

class TrajectoryFormatReward(ORM):
    """
    格式奖励：
    - `<thinking> ... </thinking>`：0.5 分
    - `Future trajectory:`：0.5 分
    """

    _THINKING_TAG_RE = re.compile(r"<thinking>.*?</thinking>", re.DOTALL | re.IGNORECASE)

    def __call__(
        self,
        completions: List[str],
        **kwargs,
    ) -> List[float]:
        rewards: List[float] = []
        for gen in completions:
            if not isinstance(gen, str) or not gen:
                rewards.append(0.0)
                continue
            score = 0.0
            if self._THINKING_TAG_RE.search(gen) is not None:
                score += 0.5
            if "Future trajectory:" in gen:
                score += 0.5
            rewards.append(float(score))
        return rewards


# ============================================================================
# 奖励函数 4: 输出长度奖励 R_length（分段：超长线性惩罚至 -1）
# ============================================================================

class TrajectoryLengthReward(ORM):
    """
    按生成长度 |y| 计算 R_length(y)，默认 |y| 为 completion 字符串字符数 len(completion)。

    论文中的 L_max、L_cache 对应参数 traj_length_l_max、traj_length_l_cache；
    GRPO 训练时与 swift 的 args 同名，可由命令行传入（如 --traj_length_l_max 4096）。
    """

    def __init__(
        self,
        args=None,
        traj_length_l_max: float = 8192.0,
        traj_length_l_cache: float = 1024.0,
        **kwargs,
    ):
        super().__init__(args, **kwargs)
        self.l_max = float(traj_length_l_max)
        self.l_cache = float(traj_length_l_cache)
        logger.info(
            f"[TrajectoryLengthReward] 初始化 L_max={self.l_max}, L_cache={self.l_cache}"
        )

    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        rewards: List[float] = []
        for gen in completions:
            if not isinstance(gen, str):
                gen = "" if gen is None else str(gen)
            n = len(gen)
            rewards.append(float(_length_reward_scalar(n, self.l_max, self.l_cache)))
        return rewards


# ============================================================================
# 注册奖励函数到 orms 字典
# ============================================================================

orms['ade_5s'] = TrajectoryADE5sReward
orms['ade_4s'] = TrajectoryADE4sReward
orms['rfs'] = TrajectoryRFSReward
orms['pdms'] = PDMSReward
orms['PDMS_reward'] = PDMSReward
orms['meta_action_acc'] = MetaActionAccReward
orms['traj_consistency'] = TrajectoryConsistencyReward
orms['traj_format'] = TrajectoryFormatReward
orms['traj_length'] = TrajectoryLengthReward

logger.info(
    "[TrajectoryReward] 已注册奖励函数: ade_5s, ade_4s, rfs, pdms/PDMS_reward, "
    "meta_action_acc, traj_consistency, traj_format, traj_length"
)
