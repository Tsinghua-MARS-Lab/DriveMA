#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
合并 ADE 与 RFS 测评：单次读取推理 jsonl，一次性输出 3s/5s ADE 与 mean RFS。

ADE：需 meta-jsonl 提供 preference_trajectories，计算与最高分偏好轨迹的 ADE。
RFS：需 meta-jsonl 提供 preference_trajectories（与 cal_rfs.py 一致）。
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# rater_feedback_utils
# ---------------------------------------------------------------------------


def _try_import_rater_feedback_utils():
    import rater_feedback_utils

    return rater_feedback_utils, None


# ---------------------------------------------------------------------------
# 轨迹重采样与 preference（RFS）
# ---------------------------------------------------------------------------


def _resample_traj_xy_np(
    traj_xy: Any,
    target_len: int,
    dst_t_start: float = 0.0,
    dst_t_end: float = 1.0,
) -> Optional[np.ndarray]:
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


def _extract_pref(pref: Any) -> Tuple[List[np.ndarray], np.ndarray]:
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


def _initial_speed_from_meta_obj(obj: Dict[str, Any]) -> Optional[float]:
    vel = obj.get("current_vel", None)
    if not (
        isinstance(vel, list)
        and len(vel) >= 2
    ):
        past_vel = obj.get("past_vel", None)
        if (
            isinstance(past_vel, list)
            and past_vel
            and isinstance(past_vel[-1], list)
            and len(past_vel[-1]) >= 2
        ):
            vel = past_vel[-1]
    if not (
        isinstance(vel, list)
        and len(vel) >= 2
    ):
        return None
    try:
        vx = float(vel[0])
        vy = float(vel[1])
    except (TypeError, ValueError):
        return None
    return float(np.sqrt(vx * vx + vy * vy))


def _load_pref_and_initial_speed_maps_from_jsonl(
    jsonl_path: str,
) -> Tuple[Dict[str, Any], Dict[str, float]]:
    pref_map: Dict[str, Any] = {}
    initial_speed_map: Dict[str, float] = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            sid = obj.get("sample_id", None)
            if isinstance(sid, str):
                pref_map[sid] = obj.get("preference_trajectories", None)
                speed = _initial_speed_from_meta_obj(obj)
                if speed is not None:
                    initial_speed_map[sid] = speed
    return pref_map, initial_speed_map


_FUTURE_TRAJ_KEY_RE = re.compile(
    r'["\']?\s*future_trajectory\s*["\']?\s*:\s*',
    re.IGNORECASE,
)

_BRACKET_PAIR_RE = re.compile(r"\[([-\d.eE+]+),\s*([-\d.eE+]+)\]")


def _xy_array_from_points(points: List[Tuple[float, float]]) -> Optional[np.ndarray]:
    if not points:
        return None
    return np.asarray(points, dtype=np.float32)


def _extract_first_future_trajectory_from_str(
    traj_str: str,
) -> List[Tuple[float, float]]:
    m = _FUTURE_TRAJ_KEY_RE.search(traj_str)
    if not m:
        return []
    start_val = m.end()
    if start_val >= len(traj_str):
        return []
    while start_val < len(traj_str) and traj_str[start_val] in " \t\n\r":
        start_val += 1
    if start_val >= len(traj_str) or traj_str[start_val] != "[":
        return []
    depth = 0
    end_val = start_val
    for i in range(start_val, len(traj_str)):
        if traj_str[i] == "[":
            depth += 1
        elif traj_str[i] == "]":
            depth -= 1
            if depth == 0:
                end_val = i + 1
                break
    if depth != 0:
        return []
    try:
        arr = json.loads(traj_str[start_val:end_val])
    except Exception:
        return []
    if not isinstance(arr, list) or not arr:
        return []
    points: List[Tuple[float, float]] = []
    for pt in arr:
        if isinstance(pt, (list, tuple)) and len(pt) >= 2:
            try:
                points.append((float(pt[0]), float(pt[1])))
            except (TypeError, ValueError):
                continue
    return points


def _extract_bracket_pairs(text: str) -> List[Tuple[float, float]]:
    points: List[Tuple[float, float]] = []
    for match in _BRACKET_PAIR_RE.finditer(text):
        try:
            points.append((float(match.group(1)), float(match.group(2))))
        except ValueError:
            continue
    return points


def _extract_from_answer_tag(traj_str: str) -> List[Tuple[float, float]]:
    m = re.search(r"<answer>(.*?)</answer>", traj_str, re.DOTALL)
    if not m:
        return []
    return _extract_bracket_pairs(m.group(1))


def parse_trajectory(traj_input: Any) -> List[Tuple[float, float]]:
    if traj_input is None:
        return []
    if isinstance(traj_input, dict):
        traj = traj_input.get("future_trajectory", None)
        if not isinstance(traj, list) or not traj:
            return []
        points: List[Tuple[float, float]] = []
        for pt in traj:
            if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                try:
                    points.append((float(pt[0]), float(pt[1])))
                except (TypeError, ValueError):
                    continue
        return points
    if isinstance(traj_input, str):
        points = _extract_from_answer_tag(traj_input)
        if points:
            return points
        points = _extract_first_future_trajectory_from_str(traj_input)
        if points:
            return points
        traj_str = traj_input.replace("Future trajectory:", "").strip()
        return _extract_bracket_pairs(traj_str)
    return []


def _first_nonempty_field(record: Dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = record.get(key, None)
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def _generation_trajectory_value(record: Dict[str, Any]) -> Any:
    return _first_nonempty_field(record, "generation_turn2", "generation")


def _parse_generation_to_xy(gen: Any) -> Optional[np.ndarray]:
    return _xy_array_from_points(parse_trajectory(gen))


def _prepare_prediction_xy_for_metric(
    pred_raw_xy: Any,
    length_seconds: int,
    frequency: int,
) -> Optional[np.ndarray]:
    target_T = int(length_seconds) * int(frequency)
    pred_raw_xy = np.asarray(pred_raw_xy, dtype=np.float32)
    if pred_raw_xy.ndim != 2 or pred_raw_xy.shape[1] != 2:
        return None
    if pred_raw_xy.shape[0] != int(length_seconds):
        return None

    origin = np.zeros((1, 2), dtype=np.float32)
    return _resample_traj_xy_np(
        np.concatenate([origin, pred_raw_xy], axis=0),
        target_T,
        dst_t_start=1.0 / float(target_T),
        dst_t_end=1.0,
    )


def _pad_or_truncate_xy_to_len(
    traj_xy: Any,
    target_len: int,
) -> Optional[np.ndarray]:
    traj_xy = np.asarray(traj_xy, dtype=np.float32)
    if traj_xy.ndim != 2 or traj_xy.shape[0] == 0 or traj_xy.shape[1] != 2:
        return None
    if traj_xy.shape[0] > target_len:
        return traj_xy[:target_len]
    if traj_xy.shape[0] < target_len:
        padding = np.repeat(
            traj_xy[-1:, :], repeats=target_len - traj_xy.shape[0], axis=0
        )
        return np.concatenate([traj_xy, padding], axis=0)
    return traj_xy


def _highest_score_preference_xy(
    pref: Any,
    target_len: int,
) -> Optional[np.ndarray]:
    rater_trajs, rater_scores = _extract_pref(pref)
    if len(rater_trajs) == 0 or rater_scores.size == 0:
        return None

    order = np.argsort(-rater_scores)
    for idx in order:
        pref_xy = _pad_or_truncate_xy_to_len(rater_trajs[int(idx)], target_len)
        if pref_xy is not None:
            return pref_xy
    return None


# ---------------------------------------------------------------------------
# ADE
# ---------------------------------------------------------------------------


def calculate_ade(
    pred_xy: np.ndarray,
    target_xy: np.ndarray,
    num_steps: int,
) -> float:
    if pred_xy.shape[0] < num_steps or target_xy.shape[0] < num_steps:
        return float("nan")
    diff = pred_xy[:num_steps] - target_xy[:num_steps]
    return float(np.mean(np.linalg.norm(diff, axis=-1)))


def compute_ade_from_records(
    records: List[Dict[str, Any]],
    pref_map: Dict[str, Any],
    scenario_id_set: Optional[set],
    frequency: int,
    length_seconds: int,
) -> Dict[str, Any]:
    target_T = int(frequency) * int(length_seconds)
    steps_3s = int(frequency) * 3
    steps_5s = int(frequency) * 5
    ade_3s_list: List[float] = []
    ade_5s_list: List[float] = []
    skipped_samples = 0
    no_pref_count = 0
    pred_parse_fail_count = 0
    pref_parse_fail_count = 0
    pred_len_counter: Counter[int] = Counter()
    scenario_filter_total = 0
    scenario_filter_kept = 0
    scenario_filter_skipped = 0
    total_samples = len(records)

    for line_num, data in enumerate(records, 1):
        if not isinstance(data, dict):
            skipped_samples += 1
            continue

        if scenario_id_set is not None:
            scenario_filter_total += 1
            sid = data.get("sample_id", None)
            if not isinstance(sid, str) or sid not in scenario_id_set:
                scenario_filter_skipped += 1
                continue
            scenario_filter_kept += 1

        sid = data.get("sample_id", None)
        generation = data.get("generation", "")

        if not isinstance(sid, str) or not generation:
            skipped_samples += 1
            continue

        pref = pref_map.get(sid, None)
        if pref is None:
            no_pref_count += 1
            skipped_samples += 1
            continue

        pred_raw_xy = _parse_generation_to_xy(generation)
        if pred_raw_xy is None or pred_raw_xy.shape[0] == 0:
            pred_parse_fail_count += 1
            skipped_samples += 1
            continue
        pred_len_counter[int(pred_raw_xy.shape[0])] += 1

        pred_xy = _prepare_prediction_xy_for_metric(
            pred_raw_xy, length_seconds, frequency
        )
        if pred_xy is None:
            pred_parse_fail_count += 1
            skipped_samples += 1
            continue

        target_xy = _highest_score_preference_xy(pref, target_T)
        if target_xy is None:
            pref_parse_fail_count += 1
            skipped_samples += 1
            continue

        ade_3s = calculate_ade(pred_xy, target_xy, steps_3s)
        if not np.isnan(ade_3s):
            ade_3s_list.append(ade_3s)

        ade_5s = calculate_ade(pred_xy, target_xy, steps_5s)
        if not np.isnan(ade_5s):
            ade_5s_list.append(ade_5s)

    results: Dict[str, Any] = {
        "total_samples": total_samples,
        "valid_samples_3s": len(ade_3s_list),
        "valid_samples_5s": len(ade_5s_list),
        "skipped_samples": skipped_samples,
        "no_preference_samples": no_pref_count,
        "prediction_parse_or_length_fail": pred_parse_fail_count,
        "preference_parse_fail": pref_parse_fail_count,
        "prediction_raw_len_distribution": dict(sorted(pred_len_counter.items())),
    }

    if scenario_id_set is not None:
        results["scenario_filter_total"] = scenario_filter_total
        results["scenario_filter_kept"] = scenario_filter_kept
        results["scenario_filter_skipped"] = scenario_filter_skipped

    if len(ade_3s_list) > 0:
        results["ade_3s"] = float(np.mean(ade_3s_list))
        results["ade_3s_std"] = float(np.std(ade_3s_list))
    else:
        results["ade_3s"] = None
        results["ade_3s_std"] = None

    if len(ade_5s_list) > 0:
        results["ade_5s"] = float(np.mean(ade_5s_list))
        results["ade_5s_std"] = float(np.std(ade_5s_list))
    else:
        results["ade_5s"] = None
        results["ade_5s_std"] = None

    return results


# ---------------------------------------------------------------------------
# scenario / records
# ---------------------------------------------------------------------------


def load_scenario_id_set(scenario_json: Optional[str]) -> Optional[set]:
    if not scenario_json:
        return None
    try:
        with open(scenario_json, "r", encoding="utf-8") as f:
            scenario_obj = json.load(f)
        scenario_ids = scenario_obj.get("scenario_ids", None)
        if isinstance(scenario_ids, list):
            s = set(str(x) for x in scenario_ids)
            print(
                f"[ADE/RFS] 从 {scenario_json} 读取 scenario_ids 数量: {len(s)}"
            )
            return s
        print(
            f"[ADE/RFS] 警告: {scenario_json} 中未找到有效的 'scenario_ids' 列表，"
            "将不进行 scenario 筛选。"
        )
    except Exception as e:
        print(f"[ADE/RFS] 读取 scenario-json 失败（将不筛选）: {e}")
    return None


def load_infer_records(infer_jsonl: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(infer_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                records.append(obj)
    return records


# ---------------------------------------------------------------------------
# RFS 主流程（与 cal_rfs 逻辑一致）
# ---------------------------------------------------------------------------


def run_rfs_evaluation(
    args: argparse.Namespace,
    records: List[Dict[str, Any]],
    pref_map: Dict[str, Any],
    initial_speed_map: Dict[str, float],
    scenario_id_set: Optional[set],
    rater_feedback_utils: Any,
) -> Optional[Dict[str, Any]]:
    freq = int(args.frequency)
    length_seconds = int(args.length_seconds)
    if freq <= 0 or length_seconds <= 0:
        raise ValueError("frequency 和 length-seconds 必须为正数。")
    target_T = freq * length_seconds
    expected_pred_T = length_seconds

    if not records:
        print("[RFS] 推理 jsonl 中没有有效样本。")
        return None

    print(f"[RFS] 从推理记录加载了 {len(records)} 条记录")

    prepared: Dict[int, Dict[str, Any]] = {}
    no_gen_count = 0
    no_sid_count = 0
    no_pref_count = 0
    no_initial_speed_count = 0
    empty_pref_count = 0
    invalid_pref_traj_count = 0
    gen_parse_fail_count = 0
    pred_len_mismatch_count = 0
    scenario_filter_skip_count = 0
    scenario_filter_keep_count = 0
    pred_traj_len_counter: Counter[int] = Counter()
    pred_len_mismatch_examples: List[Tuple[str, int]] = []
    pref_traj_len_counter: Counter[int] = Counter()
    non_target_pref_len_examples: List[Tuple[str, int]] = []
    gen_parse_fail_examples: List[Dict[str, Any]] = []
    sample_ids_infer_examples: List[Tuple[Any, str]] = []
    sample_ids_not_found_examples: List[str] = []

    for idx, rec in enumerate(records):
        sid = rec.get("sample_id", None)
        gen = rec.get("generation", None)

        if not isinstance(sid, str):
            no_sid_count += 1
            if len(sample_ids_infer_examples) < 3:
                sample_ids_infer_examples.append((sid, type(sid).__name__))
            continue
        if scenario_id_set is not None and sid not in scenario_id_set:
            scenario_filter_skip_count += 1
            continue
        if scenario_id_set is not None:
            scenario_filter_keep_count += 1
        if gen is None:
            no_gen_count += 1
            continue

        if len(sample_ids_infer_examples) < 5:
            sample_ids_infer_examples.append((sid, type(sid).__name__))

        pref = pref_map.get(sid, None)
        if pref is None:
            no_pref_count += 1
            if len(sample_ids_not_found_examples) < 5:
                sample_ids_not_found_examples.append(sid)
            continue
        init_speed = initial_speed_map.get(sid, None)
        if init_speed is None:
            no_initial_speed_count += 1
            continue
        rater_trajs_raw, rater_scores = _extract_pref(pref)
        if len(rater_trajs_raw) == 0 or rater_scores.size == 0:
            empty_pref_count += 1
            continue

        gen_xy = _parse_generation_to_xy(gen)
        if gen_xy is None or gen_xy.shape[0] == 0 or gen_xy.shape[1] != 2:
            gen_parse_fail_count += 1
            if len(gen_parse_fail_examples) < 3:
                gen_parse_fail_examples.append(
                    {
                        "sample_id": sid,
                        "generation_head": gen[:200]
                        if isinstance(gen, str)
                        else str(gen),
                    }
                )
            continue

        pred_raw_xy = gen_xy.astype(np.float32)
        pred_len = int(pred_raw_xy.shape[0])
        pred_traj_len_counter[pred_len] += 1
        if pred_len != expected_pred_T:
            pred_len_mismatch_count += 1
            if len(pred_len_mismatch_examples) < 5:
                pred_len_mismatch_examples.append((sid, pred_len))
            continue

        pred_xy = _prepare_prediction_xy_for_metric(
            pred_raw_xy, length_seconds, freq
        )
        if pred_xy is None:
            gen_parse_fail_count += 1
            continue

        rater_trajs_for_metric: List[np.ndarray] = []
        for txy in rater_trajs_raw:
            txy_arr = np.asarray(txy, dtype=np.float32)
            if (
                txy_arr.ndim != 2
                or txy_arr.shape[0] == 0
                or txy_arr.shape[1] != 2
            ):
                invalid_pref_traj_count += 1
                continue
            n_waypoints = int(txy_arr.shape[0])
            pref_traj_len_counter[n_waypoints] += 1
            if (
                n_waypoints != target_T
                and len(non_target_pref_len_examples) < 5
            ):
                non_target_pref_len_examples.append((sid, n_waypoints))
            rater_trajs_for_metric.append(txy_arr)

        if not rater_trajs_for_metric:
            continue

        prepared[idx] = {
            "sample_id": sid,
            "pred_xy": pred_xy.astype(np.float32),
            "rater_trajs": rater_trajs_for_metric,
            "rater_scores": rater_scores.astype(np.float32),
            "init_speed": float(init_speed),
        }

    if not prepared:
        print(
            "[RFS] 没有样本同时满足：有 generation 且在 meta-jsonl 中有非空 preference_trajectories。"
        )
        print("[RFS] 调试信息：")
        print(f"  - 没有 sample_id 的记录数: {no_sid_count}")
        print(f"  - 没有 generation 的记录数: {no_gen_count}")
        print(f"  - 在 meta-jsonl 中找不到 sample_id 的记录数: {no_pref_count}")
        print(f"  - 缺少 current_vel/past_vel 的记录数: {no_initial_speed_count}")
        print(f"  - preference_trajectories 为空的记录数: {empty_pref_count}")
        print(f"  - 非法 preference 轨迹条数: {invalid_pref_traj_count}")
        print(
            f"  - 预测轨迹点数不等于 {expected_pred_T} 的记录数: "
            f"{pred_len_mismatch_count}"
        )
        print(
            f"  - generation 解析失败的记录数（已命中 meta-jsonl 后）: {gen_parse_fail_count}"
        )
        if sample_ids_infer_examples:
            print(
                f"[RFS] infer-jsonl 中的 sample_id 示例（前5个）: "
                f"{[s[0] for s in sample_ids_infer_examples]}"
            )
        if sample_ids_not_found_examples:
            print(
                f"[RFS] 在 meta-jsonl 中找不到的 sample_id 示例（前5个）: "
                f"{sample_ids_not_found_examples}"
            )
        if gen_parse_fail_examples:
            print("[RFS] generation 解析失败样例（最多3个）:")
            for e in gen_parse_fail_examples:
                print(
                    f"  - sample_id={e.get('sample_id')}, "
                    f"generation_head={e.get('generation_head')}"
                )
        if pred_len_mismatch_examples:
            print(
                f"[RFS] 预测轨迹点数不匹配样例（最多5个）: "
                f"{pred_len_mismatch_examples}"
            )
        return None

    idxs = sorted(prepared.keys())
    B = len(idxs)
    pred_xy_list = [prepared[i]["pred_xy"] for i in idxs]
    rater_trajs_list = [prepared[i]["rater_trajs"] for i in idxs]
    rater_scores_list = [prepared[i]["rater_scores"] for i in idxs]
    initial_speed = np.asarray(
        [prepared[i]["init_speed"] for i in idxs], dtype=np.float32
    )

    if pred_traj_len_counter:
        print(
            "[RFS] prediction 原始轨迹点数分布（要求等于 "
            f"{expected_pred_T} 点；随后前补原点并插值到 {target_T} 点）: "
            f"{dict(sorted(pred_traj_len_counter.items()))}"
        )
    if pred_len_mismatch_examples:
        print(
            "[RFS] 已跳过预测轨迹点数不匹配样例（sample_id, 点数，最多5个）: "
            f"{pred_len_mismatch_examples}"
        )
    if pref_traj_len_counter:
        print(
            "[RFS] preference 轨迹点数分布（进入 metric 前，metric 内部会按 "
            f"{target_T} 点截断或补最后一点）: "
            f"{dict(sorted(pref_traj_len_counter.items()))}"
        )
    if non_target_pref_len_examples:
        print(
            "[RFS] 非目标点数 preference 示例（sample_id, 点数，最多5个）: "
            f"{non_target_pref_len_examples}"
        )
    if invalid_pref_traj_count > 0:
        print(f"[RFS] 跳过非法 preference 轨迹条数: {invalid_pref_traj_count}")

    prediction_trajectories = (
        np.stack(pred_xy_list, axis=0)[:, None, :, :].astype(np.float32)
    )
    prediction_probabilities = np.ones((B, 1), dtype=np.float32)

    metrics = rater_feedback_utils.get_rater_feedback_score(
        prediction_trajectories,
        prediction_probabilities,
        rater_trajs_list,
        rater_scores_list,
        initial_speed,
        frequency=freq,
        length_seconds=length_seconds,
        output_trust_region_visualization=False,
    )

    rfs = metrics.get("rater_feedback_score", None)
    if rfs is None:
        raise RuntimeError(
            "rater_feedback_utils 返回中没有 'rater_feedback_score' 字段。"
        )

    rfs_arr = np.asarray(rfs, dtype=np.float32).reshape(-1)
    if rfs_arr.shape[0] != B:
        raise RuntimeError(
            f"RFS 结果长度与样本数不一致：B={B}, got={rfs_arr.shape[0]}"
        )

    mean_rfs = float(np.mean(rfs_arr))
    print(
        f"[RFS] computed for {B} samples "
        f"(frequency={freq}Hz, length_seconds={length_seconds}s): "
        f"mean_RFS={mean_rfs:.6f}"
    )

    if args.score_th is not None:
        score_th = float(args.score_th)
        low_indices = np.where(rfs_arr < score_th)[0].tolist()
        low_records: List[Dict[str, Any]] = []
        for j in low_indices:
            info = prepared[idxs[j]]
            sid = info.get("sample_id", None)
            if sid is None:
                continue
            raw_scores = np.asarray(
                info.get("rater_scores", []), dtype=np.float32
            ).reshape(-1)
            raw_scores_list = [float(s) for s in raw_scores.tolist()]
            raw_score_mean = (
                float(np.mean(raw_scores)) if raw_scores.size > 0 else None
            )
            low_records.append(
                {
                    "sample_id": sid,
                    "RFS": float(rfs_arr[j]),
                    "raw_scores": raw_scores_list,
                    "raw_score_mean": raw_score_mean,
                }
            )

        low_count = len(low_records)
        low_ratio = (low_count / float(B)) if B > 0 else 0.0
        print(
            f"[RFS] score_th={score_th:.6f}, "
            f"低分样本数={low_count}/{B} ({low_ratio:.2%})"
        )

        low_score_output = args.low_score_output
        if low_score_output is None:
            low_score_output = f"{args.infer_jsonl}.low_score_samples.jsonl"
        with open(low_score_output, "w", encoding="utf-8") as f:
            for rec in low_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[RFS] 已导出低分样本到：{low_score_output}")

    output_records: List[Dict[str, Any]] = []
    for j, ridx in enumerate(idxs):
        info = prepared[ridx]
        sid = info.get("sample_id", None)
        pred_xy = info["pred_xy"]
        pref = pref_map.get(sid, None) if sid is not None else None

        out_obj: Dict[str, Any] = {
            "sample_id": sid,
            "pred_xy_resampled": pred_xy.tolist(),
            "preference_trajectories": pref,
            "RFS": float(rfs_arr[j]),
        }
        output_records.append(out_obj)

    if args.output_jsonl is not None:
        with open(args.output_jsonl, "w", encoding="utf-8") as f:
            for rec in output_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[RFS] 已写出带 RFS 的 jsonl：{args.output_jsonl}")

    return {
        "rfs_valid_samples": B,
        "mean_rfs": mean_rfs,
        "frequency_hz": freq,
        "length_seconds": length_seconds,
    }


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="合并计算 ADE（3s/5s）与 RFS，单次读取推理 jsonl。"
    )
    ap.add_argument(
        "--infer-jsonl",
        "--input-jsonl",
        dest="infer_jsonl",
        type=str,
        default=(
            '/cephfs/zhengwc/FluidDrive/ms-swift-3.5/output/Qwen3.5-2B/Mutil-Turn-w-mask-subsample/RL-rfs-20ep/v0-20260415-171333/checkpoint-620/val_samples_479_minus_434.jsonl'
        ),
        help="推理结果 jsonl（ADE/RFS 需含 sample_id 与 generation）",
    )
    ap.add_argument(
        "--meta-jsonl",
        type=str,
        default=(
            "/cephfs/zhengwc/FluidDrive/ms-swift-3.5/data/train_data/meta_data/val_samples_479.jsonl"
        ),
        help="原始数据 jsonl（ADE/RFS：sample_id 与 preference_trajectories；RFS 还需 current_vel 或 past_vel）",
    )
    ap.add_argument(
        "--scenario-json",
        "--scenario_json",
        dest="scenario_json",
        type=str,
        default=None,
        help="可选：JSON 含 scenario_ids，按 sample_id 筛选",
    )
    ap.add_argument(
        "--output-jsonl",
        type=str,
        default=None,
        help="可选：写出带 RFS 的逐行 jsonl",
    )
    ap.add_argument(
        "--score-th",
        type=float,
        default=4.5,
        help="RFS 低于该阈值的样本导出（见 --low-score-output）",
    )
    ap.add_argument(
        "--low-score-output",
        type=str,
        default=None,
        help="低分样本导出路径；默认 <infer-jsonl>.low_score_samples.jsonl",
    )
    ap.add_argument(
        "--frequency",
        type=int,
        default=4,
        help="RFS 目标频率 Hz",
    )
    ap.add_argument(
        "--length-seconds",
        type=int,
        default=5,
        help="RFS 轨迹长度（秒）",
    )
    ap.add_argument(
        "--output-summary-json",
        type=str,
        default=None,
        help="可选：将 ADE 与 RFS 汇总指标写入该 JSON 文件",
    )
    ap.add_argument(
        "--skip-ade",
        action="store_true",
        help="只算 RFS，跳过 ADE",
    )
    ap.add_argument(
        "--skip-rfs",
        action="store_true",
        help="只算 ADE，跳过 RFS",
    )
    return ap


def main() -> None:
    ap = build_argparser()
    args = ap.parse_args()

    if not Path(args.infer_jsonl).exists():
        print(f"[ERROR] 推理 jsonl 不存在: {args.infer_jsonl}")
        return

    scenario_id_set = load_scenario_id_set(args.scenario_json)
    records = load_infer_records(args.infer_jsonl)
    if not records:
        print("[ERROR] 推理 jsonl 无有效记录。")
        return

    print(f"[ADE/RFS] 从 {args.infer_jsonl} 读取 {len(records)} 条记录")

    summary: Dict[str, Any] = {
        "infer_jsonl": args.infer_jsonl,
        "meta_jsonl": args.meta_jsonl,
    }

    pref_map: Dict[str, Any] = {}
    initial_speed_map: Dict[str, float] = {}
    if not args.skip_ade or not args.skip_rfs:
        if not Path(args.meta_jsonl).exists():
            print(f"[ERROR] meta-jsonl 不存在，无法计算 ADE/RFS: {args.meta_jsonl}")
            return
        pref_map, initial_speed_map = (
            _load_pref_and_initial_speed_maps_from_jsonl(args.meta_jsonl)
        )
        print(
            f"[ADE/RFS] 从 {args.meta_jsonl} 加载了 "
            f"{len(pref_map)} 个 sample_id 的偏好轨迹映射，"
            f"{len(initial_speed_map)} 个 current speed"
        )

    if not args.skip_ade:
        ade_results = compute_ade_from_records(
            records,
            pref_map,
            scenario_id_set,
            frequency=int(args.frequency),
            length_seconds=int(args.length_seconds),
        )
        summary["ade"] = ade_results
        print("\n" + "=" * 60)
        print("ADE 计算结果")
        print("=" * 60)
        print(f"总样本数: {ade_results['total_samples']}")
        print(f"有效样本数 (3s): {ade_results['valid_samples_3s']}")
        print(f"有效样本数 (5s): {ade_results['valid_samples_5s']}")
        print(f"跳过样本数: {ade_results['skipped_samples']}")
        print(f"无 preference 样本数: {ade_results['no_preference_samples']}")
        print(
            "预测解析或原始点数不匹配样本数: "
            f"{ade_results['prediction_parse_or_length_fail']}"
        )
        print(f"preference 解析失败样本数: {ade_results['preference_parse_fail']}")
        print(
            "预测原始点数分布: "
            f"{ade_results['prediction_raw_len_distribution']}"
        )
        if ade_results.get("scenario_filter_total") is not None:
            print(
                f"scenario 筛选: total={ade_results['scenario_filter_total']}, "
                f"kept={ade_results['scenario_filter_kept']}, "
                f"skipped={ade_results['scenario_filter_skipped']}"
            )
        print()
        if ade_results["ade_3s"] is not None:
            print(
                f"3s ADE: {ade_results['ade_3s']:.6f} ± "
                f"{ade_results['ade_3s_std']:.6f}"
            )
        else:
            print("3s ADE: 无法计算（无有效样本）")
        if ade_results["ade_5s"] is not None:
            print(
                f"5s ADE: {ade_results['ade_5s']:.6f} ± "
                f"{ade_results['ade_5s_std']:.6f}"
            )
        else:
            print("5s ADE: 无法计算（无有效样本）")
        print("=" * 60)

    if not args.skip_rfs:
        rater_feedback_utils, import_err = _try_import_rater_feedback_utils()
        if rater_feedback_utils is None:
            print(
                f"[ERROR] 无法导入 rater_feedback_utils: {import_err}"
            )
        else:
            rfs_out = run_rfs_evaluation(
                args,
                records,
                pref_map,
                initial_speed_map,
                scenario_id_set,
                rater_feedback_utils,
            )
            if rfs_out is not None:
                summary["rfs"] = rfs_out

    if args.output_summary_json:
        out_path = Path(args.output_summary_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\n[ADE/RFS] 汇总已写入: {out_path}")

    print("\n[ADE/RFS] 全部指标统计完成。")


if __name__ == "__main__":
    main()
