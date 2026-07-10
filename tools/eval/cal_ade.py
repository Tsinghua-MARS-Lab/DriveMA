#!/usr/bin/env python3
"""
计算轨迹预测的ADE (Average Displacement Error)和FDE (Final Displacement Error)指标
读取推理结果的jsonl文件，计算3s/4s/5s ADE和3s/4s/5s FDE
"""

import json
import argparse
import re
import numpy as np
from pathlib import Path
from typing import Any, List, Tuple

# 匹配字符串中第一个 "future_trajectory": [[...], ...] 的键名（允许单/双引号）
_FUTURE_TRAJ_KEY_RE = re.compile(
    r'["\']?\s*future_trajectory\s*["\']?\s*:\s*',
    re.IGNORECASE,
)


def _extract_first_future_trajectory_from_str(traj_str: str) -> List[Tuple[float, float]]:
    """
    在字符串中找第一个 "future_trajectory": [[x,y], [x,y], ...]，解析为 [(x,y), ...]。
    """
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
    points = []
    for pt in arr:
        if isinstance(pt, (list, tuple)) and len(pt) >= 2:
            try:
                points.append((float(pt[0]), float(pt[1])))
            except (TypeError, ValueError):
                continue
    return points


# 匹配 [x, y] 形式的二维点（支持科学计数法）；可含省略号等无关字符，findall 只取合法点对
_BRACKET_PAIR_RE = re.compile(r'\[([-\d.eE+]+),\s*([-\d.eE+]+)\]')


def _extract_bracket_pairs(text: str) -> List[Tuple[float, float]]:
    """
    从任意字符串中提取所有 [num, num] 形式的二维点，例如：
    [1.0, -0.05], [2, 3], ... [5, 6]
    或 Future trajectory: [a,b], [c,d], ...
    """
    points = []
    for match in _BRACKET_PAIR_RE.finditer(text):
        try:
            points.append((float(match.group(1)), float(match.group(2))))
        except ValueError:
            continue
    return points


def _extract_from_answer_tag(traj_str: str) -> List[Tuple[float, float]]:
    """
    从 <answer>...</answer> 标签中提取轨迹点，格式如：
    [-0.0679, -0.0001], [-0.0679, -0.0001], ...
    """
    m = re.search(r'<answer>(.*?)</answer>', traj_str, re.DOTALL)
    if not m:
        return []
    return _extract_bracket_pairs(m.group(1))


def parse_trajectory(traj_input: Any) -> List[Tuple[float, float]]:
    """
    解析轨迹，提取坐标点。支持下列形式：
    1) 含 <answer>[x,y], [x,y], ...</answer> 标签的字符串；
    2) 字符串中第一个 "future_trajectory": [[x,y], [x,y], ...]；
    3) 纯文本中的 [x,y], [x,y], ...（可有 "Future trajectory:" 前缀，或中间带 ... 省略）；
    4) dict 且含 "future_trajectory" 键，值为 [[x,y], ...]。
    
    Args:
        traj_input: 轨迹字符串或 dict（如 generation/reference 字段）
    
    Returns:
        坐标点列表 [(x1, y1), (x2, y2), ...]
    """
    if traj_input is None:
        return []
    # 4) dict 且含 future_trajectory
    if isinstance(traj_input, dict):
        traj = traj_input.get("future_trajectory", None)
        if not isinstance(traj, list) or not traj:
            return []
        points = []
        for pt in traj:
            if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                try:
                    points.append((float(pt[0]), float(pt[1])))
                except (TypeError, ValueError):
                    continue
        return points
    # 字符串
    if isinstance(traj_input, str):
        # 1) 先尝试 <answer>...</answer> 标签
        points = _extract_from_answer_tag(traj_input)
        if points:
            return points
        # 2) 再尝试 future_trajectory
        points = _extract_first_future_trajectory_from_str(traj_input)
        if points:
            return points
        # 3) 回退：整段字符串中所有 [x,y] 点对（含无前缀、仅坐标；与 reference/generation 纯轨迹格式一致）
        traj_str = traj_input.replace("Future trajectory:", "").strip()
        points = _extract_bracket_pairs(traj_str)
        return points
    return []


def calculate_ade(pred_points: List[Tuple[float, float]], 
                  gt_points: List[Tuple[float, float]], 
                  num_seconds: int) -> float:
    """
    计算指定秒数的ADE
    
    Args:
        pred_points: 预测轨迹点列表
        gt_points: GT轨迹点列表
        num_seconds: 计算前N秒的ADE（例如3表示3s ADE）
    
    Returns:
        ADE值
    """
    if len(pred_points) < num_seconds or len(gt_points) < num_seconds:
        # 如果轨迹点不足，返回NaN
        return float('nan')
    
    # 取前num_seconds个点
    pred_subset = pred_points[:num_seconds]
    gt_subset = gt_points[:num_seconds]
    
    # 计算每个时刻的位移误差
    errors = []
    for pred_pt, gt_pt in zip(pred_subset, gt_subset):
        # 计算欧氏距离
        error = np.sqrt((pred_pt[0] - gt_pt[0])**2 + (pred_pt[1] - gt_pt[1])**2)
        errors.append(error)
    
    # 返回平均位移误差
    return np.mean(errors)


def calculate_fde(pred_points: List[Tuple[float, float]],
                  gt_points: List[Tuple[float, float]],
                  num_seconds: int) -> float:
    """
    计算指定秒数的FDE

    Args:
        pred_points: 预测轨迹点列表
        gt_points: GT轨迹点列表
        num_seconds: 计算第N秒终点的FDE（例如3表示3s FDE）

    Returns:
        FDE值
    """
    if len(pred_points) < num_seconds or len(gt_points) < num_seconds:
        return float('nan')

    pred_pt = pred_points[num_seconds - 1]
    gt_pt = gt_points[num_seconds - 1]
    return np.sqrt((pred_pt[0] - gt_pt[0])**2 + (pred_pt[1] - gt_pt[1])**2)


def _load_scenario_id_set(scenario_path: str):
    """
    从 JSON 或 JSONL 加载用于过滤的 sample_id 集合。
    - JSON 对象：优先读 scenario_ids 列表，其次 sample_ids、sample_id（列表）；
      sample_id 为字符串时视为单个 id。
    - JSONL：每行一个 JSON 对象，读取其 sample_id 字段（字符串去重）。
    """
    p = Path(scenario_path)
    if not p.is_file():
        return None, f"文件不存在: {scenario_path}"

    with open(scenario_path, "r", encoding="utf-8") as sf:
        raw = sf.read()
    if not raw.strip():
        return None, "文件为空"

    # 整文件 JSON（含 pretty-print 多行）
    try:
        scenario_obj = json.loads(raw)
        if isinstance(scenario_obj, dict):
            scenario_ids = scenario_obj.get("scenario_ids")
            if not isinstance(scenario_ids, list):
                scenario_ids = scenario_obj.get("sample_ids")
            if not isinstance(scenario_ids, list):
                scenario_ids = scenario_obj.get("sample_id")
            if isinstance(scenario_ids, list):
                id_set = {str(s) for s in scenario_ids if str(s).strip()}
                return id_set, None
            one = scenario_obj.get("sample_id", None)
            if isinstance(one, str) and one.strip():
                return {one.strip()}, None
        # 非上述格式的整文件 JSON：按 JSONL 再试（例如误用 .json 后缀的 jsonl）
    except json.JSONDecodeError:
        pass

    # JSONL：逐行 sample_id
    id_set = set()
    bad_lines = 0
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            bad_lines += 1
            continue
        if not isinstance(obj, dict):
            bad_lines += 1
            continue
        sid = obj.get("sample_id", None)
        if sid is None:
            continue
        s = str(sid).strip()
        if s:
            id_set.add(s)

    if id_set:
        return id_set, None
    msg = "按 JSONL 解析未收集到任何 sample_id"
    if bad_lines:
        msg += f"（{bad_lines} 行 JSON 解析失败或非对象）"
    return None, msg


def calculate_ade_from_jsonl(jsonl_path: str, scenario_json: str = None) -> dict:
    """
    从jsonl文件计算ADE/FDE指标
    
    Args:
        jsonl_path: jsonl文件路径
    
    Returns:
        包含3s/4s/5s ADE和3s/4s/5s FDE的字典
    """
    ade_3s_list = []
    ade_4s_list = []
    ade_5s_list = []
    sample_ade_5s = []
    fde_3s_list = []
    fde_4s_list = []
    fde_5s_list = []
    total_samples = 0
    skipped_samples = 0
    scenario_filter_total = 0
    scenario_filter_kept = 0
    scenario_filter_skipped = 0

    # 若提供了 scenario_json，则读取其中的 id 集合用于按 sample_id 过滤（支持 JSON 与 JSONL）
    scenario_id_set = None
    if scenario_json:
        id_set, err = _load_scenario_id_set(scenario_json)
        if id_set is not None:
            scenario_id_set = id_set
            print(
                f"[INFO] 从 {scenario_json} 读取用于过滤的 sample_id 数量: "
                f"{len(scenario_id_set)}"
            )
        else:
            print(
                f"[WARN] 读取 scenario 文件失败或未得到有效 id 列表，将不进行筛选: {err}"
            )

    print(f"[INFO] 开始读取文件: {jsonl_path}")

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
                total_samples += 1
                sample_id = data.get("sample_id", None)

                # 若有 scenario 过滤，则按 sample_id 过滤
                if scenario_id_set is not None:
                    scenario_filter_total += 1
                    # 这里假设 scenario_ids 与 infer jsonl 里的 sample_id 一致
                    if not isinstance(sample_id, str) or sample_id not in scenario_id_set:
                        scenario_filter_skipped += 1
                        continue
                    scenario_filter_kept += 1

                generation = data.get("generation_turn2", "")
                if not generation:
                    generation = data.get("generation", "")
                reference = data.get("reference", "")

                if not generation or not reference:
                    skipped_samples += 1
                    print(f"[WARN] 第 {line_num} 行缺少generation或reference字段，跳过")
                    continue

                # 解析轨迹
                pred_points = parse_trajectory(generation)
                gt_points = parse_trajectory(reference)

                if len(pred_points) == 0 or len(gt_points) == 0:
                    skipped_samples += 1
                    print(f"[WARN] 第 {line_num} 行轨迹解析失败，跳过")
                    continue

                # 计算3s ADE
                ade_3s = calculate_ade(pred_points, gt_points, 3)
                if not np.isnan(ade_3s):
                    ade_3s_list.append(ade_3s)

                # 计算4s ADE
                ade_4s = calculate_ade(pred_points, gt_points, 4)
                if not np.isnan(ade_4s):
                    ade_4s_list.append(ade_4s)

                # 计算5s ADE
                ade_5s = calculate_ade(pred_points, gt_points, 5)
                if not np.isnan(ade_5s):
                    ade_5s_list.append(ade_5s)
                    sample_ade_5s.append(
                        {
                            "sample_id": sample_id,
                            "ade_5s": float(ade_5s),
                        }
                    )

                # 计算3s FDE
                fde_3s = calculate_fde(pred_points, gt_points, 3)
                if not np.isnan(fde_3s):
                    fde_3s_list.append(fde_3s)

                # 计算4s FDE
                fde_4s = calculate_fde(pred_points, gt_points, 4)
                if not np.isnan(fde_4s):
                    fde_4s_list.append(fde_4s)

                # 计算5s FDE
                fde_5s = calculate_fde(pred_points, gt_points, 5)
                if not np.isnan(fde_5s):
                    fde_5s_list.append(fde_5s)

            except json.JSONDecodeError as e:
                skipped_samples += 1
                print(f"[WARN] 第 {line_num} 行JSON解析失败: {e}")
                continue
            except Exception as e:
                skipped_samples += 1
                print(f"[WARN] 第 {line_num} 行处理失败: {e}")
                continue
    
    # 计算平均ADE/FDE
    results = {
        "total_samples": total_samples,
        "valid_samples_3s": len(ade_3s_list),
        "valid_samples_4s": len(ade_4s_list),
        "valid_samples_5s": len(ade_5s_list),
        "valid_samples_fde_3s": len(fde_3s_list),
        "valid_samples_fde_4s": len(fde_4s_list),
        "valid_samples_fde_5s": len(fde_5s_list),
        "skipped_samples": skipped_samples,
        "sample_ade_5s": sample_ade_5s,
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
    
    if len(ade_4s_list) > 0:
        results["ade_4s"] = float(np.mean(ade_4s_list))
        results["ade_4s_std"] = float(np.std(ade_4s_list))
    else:
        results["ade_4s"] = None
        results["ade_4s_std"] = None

    if len(ade_5s_list) > 0:
        results["ade_5s"] = float(np.mean(ade_5s_list))
        results["ade_5s_std"] = float(np.std(ade_5s_list))
    else:
        results["ade_5s"] = None
        results["ade_5s_std"] = None

    if len(fde_3s_list) > 0:
        results["fde_3s"] = float(np.mean(fde_3s_list))
        results["fde_3s_std"] = float(np.std(fde_3s_list))
    else:
        results["fde_3s"] = None
        results["fde_3s_std"] = None

    if len(fde_4s_list) > 0:
        results["fde_4s"] = float(np.mean(fde_4s_list))
        results["fde_4s_std"] = float(np.std(fde_4s_list))
    else:
        results["fde_4s"] = None
        results["fde_4s_std"] = None

    if len(fde_5s_list) > 0:
        results["fde_5s"] = float(np.mean(fde_5s_list))
        results["fde_5s_std"] = float(np.std(fde_5s_list))
    else:
        results["fde_5s"] = None
        results["fde_5s_std"] = None
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="计算轨迹预测的ADE/FDE指标（3s/4s/5s ADE和3s/4s/5s FDE）"
    )
    
    parser.add_argument(
        "--input_jsonl",
        type=str,
        default="/cephfs/zhengwc/FluidDrive/ms-swift-3.5/output/Qwen3.5-2B/Mutil-Turn-w-mask-subsample/Second-Turn/v0-20260413-205852/checkpoint-950/val_samples_479_minus_434_gt.jsonl",
        help="输入JSONL文件路径（推理结果）",
    )

    parser.add_argument(
        "--scenario_json",
        type=str,
        default='',
        help=(
            "可选：JSON 文件（含 scenario_ids 或 sample_ids 或 sample_id 列表）"
            "或 JSONL（每行对象的 sample_id），仅对这些 sample_id 计算 ADE。"
        ),
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="可选：输出结果文件路径（JSON格式），不指定则只打印到控制台",
    )
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not Path(args.input_jsonl).exists():
        print(f"[ERROR] 输入文件不存在: {args.input_jsonl}")
        return
    
    # 计算ADE/FDE（如提供 scenario_json，则只对对应 sample_id 的样本计算）
    results = calculate_ade_from_jsonl(args.input_jsonl, args.scenario_json)
    
    # 打印结果
    print("\n" + "="*60)
    print("ADE/FDE计算结果")
    print("="*60)
    print(f"总样本数: {results['total_samples']}")
    print(f"有效样本数 (3s): {results['valid_samples_3s']}")
    print(f"有效样本数 (4s): {results['valid_samples_4s']}")
    print(f"有效样本数 (5s): {results['valid_samples_5s']}")
    print(f"跳过样本数: {results['skipped_samples']}")
    print()
    
    if results['ade_3s'] is not None:
        print(f"3s ADE: {results['ade_3s']:.6f} ± {results['ade_3s_std']:.6f}")
    else:
        print("3s ADE: 无法计算（无有效样本）")
    
    if results['ade_4s'] is not None:
        print(f"4s ADE: {results['ade_4s']:.6f} ± {results['ade_4s_std']:.6f}")
    else:
        print("4s ADE: 无法计算（无有效样本）")

    if results['ade_5s'] is not None:
        print(f"5s ADE: {results['ade_5s']:.6f} ± {results['ade_5s_std']:.6f}")
    else:
        print("5s ADE: 无法计算（无有效样本）")

    if results['fde_3s'] is not None:
        print(f"3s FDE: {results['fde_3s']:.6f} ± {results['fde_3s_std']:.6f}")
    else:
        print("3s FDE: 无法计算（无有效样本）")

    if results['fde_4s'] is not None:
        print(f"4s FDE: {results['fde_4s']:.6f} ± {results['fde_4s_std']:.6f}")
    else:
        print("4s FDE: 无法计算（无有效样本）")

    if results['fde_5s'] is not None:
        print(f"5s FDE: {results['fde_5s']:.6f} ± {results['fde_5s_std']:.6f}")
    else:
        print("5s FDE: 无法计算（无有效样本）")
    print("="*60)
    
    # 保存结果到文件（如果指定）
    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n[INFO] 结果已保存到: {output_path}")


if __name__ == "__main__":
    main()
