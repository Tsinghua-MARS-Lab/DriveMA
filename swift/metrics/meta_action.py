# Copyright (c) ModelScope Contributors. All rights reserved.
# 与 tools/cal_meta_action_acc.py 中判定规则保持一致（纵向 stop/wait 等价，横向三组并集）。
import json
import re
from typing import Dict, Tuple

from transformers import EvalPrediction

from swift.utils import Serializer, get_logger

from .base import EvalMetrics

logger = get_logger()

_LAT_STRAIGHT = frozenset(
    {'straight', 'lane_follow', 'left_shift_slightly', 'right_shift_slightly'})
_LAT_LEFT = frozenset({
    'left_turn', 'left_lane_change', 'left_shift_slightly', 'lane_follow', 'turn_around'
})
_LAT_RIGHT = frozenset({
    'right_turn', 'right_lane_change', 'right_shift_slightly', 'lane_follow'
})
_LAT_GROUPS = (_LAT_STRAIGHT, _LAT_LEFT, _LAT_RIGHT)

_THINK_END_TAG = '</redacted_thinking>'

_DECISION_RE = re.compile(
    r'<decision>\s*longitudinal\s+action:\s*(.+?)(?:[,;])\s*lateral\s+action:\s*([^<]+?)\s*</decision>',
    re.IGNORECASE | re.DOTALL,
)

_PLAIN_ACTION_RE = re.compile(
    r'longitudinal\s+action:\s*(.+?)\s*[,;]\s*lateral\s+action:\s*([^\n<]+)',
    re.IGNORECASE | re.DOTALL,
)


def _norm_longitudinal(s: str) -> str:
    t = (s or '').strip().lower().replace(',', ' ')
    t = ' '.join(t.split())
    return t.replace(' ', '_')


def _parse_lateral_name(raw: str) -> str:
    t = (raw or '').strip().lower()
    t = re.sub(r'\s+', '_', t)
    return t


def longitudinal_match(pred: str, ref: str) -> bool:
    if not (pred or '').strip() or not (ref or '').strip():
        return False

    def _is_stop_wait(x: str) -> bool:
        t = (x or '').strip().lower().replace(',', ' ')
        parts = [a for a in t.split() if a]
        if not parts:
            return False
        sw = {'stop', 'wait'}
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
    if text.startswith('```json'):
        text = text[7:]
    if text.startswith('```'):
        text = text[3:]
    if text.endswith('```'):
        text = text[:-3]
    return text.strip()


def _json_after_think(text: str) -> str:
    if _THINK_END_TAG in text:
        return text.split(_THINK_END_TAG, 1)[1].strip()
    return text.strip()


def parse_actions_from_generation(generation: str) -> Tuple[str, str]:
    if not generation or not isinstance(generation, str):
        return '', ''

    m = _DECISION_RE.search(generation)
    if m:
        return m.group(1).strip(), m.group(2).strip()

    tail = _json_after_think(generation)
    m = _PLAIN_ACTION_RE.search(tail)
    if m:
        return m.group(1).strip(), m.group(2).strip()

    text = _strip_markdown_json_fence(tail)
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            lo = str(obj.get('longitudinal_action', '') or '').strip()
            la = str(obj.get('lateral_action', '') or '').strip()
            return lo, la
    except json.JSONDecodeError:
        pass
    return '', ''


class MetaActionMetrics(EvalMetrics):

    def compute_metrics(self, eval_prediction: EvalPrediction) -> Dict[str, float]:
        preds, labels = eval_prediction.predictions, eval_prediction.label_ids
        total = 0
        long_correct = 0
        lat_correct = 0
        both_correct = 0

        for i in range(preds.shape[0]):
            pred_str = Serializer.from_tensor(preds[i])
            label_str = Serializer.from_tensor(labels[i])
            p_long, p_lat = parse_actions_from_generation(pred_str)
            r_long, r_lat = parse_actions_from_generation(label_str)
            if not p_long or not p_lat or not r_long or not r_lat:
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

        if total == 0:
            logger.warning_once(
                '[meta_action] 无有效样本（解析失败或空标注），指标记为 0。请检查 predict_with_generate 与验证集格式。')
            return {
                'joint_acc': 0.,
                'longitudinal_acc': 0.,
                'lateral_acc': 0.,
            }

        return {
            'joint_acc': round(both_correct / total * 100, 6),
            'longitudinal_acc': round(long_correct / total * 100, 6),
            'lateral_acc': round(lat_correct / total * 100, 6),
        }
