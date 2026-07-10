# Copyright (c) ModelScope Contributors. All rights reserved.
"""Helpers for per-turn GRPO credit assignment (multi-assistant rollouts)."""
from typing import Any, Dict, List, Tuple


def extract_two_assistant_messages(messages: Any) -> Tuple[str, str]:
    """Return (first_assistant_text, second_assistant_text) from chat messages."""
    if not messages:
        return '', ''
    texts: List[str] = []
    for m in messages:
        if m.get('role') != 'assistant':
            continue
        c = m.get('content')
        if isinstance(c, str):
            texts.append(c)
    if len(texts) >= 2:
        return texts[0], texts[1]
    if len(texts) == 1:
        return texts[0], ''
    return '', ''


def get_gt_assistant_turns(inp: Dict[str, Any]) -> Tuple[str, str]:
    """
    Read ground-truth assistant strings for turn 1 / turn 2 from the training row.

    Priority:
    - data_dict['gt_assistant_turns'] as [gt1, gt2]
    - top-level 'gt_assistant_turns'
    - 'solution' as a list/tuple of two strings
    - 'gt_assistant_1' / 'gt_assistant_2' or 'solution' string for turn 2 only
    """
    dd = inp.get('data_dict') or {}
    gt = dd.get('gt_assistant_turns') if isinstance(dd, dict) else None
    if gt is None:
        gt = inp.get('gt_assistant_turns')
    if isinstance(gt, (list, tuple)) and len(gt) >= 2:
        return str(gt[0] or ''), str(gt[1] or '')

    sol = inp.get('solution')
    if isinstance(sol, (list, tuple)) and len(sol) >= 2:
        return str(sol[0] or ''), str(sol[1] or '')

    g1 = ''
    if isinstance(dd, dict):
        g1 = str(dd.get('gt_assistant_1') or inp.get('gt_assistant_1') or '')
    g2 = None
    if isinstance(dd, dict):
        g2 = dd.get('gt_assistant_2')
    if g2 is None:
        g2 = inp.get('gt_assistant_2')
    if g2 is None:
        g2 = sol
    if isinstance(g2, (list, tuple)):
        g2 = g2[-1] if g2 else ''
    g2 = str(g2 or '')
    return g1, g2
