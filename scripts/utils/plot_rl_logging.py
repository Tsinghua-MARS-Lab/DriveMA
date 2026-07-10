#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


DEFAULT_GROUPS = {
    'optimization': [
        'loss',
        'grad_norm',
        'learning_rate',
        'kl',
        'step_time',
        'train_speed(s/it)',
        'memory(GiB)',
    ],
    'reward': [
        'reward',
        'reward_std',
        'frac_reward_zero_std',
        'rewards/meta_action_acc/mean',
        'rewards/meta_action_acc/std',
        'rewards/rfs/mean',
        'rewards/rfs/std',
        'rewards/traj_consistency/mean',
        'rewards/traj_consistency/std',
    ],
    'completion': [
        'completions/mean_length',
        'completions/min_length',
        'completions/max_length',
        'completions/clipped_ratio',
        'num_turns',
    ],
    'clipping': [
        'clip_ratio/low_mean',
        'clip_ratio/low_min',
        'clip_ratio/high_mean',
        'clip_ratio/high_max',
        'clip_ratio/region_mean',
    ],
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Plot RL training curves from logging.jsonl.'
    )
    parser.add_argument('jsonl', type=Path, help='Path to logging.jsonl')
    parser.add_argument(
        '--output-dir',
        type=Path,
        default='/cephfs/zhengwc/FluidDrive/ms-swift-3.5/output/Qwen3.5-2B/Posttrained/K2/RL-rfs-20ep/v0-20260422-115306',
        help='Directory to save figures. Defaults to <jsonl_dir>/plots',
    )
    parser.add_argument(
        '--x-axis',
        choices=['step', 'epoch', 'index'],
        default='step',
        help='X axis for all plots.',
    )
    parser.add_argument(
        '--smooth',
        type=float,
        default=0.6,
        help='EMA smoothing factor in [0, 1). Set 0 to disable smoothing.',
    )
    parser.add_argument(
        '--dpi',
        type=int,
        default=180,
        help='Figure DPI.',
    )
    parser.add_argument(
        '--metrics',
        nargs='*',
        default=None,
        help='Optional explicit metric list. When set, only one custom figure is generated.',
    )
    return parser.parse_args()


def load_rows(path: Path):
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f'No rows found in {path}')
    return rows


def parse_step(raw_value, fallback):
    if isinstance(raw_value, str) and '/' in raw_value:
        head = raw_value.split('/', 1)[0]
        try:
            return int(head)
        except ValueError:
            return fallback
    if isinstance(raw_value, (int, float)):
        return raw_value
    return fallback


def build_series(rows):
    xs = {'step': [], 'epoch': [], 'index': []}
    metrics = {}
    for idx, row in enumerate(rows, start=1):
        xs['index'].append(idx)
        xs['step'].append(parse_step(row.get('global_step/max_steps'), idx))
        epoch = row.get('epoch')
        xs['epoch'].append(epoch if isinstance(epoch, (int, float)) else idx)
        for key, value in row.items():
            if key in {'elapsed_time', 'remaining_time', 'global_step/max_steps'}:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            metrics.setdefault(key, []).append(float(value))

    valid_metrics = {}
    expected_len = len(rows)
    for key, values in metrics.items():
        if len(values) == expected_len:
            valid_metrics[key] = values
    return xs, valid_metrics


def ema(values, factor):
    if factor <= 0:
        return values
    smoothed = []
    prev = None
    for value in values:
        if prev is None:
            prev = value
        else:
            prev = factor * prev + (1 - factor) * value
        smoothed.append(prev)
    return smoothed


def chunked(items, size):
    for start in range(0, len(items), size):
        yield items[start:start + size]


def sanitize_metric_name(name):
    safe = []
    for char in name:
        if char.isalnum() or char in {'-', '_'}:
            safe.append(char)
        else:
            safe.append('_')
    return ''.join(safe).strip('_') or 'metric'


def pick_groups(metrics, custom_metrics):
    if custom_metrics:
        return {'custom': [metric for metric in custom_metrics if metric in metrics]}

    groups = {}
    used = set()
    for group_name, group_metrics in DEFAULT_GROUPS.items():
        present = [metric for metric in group_metrics if metric in metrics]
        if present:
            groups[group_name] = present
            used.update(present)

    remaining = [metric for metric in sorted(metrics) if metric not in used]
    for index, subset in enumerate(chunked(remaining, 6), start=1):
        groups[f'other_{index}'] = subset
    return groups


def plot_group(group_name, group_metrics, xs, x_axis, metrics, output_path, smooth, dpi):
    cols = 2
    rows = math.ceil(len(group_metrics) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 7, rows * 3.6), squeeze=False)
    axes_flat = axes.flatten()
    x_values = xs[x_axis]

    for axis, metric in zip(axes_flat, group_metrics):
        values = metrics[metric]
        axis.plot(x_values, values, label='raw', linewidth=1.2, alpha=0.45)
        if smooth > 0:
            axis.plot(x_values, ema(values, smooth), label=f'ema({smooth})', linewidth=1.8)
        axis.set_title(metric, fontsize=10)
        axis.set_xlabel(x_axis)
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=8)

    for axis in axes_flat[len(group_metrics):]:
        axis.axis('off')

    fig.suptitle(f'RL training curves: {group_name}', fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def write_summary(path: Path, rows, xs, metrics):
    last_row = rows[-1]
    summary = {
        'num_points': len(rows),
        'last_step': xs['step'][-1],
        'last_epoch': xs['epoch'][-1],
        'available_metrics': sorted(metrics),
        'last_row': last_row,
    }
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')


def main():
    args = parse_args()
    rows = load_rows(args.jsonl)
    xs, metrics = build_series(rows)
    groups = pick_groups(metrics, args.metrics)

    if args.metrics and not groups['custom']:
        raise ValueError('None of the requested metrics were found in the log.')

    output_dir = args.output_dir or args.jsonl.parent / 'plots'
    output_dir.mkdir(parents=True, exist_ok=True)

    generated = []
    for group_name, group_metrics in groups.items():
        if not group_metrics:
            continue
        file_name = f'rl_{group_name}_{args.x_axis}.png'
        output_path = output_dir / file_name
        plot_group(
            group_name=group_name,
            group_metrics=group_metrics,
            xs=xs,
            x_axis=args.x_axis,
            metrics=metrics,
            output_path=output_path,
            smooth=args.smooth,
            dpi=args.dpi,
        )
        generated.append(output_path)

    write_summary(output_dir / 'rl_summary.json', rows, xs, metrics)

    print(f'Loaded {len(rows)} rows from {args.jsonl}')
    print(f'Last step: {xs["step"][-1]}')
    print(f'Last epoch: {xs["epoch"][-1]:.6f}')
    print(f'Generated {len(generated)} figure(s):')
    for path in generated:
        print(path)
    print(output_dir / 'rl_summary.json')


if __name__ == '__main__':
    main()
