"""Plots training curves (accuracies and loss) from the log.txt of one or more run directories.

When training was resumed, pass all run directories in order; epochs are concatenated (a later
directory overrides epochs it repeats) and the resume points are marked with a dashed line.
The first epoch where an accuracy reaches 100% is marked with a ring and the epoch number.

Understood log formats:
    Epoch   3 | Loss: 0.4132
        Train objects  : exact 97.6%, left object 97.9%, ...
        Held-out object: exact 16.5%, ...
and the single-line form
    Epoch   3 | Loss: 0.4132 | Train accuracy: 90.00% | Test accuracy: 80.00%

Usage:
    python plot_training_curves.py out_vqa/run1 out_vqa/run2_resumed
    python plot_training_curves.py out_multi_captioner/20261031-224448-encoder-dim-32 --metric shape
    python plot_training_curves.py out_vqa/run1 --metrics all color shape relation --splits Held-out
"""
import argparse
import os
import re

import matplotlib.pyplot as plt

EPOCH_PATTERN = re.compile(r'^Epoch\s+(\d+)\s*\|\s*Loss:\s*([\d.eE+-]+)(.*)$')
# "Train objects  : exact 97.6%, left object 97.9%" -> name, list of (metric, value)
RESULT_LINE_PATTERN = re.compile(r'^\s+([^:]+?)\s*:\s*(.*%.*)$')
METRIC_PATTERN = re.compile(r'([A-Za-z][\w\- ]*?)\s+([\d.]+)%')
# "| Train accuracy: 90.00% | Test accuracy: 80.00%" -> name, value
INLINE_ACCURACY_PATTERN = re.compile(r'\|\s*([^|:]+?)\s*:\s*([\d.]+)%')

# Categorical palette in fixed order (blue, orange, aqua, yellow, magenta, green, violet, red).
SERIES_COLORS = ('#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948')
INK, MUTED_INK, GRID, BASELINE, SURFACE = '#0b0b0b', '#898781', '#e1e0d9', '#c3c2b7', '#fcfcfb'


LINE_STYLES = ('-', '--', ':', '-.')


def parse_log(log_path) -> dict:
    """Parses one log file.

    :param log_path: (str) path of the log.txt
    :return: (dict) epoch -> {'loss': float, 'accuracy': {split name: {metric name: accuracy in percent}}},
        metrics in log order. In the single-line format each split has the single metric 'accuracy'.
    """
    epochs, current = {}, None
    with open(log_path) as log_file:
        for line in log_file:
            line = line.rstrip('\n')
            match = EPOCH_PATTERN.match(line)
            if match:
                current = {'loss': float(match.group(2)), 'accuracy': {}}
                epochs[int(match.group(1))] = current
                for name, value in INLINE_ACCURACY_PATTERN.findall(match.group(3)):
                    current['accuracy'][name] = {'accuracy': float(value)}
                continue
            match = RESULT_LINE_PATTERN.match(line)
            if current is None or not match:
                current = None if not line.startswith(' ') else current
                continue
            metrics = METRIC_PATTERN.findall(match.group(2))
            if metrics:
                current['accuracy'][match.group(1)] = {name.strip(): float(value) for name, value in metrics}
    return epochs


def select_series(epochs, metrics, split_filters) -> list:
    """Picks the curves to plot.

    :param metrics: (list) metric names; None takes the first metric of each split (e.g. 'exact', 'all')
    :param split_filters: (list) keep only splits containing one of these substrings (case-insensitive); None keeps all
    :return: (list) (split, metric, [(epoch, accuracy), ...]) per curve, splits outer, metrics inner
    """
    splits = list(dict.fromkeys(split for values in epochs.values() for split in values['accuracy']))
    if split_filters:
        splits = [s for s in splits if any(f.lower() in s.lower() for f in split_filters)]
    series = []
    for split in splits:
        available = list(dict.fromkeys(m for v in epochs.values() for m in v['accuracy'].get(split, {})))
        for metric in (metrics or available[:1]):
            points = [(epoch, values['accuracy'][split][metric]) for epoch, values in epochs.items()
                      if metric in values['accuracy'].get(split, {})]
            if points:
                series.append((split, metric, points))
            else:
                print(f"Warning: no metric '{metric}' for '{split}' (available: {', '.join(available)})")
    return series


def collect_runs(run_directories) -> tuple:
    """Parses and concatenates the logs of all run directories.

    :return: (tuple) epoch -> parsed values, list of epochs where a resumed run starts
    """
    all_epochs, resume_epochs = {}, []
    for index, directory in enumerate(run_directories):
        log_path = directory if directory.endswith('.txt') else os.path.join(directory, 'log.txt')
        if not os.path.isfile(log_path):
            raise FileNotFoundError(f'No log found: {log_path}')
        epochs = parse_log(log_path)
        if not epochs:
            print(f'Warning: no epochs found in {log_path}')
            continue
        if index > 0 and all_epochs:
            resume_epochs.append(min(epochs))
        all_epochs.update(epochs)
    return dict(sorted(all_epochs.items())), resume_epochs


def style_axis(axis):
    axis.set_facecolor(SURFACE)
    axis.grid(axis='y', color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    for side in ('top', 'right'):
        axis.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        axis.spines[side].set_color(BASELINE)
    axis.tick_params(colors=MUTED_INK, labelcolor=INK)


def plot_curves(epochs, series, resume_epochs, title, output_path, metrics):
    """Saves an accuracy plot (top) and a loss plot (bottom) sharing the epoch axis.

    With one metric each split gets its own color. With several metrics the color encodes the metric
    and the line style the split (solid, dashed, dotted, dash-dotted).
    """
    splits = list(dict.fromkeys(split for split, _, _ in series))
    series_metrics = list(dict.fromkeys(metric for _, metric, _ in series))
    multi_metric = len(series_metrics) > 1
    # Direct end labels only for few lines; with many lines the legend (outside the plot) does the job.
    direct_labels = len(series) <= 4
    figure, (accuracy_axis, loss_axis) = plt.subplots(
        2, 1, figsize=(9 if direct_labels else 12, 7), sharex=True,
        gridspec_kw={'height_ratios': (3, 1.4)}, facecolor=SURFACE)
    style_axis(accuracy_axis)
    style_axis(loss_axis)

    end_labels = []  # [y, x, text] of the direct labels at the line ends
    annotated_epochs = set()
    for split, metric, points in series:
        if multi_metric:
            color = SERIES_COLORS[series_metrics.index(metric) % len(SERIES_COLORS)]
            line_style = LINE_STYLES[splits.index(split) % len(LINE_STYLES)]
            name = f'{split} · {metric}'
        else:
            color = SERIES_COLORS[splits.index(split) % len(SERIES_COLORS)]
            line_style = '-'
            name = split
        xs, ys = zip(*points)
        first_perfect = next((x for x, y in points if y >= 100.0), None)
        legend_name = name if first_perfect is None else f'{name} (100% at epoch {first_perfect})'
        accuracy_axis.plot(xs, ys, color=color, linestyle=line_style, linewidth=2, label=legend_name,
                           marker='o' if direct_labels else None, markersize=4)
        end_labels.append([ys[-1], xs[-1], f'{name} {ys[-1]:.1f}%'])
        if first_perfect is not None:
            accuracy_axis.plot(first_perfect, 100.0, marker='o', markersize=11, markerfacecolor='none',
                               markeredgecolor=color, markeredgewidth=2, zorder=5)
            # One text per epoch, even if several curves reach 100% there.
            if direct_labels and first_perfect not in annotated_epochs:
                accuracy_axis.annotate(f'100% at epoch {first_perfect}', (first_perfect, 100.0),
                                       xytext=(0, -18 - 12 * len(annotated_epochs)), textcoords='offset points',
                                       ha='center', fontsize=8, color=INK,
                                       arrowprops=dict(arrowstyle='-', color=color, linewidth=1))
                annotated_epochs.add(first_perfect)
            print(f'{name}: first reaches 100% at epoch {first_perfect}')
        else:
            print(f'{name}: max {max(ys):.1f}% (never reaches 100%)')

    if direct_labels:
        # Direct labels at the line ends, in ink color, pushed apart vertically so equal values don't overlap.
        end_labels.sort(key=lambda label: -label[0])
        min_gap = 4.5  # in percent (data units of the 0..105 accuracy axis)
        label_x_offset = max(0.3, 0.01 * len(epochs))
        previous_y = float('inf')
        for y, x, text in end_labels:
            previous_y = min(y, previous_y - min_gap)
            accuracy_axis.annotate(text, (x, y), xytext=(x + label_x_offset, previous_y), textcoords='data',
                                   va='center', fontsize=8, color=INK, annotation_clip=False)

    xs = list(epochs)
    loss_axis.plot(xs, [epochs[x]['loss'] for x in xs], color=MUTED_INK, linewidth=2)
    loss_axis.set_yscale('log')
    loss_axis.set_ylabel('Loss (log)', color=INK)
    loss_axis.set_xlabel('Epoch', color=INK)

    for resume_epoch in resume_epochs:
        for axis in (accuracy_axis, loss_axis):
            axis.axvline(resume_epoch - 0.5, color=MUTED_INK, linestyle='--', linewidth=1)
        accuracy_axis.text(resume_epoch - 0.5, 0.02, ' resumed', transform=accuracy_axis.get_xaxis_transform(),
                           fontsize=8, color=MUTED_INK, va='bottom')

    accuracy_axis.set_ylim(0, 105)
    accuracy_axis.set_ylabel(f'Accuracy ({", ".join(series_metrics) if metrics else "first metric"}) %', color=INK)
    if len(series) > 1 and direct_labels:
        accuracy_axis.legend(loc='lower right', frameon=False, fontsize=9)
        accuracy_axis.set_xlim(min(xs) - 0.5, max(xs) + 0.5 + max(1.0, 0.15 * len(xs)))  # room for end labels
    elif len(series) > 1:
        accuracy_axis.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=8,
                             handlelength=3.5)
        accuracy_axis.set_xlim(min(xs) - 0.5, max(xs) + 0.5)
    accuracy_axis.set_title(title, color=INK, fontsize=11, loc='left')
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, facecolor=SURFACE)
    plt.close(figure)
    print(f'Saved {output_path}')


def main():
    parser = argparse.ArgumentParser(description='Plot accuracy/loss curves from training logs.')
    parser.add_argument('run_directories', nargs='+',
                        help='run directories (or log.txt files), in training order if resumed')
    parser.add_argument('--metric', '--metrics', dest='metrics', nargs='+', default=None,
                        help="one or more metrics per result line, e.g. 'exact' shape color 'left object' "
                             "(default: the first one of each line)")
    parser.add_argument('--splits', nargs='+', default=None,
                        help="only plot result lines containing one of these substrings, e.g. Held-out")
    parser.add_argument('--output', default=None,
                        help='output image (default: training_curves.png in the last run directory)')
    args = parser.parse_args()

    epochs, resume_epochs = collect_runs(args.run_directories)
    if not epochs:
        raise SystemExit('No epochs found in the given logs.')
    series = select_series(epochs, args.metrics, args.splits)
    if not series:
        raise SystemExit('None of the requested metrics/splits found in the given logs.')
    last_directory = args.run_directories[-1]
    if last_directory.endswith('.txt'):
        last_directory = os.path.dirname(last_directory) or '.'
    suffix = ('_' + '-'.join(m.replace(' ', '_') for m in args.metrics)) if args.metrics else ''
    output_path = args.output or os.path.join(last_directory, f'training_curves{suffix}.png')
    title = ' + '.join(os.path.basename(os.path.normpath(d)) for d in args.run_directories)
    plot_curves(epochs, series, resume_epochs, title, output_path, args.metrics)


if __name__ == '__main__':
    main()
