"""Portable redraw of Figure A from the accompanying CSV/JSON files only."""
from __future__ import annotations

import argparse
import csv
import io
import math
from pathlib import Path


def load_data(directory: Path):
    data = {'pruning': [], 'no_pruning': []}
    with (directory / 'plot_data.csv').open(newline='') as stream:
        for raw in csv.DictReader(stream):
            record = {'condition': raw['condition'], 'step': int(raw['step']),
                      'cell_fb_ms': float(raw['cell_fb_ms']),
                      'activation_percent': float(raw['activation_percent']),
                      'active_candidates': int(raw['active_candidates']),
                      'saved_activation_bytes': int(raw['saved_activation_bytes']),
                      'initial_activation_bytes': int(raw['initial_activation_bytes'])}
            assert record['condition'] in data
            assert math.isfinite(record['cell_fb_ms']) and record['cell_fb_ms'] > 0
            expected = 100 * record['saved_activation_bytes'] / record['initial_activation_bytes']
            assert record['activation_percent'] == expected
            data[record['condition']].append(record)
    for records in data.values():
        assert [record['step'] for record in records] == list(range(0, 2001, 20))
        assert records[0]['activation_percent'] == 100 and records[0]['active_candidates'] == 45
    with (directory / 'pruning_steps.csv').open(newline='') as stream:
        events = [{key: int(value) for key, value in row.items()} for row in csv.DictReader(stream)]
    assert len(events) == 15 and sum(event['removed_candidates'] for event in events) == 19
    remaining = 45
    for event in events:
        remaining -= event['removed_candidates']
        assert remaining == event['active_candidates_after']
        assert data['pruning'][event['step'] // 20]['active_candidates'] == remaining
    assert remaining == data['pruning'][-1]['active_candidates'] == 26
    return data, events


def make_figure(data, events, *, large=False):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from matplotlib.ticker import MaxNLocator, PercentFormatter

    font = 11 if large else 7
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': font,
                         'axes.labelsize': font, 'xtick.labelsize': font - .5,
                         'ytick.labelsize': font - .5, 'pdf.fonttype': 42, 'svg.fonttype': 'none'}):
        fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 7) if large else (3.5, 5.2))
        fig.subplots_adjust(left=.13 if large else .20, right=.97, top=.86,
                            bottom=.23 if large else .27, hspace=.16)
        fields = ('cell_fb_ms', 'activation_percent', 'active_candidates')
        for condition in ('no_pruning', 'pruning'):
            records = data[condition]
            x = [record['step'] for record in records]
            for index, (axis, field) in enumerate(zip(axes, fields)):
                axis.plot(x, [record[field] for record in records],
                          color='#009E73' if condition == 'pruning' else '#777777',
                          label='Pruning' if condition == 'pruning' else 'w/o pruning (measured)',
                          linestyle='-' if condition == 'pruning' else '--',
                          marker='^' if condition == 'pruning' else None,
                          markersize=2.8 if large else 1.8, linewidth=1.1 if large else .9,
                          drawstyle='steps-post' if index == 2 else 'default')
        for axis in axes:
            for event in events:
                axis.axvline(event['step'], color='#009E73', alpha=.13, linewidth=.6, zorder=0)
            axis.set_xlim(0, 2000)
            axis.set_ylim(bottom=0)
            axis.grid(axis='y', color='#dedede', linewidth=.5)
            axis.spines[['top', 'right']].set_visible(False)
            axis.yaxis.set_major_locator(MaxNLocator(nbins=3))
        axes[0].set_ylabel('Cell F+B\n(ms)')
        axes[1].set_ylabel('Saved activation\n(% of initial)')
        axes[2].set_ylabel('Active\ncandidates')
        axes[1].set_ylim(0, max(105, 1.05 * max(record['activation_percent']
                                             for records in data.values() for record in records)))
        axes[1].set_yticks([0, 25, 50, 75, 100])
        axes[1].yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
        axes[2].set_yticks([0, 15, 30, 45])
        axes[2].set_xticks([0, 500, 1000, 1500, 2000])
        axes[2].set_xlabel('Completed graph updates')
        axes[0].legend(loc='lower left', frameon=False, fontsize=font - .6, handlelength=2)
        axes[1].text(.98, .04, 'Vertical lines: actual pruning steps', transform=axes[1].transAxes,
                     ha='right', va='bottom', fontsize=font - .9)
        fig.suptitle('Citation: pruning dynamics', x=.53, y=.98, fontweight='bold', fontsize=font + 1)
        fig.text(.53, .925, '2,000 updates', ha='center', va='center', fontsize=font)
        footer = fig.text(.025, .025,
                          'All 101 fixed probes retained, including step 0.\n'
                          'Saved activation: measured step 0 = 100%.\n'
                          'Graph-side candidate-module measurements.',
                          fontsize=font - 1, va='bottom', linespacing=1.35)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        gap = axes[2].xaxis.label.get_window_extent(renderer).y0 - footer.get_window_extent(renderer).y1
        assert gap > 4, 'Footer overlaps x-axis label'
        assert footer.get_window_extent(renderer).x1 <= fig.bbox.x1
        for index, (axis, field) in enumerate(zip(axes, fields)):
            lines = [line for line in axis.lines if len(line.get_xdata()) == 101]
            assert len(lines) == 2
            for condition, line in zip(('no_pruning', 'pruning'), lines):
                assert list(line.get_xdata()) == [row['step'] for row in data[condition]]
                assert list(line.get_ydata()) == [row[field] for row in data[condition]]
        visible = [text.get_text() for text in fig.texts]
        visible += [text.get_text() for axis in axes for text in axis.texts]
        visible += [text.get_text() for text in axes[0].get_legend().get_texts()]
        assert not any(word in text.lower() for word in ('rho', 'ρ', 'seed', 'gpu') for text in visible)
        return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    directory = args.data_dir.resolve()
    output = (args.output_dir or directory / 'redrawn').resolve()
    output.mkdir(parents=True, exist_ok=False)
    data, events = load_data(directory)
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    for large in (False, True):
        fig = make_figure(data, events, large=large)
        for extension in (('png',) if large else ('png', 'pdf', 'svg')):
            filename = 'figure_large.png' if large else f'figure.{extension}'
            buffer = io.BytesIO()
            fig.savefig(buffer, format=extension, dpi=240 if large else 400, facecolor='white')
            raw = buffer.getvalue()
            with (output / filename).open('xb') as stream:
                stream.write(raw)
        plt.close(fig)
    print(f'Figures saved to {output}')


if __name__ == '__main__':
    main()
