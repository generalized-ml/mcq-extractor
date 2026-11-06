"""Plot quality scores and average recorded LLM API cost per PDF."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def read_experiments(path: Path) -> list[dict]:
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        required = {'experiment_number', 'experiment_name', 'keyword_score', 'semantic_score'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f'CSV must include {sorted(required)}')
        if not {'llm_cost_per_pdf_usd', 'llm_cost_usd', 'llm_api_cost'}.intersection(reader.fieldnames or []):
            raise ValueError('CSV must include llm_cost_per_pdf_usd')
        rows = list(reader)
    if not rows:
        raise ValueError('No experiments to plot')
    for row in rows:
        for key in ('keyword_score', 'semantic_score'):
            value = float(row[key])
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f'Invalid {key} for experiment {row["experiment_number"]}')
            row[key] = value
        raw_aggregate = row.get('aggregate_score', '').strip()
        row['aggregate_score'] = (float(raw_aggregate) if raw_aggregate else
                                  (row['keyword_score'] + row['semantic_score']) / 2)
        if not math.isfinite(row['aggregate_score']) or not 0 <= row['aggregate_score'] <= 1:
            raise ValueError(f'Invalid aggregate_score for experiment {row["experiment_number"]}')
        # Historical totals have no reliable PDF denominator; leave them unknown.
        raw = row.get('llm_cost_per_pdf_usd', '').strip()
        row['cost'] = None if raw.lower() in ('', 'none', 'null', 'nan') else float(raw)
        if row['cost'] is not None and (not math.isfinite(row['cost']) or row['cost'] < 0):
            raise ValueError('LLM API cost must be nonnegative and finite')
        if row.get('llm_cost_currency', 'USD') not in ('', 'USD'):
            raise ValueError('Only USD costs can be compared on this chart')
        row['partial_cost'] = (int(row.get('llm_missing_cost_calls') or 0) > 0
                               or int(row.get('llm_missing_cost_pdfs') or 0) > 0)
    return sorted(rows, key=lambda row: int(row['experiment_number']))


def plot_experiments(csv_path: Path, output: Path) -> Path:
    rows = read_experiments(csv_path)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, MultipleLocator
    from matplotlib.patches import Patch

    colors = ('#3979D6', '#19A78D', '#8266D7', '#E99A3B')
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.labelcolor': '#475569', 'text.color': '#172B45',
                         'xtick.color': '#475569', 'ytick.color': '#475569'}):
        fig, ax = plt.subplots(figsize=(max(10, len(rows)*2.2), 6.5), facecolor='#F5F7FB')
        ax.set_facecolor('#FFFFFF')
        cost_ax = ax.twinx()
        width = .18
        x = list(range(len(rows)))
        for offset, key, color in ((-1.5*width, 'keyword_score', colors[0]),
                                   (-.5*width, 'semantic_score', colors[1]),
                                   (.5*width, 'aggregate_score', colors[2])):
            bars = ax.bar([i+offset for i in x], [r[key] for r in rows], width*.88,
                          color=color, zorder=3)
            ax.bar_label(bars, labels=[f'{r[key]:.3f}' for r in rows], padding=7,
                         fontsize=10, color='#172B45')
        costs = [r['cost'] for r in rows if r['cost'] is not None]
        cost_ax.set_ylim(0, max(costs, default=0)*1.3 or .01)
        for i, row in enumerate(rows):
            value = row['cost']
            if value is None:
                cost_ax.annotate('Unknown', (i+1.5*width, 0), xytext=(0, 8),
                                 textcoords='offset points', ha='center', fontsize=9, color=colors[3])
                continue
            bars = cost_ax.bar(i+1.5*width, value, width*.88, color=colors[3], zorder=3,
                               hatch='///' if row['partial_cost'] else None,
                               edgecolor='#BA711F', linewidth=0)
            cost_ax.bar_label(bars, labels=[f'${value:.4f}' + ('*' if row['partial_cost'] else '')],
                              padding=7, fontsize=10, color='#8D541A')
        ax.set_ylim(0, 1.08)
        ax.yaxis.set_major_locator(MultipleLocator(.2))
        ax.set_ylabel('Quality score · higher is better', labelpad=12)
        cost_ax.set_ylabel('Recorded LLM API cost/PDF (USD) · lower is better', labelpad=14, color='#8D541A')
        cost_ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f'${value:.3f}'))
        ax.set_xticks(x, [f'Experiment {r["experiment_number"]}\n{r["experiment_name"]}' for r in rows])
        ax.tick_params(axis='x', length=0, pad=13, labelrotation=20 if len(rows)>4 else 0)
        ax.set_xlabel('Experiments', labelpad=16)
        ax.grid(axis='y', color='#E5EBF2', linewidth=.8, zorder=0)
        for axis in (ax, cost_ax):
            for spine in axis.spines.values():
                spine.set_visible(False)
        fig.text(.09, .95, 'Extraction quality & LLM cost per PDF', fontsize=21, weight='bold')
        fig.text(.09, .90, 'Keyword, semantic, and mean scores use the left axis; API cost/PDF uses the right axis.',
                 fontsize=10, color='#64748B')
        fig.legend(handles=[Patch(facecolor=c, label=l) for c,l in zip(colors,
                   ('keyword_score', 'semantic_score', 'aggregate_score', 'llm_cost_per_pdf_usd'))],
                   loc='upper left', bbox_to_anchor=(.08,.87), ncol=4, frameon=False)
        note = '* Costs have incomplete coverage: some calls or PDFs have unknown cost. Unknown costs are not zero.'
        if not any(r['partial_cost'] or r['cost'] is None for r in rows):
            note = 'Cost is reported extraction/review API usage; local computation is excluded.'
        fig.text(.09, .025, note, fontsize=9, color='#64748B')
        fig.subplots_adjust(left=.09, right=.88, bottom=.22, top=.77)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=200, facecolor=fig.get_facecolor(), bbox_inches='tight')
        plt.close(fig)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, default=ROOT/'results/experiment.csv')
    parser.add_argument('--output', type=Path, default=ROOT/'results/experiments.png',
                        help='PNG, SVG, or PDF destination')
    args = parser.parse_args()
    try:
        print(f'Saved chart: {plot_experiments(args.csv, args.output)}')
    except (ValueError, OSError, ImportError) as exc:
        parser.exit(1, f'Plot failed: {exc}\n')


if __name__ == '__main__':
    main()
