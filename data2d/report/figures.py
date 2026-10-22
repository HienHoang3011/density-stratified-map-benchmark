"""Render every figure and the data of every results table of the article.

Usage: python -m data2d figures <data dir> <output dir>

Example maps are not hand-picked: for each target, the benchmarked map whose
achieved occupancy is closest to the median of all maps in that target is used
(ties go to the lower map index). The planner comparison uses that rule at the
40% target and its first query on which all seven planners returned a valid path.
"""
import argparse
import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path
from statistics import median

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from shapely import wkt

from data2d.layout import ALGORITHMS, Layout, query, read_json, read_jsonl
from data2d.report.plots import path_comparison

GRID = ('A*', 'Dijkstra', 'BFS')
SAMPLING = ('PRM', 'RRT', 'RRT*', 'Informed RRT*')
# One fixed colour and marker per planner across every figure (validated categorical order).
COLOR = {'A*': '#2a78d6', 'Dijkstra': '#eb6834', 'BFS': '#1baf7a', 'PRM': '#eda100',
         'RRT': '#e87ba4', 'RRT*': '#008300', 'Informed RRT*': '#6250d6'}
MARKER = {'A*': 'o', 'Dijkstra': 's', 'BFS': '^', 'PRM': 'D', 'RRT': 'v', 'RRT*': 'P',
          'Informed RRT*': 'X'}
INK, MUTED, OBSTACLE = '#0b0b0b', '#52514e', '#445364'

# Applied only while the article figures are drawn (see main), so importing this
# module does not change the style of any other image.
STYLE = {'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
         'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.spines.top': False,
         'axes.spines.right': False, 'legend.frameon': False}


def save(fig, out, name):
    for ext in ('png', 'pdf'):
        fig.savefig(out / f'{name}.{ext}', dpi=300)
    plt.close(fig)


def read_csv(path):
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def representative_maps(data, densities, benchmark_maps):
    inventory = read_csv(Layout(data).inventory_csv)
    chosen = {}
    for d in densities:
        rows = [r for r in inventory if int(r['Density_pct']) == d]
        mid = median(float(r['AchievedDensity_pct']) for r in rows)
        bench = [r for r in rows if int(r['Map_ID']) <= benchmark_maps]
        best = min(bench, key=lambda r: (abs(float(r['AchievedDensity_pct']) - mid), int(r['Map_ID'])))
        chosen[d] = int(best['Map_ID'])
    return chosen


def figure_sample_maps(out, data, densities, chosen):
    fig, axes = plt.subplots(2, len(densities), figsize=(7.2, 3.35), constrained_layout=True)
    for col, d in enumerate(densities):
        stem = f'D{d:02d}_M{chosen[d]:02d}'
        m = read_json(Layout(data).map_file(d, chosen[d]))
        top, bottom = axes[0, col], axes[1, col]
        for s in m['obstacles_wkt']:
            x, y = wkt.loads(s).exterior.xy
            top.fill(x, y, facecolor=OBSTACLE, edgecolor='none')
        raster = np.asarray(Image.open(Layout(data).raster_file(d, chosen[d])))
        bottom.imshow(raster, cmap='gray', vmin=0, vmax=255, interpolation='nearest',
                      extent=(0, m['width'], 0, m['height']), origin='upper')
        for pair in range(1, 6):
            q = query(m, pair)
            (sx, sy), (gx, gy) = q['start'], q['goal']
            for ax in (top, bottom):
                # No connecting line: a straight S-G segment would read as a route through obstacles.
                ax.scatter([sx], [sy], s=16, color='#13895f', edgecolor='white', linewidth=0.5, zorder=4)
                ax.scatter([gx], [gy], s=34, marker='*', color='#d44435', edgecolor='white', linewidth=0.4, zorder=4)
                for x, y in ((sx, sy), (gx, gy)):
                    dx = -1.1 if x > m['width'] - 4 else 1.1      # keep labels inside the map
                    dy = -1.1 if y > m['height'] - 4 else 1.1
                    ax.text(x + dx, y + dy, str(pair), fontsize=5.5, color=INK, ha='center', va='center',
                            zorder=5, bbox=dict(boxstyle='round,pad=0.12', facecolor='white',
                                                edgecolor='none', alpha=0.85))
        top.set_title(f'{d}% target  ·  {stem}\nachieved {m["achieved_density_pct"]:.2f}%', fontsize=7.2)
        for ax in (top, bottom):
            ax.set(xlim=(0, m['width']), ylim=(0, m['height']), xticks=(0, 15, 30), yticks=(0, 15, 30))
            ax.set_aspect('equal'); ax.tick_params(labelsize=6, length=2)
        if col == 0:
            top.set_ylabel('geometry\ny (m)', fontsize=7)
            bottom.set_ylabel('occupancy raster\ny (m)', fontsize=7)
        bottom.set_xlabel('x (m)', fontsize=7)
    save(fig, out, 'sample_maps_with_queries')


def benchmark_table(data):
    groups = defaultdict(list)
    for r in read_csv(Layout(data).benchmark_csv):
        groups[(int(r['Density_pct']), r['Algorithm'])].append(r)
    return groups


def figure_length_runtime(out, groups, densities):
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.6), sharex=True, sharey='row', constrained_layout=True)
    for col, family in enumerate((GRID, SAMPLING)):
        for algo in family:
            rows = {d: groups[(d, algo)] for d in densities}
            length = [median(float(r['PathLength']) for r in rows[d] if r['Success'] == 'Yes') for d in densities]
            time = [median(float(r['Time_sec']) for r in rows[d]) for d in densities]
            style = dict(color=COLOR[algo], marker=MARKER[algo], markersize=4.5, lw=1.8,
                         markeredgecolor='white', markeredgewidth=0.5, label=algo)
            # A* and Dijkstra return equal-cost paths; dash Dijkstra so both stay visible.
            if algo == 'Dijkstra':
                style['ls'] = (0, (3, 2))
            axes[0, col].plot(densities, length, **style)
            axes[1, col].plot(densities, time, **style)
        axes[0, col].set_title('Grid planners' if col == 0 else 'Sampling planners', fontsize=8.5)
        axes[0, col].legend(fontsize=7, loc='upper left')
        axes[1, col].set_xlabel('Target occupied-cell percentage')
        axes[1, col].set_yscale('log')
        for ax in axes[:, col]:
            ax.set_xticks(densities); ax.grid(alpha=0.22)
    axes[0, 0].set_ylabel('Median valid path\nlength (m)')
    axes[1, 0].set_ylabel('Median planner-call\ntime (s, log scale)')
    save(fig, out, 'length_runtime_by_density')


def first_all_valid_pair(data, density, map_id):
    status = defaultdict(set)
    for r in read_jsonl(Layout(data).paths_file(density, map_id)):
        if r['validation_status'] == 'valid':
            status[r['pair_idx']].add(r['algorithm'])
    return min(p for p, names in status.items() if len(names) == len(GRID + SAMPLING))


def write_table(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def article_tables(out, data, densities, groups):
    """Values of Tables 1, 2, 4, 5 and 6 of the article, one CSV per table."""
    layout = Layout(data)
    inventory = read_csv(layout.inventory_csv)
    maps = [read_json(p) for p in layout.map_files()]
    table1, table2 = [], []
    for d in densities:
        rows = [r for r in inventory if int(r['Density_pct']) == d]
        achieved = [float(r['AchievedDensity_pct']) for r in rows]
        table1.append({'Target_pct': d, 'Maps': len(rows),
                       'TargetMet': sum(r['TargetMet'] == 'Yes' for r in rows),
                       'AchievedMean_pct': f"{np.mean(achieved):.2f}",
                       'AchievedSD_pct': f"{np.std(achieved, ddof=1):.2f}",
                       'AchievedMin_pct': f"{min(achieved):.2f}", 'AchievedMax_pct': f"{max(achieved):.2f}",
                       'Queries': sum(int(r['ScenarioCount']) for r in rows)})
        queries = [q for m in maps if m['density_pct'] == d for q in m['queries']]
        table2.append({'Target_pct': d, 'Queries': len(queries),
                       'MedianSeparation_m': f"{median(q['euclidean_separation_m'] for q in queries):.2f}",
                       'QueriesNeedingMoreThanOneAttempt': sum(q['selection_attempts'] > 1 for q in queries),
                       'MaxAttempts': max(q['selection_attempts'] for q in queries)})
    table4, table5 = [], []
    for algo in ALGORITHMS:
        rows = [r for d in densities for r in groups[(d, algo)]]
        valid = [r for r in rows if r['Success'] == 'Yes']
        table4.append({'Planner': algo, 'ValidPaths': len(valid),
                       'Success_pct': f"{100 * len(valid) / len(rows):.2f}",
                       'MedianTime_s': f"{median(float(r['Time_sec']) for r in rows):.4f}",
                       'MedianLength_m': f"{median(float(r['PathLength']) for r in valid):.3f}"})
    for d in densities:
        row = {'Target_pct': d}
        for algo in ALGORITHMS:
            rows = groups[(d, algo)]
            row[algo] = f"{100 * sum(r['Success'] == 'Yes' for r in rows) / len(rows):.1f}"
        table5.append(row)
    write_table(out / 'table1_inventory.csv', table1)
    write_table(out / 'table2_queries.csv', table2)
    write_table(out / 'table4_overall.csv', table4)
    write_table(out / 'table5_success_by_target.csv', table5)
    shutil.copyfile(layout.repeated_summary_overall, out / 'table6_five_seed.csv')


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=Path, help='Data2D data directory')
    parser.add_argument('out', type=Path, help='Output directory for figures and tables')
    args = parser.parse_args(argv)
    with plt.rc_context(STYLE):
        render(args.data, args.out)


def render(data, out):
    out.mkdir(parents=True, exist_ok=True)
    manifest = read_json(Layout(data).run_manifest)
    densities = manifest['density_targets_pct']
    chosen = representative_maps(data, densities, manifest['benchmark_maps_per_density'])
    groups = benchmark_table(data)

    figure_sample_maps(out, data, densities, chosen)
    figure_length_runtime(out, groups, densities)
    example_density = 40 if 40 in densities else densities[-1]
    pair = first_all_valid_pair(data, example_density, chosen[example_density])
    path_comparison(data, out / 'planners_same_scenario', density=example_density,
                    map_id=chosen[example_density], pair_id=pair)
    for ext in ('png', 'pdf'):
        shutil.copyfile(Layout(data).repeated_figure.with_suffix(f'.{ext}'),
                        out / f'repeat_subset_success.{ext}')
    selection = {'representative_maps': {str(d): f'D{d:02d}_M{m:02d}' for d, m in chosen.items()},
                 'planner_comparison': f'D{example_density:02d}_M{chosen[example_density]:02d}_P{pair:02d}',
                 'rule': 'benchmarked map with achieved occupancy closest to the stratum median; '
                         'first query with seven valid paths'}
    (out / 'figure_selection.json').write_text(json.dumps(selection, indent=2) + '\n')
    article_tables(out, data, densities, groups)
    print(json.dumps(selection, indent=2))


if __name__ == '__main__':
    main()
