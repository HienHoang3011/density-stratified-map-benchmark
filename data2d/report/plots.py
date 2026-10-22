"""All drawing: map previews, the seven-planner gallery, the five-seed figure and
optional per-call diagnostic plots. Images are for browsing; the data files are
the source of every value.
"""
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from PIL import Image

from data2d.layout import ALGORITHMS, Layout, map_key, query, read_json, read_jsonl

# One fixed colour per sampling planner (PRM, RRT, RRT*, Informed RRT*), shared with figures.py.
SAMPLING_COLORS = ("#eda100", "#e87ba4", "#008300", "#6250d6")


def map_preview(map_data, shapes, occupancy, output):
    """Final obstacle geometry on metre axes, with the map key and achieved occupancy."""
    fig, ax = plt.subplots(figsize=(4.8, 5.1), constrained_layout=True)
    for shape in shapes:
        x, y = shape.exterior.xy
        ax.fill(x, y, facecolor='#445364', edgecolor='#283440', linewidth=0.35)
    ax.set(xlim=(0, map_data['width']), ylim=(0, map_data['height']),
           xlabel='x (m)', ylabel='y (m)',
           title=(f"Map D{map_data['density_pct']:02d}_M{map_data['map_idx']:02d}  |  "
                  f"occupied {occupancy:.2f}%"))
    ax.set_aspect('equal')
    ax.set_xticks([0, 10, 20, 30]); ax.set_yticks([0, 10, 20, 30])
    ax.grid(alpha=0.12, linewidth=0.5)
    fig.savefig(output, dpi=140)
    plt.close(fig)


def path_comparison(data_dir, output, density=50, map_id=1, pair_id=1):
    """Draw the seven saved paths of one query on its occupancy raster (.png and .pdf)."""
    layout, output = Layout(data_dir), Path(output)
    stem = map_key(density, map_id)
    scenario = query(read_json(layout.map_file(density, map_id)), pair_id)
    raster = np.asarray(Image.open(layout.raster_file(density, map_id)).convert("L"))
    records = {r["algorithm"]: r for r in read_jsonl(layout.paths_file(density, map_id))
               if r["pair_idx"] == pair_id}
    if set(records) != set(ALGORITHMS):
        raise ValueError(f"Incomplete path records for {stem}, pair {pair_id}")

    fig, axes = plt.subplots(2, 4, figsize=(12.8, 6.5), constrained_layout=True)
    for ax, algorithm in zip(axes.flat, ALGORITHMS):
        record = records[algorithm]
        ax.imshow(raster, cmap="gray", vmin=0, vmax=255,
                  interpolation="nearest", extent=(0, 30, 0, 30), origin="upper")
        route = record["path"]
        if record["validation_status"] == "valid" and route:
            points = np.asarray(route)
            ax.plot(points[:, 0], points[:, 1], color="#1676c5", linewidth=1.5)
        else:
            ax.text(0.5, 0.06, "No valid path", transform=ax.transAxes,
                    ha="center", va="bottom", fontsize=8,
                    bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"})
        start, goal = scenario["start"], scenario["goal"]
        ax.scatter([start[0]], [start[1]], color="#13895f", s=42,
                   edgecolor="white", linewidth=0.6, zorder=5)
        ax.scatter([goal[0]], [goal[1]], color="#d44435", s=70,
                   marker="*", edgecolor="white", linewidth=0.5, zorder=5)
        ax.set(xlim=(0, 30), ylim=(0, 30), title=algorithm)
        ax.set_aspect("equal")
        ax.set_xticks((0, 10, 20, 30))
        ax.set_yticks((0, 10, 20, 30))
        ax.tick_params(labelsize=7)
    axes.flat[-1].axis("off")
    fig.suptitle(f"{stem}, query P{pair_id:02d}: saved paths on exact occupancy raster", fontsize=12)
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(output.with_suffix(f".{suffix}"), dpi=180)
    plt.close(fig)


def repeated_success(by_density, by_query, design, output):
    """Pooled and per-query valid-run percentages of the five-seed subset (.png and .pdf)."""
    lookup = {(int(r["TargetDensity_pct"]), r["Algorithm"]): r for r in by_density}
    fig, axes = plt.subplots(2, 2, figsize=(10.3, 6.9), sharex=True, sharey=True,
                             constrained_layout=True)
    densities = design["densities"]
    repeats = design["repeats"]
    scenarios_per_density = len(design["maps"]) * len(design["pairs"])
    for ax, algorithm, color in zip(axes.flat, design["algorithms"], SAMPLING_COLORS):
        ax.plot(densities, [float(lookup[(d, algorithm)]["SuccessRate_pct"]) for d in densities],
                "o-", color=color, linewidth=1.8, markersize=4,
                label=f"{scenarios_per_density * repeats} calls / target")
        for density in densities:
            proportions = [sum(r["Success"] == "Yes" for r in qrows) / repeats * 100
                           for (d, m, p, name), qrows in by_query.items()
                           if d == density and name == algorithm]
            assert len(proportions) == scenarios_per_density
            ax.scatter([density] * len(proportions), proportions, color=color, s=11, alpha=0.23)
        ax.set_title(algorithm, fontsize=10.5, weight="bold")
        ax.set_ylim(-4, 104)
        margin = 0.05 * (max(densities) - min(densities) or 1)
        ax.set_xlim(min(densities) - margin, max(densities) + margin)
        ax.grid(True, alpha=0.22)
        ax.set_xticks(densities)
    for ax in axes[:, 0]:
        ax.set_ylabel("Valid runs (%)")
    for ax in axes[1]:
        ax.set_xlabel("Target occupied-cell percentage")
    axes[0, 0].legend([axes[0, 0].lines[0], axes[0, 0].collections[0]],
                      ["Pooled rate", "Individual query"],
                      loc="lower left", frameon=True, fontsize=8)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(output.with_suffix(f".{ext}"), dpi=220)
    plt.close(fig)


# ---------------------------------------------------------------- optional diagnostics
# Saved by `python -m data2d generate` without --no-plots: one image per planner call.

def draw_base_map(ax, generator, obstacles, start, goal, title):
    ax.set_xlim(0, generator.width)
    ax.set_ylim(0, generator.height)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=12, fontweight='bold')
    
    for geom in obstacles:
        if geom.geom_type == 'Polygon':
            x, y = geom.exterior.xy
            ax.fill(x, y, color='#2c3e50', alpha=0.9, edgecolor='black')
            
    ax.plot(start[0], start[1], 'bo', markersize=10, label='Start')
    ax.plot(goal[0], goal[1], 'ro', markersize=10, label='Goal')
    plt.grid(True, linestyle=':', alpha=0.4)

def plot_raw_scenario(out_dir, generator, obstacles, start, goal, density, map_idx, pair_idx):
    fig, ax = plt.subplots(figsize=(8, 8))
    title = f"Raw Map Data\nTarget occupancy: {density}% | Map ID: {map_idx:02d} | Scenario: {pair_idx:02d}"
    draw_base_map(ax, generator, obstacles, start, goal, title)
    ax.legend(loc='upper right')
    
    folder = os.path.join(out_dir, f"D{density}_Map_{map_idx:02d}_Pair_{pair_idx:02d}")
    os.makedirs(folder, exist_ok=True)
    
    size_str = f"S{generator.min_size}to{generator.max_size}"
    filename = f"Raw_D{density}_{size_str}_M{map_idx:02d}_P{pair_idx:02d}.png"
    fig.savefig(os.path.join(folder, filename), bbox_inches='tight', dpi=150)
    plt.close(fig)

def plot_algorithm_process(out_dir, generator, obstacles, start, goal, path, process_data, extra_data,
                           algo_name, density, map_idx, pair_idx):
    fig, ax = plt.subplots(figsize=(8, 8))
    title = f"{algo_name} Algorithm Search Process\nTarget occupancy: {density}% | Map ID: {map_idx:02d} | Scenario: {pair_idx:02d}"
    draw_base_map(ax, generator, obstacles, start, goal, title)
    
    if algo_name in ['A*', 'Dijkstra', 'BFS']:
        if process_data:
            px, py = zip(*process_data)
            ax.scatter(px, py, c="#03896c", s=8, alpha=0.3, edgecolors='#03896c', label='Explored Nodes')
    else:
        if process_data:
            lc = LineCollection(process_data, color="#1edb09", linewidth=0.8, alpha=0.5, label='Search Tree')
            ax.add_collection(lc)

    if algo_name == 'Informed RRT*' and extra_data:
        center = extra_data['center']
        angle = np.degrees(extra_data['angle'])
        width = extra_data['a'] * 2
        height = extra_data['b'] * 2
        
        from matplotlib.patches import Ellipse
        ell_patch = Ellipse(xy=center, width=width, height=height, angle=angle,
                            edgecolor="#74370E", facecolor='#74370E', alpha=0.1, linewidth=2.5, linestyle='-', label='Ellipse Space')
        ax.add_patch(ell_patch)
        ell_border = Ellipse(xy=center, width=width, height=height, angle=angle,
                            edgecolor='#d35400', facecolor='none', alpha=1.0, linewidth=3.0, label='_nolegend_')
        ax.add_patch(ell_border)

    if path:
        path_x, path_y = zip(*path)
        color = '#e74c3c' if algo_name in ['A*', 'Dijkstra', 'BFS'] else '#f39c12'
        ax.plot(path_x, path_y, color=color, linewidth=3.5, alpha=0.9, label='Final Path')
    
    handles, labels = ax.get_legend_handles_labels()
    for i, label in enumerate(labels):
        if label == 'Ellipse Space' and extra_data:
            from matplotlib.lines import Line2D
            handles[i] = Line2D([0], [0], color='#d35400', linestyle='--', linewidth=2.5)

    ax.legend(handles=handles, labels=labels, loc='upper right', fontsize=9)
    
    folder = os.path.join(out_dir, f"D{density}_Map_{map_idx:02d}_Pair_{pair_idx:02d}")
    os.makedirs(folder, exist_ok=True)
    
    algo_safe = algo_name.replace('*', '_star').replace(' ', '_')
    filename = os.path.join(folder, f"{algo_safe}_D{density}_M{map_idx:02d}_P{pair_idx:02d}.png")
    fig.savefig(filename, bbox_inches='tight', dpi=150)
    plt.close(fig)
