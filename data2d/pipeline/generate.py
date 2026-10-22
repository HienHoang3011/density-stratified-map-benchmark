"""Generate the maps, their start-goal queries and the seven-planner baseline.

Usage:
    python -m data2d generate --output-dir new_run --densities 10 20 30 40 50 \\
        --maps-per-density 300 --benchmark-maps-per-density 300 --pairs-per-map 5 \\
        --base-seed 20261010 --workers 8 --no-plots

Writes maps/, paths/ and metadata/{benchmark.csv, map_inventory.csv,
run_manifest.json} into a new output directory (see data2d/layout.py).
"""
import argparse
import csv
import importlib.metadata
import math
import multiprocessing
import os
import platform
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.ndimage import label

from data2d.layout import ALGORITHMS, Layout, code_sha256, write_json, write_jsonl
from data2d.core.maps import MapGenerator
from data2d.core.planners import PathPlanners, calculate_path_length
from data2d.core.seeds import map_seed, planner_seed, query_seed

GENERATOR_VERSION = "1.0"
DEFAULT_DENSITIES = (10, 20, 30, 40, 50)
WORKSPACE = dict(width=30.0, height=30.0, min_size=1.5, max_size=6.0,
                 robot_radius=0.1, grid_resolution=0.1)
QUERY_MIN_SEPARATION_M = 12.0
QUERY_MAX_ATTEMPTS = 1000
PRM_K_NEIGHBORS = 15
RRT_STEP_M = 1.5
CSV_HEADERS = ['Density_pct', 'Map_ID', 'Pair_ID', 'Algorithm', 'Success', 'PathLength', 'Time_sec',
               'Nodes_Expanded']
INVENTORY_HEADERS = ['Density_pct', 'Map_ID', 'Seed', 'AchievedDensity_pct', 'TargetMet',
                     'ObstacleCount', 'ScenarioCount', 'AlgorithmRecords']


# ---------------------------------------------------------------- one map
def map_metadata(density, map_idx, gen, obstacles, grid, seed, target_met):
    """Map JSON fields; queries and the WKT list are added by `map_record_json`."""
    return {
        "density_pct": density, "map_idx": map_idx,
        "width": gen.width, "height": gen.height,
        "generator_version": GENERATOR_VERSION,
        "map_seed": seed,
        "achieved_density_pct": 100.0 * float(np.count_nonzero(grid)) / grid.size,
        "target_met": bool(target_met),
        "obstacle_count": len(obstacles),
        "grid_resolution_m": gen.grid_resolution,
        "robot_radius_m": gen.robot_radius,
        "obstacle_types": gen.obstacle_types,
        "min_size_m": gen.min_size,
        "max_size_m": gen.max_size,
        "raster_rule": "cell intersects obstacle buffered by robot radius",
        "motion_boundary_inset_m": gen.robot_radius,
        "motion_rule": ("robot centre stays inside the workspace inset by the robot radius; "
                        "grid planners and query selection also block cells whose square "
                        "leaves that inset"),
    }


def map_record_json(metadata, queries, missing_pairs, obstacles):
    """Final map JSON: metadata, the saved queries, then the long WKT list last."""
    return {
        **metadata,
        "scenario_count": len(queries),
        "missing_pair_indices": missing_pairs,
        "query_min_separation_m": QUERY_MIN_SEPARATION_M,
        "query_free_component_connectivity": 4,
        "queries": queries,
        "obstacles_wkt": [geom.wkt for geom in obstacles],
    }


def select_query(gen, motion, components, seed):
    """Draw candidate pairs until both endpoints share a four-connected free region
    of the motion grid and lie at least 12 m apart. Returns (start, goal, attempts)."""
    random.seed(seed)
    res = gen.grid_resolution
    for attempt in range(1, QUERY_MAX_ATTEMPTS + 1):
        start = gen.get_random_free_point(motion)
        goal = gen.get_random_free_point(motion)
        if start and goal and math.dist(start, goal) >= QUERY_MIN_SEPARATION_M:
            sg = (int(start[0] / res), int(start[1] / res))
            gg = (int(goal[0] / res), int(goal[1] / res))
            if components[sg[1], sg[0]] > 0 and components[sg[1], sg[0]] == components[gg[1], gg[0]]:
                return start, goal, attempt
    return None, None, QUERY_MAX_ATTEMPTS


def planner_calls(planners, start, goal, config):
    """The seven baseline calls, in the order of ALGORITHMS."""
    iters = config['rrt_iterations']
    return [
        ('A*', lambda: planners.grid_search(start, goal, mode='A*')),
        ('Dijkstra', lambda: planners.grid_search(start, goal, mode='Dijkstra')),
        ('BFS', lambda: planners.grid_search(start, goal, mode='BFS')),
        ('PRM', lambda: planners.prm(start, goal, num_samples=config['prm_samples'], k_neighbors=PRM_K_NEIGHBORS)),
        ('RRT', lambda: planners.rrt_family(start, goal, mode='RRT', max_iter=iters)),
        ('RRT*', lambda: planners.rrt_family(start, goal, mode='RRT*', max_iter=iters)),
        ('Informed RRT*', lambda: planners.rrt_family(start, goal, mode='Informed RRT*', max_iter=iters)),
    ]


def run_map(task):
    """Generate one map, select its queries and run the baseline. Runs in a worker."""
    density, map_idx, config = task
    layout = Layout(config['output_root'])
    base = config['base_seed']
    seed = map_seed(base, density, map_idx)
    print(f"Map D{density}_M{map_idx:02d}: starting (seed={seed})", flush=True)

    # 1. Obstacles are placed until the occupied-cell threshold is reached.
    random.seed(seed)
    gen = MapGenerator(**WORKSPACE)
    obstacles, grid, target_met = gen.create_exact_density_map(density, max_no_progress=config['max_no_progress'])
    metadata = map_metadata(density, map_idx, gen, obstacles, grid, seed, target_met)
    inventory = {
        'Density_pct': density, 'Map_ID': map_idx, 'Seed': seed,
        'AchievedDensity_pct': round(metadata['achieved_density_pct'], 6),
        'TargetMet': 'Yes' if target_met else 'No',
        'ObstacleCount': len(obstacles), 'ScenarioCount': 0, 'AlgorithmRecords': 0,
    }
    stats = {'prm_calls': 0, 'prm_limit_reached': 0}
    if not target_met:
        print(f"Map D{density}_M{map_idx:02d}: target not reached; no queries", flush=True)
        write_json(layout.map_file(density, map_idx),
                   map_record_json(metadata, [], list(range(1, config['pairs_per_map'] + 1)), obstacles))
        return [], inventory, stats

    # 2. Queries are selected on the motion grid (occupancy grid plus the blocked boundary ring).
    motion = gen.motion_grid(grid)
    components, _ = label(motion == 0)            # four-connectivity in two dimensions
    gen.prepare_free_point_cache(motion)
    planners = PathPlanners(gen, obstacles, motion)
    benchmark = map_idx <= config['benchmark_maps_per_density']
    queries, missing, rows, path_records = [], [], [], []

    for pair_idx in range(1, config['pairs_per_map'] + 1):
        q_seed = query_seed(base, density, map_idx, pair_idx)
        start, goal, attempts = select_query(gen, motion, components, q_seed)
        if start is None:
            print(f"Map D{density}_M{map_idx:02d}: no connected pair P{pair_idx:02d}", flush=True)
            missing.append(pair_idx)
            continue
        queries.append({
            "pair_idx": pair_idx, "start": start, "goal": goal,
            "scenario_seed": q_seed, "selection_attempts": attempts,
            "euclidean_separation_m": math.dist(start, goal),
        })
        if not benchmark:
            continue

        # 3. Seven planner calls, each with its own seed, each path validated.
        if config['plots_dir']:
            from data2d.report.plots import plot_raw_scenario
            plot_raw_scenario(config['plots_dir'], gen, obstacles, start, goal, density, map_idx, pair_idx)
        for algorithm, call in planner_calls(planners, start, goal, config):
            p_seed = planner_seed(base, density, map_idx, pair_idx, algorithm)
            random.seed(p_seed)
            raw_path, elapsed, nodes, process_data, *extra = call()
            valid, status = planners.validate_path(raw_path, start, goal)
            path = raw_path if valid else None
            rows.append([density, map_idx, pair_idx, algorithm, 'Yes' if valid else 'No',
                         round(calculate_path_length(path), 3), round(elapsed, 4), nodes])
            record = {
                'density_pct': density, 'map_idx': map_idx, 'pair_idx': pair_idx,
                'algorithm': algorithm, 'map_seed': seed,
                'scenario_seed': q_seed, 'planner_seed': p_seed,
                'validation_status': status, 'path': raw_path,
            }
            if algorithm == 'PRM':
                record.update(planners.last_prm_sampling)
                stats['prm_calls'] += 1
                stats['prm_limit_reached'] += record['prm_sample_limit_reached']
            path_records.append(record)
            if config['plots_dir']:
                from data2d.report.plots import plot_algorithm_process
                plot_algorithm_process(config['plots_dir'], gen, obstacles, start, goal, path, process_data,
                                       extra[0] if extra else None, algorithm, density, map_idx, pair_idx)

    # 4. The paths are written before the map file: a complete map file marks a finished map.
    if benchmark:
        write_jsonl(layout.paths_file(density, map_idx), path_records)
    inventory['ScenarioCount'] = len(queries)
    inventory['AlgorithmRecords'] = len(rows)
    write_json(layout.map_file(density, map_idx), map_record_json(metadata, queries, missing, obstacles))
    print(f"Map D{density}_M{map_idx:02d}: completed ({len(queries)} pairs)", flush=True)
    return rows, inventory, stats


# ---------------------------------------------------------------- whole run
def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Generate Data2D maps, queries and the planner baseline")
    parser.add_argument('--output-dir', type=Path, required=True, help='New, non-existing directory for this run')
    parser.add_argument('--densities', type=int, nargs='+', default=list(DEFAULT_DENSITIES))
    parser.add_argument('--maps-per-density', type=int, default=300)
    parser.add_argument('--pairs-per-map', type=int, default=5)
    parser.add_argument('--benchmark-maps-per-density', type=int, default=300,
                        help='Run the seven planners on the first N maps of each target; all maps get queries')
    parser.add_argument('--base-seed', type=int, default=20261010)
    parser.add_argument('--workers', type=int, default=max(1, min(8, multiprocessing.cpu_count() - 1)))
    parser.add_argument('--prm-samples', type=int, default=600)
    parser.add_argument('--rrt-iterations', type=int, default=2500)
    parser.add_argument('--max-no-progress', type=int, default=1000)
    parser.add_argument('--no-plots', action='store_true', help='Skip the diagnostic image of every planner call')
    args = parser.parse_args(argv)
    if (not args.densities or any(d <= 0 or d >= 100 for d in args.densities)
            or len(args.densities) != len(set(args.densities))):
        parser.error('densities must be unique integers between 1 and 99')
    for name in ('maps_per_density', 'pairs_per_map', 'benchmark_maps_per_density', 'workers', 'prm_samples',
                 'rrt_iterations', 'max_no_progress'):
        if getattr(args, name) < 1:
            parser.error(f'{name} must be positive')
    if args.base_seed < 0:
        parser.error('base_seed must be nonnegative')
    if args.benchmark_maps_per_density > args.maps_per_density:
        parser.error('benchmark-maps-per-density cannot exceed maps-per-density')
    return args


def runtime_info():
    cpu_model = platform.processor() or None
    try:   # platform.processor() only gives the architecture on Linux
        with open('/proc/cpuinfo', encoding='utf-8') as stream:
            cpu_model = next(line.split(':', 1)[1].strip() for line in stream if line.startswith('model name'))
    except (OSError, StopIteration):
        pass
    try:
        ram_bytes = os.sysconf('SC_PHYS_PAGES') * os.sysconf('SC_PAGE_SIZE')
    except (AttributeError, ValueError, OSError):
        ram_bytes = None
    return {
        'platform': platform.platform(), 'processor': platform.processor(), 'cpu_model': cpu_model,
        'cpu_count': multiprocessing.cpu_count(), 'ram_bytes': ram_bytes,
        'python': platform.python_version(),
        'packages': {name: importlib.metadata.version(name) for name in ('numpy', 'scipy', 'shapely', 'matplotlib')},
    }


def sort_csv(path, key):
    """Rewrite a CSV in key order so worker completion order does not affect the file."""
    with open(path, newline='', encoding='utf-8') as stream:
        reader = csv.reader(stream)
        header = next(reader)
        rows = sorted(reader, key=key)
    with open(path, 'w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def main(argv=None):
    args = parse_args(argv)
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    layout = Layout(root)
    for folder in (layout.maps, layout.paths, layout.metadata):
        folder.mkdir(parents=True)
    config = {
        'output_root': str(root), 'base_seed': args.base_seed,
        'pairs_per_map': args.pairs_per_map, 'prm_samples': args.prm_samples,
        'rrt_iterations': args.rrt_iterations,
        'benchmark_maps_per_density': args.benchmark_maps_per_density,
        'max_no_progress': args.max_no_progress,
        'plots_dir': None if args.no_plots else str(layout.diagnostics),
    }
    manifest = {
        'generator_version': GENERATOR_VERSION,
        'status': 'running', 'started_at': datetime.now().astimezone().isoformat(),
        'code_sha256': code_sha256(),
        'density_targets_pct': args.densities,
        'maps_per_density': args.maps_per_density,
        'benchmark_maps_per_density': args.benchmark_maps_per_density,
        'pairs_per_map': args.pairs_per_map,
        'base_seed': args.base_seed,
        'seed_rule': ('first 32 bits of SHA256 over colon-joined fields; '
                      'map: base_seed:density:map_idx; '
                      'scenario: base_seed:density:map_idx:pair_idx:scenario; '
                      'baseline planner: base_seed:density:map_idx:pair_idx:algorithm'),
        'workers': args.workers,
        'prm_samples': args.prm_samples, 'prm_k_neighbors': PRM_K_NEIGHBORS,
        'prm_max_sample_attempts': 100 * args.prm_samples,
        'rrt_iterations': args.rrt_iterations, 'rrt_step_m': RRT_STEP_M,
        'rrt_star_radius_rule': 'min(eta, gamma*sqrt(log(n)/n))',
        'rrt_star_eta_m': RRT_STEP_M,
        'rrt_star_gamma_m': 2.2 * math.sqrt(1.5 * WORKSPACE['width'] * WORKSPACE['height'] / math.pi),
        'motion_region': 'workspace inset by robot_radius on every side; required for every valid waypoint',
        'occupancy_density_grid': 'buffered obstacles only; the motion inset does not change density',
        'sampling_region': 'uniform over the motion region for PRM, RRT and RRT*',
        'prm_sample_acceptance': 'continuous point collision against buffered obstacles',
        'informed_sampling_boundary_rule': 'reject and resample outside the motion region',
        'goal_cost_update': 'all goal candidates reevaluated after rewiring',
        'rrt_and_rrt_star_goal_bias': 0.1,
        'informed_rrt_star_goal_bias_before_first_solution': 0.0,
        'max_no_progress': args.max_no_progress,
        'plots_saved': not args.no_plots,
        'timing_scope': 'planner function only; excludes map generation, path audit and plotting',
        'environment': runtime_info(),
    }
    write_json(layout.run_manifest, manifest)

    tasks = [(d, m, config) for d in args.densities for m in range(1, args.maps_per_density + 1)]
    with layout.benchmark_csv.open('w', newline='', encoding='utf-8') as stream:
        csv.writer(stream).writerow(CSV_HEADERS)
    with layout.inventory_csv.open('w', newline='', encoding='utf-8') as stream:
        csv.DictWriter(stream, fieldnames=INVENTORY_HEADERS).writeheader()

    began = time.perf_counter()
    pool = multiprocessing.Pool(min(args.workers, len(tasks))) if args.workers > 1 else None
    results = pool.imap_unordered(run_map, tasks) if pool else map(run_map, tasks)
    totals = dict(queries=0, records=0, target_met=0, benchmark_maps=0, prm_calls=0, prm_limit=0)
    try:
        for rows, inventory, stats in results:
            with layout.benchmark_csv.open('a', newline='') as stream:
                csv.writer(stream).writerows(rows)
            with layout.inventory_csv.open('a', newline='', encoding='utf-8') as stream:
                csv.DictWriter(stream, fieldnames=INVENTORY_HEADERS).writerow(inventory)
            totals['queries'] += inventory['ScenarioCount']
            totals['records'] += inventory['AlgorithmRecords']
            totals['target_met'] += inventory['TargetMet'] == 'Yes'
            totals['benchmark_maps'] += inventory['AlgorithmRecords'] > 0
            totals['prm_calls'] += stats['prm_calls']
            totals['prm_limit'] += stats['prm_limit_reached']
    except BaseException:
        if pool:
            pool.terminate()
            pool.join()
        manifest['status'] = 'failed'
        write_json(layout.run_manifest, manifest)
        raise
    if pool:
        pool.close()
        pool.join()

    sort_csv(layout.benchmark_csv, key=lambda r: (int(r[0]), int(r[1]), int(r[2]), ALGORITHMS.index(r[3])))
    sort_csv(layout.inventory_csv, key=lambda r: (int(r[0]), int(r[1])))
    manifest.update({
        'status': 'complete', 'ended_at': datetime.now().astimezone().isoformat(),
        'wall_time_sec': round(time.perf_counter() - began, 3),
        'map_files': len(tasks), 'maps_meeting_target': totals['target_met'],
        'saved_queries': totals['queries'], 'algorithm_records': totals['records'],
        'benchmark_map_files': totals['benchmark_maps'],
        'prm_calls': totals['prm_calls'],
        'prm_calls_reaching_sample_limit': totals['prm_limit'],
    })
    write_json(layout.run_manifest, manifest)
    print(f"Finished: {totals['queries']} queries, {totals['records']} records in {root}")


if __name__ == '__main__':
    main()
