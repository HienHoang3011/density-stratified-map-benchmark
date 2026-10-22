"""Independent audit of a Data2D data directory.

Usage: python -m data2d audit <data dir> [--workers N]

The checks re-derive everything from the published rules and never call the
generator (this module imports no other data2d module except the file layout):
occupancy is recomputed cell by cell from the WKT geometry with its own
rasterization, the motion region from shapely geometry, and every seed from its
documented SHA-256 string. Every valid baseline path is checked for endpoints,
motion-region bounds, zero-length segments, collisions, its CSV length and a
lower bound on Nodes_Expanded. A separate count confirms that no valid path
(baseline or five-seed) has a waypoint closer than the robot radius to the edge.
Writes metadata/audit_report.json and metadata/boundary_clearance_audit.json.
`audit_map` is also used by data2d.checks.live_audit while a run is in progress.
"""
import argparse
import csv
import hashlib
import json
import math
import multiprocessing as mp
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import label
from shapely import box as vector_box, covers, intersects, prepare, wkt
from shapely.geometry import LineString, box
from shapely.ops import unary_union
from shapely.strtree import STRtree

from data2d.layout import Layout, read_json, read_jsonl, write_json

GRID_METHODS = ('A*', 'Dijkstra', 'BFS')
TREE_METHODS = ('RRT', 'RRT*', 'Informed RRT*')


def seed_of(*parts):
    """Documented rule: first 32 bits of SHA-256 over colon-joined fields."""
    digest = hashlib.sha256(':'.join(str(p) for p in parts).encode('utf-8')).digest()
    return int.from_bytes(digest[:4], 'big')


def cell_boxes(m):
    res = m['grid_resolution_m']
    w, h = round(m['width'] / res), round(m['height'] / res)
    xs, ys = np.meshgrid(np.arange(w), np.arange(h))
    x0, y0 = xs.ravel() * res, ys.ravel() * res
    return vector_box(x0, y0, x0 + res, y0 + res), (h, w)


def independent_grids(m):
    """Obstacle grid (density) and motion grid, rows bottom-up, True = blocked."""
    cells, shape = cell_boxes(m)
    buffered = unary_union([wkt.loads(s).buffer(m['robot_radius_m']) for s in m['obstacles_wkt']])
    prepare(buffered)
    obstacle = intersects(buffered, cells).reshape(shape)
    # Cell edges are built as index*res (+res), which overshoots by ~1e-15
    # (e.g. 29.8 + 0.1 = 29.900000000000002); tolerate that rounding only.
    r, eps = m['motion_boundary_inset_m'], 1e-9
    inset = box(r - eps, r - eps, m['width'] - r + eps, m['height'] - r + eps)
    motion = obstacle | ~covers(inset, cells).reshape(shape)
    return obstacle, motion


def audit_map(run, map_path, manifest, csv_rows=None, check_png=True):
    """Audit one map and its queries/paths. Raises AssertionError on any failure."""
    stats = Counter()
    layout = Layout(run)
    m = read_json(map_path)
    d, mid = m['density_pct'], m['map_idx']
    stem = f'D{d:02d}_M{mid:02d}'
    base = manifest['base_seed']
    res = m['grid_resolution_m']
    r = m['motion_boundary_inset_m']
    assert m['map_seed'] == seed_of(base, d, mid), (stem, 'map seed')

    obstacle, motion = independent_grids(m)
    assert abs(100 * int(obstacle.sum()) / obstacle.size - m['achieved_density_pct']) < 1e-8, \
        (stem, 'achieved density differs from WKT occupancy')
    if check_png:
        raster = np.asarray(Image.open(layout.raster_file(d, mid)).convert('L'))
        assert set(np.unique(raster)).issubset({0, 255}), (stem, 'non-binary PNG')
        mismatch = int(((raster[::-1] == 0) != obstacle).sum())
        assert mismatch == 0, (stem, f'{mismatch} PNG cells differ from WKT occupancy')
        stats['png_cells_checked'] += obstacle.size
    components, _ = label(~motion)
    stats['maps'] += 1

    def cell(pt):
        return int(pt[1] / res), int(pt[0] / res)

    scenarios = {}
    pairs = [q['pair_idx'] for q in m['queries']]
    assert len(pairs) == len(set(pairs)), (stem, 'duplicate pair index')
    assert sorted(pairs + m['missing_pair_indices']) == list(range(1, manifest['pairs_per_map'] + 1)), \
        (stem, 'saved and missing pair indices do not cover every pair')
    for q in m['queries']:
        pair = q['pair_idx']
        start, goal = q['start'], q['goal']
        assert q['scenario_seed'] == seed_of(base, d, mid, pair, 'scenario'), (stem, pair, 'scenario seed')
        (sy, sx), (gy, gx) = cell(start), cell(goal)
        assert not motion[sy, sx] and not motion[gy, gx], (stem, pair, 'endpoint not in free motion cell')
        assert components[sy, sx] > 0 and components[sy, sx] == components[gy, gx], \
            (stem, pair, 'endpoints not four-connected in the motion grid')
        assert math.dist(start, goal) >= 12 - 1e-8
        assert abs(math.dist(start, goal) - q['euclidean_separation_m']) < 1e-8
        scenarios[pair] = q
        stats['scenarios'] += 1
    assert len(scenarios) == m['scenario_count'], (stem, 'scenario_count')

    ppath = layout.paths_file(d, mid)
    if not ppath.exists():
        return stats
    obstacles = [wkt.loads(x).buffer(m['robot_radius_m']) for x in m['obstacles_wkt']]
    tree = STRtree(obstacles)
    stats['benchmark_maps'] += 1
    for line in ppath.read_text().splitlines():
        rec = json.loads(line)
        stats['planner_calls'] += 1
        pair, algo = rec['pair_idx'], rec['algorithm']
        key = (d, mid, pair, algo)
        q = scenarios[pair]
        assert rec['scenario_seed'] == q['scenario_seed'] and rec['map_seed'] == m['map_seed']
        assert rec['planner_seed'] == seed_of(base, d, mid, pair, algo), (key, 'planner seed')
        if algo == 'PRM':
            limit = manifest['prm_max_sample_attempts']
            accepted, attempts = rec['prm_samples_accepted'], rec['prm_sample_attempts']
            assert accepted <= attempts <= limit, (key, 'PRM attempts')
            assert rec['prm_sample_limit_reached'] == (accepted < manifest['prm_samples']), (key, 'PRM limit flag')
            stats['prm_limit_reached'] += rec['prm_sample_limit_reached']
        route, status = rec['path'], rec['validation_status']
        row = csv_rows.get(key) if csv_rows is not None else None
        if status == 'no_path':
            assert route is None
            stats['no_path'] += 1
            if row:
                assert row['Success'] == 'No' and float(row['PathLength']) == 0, key
            continue
        if status != 'valid':
            stats['invalid_returned'] += 1
            if row:
                assert row['Success'] == 'No', key
            continue
        assert route and len(route) >= 2
        assert math.dist(route[0], q['start']) < 1e-8 and math.dist(route[-1], q['goal']) < 1e-8, key
        for x, y in route:
            assert r <= x <= m['width'] - r and r <= y <= m['height'] - r, (key, 'waypoint outside motion region')
        for a, b in zip(route, route[1:]):
            assert math.dist(a, b) > 0, (key, 'zero-length segment')
            segment = LineString((a, b))
            assert not any(segment.intersects(obstacles[int(i)]) for i in tree.query(segment)), (key, 'collision')
        if row:
            length = sum(math.dist(a, b) for a, b in zip(route, route[1:]))
            assert row['Success'] == 'Yes' and abs(round(length, 3) - float(row['PathLength'])) < 1e-6, key
            nodes = int(row['Nodes_Expanded'])
            # Every grid/PRM path vertex was expanded; every tree path vertex but the goal is a tree node.
            minimum = len(route) - 1 if algo in TREE_METHODS else len(route)
            assert nodes >= minimum, (key, 'Nodes_Expanded below path vertex count')
            stats['csv_rows_checked'] += 1
        stats['valid_paths'] += 1
    return stats


def _audit_job(job):
    run, map_path, manifest, csv_rows = job
    return audit_map(run, map_path, manifest, csv_rows)


def boundary_clearance(layout, radius):
    """Count valid paths with a waypoint closer to the workspace edge than the robot radius."""
    def inspect(files):
        by_algorithm, by_density, valid = Counter(), Counter(), 0
        sizes = {}
        for file in files:
            for row in read_jsonl(file):
                if row["validation_status"] != "valid":
                    continue
                valid += 1
                key = (row['density_pct'], row['map_idx'])
                if key not in sizes:
                    m = read_json(layout.map_file(*key))
                    sizes[key] = (m["width"], m["height"])
                width, height = sizes[key]
                clearance = min(min(x, width - x, y, height - y) for x, y in row["path"])
                if clearance < radius:
                    by_algorithm[row["algorithm"]] += 1
                    by_density[row["density_pct"]] += 1
        return {"valid_paths": valid,
                "paths_with_waypoint_clearance_below_robot_radius": sum(by_algorithm.values()),
                "by_algorithm": dict(sorted(by_algorithm.items())),
                "by_density_pct": dict(sorted(by_density.items()))}

    repeated = [layout.repeated_paths] if layout.repeated_paths.exists() else []
    return {
        "criterion": f"For a valid archived path, min over waypoints of x, width-x, y, height-y < robot radius {radius} m",
        "baseline": inspect(layout.baseline_path_files()),
        "repeated_subset": inspect(repeated),
        "interpretation": ("Valid-path waypoints lie within the workspace inset by the robot radius. "
                           "Because this region is convex, the connecting segments and the full "
                           "circular robot footprint also stay within the workspace."),
    }


def audit(root, workers):
    layout = Layout(root)
    manifest = read_json(layout.run_manifest)
    csv_rows = {}
    with layout.benchmark_csv.open(newline='', encoding='utf-8') as stream:
        for row in csv.DictReader(stream):
            key = (int(row['Density_pct']), int(row['Map_ID']), int(row['Pair_ID']), row['Algorithm'])
            assert key not in csv_rows, (key, 'duplicate CSV row')
            csv_rows[key] = row
    by_map = {}
    for key, row in csv_rows.items():
        by_map.setdefault(key[:2], {})[key] = row
    jobs = []
    for path in layout.map_files():
        m = read_json(path)
        jobs.append((layout.root, path, manifest, by_map.get((m['density_pct'], m['map_idx']), {})))
    stats = Counter()
    with mp.Pool(workers) as pool:
        for result in pool.imap_unordered(_audit_job, jobs, chunksize=4):
            stats.update(result)
    assert stats['csv_rows_checked'] == stats['valid_paths']
    assert stats['planner_calls'] == len(csv_rows)
    for key, expected in [('maps', manifest['map_files']), ('scenarios', manifest['saved_queries']),
                          ('benchmark_maps', manifest['benchmark_map_files']),
                          ('planner_calls', manifest['algorithm_records']),
                          ('prm_limit_reached', manifest['prm_calls_reaching_sample_limit'])]:
        assert stats[key] == expected, (key, stats[key], expected)
    out = {'checks': 'passed', 'counts': dict(sorted(stats.items())),
           'method': ('Cell-by-cell occupancy recomputed from WKT and compared with every PNG; '
                      'queries checked for free, four-connected motion-grid cells; map, scenario and '
                      'planner seeds recomputed from the documented rule; every valid baseline path checked '
                      'for endpoints, motion-region bounds, zero-length segments, continuous collision '
                      'against buffered WKT, CSV length and Nodes_Expanded lower bound; PRM sampling '
                      'counters checked against the attempt limit')}
    write_json(layout.audit_report, out)
    boundary = boundary_clearance(layout, radius=read_json(jobs[0][1])['robot_radius_m'])
    write_json(layout.boundary_report, boundary)
    print(json.dumps(out, indent=2))
    print('Waypoints closer than the robot radius to the edge:',
          boundary['baseline']['paths_with_waypoint_clearance_below_robot_radius'], 'baseline,',
          boundary['repeated_subset']['paths_with_waypoint_clearance_below_robot_radius'], 'five-seed')


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=Path, help='Data2D data directory')
    parser.add_argument('--workers', type=int, default=max(1, min(8, mp.cpu_count() - 1)))
    args = parser.parse_args(argv)
    audit(args.data, args.workers)


if __name__ == '__main__':
    main()
