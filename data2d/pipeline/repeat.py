"""Five-seed reruns of the sampling planners on 30 fixed queries.

Usage:
    python -m data2d repeat <data dir>                   run, check and summarize
    python -m data2d repeat <data dir> --summarize-only  check and summarize existing records
    python -m data2d repeat <data dir> --output-dir DIR  run only, writing to DIR

The subset uses maps 1, 10 and 20 and queries 1 and 5 at every target. Each of
PRM, RRT, RRT* and Informed RRT* runs five times per query with seeds from the
`Data2D-repeat-subset` namespace; maps, endpoints and parameters are unchanged.
Writes paths/repeated_subset.jsonl, metadata/repeated_trials.csv, the two
summary tables, metadata/repeat_manifest.json, metadata/repeat_validation.json
and images/repeated_subset_success.png/.pdf. The baseline is never modified.
"""
import argparse
import csv
import importlib.metadata
import math
import os
import platform
import random
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from statistics import median

import numpy as np
from PIL import Image
from shapely import wkt
from shapely.geometry import LineString
from shapely.strtree import STRtree

from data2d.layout import (SAMPLING_ALGORITHMS, Layout, code_sha256, map_key, query, read_json, read_jsonl,
                     write_json, write_jsonl)
from data2d.core.maps import MapGenerator
from data2d.core.planners import PathPlanners, calculate_path_length
from data2d.report.plots import repeated_success
from data2d.core.seeds import repeat_seed

MAP_INDICES = (1, 10, 20)
PAIR_INDICES = (1, 5)
SEED_COUNT = 5
FIELDS = ("Density_pct", "Map_ID", "Pair_ID", "Algorithm", "Repeat",
          "PlannerSeed", "Success", "ValidationStatus", "PathLength_m",
          "Time_sec", "Nodes_Expanded")


# ---------------------------------------------------------------- run
def run_map(task):
    density, map_idx, root = task
    layout = Layout(root)
    map_data = read_json(layout.map_file(density, map_idx))
    obstacles = [wkt.loads(value) for value in map_data["obstacles_wkt"]]
    generator = MapGenerator(
        width=map_data["width"], height=map_data["height"],
        obstacle_types=map_data["obstacle_types"],
        min_size=map_data["min_size_m"], max_size=map_data["max_size_m"],
        robot_radius=map_data["robot_radius_m"],
        grid_resolution=map_data["grid_resolution_m"])
    raster = np.asarray(Image.open(layout.raster_file(density, map_idx)).convert("L"))
    grid = (raster[::-1] == 0).astype(np.uint8)
    assert grid.shape == (generator.grid_h, generator.grid_w)
    planners = PathPlanners(generator, obstacles, generator.motion_grid(grid))
    rows, paths = [], []
    for pair_idx in PAIR_INDICES:
        scenario = query(map_data, pair_idx)
        start, goal = tuple(scenario["start"]), tuple(scenario["goal"])
        for repeat in range(1, SEED_COUNT + 1):
            for algorithm in SAMPLING_ALGORITHMS:
                seed = repeat_seed(density, map_idx, pair_idx, algorithm, repeat)
                random.seed(seed)
                if algorithm == "PRM":
                    result = planners.prm(start, goal, num_samples=600, k_neighbors=15)
                else:
                    result = planners.rrt_family(start, goal, mode=algorithm, max_iter=2500, step_size=1.5)
                path, elapsed, nodes = result[:3]
                valid, status = planners.validate_path(path, start, goal)
                length = calculate_path_length(path) if valid else 0.0
                rows.append((density, map_idx, pair_idx, algorithm, repeat, seed,
                             "Yes" if valid else "No", status,
                             f"{length:.6f}", f"{elapsed:.6f}", nodes))
                record = {"density_pct": density, "map_idx": map_idx, "pair_idx": pair_idx,
                          "algorithm": algorithm, "repeat": repeat, "planner_seed": seed,
                          "validation_status": status, "path": path}
                if algorithm == "PRM":
                    record.update(planners.last_prm_sampling)
                paths.append(record)
    return map_key(density, map_idx), rows, paths


def run(layout, densities, workers, trials_file, paths_file):
    tasks = [(d, m, str(layout.root)) for d in densities for m in MAP_INDICES]
    rows, paths = [], []
    began = time.monotonic()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_map, task) for task in tasks]
        for done, future in enumerate(as_completed(futures), 1):
            stem, map_rows, map_paths = future.result()
            rows.extend(map_rows)
            paths.extend(map_paths)
            print(f"{done}/{len(tasks)} maps: {stem}, {len(map_rows)} calls, "
                  f"{sum(r[6] == 'Yes' for r in map_rows)} valid; elapsed {time.monotonic() - began:.1f}s",
                  flush=True)
    # One file each, in key order, so worker completion order does not matter.
    order = {name: i for i, name in enumerate(SAMPLING_ALGORITHMS)}
    rows.sort(key=lambda r: (r[0], r[1], r[2], r[4], order[r[3]]))
    paths.sort(key=lambda r: (r["density_pct"], r["map_idx"], r["pair_idx"], r["repeat"], order[r["algorithm"]]))
    for target in (trials_file, paths_file):
        target.parent.mkdir(parents=True, exist_ok=True)
    tmp = trials_file.with_name(trials_file.name + ".tmp")
    with tmp.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(FIELDS)
        writer.writerows(rows)
    os.replace(tmp, trials_file)
    write_jsonl(paths_file, paths)
    return len(tasks), len(rows)


# ---------------------------------------------------------------- check and summarize
def load_and_verify(layout, design):
    """Check keys, seeds, endpoints, lengths, bounds, zero-length segments, collisions
    and the one-to-one link between the CSV rows and the JSONL lines."""
    algorithms, pairs, repeats = design["algorithms"], design["pairs"], design["repeats"]
    calls_per_map = len(pairs) * len(algorithms) * repeats
    with layout.repeated_trials_csv.open(newline="") as stream:
        all_rows = list(csv.DictReader(stream))
    paths = {}
    for record in read_jsonl(layout.repeated_paths):
        key = (record["density_pct"], record["map_idx"], record["pair_idx"], record["algorithm"], record["repeat"])
        assert key not in paths
        paths[key] = record
    expected_maps = {(d, m) for d in design["densities"] for m in design["maps"]}
    by_map = defaultdict(list)
    for row in all_rows:
        by_map[(int(row["Density_pct"]), int(row["Map_ID"]))].append(row)
    assert set(by_map) == expected_maps, sorted(set(by_map) ^ expected_maps)
    seen = set()
    valid_count = invalid_count = 0
    for (density, map_idx), rows in sorted(by_map.items()):
        map_data = read_json(layout.map_file(density, map_idx))
        obstacles = [wkt.loads(s).buffer(map_data["robot_radius_m"]) for s in map_data["obstacles_wkt"]]
        tree = STRtree(obstacles)
        inset = map_data["motion_boundary_inset_m"]
        assert len(rows) == calls_per_map, (density, map_idx, len(rows))
        for row in rows:
            pair, algorithm, repeat = int(row["Pair_ID"]), row["Algorithm"], int(row["Repeat"])
            assert pair in pairs and algorithm in algorithms and 1 <= repeat <= repeats
            key = (density, map_idx, pair, algorithm, repeat)
            assert key not in seen
            seen.add(key)
            saved = paths[key]
            assert int(row["PlannerSeed"]) == saved["planner_seed"]
            scenario = query(map_data, pair)
            if row["Success"] == "Yes":
                assert row["ValidationStatus"] == "valid" == saved["validation_status"]
                path = saved["path"]
                assert path and math.dist(path[0], scenario["start"]) < 1e-8
                assert math.dist(path[-1], scenario["goal"]) < 1e-8
                length = sum(math.dist(a, b) for a, b in zip(path, path[1:]))
                assert abs(length - float(row["PathLength_m"])) < 1e-5
                for x, y in path:
                    assert inset <= x <= map_data["width"] - inset and inset <= y <= map_data["height"] - inset
                for a, b in zip(path, path[1:]):
                    assert math.dist(a, b) > 0, "zero-length segment"
                    line = LineString((a, b))
                    assert not any(line.intersects(obstacles[int(i)]) for i in tree.query(line))
                valid_count += 1
            else:
                # A returned route that failed validation is kept for diagnosis.
                assert row["ValidationStatus"] == saved["validation_status"] != "valid"
                if saved["validation_status"] == "no_path":
                    assert saved["path"] is None
                else:
                    invalid_count += 1
                assert float(row["PathLength_m"]) == 0
    total_calls = len(expected_maps) * calls_per_map
    assert len(all_rows) == total_calls == len(seen) == len(paths)
    assert len({(r["Density_pct"], r["Map_ID"], r["Pair_ID"]) for r in all_rows}) == len(expected_maps) * len(pairs)
    assert len({r["PlannerSeed"] for r in all_rows}) == total_calls
    return all_rows, valid_count, invalid_count


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(layout, rows, design):
    repeats = design["repeats"]
    per_density = len(design["maps"]) * len(design["pairs"])
    by_query = defaultdict(list)
    for row in rows:
        by_query[(int(row["Density_pct"]), int(row["Map_ID"]), int(row["Pair_ID"]), row["Algorithm"])].append(row)
    overall, by_density = [], []
    for algorithm in design["algorithms"]:
        selected = [r for r in rows if r["Algorithm"] == algorithm]
        successes = [r for r in selected if r["Success"] == "Yes"]
        counts = Counter(sum(r["Success"] == "Yes" for r in qrows)
                         for (d, m, p, name), qrows in by_query.items() if name == algorithm)
        overall.append({
            "Algorithm": algorithm, "Scenarios": per_density * len(design["densities"]), "Calls": len(selected),
            "ValidPaths": len(successes),
            "SuccessRate_pct": f"{len(successes) / len(selected) * 100:.2f}",
            f"Scenarios_0of{repeats}": counts[0],
            f"Scenarios_1to{repeats - 1}of{repeats}": sum(counts[i] for i in range(1, repeats)),
            f"Scenarios_{repeats}of{repeats}": counts[repeats],
            "MedianTime_sec": f"{median(float(r['Time_sec']) for r in selected):.4f}",
            "MedianSuccessfulPathLength_m": f"{median(float(r['PathLength_m']) for r in successes):.3f}",
        })
        for density in design["densities"]:
            group = [r for r in selected if int(r["Density_pct"]) == density]
            valid = [r for r in group if r["Success"] == "Yes"]
            assert len(group) == per_density * repeats
            by_density.append({
                "TargetDensity_pct": density, "Algorithm": algorithm,
                "Scenarios": per_density, "Calls": len(group), "ValidPaths": len(valid),
                "SuccessRate_pct": f"{len(valid) / len(group) * 100:.2f}",
                "MedianTime_sec": f"{median(float(r['Time_sec']) for r in group):.4f}",
                "MedianSuccessfulPathLength_m": f"{median(float(r['PathLength_m']) for r in valid):.3f}" if valid else "",
            })
    write_csv(layout.repeated_summary_overall, overall)
    write_csv(layout.repeated_summary_by_density, by_density)
    return overall, by_density, by_query


def check_and_summarize(layout):
    manifest = read_json(layout.repeat_manifest)
    design = {"densities": manifest["density_targets_pct"], "maps": manifest["map_indices"],
              "pairs": manifest["pair_indices"], "algorithms": manifest["algorithms"],
              "repeats": manifest["repeats_per_scenario_algorithm"]}
    rows, valid, invalid = load_and_verify(layout, design)
    overall, by_density, by_query = summarize(layout, rows, design)
    repeated_success(by_density, by_query, design, layout.repeated_figure)
    maps = len(design["densities"]) * len(design["maps"])
    write_json(layout.repeat_validation, {
        "map_files": maps, "scenarios": maps * len(design["pairs"]), "planner_calls": len(rows),
        "valid_paths": valid, "no_path": len(rows) - valid - invalid,
        "csv_jsonl_join_mismatches": 0, "duplicate_seeds": 0, "invalid_returned_paths": invalid,
        "checks": ["unique density-map-pair-algorithm-repeat keys", "scenario endpoints", "waypoint path length",
                   "motion-region bounds (robot-radius inset)", "no zero-length segment",
                   "continuous collision against buffered WKT", "one-to-one CSV and JSONL linkage"]})
    print(f"Five-seed subset: {len(rows)} calls, {valid} valid paths, {len(rows) - valid} without a path")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path, help="Data2D data directory")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--summarize-only", action="store_true", help="Check and summarize existing records")
    parser.add_argument("--output-dir", type=Path,
                        help="Run only, writing repeated_trials.csv and repeated_subset.jsonl here")
    args = parser.parse_args(argv)
    layout = Layout(args.data.resolve())
    if args.summarize_only:
        check_and_summarize(layout)
        return
    densities = read_json(layout.run_manifest)["density_targets_pct"]
    if args.output_dir:
        run(layout, densities, args.workers,
            args.output_dir / "repeated_trials.csv", args.output_dir / "repeated_subset.jsonl")
        return
    maps, calls = run(layout, densities, args.workers, layout.repeated_trials_csv, layout.repeated_paths)
    write_json(layout.repeat_manifest, {
        "design": "stratified subset with 3 map indices and 2 pair indices per density",
        "density_targets_pct": densities,
        "map_indices": list(MAP_INDICES), "pair_indices": list(PAIR_INDICES),
        "algorithms": list(SAMPLING_ALGORITHMS), "repeats_per_scenario_algorithm": SEED_COUNT,
        "seed_rule": "first 32 bits of SHA-256(Data2D-repeat-subset:density:map:pair:algorithm:repeat)",
        "prm_samples": 600, "prm_k_neighbors": 15, "prm_max_sample_attempts": 60000,
        "motion_region": "workspace inset by robot_radius; see run_manifest.json",
        "rrt_iterations": 2500, "rrt_step_m": 1.5,
        "map_files": maps, "scenarios": maps * len(PAIR_INDICES), "planner_calls": calls,
        "benchmark_workers": args.workers,
        "timing_scope": "planner function only; excludes map load, validation, and file writing",
        "code_sha256": code_sha256(),
        "environment": {
            "platform": platform.platform(), "python": platform.python_version(),
            **{name: importlib.metadata.version(name) for name in ("numpy", "scipy", "shapely", "matplotlib")},
        },
        "note": "Supplementary repeated trials; the baseline benchmark.csv remains unchanged."})
    check_and_summarize(layout)


if __name__ == "__main__":
    main()
