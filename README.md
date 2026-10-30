# Data2D: density-stratified 2D maps for path-planning benchmarking

This repository contains the code that generates, benchmarks and checks the Data2D dataset, described in "A density-stratified dataset of synthetic two-dimensional obstacle maps and path-planning records" by Nguyen Hai Long, Hoang Chi Hien, Pham Quang Toan, Nguyen Duc Trung and Nguyen Kieu Linh. The dataset itself is on Zenodo: https://doi.org/10.5281/zenodo.23795405.

The pipeline

1. places circles, ellipses and rotated rectangles in a 30 m × 30 m workspace until a target occupied-cell ratio (10, 20, 30, 40 or 50%) is reached,
2. rasterizes each map into an exact 300 × 300 occupancy grid, with obstacles inflated by a 0.1 m robot radius,
3. selects five start–goal queries per map that are guaranteed to be solvable by the robot and at least 12 m apart,
4. runs seven planners on every query (A*, Dijkstra and BFS on the grid; PRM, RRT, RRT* and Informed RRT* in continuous space) and validates every returned path,
5. reruns the four sampling-based planners with five seeds on 30 fixed queries, and
6. audits the result independently of the generator.

All random choices come from one base seed, so a rerun with the pinned package versions reproduces the maps, queries, rasters, paths and every benchmark field except the run times, byte for byte.

## Repository structure

```text
data2d/
├── __main__.py           the command line: python -m data2d <command>
├── layout.py             folder structure of a data directory, small I/O helpers
├── core/                 the algorithms that define the data
│   ├── seeds.py          the seed rule (first 32 bits of SHA-256 over colon-joined fields)
│   ├── maps.py           obstacle sampling, rasterization and the motion grid
│   └── planners.py       the seven planners and the path validator
├── pipeline/             the steps that write a data directory
│   ├── generate.py       maps, their five queries and the seven-planner baseline
│   ├── visuals.py        occupancy rasters, previews and planner gallery
│   └── repeat.py         five-seed reruns of the sampling-based planners
├── checks/               checks that do not reuse the generator's code
│   ├── audit.py          independent audit of a finished data directory
│   └── live_audit.py     the same per-map checks while generate is running
└── report/               figures and tables
    ├── plots.py          previews, gallery, five-seed figure, diagnostic plots
    └── figures.py        figures and table values of the article
tests/test_planners.py    planner tests against independent oracles
requirements.txt          pinned package versions used for the released data
```

`checks/audit.py` imports nothing from the package except `layout.py`: it re-derives the rasters, the motion grid and every seed from the documented rules, so it checks the data without reusing the generator's code.

## Requirements

Python 3.12 and the pinned packages:

```bash
pip install -r requirements.txt
```

Everything runs on the CPU. The released run took about one hour with 8 worker processes on an Intel Core i7-1255U.

## Usage

Run the commands from the repository root. One command runs the whole pipeline with the settings of the released dataset:

```bash
python tests/test_planners.py
python -m data2d run --output-dir new_run --live-audit
```

This writes the data directory `new_run/` and the article figures and tables to `new_run_article/`. Add `--quick` for a small test run (targets 10 and 50, 20 maps each, about two minutes).

The steps can also be run one by one; `python -m data2d <command> --help` lists their options:

```bash
python -m data2d generate --output-dir new_run --no-plots   # maps, queries, baseline
python -m data2d visuals new_run                            # rasters, previews, gallery
python -m data2d repeat new_run                             # five-seed subset
python -m data2d audit new_run                              # independent audit
python -m data2d figures new_run article                    # article figures and tables
```

Without `--no-plots`, `generate` also saves a diagnostic image of every planner call in `images/diagnostics/`.

The same commands work on the released dataset: unpack `Data2D.zip` and pass the `Data2D/` folder, for example `python -m data2d audit Data2D` or `python -m data2d figures Data2D article`. `repeat` and `audit` write their records and reports into that folder; use `python -m data2d repeat Data2D --output-dir DIR` to rerun the five-seed subset without touching the released files.

## Output structure

`generate`, `visuals` and `repeat` write the same layout as the released dataset:

```text
new_run/
├── maps/        Map_D10_M01.json, ...    metadata, the five start–goal queries, obstacle WKT
├── rasters/     Map_D10_M01.png, ...     exact 300 × 300 occupancy rasters (0 = occupied)
├── paths/       Paths_D10_M01.jsonl, ... baseline paths; repeated_subset.jsonl (five seeds)
├── images/      previews/, gallery/, repeated_subset_success.png/.pdf
└── metadata/    benchmark.csv, map_inventory.csv, repeated_trials.csv, summaries,
                 run_manifest.json, repeat_manifest.json, audit reports
```

Keys such as `D50_M01_P01` (target 50%, map 1, query 1) link maps, queries, benchmark rows and paths. Coordinates are in metres with the origin at the lower-left corner; raster rows run top to bottom, so grid cell `(gx, gy)` is pixel `(gx, 299 - gy)`. The manifests record the SHA-256 hash of every file in `data2d/`, and for the released data these match the files in this repository.

## How to cite

If you use this code or dataset in your research, please cite:

Nguyen, H. L., Hoang, C. H., Pham, Q. T., Nguyen, D. T., & Nguyen, K. L. (2026).
"Data2D: density-stratified synthetic two-dimensional maps and path-planning records".
Zenodo. https://doi.org/10.5281/zenodo.23795405

and the associated article:

Nguyen, H. L., Hoang, C. H., Pham, Q. T., Nguyen, D. T., & Nguyen, K. L. (2026).
"A density-stratified dataset of synthetic two-dimensional obstacle maps and path-planning records".
Data in Brief.

## License

The code is released under the MIT License (see `LICENSE`).
