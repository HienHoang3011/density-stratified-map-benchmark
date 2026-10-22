"""File layout of a Data2D data directory and small I/O helpers.

Every module reads and writes the archive through `Layout`, so the folder
structure is defined in this one place:

    maps/        Map_D10_M01.json      map metadata, five queries, obstacle WKT
    rasters/     Map_D10_M01.png       300 x 300 occupancy raster
    paths/       Paths_D10_M01.jsonl   baseline paths; repeated_subset.jsonl
    images/      previews/, gallery/, repeated_subset_success.png/.pdf
    metadata/    benchmark.csv, map_inventory.csv, manifests, audit reports
"""
import hashlib
import json
import os
from pathlib import Path

ALGORITHMS = ('A*', 'Dijkstra', 'BFS', 'PRM', 'RRT', 'RRT*', 'Informed RRT*')
SAMPLING_ALGORITHMS = ('PRM', 'RRT', 'RRT*', 'Informed RRT*')
PACKAGE_DIR = Path(__file__).resolve().parent


def map_key(density, map_idx):
    return f"D{density:02d}_M{map_idx:02d}"


class Layout:
    """Paths inside one Data2D data directory."""

    def __init__(self, root):
        self.root = Path(root)
        self.maps = self.root / 'maps'
        self.rasters = self.root / 'rasters'
        self.paths = self.root / 'paths'
        self.images = self.root / 'images'
        self.previews = self.images / 'previews'
        self.gallery = self.images / 'gallery'
        self.diagnostics = self.images / 'diagnostics'
        self.metadata = self.root / 'metadata'

        self.benchmark_csv = self.metadata / 'benchmark.csv'
        self.inventory_csv = self.metadata / 'map_inventory.csv'
        self.run_manifest = self.metadata / 'run_manifest.json'
        self.audit_report = self.metadata / 'audit_report.json'
        self.boundary_report = self.metadata / 'boundary_clearance_audit.json'
        self.repeat_manifest = self.metadata / 'repeat_manifest.json'
        self.repeat_validation = self.metadata / 'repeat_validation.json'
        self.repeated_trials_csv = self.metadata / 'repeated_trials.csv'
        self.repeated_summary_overall = self.metadata / 'repeated_summary_overall.csv'
        self.repeated_summary_by_density = self.metadata / 'repeated_summary_by_density.csv'
        self.repeated_paths = self.paths / 'repeated_subset.jsonl'
        self.repeated_figure = self.images / 'repeated_subset_success'   # .png and .pdf

    def map_file(self, density, map_idx):
        return self.maps / f"Map_{map_key(density, map_idx)}.json"

    def raster_file(self, density, map_idx):
        return self.rasters / f"Map_{map_key(density, map_idx)}.png"

    def preview_file(self, density, map_idx):
        return self.previews / f"Map_{map_key(density, map_idx)}.png"

    def paths_file(self, density, map_idx):
        return self.paths / f"Paths_{map_key(density, map_idx)}.jsonl"

    def gallery_file(self, density, map_idx, pair_idx):
        """Output path without extension; the gallery is saved as .png and .pdf."""
        return self.gallery / f"Planners_{map_key(density, map_idx)}_P{pair_idx:02d}"

    def map_files(self):
        return sorted(self.maps.glob('Map_*.json'))

    def baseline_path_files(self):
        return sorted(self.paths.glob('Paths_*.jsonl'))


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, data):
    """Write JSON atomically, so a reader never sees a half-written file."""
    tmp = f"{path}.tmp"
    with open(tmp, 'w', encoding='utf-8') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
    os.replace(tmp, path)


def write_jsonl(path, records):
    tmp = Path(f"{path}.tmp")
    with tmp.open('w', encoding='utf-8') as stream:
        for record in records:
            stream.write(json.dumps(record, allow_nan=False) + '\n')
    os.replace(tmp, path)


def read_jsonl(path):
    with open(path, encoding='utf-8') as stream:
        return [json.loads(line) for line in stream]


def query(map_data, pair_idx):
    """The saved query with this pair index from a map JSON."""
    return next(q for q in map_data['queries'] if q['pair_idx'] == pair_idx)


def code_sha256():
    """SHA-256 of every source file of the package, recorded in the manifests."""
    return {f"data2d/{p.relative_to(PACKAGE_DIR).as_posix()}": hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(PACKAGE_DIR.rglob('*.py')) if '__pycache__' not in p.parts}
