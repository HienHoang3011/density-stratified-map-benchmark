"""Export the occupancy rasters, the geometry previews and the planner gallery.

Usage: python -m data2d visuals <data dir> [--workers 8]

Run after `data2d.generate`. The raster is recomputed from the stored WKT and
must reproduce the recorded occupancy exactly.
"""
import argparse
import multiprocessing as mp

import numpy as np
from PIL import Image
from shapely import wkt

from data2d.layout import Layout, code_sha256, read_json, write_json
from data2d.core.maps import MapGenerator
from data2d.report.plots import map_preview, path_comparison


def export_map(job):
    source, root = job
    layout = Layout(root)
    item = read_json(source)
    shapes = [wkt.loads(value) for value in item['obstacles_wkt']]
    generator = MapGenerator(width=item['width'], height=item['height'],
                             robot_radius=item['robot_radius_m'],
                             grid_resolution=item['grid_resolution_m'])
    grid = generator.convert_to_grid(shapes)
    occupancy = 100.0 * np.count_nonzero(grid) / grid.size
    if abs(occupancy - item['achieved_density_pct']) > 1e-8:
        raise ValueError(f"Raster mismatch: {source.name}")
    # PNG rows run top to bottom: grid cell (gx, gy) is pixel (gx, 299 - gy); 0 = occupied.
    raster = np.flipud(np.where(grid == 1, 0, 255).astype(np.uint8))
    Image.fromarray(raster, mode='L').save(layout.raster_file(item['density_pct'], item['map_idx']), optimize=True)
    map_preview(item, shapes, occupancy, layout.preview_file(item['density_pct'], item['map_idx']))
    return source.name


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('data', help='Data2D data directory')
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args(argv)
    layout = Layout(args.data)
    manifest = read_json(layout.run_manifest)
    if manifest['status'] != 'complete':
        raise RuntimeError('Export only from a complete run')
    map_paths = layout.map_files()
    if len(map_paths) != manifest['map_files']:
        raise RuntimeError('Map files are incomplete')
    for folder in (layout.rasters, layout.previews, layout.gallery):
        folder.mkdir(parents=True, exist_ok=True)
    with mp.Pool(args.workers) as pool:
        exported = list(pool.imap_unordered(export_map, ((p, layout.root) for p in map_paths)))
    if len(exported) != manifest['map_files']:
        raise RuntimeError('Missing map visualization')

    densities = manifest['density_targets_pct']
    for density in densities:
        path_comparison(layout.root, layout.gallery_file(density, 1, 1), density=density, map_id=1, pair_id=1)

    counts = {
        'occupancy_png_maps': len(list(layout.rasters.glob('*.png'))),
        'geometry_preview_png_maps': len(list(layout.previews.glob('*.png'))),
        'planner_comparison_png': len(list(layout.gallery.glob('*.png'))),
        'planner_comparison_pdf': len(list(layout.gallery.glob('*.pdf'))),
    }
    expected = [manifest['map_files'], manifest['map_files'], len(densities), len(densities)]
    if list(counts.values()) != expected:
        raise RuntimeError(f'Visual counts {counts} != {expected}')
    manifest['public_visuals'] = {
        **counts,
        'exported_after_benchmark': True,
        'raster_convention': 'PNG 0 occupied, 255 free; pixel (gx, 299-gy) for map grid cell (gx,gy)',
        'code_sha256': code_sha256(),
    }
    write_json(layout.run_manifest, manifest)
    print('Exported:', counts)


if __name__ == '__main__':
    main()
