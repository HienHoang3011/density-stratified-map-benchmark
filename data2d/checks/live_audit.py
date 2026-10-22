"""Audit a Data2D run while the generator is still writing it.

Usage: python -m data2d live-audit <run dir> --gen-pid PID

Every benchmark map and every Nth other map is audited as soon as its final map
JSON appears (the generator writes paths and queries before that file, and all
writes are atomic). On the first failure the generator's process group is
terminated so a bad run stops early. Progress lines report running success
rates so problems in the numbers are visible before the run finishes.
"""
import argparse
import json
import os
import signal
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from data2d.checks.audit import audit_map
from data2d.layout import Layout


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def lower_priority():
    os.nice(10)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('--gen-pid', type=int, required=True)
    parser.add_argument('--sample-every', type=int, default=10,
                        help='Also audit non-benchmark maps whose index is a multiple of this')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--report-every', type=float, default=30.0)
    args = parser.parse_args(argv)
    layout = Layout(args.run)
    manifest = None
    while manifest is None:
        manifest = read_json(layout.run_manifest)
        time.sleep(0.5)

    submitted, pending = set(), {}
    stats = Counter()
    outcomes = defaultdict(Counter)      # (density, algorithm) -> Counter(valid/total)
    last_report = time.monotonic()
    failed = None
    final_scan_done = False

    def report(final=False):
        rates = []
        for density in manifest['density_targets_pct']:
            parts = []
            for algo in ('PRM', 'RRT', 'RRT*', 'Informed RRT*'):
                c = outcomes[(density, algo)]
                if c['total']:
                    parts.append(f"{algo} {100 * c['valid'] / c['total']:.0f}%")
            if parts:
                rates.append(f"D{density}: " + ', '.join(parts))
        head = 'FINAL' if final else time.strftime('%H:%M:%S')
        print(f"[live-audit {head}] maps audited {stats['maps']} (benchmark {stats['benchmark_maps']}), "
              f"queries {stats['scenarios']}, valid paths {stats['valid_paths']}, "
              f"no-path {stats['no_path']}, PRM limit hits {stats['prm_limit_reached']}", flush=True)
        for line in rates:
            print(f"    {line}", flush=True)

    with ProcessPoolExecutor(args.workers, initializer=lower_priority) as pool:
        while True:
            # `generate` is finished once its process has exited or its manifest is no longer
            # 'running' (`python -m data2d run` keeps the process alive for the next steps).
            status = (read_json(layout.run_manifest) or {}).get('status')
            generator_running = alive(args.gen_pid) and status == 'running'
            for path in layout.map_files():
                if path in submitted:
                    continue
                m = read_json(path)
                if m is None or 'scenario_count' not in m:
                    continue                       # first, pre-query write; wait for the final one
                is_bench = m['map_idx'] <= manifest['benchmark_maps_per_density']
                submitted.add(path)
                if is_bench or m['map_idx'] % args.sample_every == 0:
                    pending[path] = (m, pool.submit(audit_map, args.run, path, manifest, None, False))

            for path, (m, future) in list(pending.items()):
                if not future.done():
                    continue
                del pending[path]
                try:
                    stats.update(future.result())
                except Exception as error:          # AssertionError or unreadable data
                    failed = f"{path.name}: {error!r}"
                    break
                paths = layout.paths_file(m['density_pct'], m['map_idx'])
                if paths.exists():
                    for line in paths.read_text().splitlines():
                        rec = json.loads(line)
                        c = outcomes[(rec['density_pct'], rec['algorithm'])]
                        c['total'] += 1
                        c['valid'] += rec['validation_status'] == 'valid'

            if failed:
                print(f"[live-audit] FAILED {failed}", flush=True)
                if alive(args.gen_pid):
                    os.killpg(os.getpgid(args.gen_pid), signal.SIGTERM)
                    print('[live-audit] generator stopped', flush=True)
                for _, future in pending.values():
                    future.cancel()
                sys.exit(1)

            if time.monotonic() - last_report >= args.report_every:
                report()
                last_report = time.monotonic()
            if not generator_running and not pending:
                # The scan above ran after the generator finished, so nothing more will appear.
                if final_scan_done:
                    break
                final_scan_done = True
                continue
            time.sleep(1.0)

    final_manifest = read_json(layout.run_manifest) or {}
    report(final=True)
    if final_manifest.get('status') != 'complete':
        print(f"[live-audit] generator status: {final_manifest.get('status')}", flush=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
