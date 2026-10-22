"""Command line: `python -m data2d <command> ...` (run from the repository root).

    run        the whole pipeline in one go (generate, visuals, repeat, audit, figures)
    generate   step 1: maps, queries and the seven-planner baseline
    visuals    step 2: occupancy rasters, previews and planner gallery
    repeat     step 3: five-seed reruns of the sampling-based planners
    audit      step 4: independent audit of a data directory
    figures    step 5: figures and table values of the article
    live-audit audit maps while `generate` is still running

`python -m data2d <command> --help` lists the options of each command.
"""
import argparse
import importlib
import os
import subprocess
import sys
from pathlib import Path


def run(argv):
    parser = argparse.ArgumentParser(
        prog='python -m data2d run',
        description='Generate, benchmark, check and summarize a complete Data2D data directory. '
                    'The defaults reproduce the released dataset.')
    parser.add_argument('--output-dir', type=Path, required=True, help='New, non-existing data directory')
    parser.add_argument('--article-dir', type=Path,
                        help='Where to write the article figures and tables (default: <output-dir>_article)')
    parser.add_argument('--quick', action='store_true',
                        help='Small test run: targets 10 and 50 with 20 maps each (about two minutes)')
    parser.add_argument('--workers', type=int, default=8, help='Worker processes for generate and visuals')
    parser.add_argument('--repeat-workers', type=int, default=6, help='Worker processes for the five-seed subset')
    parser.add_argument('--live-audit', action='store_true',
                        help='Audit every map while it is generated and stop on the first failure')
    args = parser.parse_args(argv)

    generate, visuals, repeat, audit, figures = (
        importlib.import_module(STEPS[name]) for name in ('generate', 'visuals', 'repeat', 'audit', 'figures'))

    data = str(args.output_dir)
    article = str(args.article_dir or Path(f"{args.output_dir}_article"))
    maps = ['--densities', '10', '50', '--maps-per-density', '20', '--benchmark-maps-per-density', '20'] \
        if args.quick else []

    print('== 1/5 generate: maps, queries and seven-planner baseline', flush=True)
    watcher = None
    if args.live_audit:
        # The watcher waits for the run manifest, then audits each finished map.
        watcher = subprocess.Popen([sys.executable, '-m', 'data2d', 'live-audit', data, '--gen-pid', str(os.getpid())])
    generate.main(['--output-dir', data, '--workers', str(args.workers), '--no-plots', *maps])
    if watcher and watcher.wait() != 0:
        sys.exit('Live audit failed')

    print('== 2/5 visuals: rasters, previews and gallery', flush=True)
    visuals.main([data, '--workers', str(args.workers)])
    print('== 3/5 repeat: five-seed subset', flush=True)
    repeat.main([data, '--workers', str(args.repeat_workers)])
    print('== 4/5 audit: independent checks', flush=True)
    audit.main([data])
    print('== 5/5 figures: article figures and tables', flush=True)
    figures.main([data, article])
    print(f'Done: data in {data}, article figures and tables in {article}')


STEPS = {
    'generate': 'data2d.pipeline.generate',
    'visuals': 'data2d.pipeline.visuals',
    'repeat': 'data2d.pipeline.repeat',
    'audit': 'data2d.checks.audit',
    'figures': 'data2d.report.figures',
    'live-audit': 'data2d.checks.live_audit',
}


def main():
    command = sys.argv[1] if len(sys.argv) > 1 else None
    if command == 'run':
        run(sys.argv[2:])
    elif command in STEPS:
        importlib.import_module(STEPS[command]).main(sys.argv[2:])
    else:
        sys.exit(__doc__)


if __name__ == '__main__':
    main()
