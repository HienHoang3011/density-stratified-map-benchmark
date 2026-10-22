"""Data2D: density-stratified synthetic 2D maps and path-planning records.

    layout.py   folder structure of a data directory and small I/O helpers
    core/       the algorithms: seeds, map generation, the seven planners
    pipeline/   the steps that write a data directory: generate, visuals, repeat
    checks/     independent checks: audit, live_audit
    report/     figures and tables

Run everything with `python -m data2d run --output-dir <new dir>`; see __main__.py.
"""
