"""Deterministic seeds: the first 32 bits of SHA-256 over colon-joined fields.

Because a seed depends only on identifiers, any single map, query or planner
call can be reproduced on its own, independently of worker order.
"""
import hashlib


def stable_seed(*parts):
    payload = ':'.join(str(part) for part in parts).encode('utf-8')
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], 'big')


def map_seed(base_seed, density, map_idx):
    return stable_seed(base_seed, density, map_idx)


def query_seed(base_seed, density, map_idx, pair_idx):
    return stable_seed(base_seed, density, map_idx, pair_idx, 'scenario')


def planner_seed(base_seed, density, map_idx, pair_idx, algorithm):
    return stable_seed(base_seed, density, map_idx, pair_idx, algorithm)


def repeat_seed(density, map_idx, pair_idx, algorithm, repeat):
    """Seeds of the five-seed subset, in their own namespace."""
    return stable_seed('Data2D-repeat-subset', density, map_idx, pair_idx, algorithm, repeat)
