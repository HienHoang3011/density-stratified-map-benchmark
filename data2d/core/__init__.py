"""The algorithms that define the data: how maps, queries and paths are made.

seeds     the seed rule (first 32 bits of SHA-256 over colon-joined fields)
maps      obstacle sampling, robot-aware rasterization and the motion grid
planners  A*, Dijkstra, BFS, PRM, RRT, RRT*, Informed RRT* and the path validator
"""
