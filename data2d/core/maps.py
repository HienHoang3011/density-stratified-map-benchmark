"""Map generation: random obstacles, robot-aware rasterization and the motion grid."""
import math
import random

import numpy as np
from shapely import intersects, box as vector_box
from shapely.affinity import rotate, scale
from shapely.geometry import Point, box


class MapGenerator:
    def __init__(self, width=30.0, height=30.0, 
                 obstacle_types=['circle', 'ellipse', 'rectangle'],
                 min_size=1.5, max_size=6.0, robot_radius=0.1, grid_resolution=0.1):
        self.width = width
        self.height = height
        self.obstacle_types = obstacle_types
        self.min_size = min_size
        self.max_size = max_size
        self.robot_radius = robot_radius
        self.grid_resolution = grid_resolution
        
        self.grid_w = int(math.ceil(width / grid_resolution))
        self.grid_h = int(math.ceil(height / grid_resolution))
        self._valid_free_cells = None

    def generate_random_obstacle(self):
        obs_type = random.choice(self.obstacle_types)
        x = random.uniform(self.min_size, self.width - self.min_size)
        y = random.uniform(self.min_size, self.height - self.min_size)
        angle = random.uniform(0, 360)
        
        if obs_type == 'circle':
            return Point(x, y).buffer(random.uniform(self.min_size, self.max_size)/2)
        elif obs_type == 'ellipse':
            geom = Point(x, y).buffer(1.0)
            geom = scale(geom, xfact=random.uniform(self.min_size, self.max_size)/2, 
                               yfact=random.uniform(self.min_size, self.max_size)/2)
            return rotate(geom, angle, origin='centroid')
        else:
            w = random.uniform(self.min_size, self.max_size)
            h = random.uniform(self.min_size, self.max_size)
            geom = box(x - w/2, y - h/2, x + w/2, y + h/2)
            return rotate(geom, angle, origin='centroid')

    def add_obstacle_to_grid(self, geom, grid):
        expanded_geom = geom.buffer(self.robot_radius)
        minx, miny, maxx, maxy = expanded_geom.bounds
        min_gx = max(0, int(minx / self.grid_resolution))
        max_gx = min(self.grid_w - 1, int(maxx / self.grid_resolution))
        min_gy = max(0, int(miny / self.grid_resolution))
        max_gy = min(self.grid_h - 1, int(maxy / self.grid_resolution))
        region = grid[min_gy:max_gy + 1, min_gx:max_gx + 1]
        ys, xs = np.where(region == 0)
        if len(xs) == 0:
            return
        cell_x = (xs + min_gx) * self.grid_resolution
        cell_y = (ys + min_gy) * self.grid_resolution
        cells = vector_box(cell_x, cell_y,
                    cell_x + self.grid_resolution, cell_y + self.grid_resolution)
        occupied = intersects(expanded_geom, cells)
        region[ys[occupied], xs[occupied]] = 1

    def convert_to_grid(self, obstacles_geom):
        grid = np.zeros((self.grid_h, self.grid_w), dtype=int)
        for geom in obstacles_geom:
            self.add_obstacle_to_grid(geom, grid)
        return grid

    def create_exact_density_map(self, target_density_pct, max_no_progress=1000):
        """Place shapes until the occupied-cell threshold or a stall limit is reached."""
        obstacles = []
        target_ratio = target_density_pct / 100.0
        current_ratio = 0.0
        grid = np.zeros((self.grid_h, self.grid_w), dtype=int)
        
        no_progress = 0
        while current_ratio < target_ratio and no_progress < max_no_progress:
            geom = self.generate_random_obstacle()
            minx, miny, maxx, maxy = geom.bounds
            
            # Reject candidates that extend beyond the workspace.
            if minx < 0 or miny < 0 or maxx > self.width or maxy > self.height:
                no_progress += 1
                continue
            
            obstacles.append(geom)
            self.add_obstacle_to_grid(geom, grid)
            
            # Occupancy is the fraction of occupied cells of the full grid.
            updated_ratio = np.count_nonzero(grid) / grid.size
            no_progress = 0 if updated_ratio > current_ratio else no_progress + 1
            current_ratio = updated_ratio
            
        return obstacles, grid, current_ratio >= target_ratio

    def motion_bounds(self):
        """Region the robot centre may occupy: the workspace inset by the robot radius."""
        r = self.robot_radius
        return r, r, self.width - r, self.height - r

    def motion_grid(self, grid):
        """Obstacle grid plus every cell whose square leaves the motion region.

        The obstacle grid alone defines occupancy density; this grid is used for
        query selection and grid search so that cell paths respect the boundary.
        """
        motion = grid.copy()
        res = self.grid_resolution
        min_x, min_y, max_x, max_y = self.motion_bounds()
        eps = 1e-9
        lo_x = int(math.ceil(min_x / res - eps))
        lo_y = int(math.ceil(min_y / res - eps))
        hi_x = int(math.floor(max_x / res + eps))   # first cell index whose right edge exceeds max_x
        hi_y = int(math.floor(max_y / res + eps))
        motion[:, :lo_x] = 1
        motion[:, hi_x:] = 1
        motion[:lo_y, :] = 1
        motion[hi_y:, :] = 1
        return motion

    def prepare_free_point_cache(self, grid):
        """Cache eligible free-cell centers once per map, keeping uniform sampling."""
        free_y, free_x = np.where(grid == 0)
        min_g = int(math.ceil(self.robot_radius * 2 / self.grid_resolution))
        valid_mask = (
            (free_x >= min_g) & (free_x < self.grid_w - min_g)
            & (free_y >= min_g) & (free_y < self.grid_h - min_g)
        )
        self._valid_free_cells = (free_x[valid_mask], free_y[valid_mask])

    def get_random_free_point(self, grid):
        if self._valid_free_cells is None:
            self.prepare_free_point_cache(grid)
        valid_x, valid_y = self._valid_free_cells
        
        if len(valid_x) == 0:
            return None
        
        idx = random.randint(0, len(valid_x) - 1)
        x_coord = (valid_x[idx] + 0.5) * self.grid_resolution
        y_coord = (valid_y[idx] + 0.5) * self.grid_resolution
        return (x_coord, y_coord)
