"""The seven reference planners and the path validator.

Grid planners (A*, Dijkstra, BFS) search the motion grid; PRM, RRT, RRT* and
Informed RRT* work in the continuous motion region against obstacles buffered
by the robot radius. All randomness comes from the module-level `random`
generator, which callers seed before each planner call.
"""
import heapq
import math
import random
import time
from collections import deque

from shapely.geometry import LineString, Point
from shapely.strtree import STRtree


def calculate_path_length(path):
    if not path or len(path) < 2: return 0.0
    return sum(math.hypot(path[i+1][0]-path[i][0], path[i+1][1]-path[i][1]) for i in range(len(path)-1))

class PathPlanners:
    def __init__(self, generator, obstacles, grid):
        """`grid` is the motion grid (see MapGenerator.motion_grid)."""
        self.gen = generator
        self.obstacles = obstacles
        self.grid = grid
        self.buffered_obs = [obs.buffer(self.gen.robot_radius) for obs in obstacles]
        self.buffered_tree = STRtree(self.buffered_obs)
        self.min_x, self.min_y, self.max_x, self.max_y = generator.motion_bounds()
        self.last_prm_sampling = None

    def in_motion_region(self, point):
        # The region is convex, so in-region waypoints imply in-region segments.
        return (self.min_x <= point[0] <= self.max_x and
                self.min_y <= point[1] <= self.max_y)

    def sample_motion_region(self):
        return (random.uniform(self.min_x, self.max_x), random.uniform(self.min_y, self.max_y))

    def is_collision_free_continuous(self, p1, p2):
        line = LineString([p1, p2])
        for idx in self.buffered_tree.query(line):
            if line.intersects(self.buffered_obs[int(idx)]):
                return False
        return True

    def is_free_configuration(self, point):
        if not self.in_motion_region(point):
            return False
        geometry = Point(point)
        return not any(geometry.intersects(self.buffered_obs[int(idx)])
                       for idx in self.buffered_tree.query(geometry))

    def is_valid_grid_step(self, current, nxt):
        """Reject diagonal moves through either blocked side-adjacent cell."""
        x, y = nxt
        if not (0 <= x < self.gen.grid_w and 0 <= y < self.gen.grid_h):
            return False
        if self.grid[y, x] != 0:
            return False
        dx, dy = x - current[0], y - current[1]
        if dx and dy:
            if self.grid[current[1], x] != 0 or self.grid[y, current[0]] != 0:
                return False
        return True

    def validate_path(self, path, start, goal):
        """Validate saved waypoints against the buffered continuous geometry."""
        if not path:
            return False, "no_path"
        if len(path) < 2 or math.dist(path[0], start) > 1e-8 or math.dist(path[-1], goal) > 1e-8:
            return False, "wrong_endpoints"
        for point in path:
            if not self.in_motion_region(point):
                return False, "out_of_bounds"
        for p1, p2 in zip(path, path[1:]):
            if not self.is_collision_free_continuous(p1, p2):
                return False, "collision"
        return True, "valid"

    def grid_search(self, start, goal, mode='A*'):
        start_t = time.perf_counter()
        res = self.gen.grid_resolution
        start_g = (int(start[0]/res), int(start[1]/res))
        goal_g = (int(goal[0]/res), int(goal[1]/res))
        
        # Clip to ensure valid coordinates within grid bounds
        start_g = (max(0, min(self.gen.grid_w - 1, start_g[0])), max(0, min(self.gen.grid_h - 1, start_g[1])))
        goal_g = (max(0, min(self.gen.grid_w - 1, goal_g[0])), max(0, min(self.gen.grid_h - 1, goal_g[1])))
        
        if self.grid[start_g[1], start_g[0]] == 1 or self.grid[goal_g[1], goal_g[0]] == 1:
            return None, time.perf_counter() - start_t, 0, []

        if start_g == goal_g:
            return [start, goal], time.perf_counter() - start_t, 0, [((start_g[0]+0.5)*res, (start_g[1]+0.5)*res)]

        neighbors = [(0,1),(1,0),(0,-1),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]
        nodes_expanded = 0
        came_from = {}
        visited_nodes = set()
        
        if mode == 'BFS':
            queue = deque([start_g])
            visited_nodes.add(start_g)
            while queue:
                current = queue.popleft()
                nodes_expanded += 1
                if current == goal_g: break
                for dx, dy in neighbors:
                    nxt = (current[0]+dx, current[1]+dy)
                    if nxt not in visited_nodes and self.is_valid_grid_step(current, nxt):
                        visited_nodes.add(nxt)
                        came_from[nxt] = current
                        queue.append(nxt)
        else:
            open_set = []
            heapq.heappush(open_set, (0, start_g))
            g_score = {start_g: 0}
            
            while open_set:
                _, current = heapq.heappop(open_set)
                if current in visited_nodes:
                    continue
                visited_nodes.add(current)
                nodes_expanded += 1
                if current == goal_g: break

                for dx, dy in neighbors:
                    nxt = (current[0]+dx, current[1]+dy)
                    if not self.is_valid_grid_step(current, nxt):
                        continue
                    move_cost = 1.0 if dx == 0 or dy == 0 else math.sqrt(2)
                    tentative_g = g_score[current] + move_cost

                    if nxt not in g_score or tentative_g < g_score[nxt]:
                        came_from[nxt] = current
                        g_score[nxt] = tentative_g
                        if mode == 'Dijkstra':
                            h = 0.0
                        else:
                            # Octile distance heuristic for 8-connected grid
                            dx_g = abs(nxt[0] - goal_g[0])
                            dy_g = abs(nxt[1] - goal_g[1])
                            h = max(dx_g, dy_g) + (math.sqrt(2) - 1.0) * min(dx_g, dy_g)
                        heapq.heappush(open_set, (tentative_g + h, nxt))

        path = None
        if goal_g in came_from:
            path = []; curr = goal_g
            while curr in came_from:
                path.append(((curr[0]+0.5)*res, (curr[1]+0.5)*res))
                curr = came_from[curr]
            path.append(((start_g[0]+0.5)*res, (start_g[1]+0.5)*res))
            path.reverse()
            path[0] = start; path[-1] = goal
        # Stop the clock before building visualization-only data.
        elapsed = time.perf_counter() - start_t
        explored_continuous = [((p[0]+0.5)*res, (p[1]+0.5)*res) for p in visited_nodes]
        return path, elapsed, nodes_expanded, explored_continuous

    def prm(self, start, goal, num_samples=600, k_neighbors=15, max_sample_attempts=None):
        start_t = time.perf_counter()
        nodes = [start, goal]
        # Bound rejection sampling so a nearly full map cannot loop forever.
        if max_sample_attempts is None:
            max_sample_attempts = 100 * num_samples
        attempts = 0
        while len(nodes) < num_samples + 2 and attempts < max_sample_attempts:
            attempts += 1
            p = self.sample_motion_region()
            if self.is_free_configuration(p):
                nodes.append(p)
        self.last_prm_sampling = {
            'prm_sample_attempts': attempts,
            'prm_samples_accepted': len(nodes) - 2,
            'prm_sample_limit_reached': len(nodes) < num_samples + 2,
        }

        graph = {i: [] for i in range(len(nodes))}
        for i, n1 in enumerate(nodes):
            distances = [(math.hypot(n1[0]-n2[0], n1[1]-n2[1]), j) for j, n2 in enumerate(nodes) if i != j]
            distances.sort()
            for dist, j in distances[:k_neighbors]:
                if self.is_collision_free_continuous(n1, nodes[j]):
                    if j not in (neighbor for neighbor, _ in graph[i]):
                        graph[i].append((j, dist))
                    if i not in (neighbor for neighbor, _ in graph[j]):
                        graph[j].append((i, dist))

        open_set = [(0, 0)]
        came_from = {}
        g_score = {0: 0}
        nodes_expanded = 0
        visited = set()
        path = None

        while open_set:
            _, current = heapq.heappop(open_set)
            if current in visited:
                continue
            visited.add(current)
            nodes_expanded += 1
            if current == 1:
                path = []; curr = 1
                while curr in came_from:
                    path.append(nodes[curr])
                    curr = came_from[curr]
                path.append(start); path.reverse()
                break

            for neighbor, cost in graph[current]:
                tentative_g = g_score[current] + cost
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    h = math.hypot(nodes[neighbor][0]-goal[0], nodes[neighbor][1]-goal[1])
                    heapq.heappush(open_set, (tentative_g + h, neighbor))

        # Stop the clock before building visualization-only data.
        elapsed = time.perf_counter() - start_t
        edges_for_vis = [(nodes[i], nodes[j]) for i, edges in graph.items()
                         for j, _ in edges if i < j]
        return path, elapsed, nodes_expanded, edges_for_vis

    def sample_informed_rrt_star(self, start, goal, c_best):
        """Uniformly sample the informed ellipse intersected with the motion region."""
        if not math.isfinite(c_best):
            return self.sample_motion_region()
        c_min = math.dist(start, goal)
        c_best = max(c_best, c_min)
        center = ((start[0] + goal[0]) / 2, (start[1] + goal[1]) / 2)
        theta = math.atan2(goal[1] - start[1], goal[0] - start[0])
        a = c_best / 2
        b = math.sqrt(max(0.0, (c_best - c_min) * (c_best + c_min))) / 2
        ct, st = math.cos(theta), math.sin(theta)
        while True:
            r = math.sqrt(random.random())
            phi = random.uniform(0, 2 * math.pi)
            u, v = r * math.cos(phi), r * math.sin(phi)
            x = center[0] + a * u * ct - b * v * st
            y = center[1] + a * u * st + b * v * ct
            if self.in_motion_region((x, y)):
                return (x, y)

    def rrt_neighbor_radius(self, node_count, step_size):
        # d=2. Workspace area upper-bounds free-space area; gamma exceeds
        # 2*sqrt((1+1/d)*mu(X_free)/pi), the RRT* sufficient threshold.
        gamma = 2.2 * math.sqrt(1.5 * self.gen.width * self.gen.height / math.pi)
        return min(step_size, gamma * math.sqrt(math.log(node_count) / node_count))

    def rrt_family(self, start, goal, mode='RRT*', max_iter=2500, step_size=1.5):
        start_t = time.perf_counter()
        nodes = [start]
        parents, costs, children = {0: None}, {0: 0.0}, {0: []}
        goal_candidates = {}
        best_goal_idx, c_best = None, float('inf')
        tolerance = 1e-12

        def propagate_costs(parent_idx):
            stack = [parent_idx]
            while stack:
                curr = stack.pop()
                for child in children[curr]:
                    costs[child] = costs[curr] + math.dist(nodes[child], nodes[curr])
                    stack.append(child)

        for i in range(1, max_iter + 1):
            if mode == 'Informed RRT*':
                rand_pt = self.sample_informed_rrt_star(start, goal, c_best)
            else:
                rand_pt = goal if random.random() < 0.1 else self.sample_motion_region()
            nearest_idx = min(range(len(nodes)), key=lambda idx: math.dist(nodes[idx], rand_pt))
            nearest_node = nodes[nearest_idx]
            dist = math.dist(nearest_node, rand_pt)
            if dist <= tolerance:
                continue
            ratio = min(1.0, step_size / dist)
            new_node = (nearest_node[0] + ratio * (rand_pt[0] - nearest_node[0]),
                        nearest_node[1] + ratio * (rand_pt[1] - nearest_node[1]))
            if not self.in_motion_region(new_node):
                continue
            if not self.is_collision_free_continuous(nearest_node, new_node):
                continue
            new_idx = len(nodes)
            actual_step = math.dist(nearest_node, new_node)
            near_indices = []
            best_parent, min_cost = nearest_idx, costs[nearest_idx] + actual_step
            if mode != 'RRT':
                search_radius = self.rrt_neighbor_radius(new_idx + 1, step_size)
                near_indices = [idx for idx in range(new_idx)
                                if math.dist(nodes[idx], new_node) <= search_radius]
                for near_idx in near_indices:
                    candidate_cost = costs[near_idx] + math.dist(nodes[near_idx], new_node)
                    if (candidate_cost + tolerance < min_cost and
                            self.is_collision_free_continuous(nodes[near_idx], new_node)):
                        best_parent, min_cost = near_idx, candidate_cost
            nodes.append(new_node)
            parents[new_idx], costs[new_idx], children[new_idx] = best_parent, min_cost, []
            children[best_parent].append(new_idx)
            for near_idx in near_indices:
                cost_via_new = costs[new_idx] + math.dist(new_node, nodes[near_idx])
                if (cost_via_new + tolerance < costs[near_idx] and
                        self.is_collision_free_continuous(new_node, nodes[near_idx])):
                    children[parents[near_idx]].remove(near_idx)
                    parents[near_idx] = new_idx
                    children[new_idx].append(near_idx)
                    costs[near_idx] = cost_via_new
                    propagate_costs(near_idx)
            dist_to_goal = math.dist(new_node, goal)
            if dist_to_goal <= step_size and self.is_collision_free_continuous(new_node, goal):
                goal_candidates[new_idx] = dist_to_goal
                if mode == 'RRT':
                    best_goal_idx, c_best = new_idx, costs[new_idx] + dist_to_goal
                    break
            # Rewiring can improve any existing goal connection or its ancestor.
            if goal_candidates:
                best_goal_idx = min(goal_candidates,
                                    key=lambda idx: costs[idx] + goal_candidates[idx])
                c_best = costs[best_goal_idx] + goal_candidates[best_goal_idx]

        path = None
        if best_goal_idx is not None:
            path, curr = [goal], best_goal_idx
            while curr is not None:
                path.append(nodes[curr])
                curr = parents[curr]
            path.reverse()
            # A goal-biased sample can place a tree vertex exactly on the goal;
            # drop it so the saved route has no zero-length final segment.
            if math.dist(path[-2], goal) <= tolerance:
                del path[-2]
        # Stop the clock before building visualization-only data.
        elapsed = time.perf_counter() - start_t

        edges_for_vis = [(nodes[child], nodes[parent]) for child, parent in parents.items()
                         if parent is not None]
        final_ellipse = None
        if best_goal_idx is not None:
            edges_for_vis.append((goal, nodes[best_goal_idx]))
            if mode == 'Informed RRT*':
                c_min = math.dist(start, goal)
                final_ellipse = {
                    'center': ((start[0] + goal[0]) / 2, (start[1] + goal[1]) / 2),
                    'angle': math.atan2(goal[1] - start[1], goal[0] - start[0]),
                    'a': c_best / 2,
                    'b': math.sqrt(max(0.0, (c_best - c_min) * (c_best + c_min))) / 2}
        # Nodes_Expanded counts tree vertices only, whether or not the goal was reached.
        return path, elapsed, len(nodes), edges_for_vis, final_ellipse
