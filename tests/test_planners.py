"""Independent graph oracles and tree invariants for the seven planners.

Run from the repository root: python tests/test_planners.py
"""
import json, math, random, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data2d.core import maps, planners, seeds

m = SimpleNamespace(MapGenerator=maps.MapGenerator, PathPlanners=planners.PathPlanners,
                    calculate_path_length=planners.calculate_path_length,
                    stable_seed=seeds.stable_seed, map_seed=seeds.map_seed)

def capture(fn, *args, **kwargs):
    state={}
    def profile(frame,event,arg):
        if event=='return' and frame.f_code is fn.__func__.__code__:
            state.update(frame.f_locals)
    sys.setprofile(profile)
    try: result=fn(*args, **kwargs)
    finally: sys.setprofile(None)
    return result,state

def tree_errors(p, state):
    errors=[]; nodes=state['nodes']; parents=state['parents']; costs=state['costs']
    for idx in range(len(nodes)):
        actual=0.; cur=idx; seen=set()
        while parents[cur] is not None:
            assert cur not in seen, 'parent cycle'
            seen.add(cur); par=parents[cur]
            assert p.is_collision_free_continuous(nodes[cur],nodes[par]), 'tree collision'
            actual+=math.dist(nodes[cur],nodes[par]); cur=par
        assert cur==0
        if not math.isclose(actual,costs[idx],abs_tol=1e-8): errors.append(idx)
    return errors

summary={}
# Oracle builds its own adjacency and uses SciPy shortest paths, without calling planner moves.
random.seed(107); grid_cases=100
for case in range(grid_cases):
    grid=(np.random.default_rng(case).random((12,12))<.25).astype(int)
    grid[0,0]=grid[-1,-1]=0
    gen=m.MapGenerator(width=12,height=12,grid_resolution=1,robot_radius=0)
    p=m.PathPlanners(gen,[],grid)
    rows=[]; cols=[]; weights=[]
    for y in range(12):
        for x in range(12):
            if grid[y,x]: continue
            for dy in (-1,0,1):
                for dx in (-1,0,1):
                    xx,yy=x+dx,y+dy
                    if (dx==dy==0 or not(0<=xx<12 and 0<=yy<12) or grid[yy,xx]): continue
                    if dx and dy and (grid[y,xx] or grid[yy,x]): continue
                    rows.append(y*12+x);cols.append(yy*12+xx);weights.append(math.hypot(dx,dy))
    graph=csr_matrix((weights,(rows,cols)),shape=(144,144))
    weighted=shortest_path(graph,indices=0)[143]
    hops=shortest_path(graph,indices=0,unweighted=True)[143]
    for mode in ('A*','Dijkstra','BFS'):
        path=p.grid_search((.5,.5),(11.5,11.5),mode)[0]
        oracle=hops if mode=='BFS' else weighted
        if not math.isfinite(oracle): assert path is None
        else:
            assert path is not None
            got=len(path)-1 if mode=='BFS' else m.calculate_path_length(path)
            assert math.isclose(got,oracle,abs_tol=1e-8),(case,mode,got,oracle)
summary['grid_oracle_cases']=grid_cases
# Diagonal blocked by two side cells; start==goal and no-path are explicit cases.
gen=m.MapGenerator(width=2,height=2,grid_resolution=1,robot_radius=0)
p=m.PathPlanners(gen,[],np.array([[0,1],[1,0]]))
for mode in ('A*','Dijkstra','BFS'):
    assert p.grid_search((.5,.5),(1.5,1.5),mode)[0] is None
    assert p.grid_search((.5,.5),(.5,.5),mode)[0]==[(.5,.5),(.5,.5)]

# Continuous planners: open and obstructed environments, multiple deterministic seeds.
prm_count=rrt_count=0
for obstacles in ([],[box(4,0,6,7)]):
    gen=m.MapGenerator(width=10,height=10,grid_resolution=.2,robot_radius=.1)
    grid=gen.convert_to_grid(obstacles); p=m.PathPlanners(gen,obstacles,grid)
    start,goal=(1.,1.),(9.,9.)
    for seed in range(8):
        random.seed(seed)
        result,state=capture(p.prm,start,goal,num_samples=120,k_neighbors=15)
        assert all(p.is_free_configuration(q) for q in state['nodes'])
        rows=[];cols=[];weights=[]
        for i,edges in state['graph'].items():
            for j,w in edges:
                assert p.is_collision_free_continuous(state['nodes'][i],state['nodes'][j])
                assert math.isclose(w,math.dist(state['nodes'][i],state['nodes'][j]))
                assert (i,w) in state['graph'][j]
                rows.append(i);cols.append(j);weights.append(w)
        oracle=shortest_path(csr_matrix((weights,(rows,cols)),shape=(122,122)),indices=0)[1]
        path=result[0]
        assert (path is not None)==math.isfinite(oracle)
        if path:
            assert math.isclose(m.calculate_path_length(path),oracle,abs_tol=1e-8)
            assert p.validate_path(path,start,goal)[0]
        prm_count+=1
        for mode in ('RRT','RRT*','Informed RRT*'):
            random.seed(seed)
            result,state=capture(p.rrt_family,start,goal,mode=mode,max_iter=500)
            assert not tree_errors(p,state),(seed,mode)
            assert result[2]==len(state['nodes'])
            if result[0]:
                assert p.validate_path(result[0],start,goal)[0]
                assert all(math.dist(a,b)>0 for a,b in zip(result[0],result[0][1:])),(seed,mode)
                assert math.isclose(m.calculate_path_length(result[0]),state['c_best'],abs_tol=1e-8)
                assert math.isclose(state['c_best'],min(state['costs'][idx]+d for idx,d in state['goal_candidates'].items()),abs_tol=1e-8)
            rrt_count+=1
# PRM rejection sampling stops on a fully blocked workspace instead of looping forever.
gen=m.MapGenerator(width=10,height=10,grid_resolution=.2,robot_radius=.1)
blocked=[box(0,0,10,10)]; p=m.PathPlanners(gen,blocked,gen.convert_to_grid(blocked))
random.seed(0)
assert p.prm((1.,1.),(9.,9.),num_samples=50,max_sample_attempts=500)[0] is None
summary['prm_graph_oracle_cases']=prm_count
summary['rrt_tree_invariant_cases']=rrt_count
# Boundary-intersecting ellipse: current implementation never clamps a sample.
gen=m.MapGenerator(width=10,height=10,grid_resolution=.2,robot_radius=.1)
p=m.PathPlanners(gen,[],np.zeros((50,50),dtype=int))
random.seed(9)
count=0
for _ in range(10000):
    q=p.sample_informed_rrt_star((.2,.2),(9.8,.2),15.)
    assert 0<=q[0]<=10 and 0<=q[1]<=10
    assert math.dist(q,(.2,.2))+math.dist(q,(9.8,.2))<=15.+1e-10
    count+=q[0] in (0,10) or q[1] in (0,10)
assert count==0
summary['informed_current_exact_boundary_samples']=count
assert p.rrt_neighbor_radius(100000,1.5)<p.rrt_neighbor_radius(100,1.5)

# Motion region: border cells are blocked and every planner keeps waypoints inside the inset.
gen=m.MapGenerator(width=10,height=10,grid_resolution=.1,robot_radius=.1)
motion=gen.motion_grid(np.zeros((100,100),dtype=int))
assert motion[:,0].all() and motion[:,-1].all() and motion[0,:].all() and motion[-1,:].all()
assert motion[1:-1,1:-1].sum()==0
wall=[box(4,.15,6,9.85)]  # buffered wall leaves 0.05 m gaps against the top and bottom edges
p=m.PathPlanners(gen,wall,gen.motion_grid(gen.convert_to_grid(wall)))
assert not p.is_free_configuration((5.,.02)) and not p.is_free_configuration((5.,9.98))
assert p.validate_path([(1.,5.),(1.,.02),(9.,.02),(9.,5.)],(1.,5.),(9.,5.))==(False,'out_of_bounds')
boundary_cases=0
for seed in range(6):
    for name,fn in (('A*',lambda: p.grid_search((1.,5.),(9.,5.),'A*')),
                    ('PRM',lambda: p.prm((1.,5.),(9.,5.),num_samples=150)),
                    ('RRT*',lambda: p.rrt_family((1.,5.),(9.,5.),mode='RRT*',max_iter=400))):
        random.seed(seed)
        assert fn()[0] is None,(name,'crossed the wall through a gap thinner than the inset')
        boundary_cases+=1
open_p=m.PathPlanners(gen,[],gen.motion_grid(np.zeros((100,100),dtype=int)))
for seed in range(6):
    for mode in ('RRT','RRT*','Informed RRT*'):
        random.seed(seed)
        route=open_p.rrt_family((.25,.25),(9.75,9.75),mode=mode,max_iter=400)[0]
        assert route and all(open_p.in_motion_region(q) for q in route)
        boundary_cases+=1
    random.seed(seed)
    route=open_p.prm((.25,.25),(9.75,9.75),num_samples=80)[0]
    assert route and all(open_p.in_motion_region(q) for q in route)
    assert open_p.last_prm_sampling['prm_sample_limit_reached'] is False
    boundary_cases+=1
summary['motion_region_cases']=boundary_cases

# Seed rule documented in the manifest: SHA-256 over colon-joined fields, first 32 bits.
import hashlib
for parts in ((20261010,50,1),(20261010,50,1,1,'scenario'),(20261010,50,1,1,'Informed RRT*')):
    expected=int(hashlib.sha256(':'.join(map(str,parts)).encode()).hexdigest()[:8],16)
    assert m.stable_seed(*parts)==expected
assert m.map_seed(20261010,50,1)==m.stable_seed(20261010,50,1)
summary['seed_rule_cases']=3
summary['status']='passed'
print(json.dumps(summary,indent=2))
