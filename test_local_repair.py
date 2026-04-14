"""Unit tests for local_repair A0 module."""
import numpy as np
from addict import Dict
from local_repair import repair_endpoints_a0, print_repair_stats

def make_config(**overrides):
    base = {
        'REPAIR_SEARCH_RADIUS': 80,
        'REPAIR_MIN_NODE_DISTANCE': 12,
        'REPAIR_MIN_PATH_MEAN_SCORE': 0.35,
        'REPAIR_MIN_PATH_MIN_SCORE': 0.10,
        'REPAIR_MAX_LOW_SCORE_RATIO': 0.30,
        'REPAIR_LOW_SCORE_THRESHOLD': 0.20,
        'REPAIR_USE_ASTAR': False,  # skip A* for fast unit tests
        'REPAIR_CONE_ANGLE': 100,
        'REPAIR_MIN_DIRECTION_EDGE_LEN': 20,
        'REPAIR_MAX_EDGES_PER_ENDPOINT': 1,
        'REPAIR_MAX_ADDED_EDGES_PER_IMAGE': 30,
        'REPAIR_MAX_ADDED_EDGE_RATIO': 0.10,
        'REPAIR_SKIP_SAME_COMPONENT': True,
    }
    base.update(overrides)
    return Dict(base)


def test_basic_reconnection():
    """A--B    C: gap between B and C, road mask is continuous -> should reconnect."""
    config = make_config()
    points = np.array([[100,100],[150,100],[200,100]], dtype=np.float32)
    edges = np.array([[0, 1]], dtype=np.int32)
    mask = np.zeros((400, 400), dtype=np.uint8)
    mask[95:106, 90:210] = 200  # continuous road signal

    repaired, stats = repair_endpoints_a0(points, edges, mask, config)
    print_repair_stats(stats)
    print(f"  Original: {edges.tolist()}, Repaired: {repaired.tolist()}")
    assert stats['n_edges_added'] > 0, "Should have added edge"
    assert any((e[0]==1 and e[1]==2) or (e[0]==2 and e[1]==1) for e in repaired)
    print("  TEST 1 PASSED: basic reconnection\n")


def test_same_component_blocked():
    """A-B-C-D chain: A and D are endpoints in same component -> no shortcut."""
    config = make_config()
    points = np.array([[100,100],[150,100],[200,100],[200,150]], dtype=np.float32)
    edges = np.array([[0,1],[1,2],[2,3]], dtype=np.int32)
    mask = np.zeros((400, 400), dtype=np.uint8)
    mask[:] = 180  # road signal everywhere

    repaired, stats = repair_endpoints_a0(points, edges, mask, config)
    print_repair_stats(stats)
    assert stats['n_edges_added'] == 0, f"Same component shortcut should be blocked, got {stats['n_edges_added']}"
    print("  TEST 2 PASSED: same-component shortcut blocked\n")


def test_direction_cone():
    """Direction cone should block candidates that are not in the extension direction.
    
    Two parallel horizontal segments separated by 60px vertically (within radius=80):
    Seg1: 0(100,100)--1(140,100)  endpoints extend horizontally
    Seg2: 2(100,160)--3(140,160)  endpoints extend horizontally
    
    Endpoint 1 (dir=right(1,0)): candidate 2 at (100,160), delta=(-40,60), 
      normalized ~ (-0.55, 0.83). dot with (1,0) = -0.55 < cos(50°)=0.64 -> FILTERED
    Endpoint 1: candidate 3 at (140,160), delta=(0,60),
      normalized = (0,1). dot with (1,0) = 0 < 0.64 -> FILTERED
    """
    config = make_config(REPAIR_SKIP_SAME_COMPONENT=False, REPAIR_SEARCH_RADIUS=80)
    
    points = np.array([
        [100, 100],  # 0: seg1 left endpoint
        [140, 100],  # 1: seg1 right endpoint (dir=right)
        [100, 160],  # 2: seg2 left endpoint (dir=left)
        [140, 160],  # 3: seg2 right endpoint (dir=right)
    ], dtype=np.float32)
    edges = np.array([[0, 1], [2, 3]], dtype=np.int32)
    mask = np.zeros((400, 400), dtype=np.uint8)
    mask[:] = 200  # road everywhere

    repaired, stats = repair_endpoints_a0(points, edges, mask, config)
    print_repair_stats(stats)
    assert stats['n_filtered_direction'] > 0, "Direction cone should have filtered candidates"
    assert stats['n_edges_added'] == 0, "No edges should be added (all out of cone)"
    print("  TEST 3 PASSED: direction cone filtering\n")


def test_weak_mask_rejected():
    """Two endpoints with weak road mask between them -> should NOT connect."""
    config = make_config()
    points = np.array([[100,100],[150,100],[200,100]], dtype=np.float32)
    edges = np.array([[0, 1]], dtype=np.int32)
    mask = np.zeros((400, 400), dtype=np.uint8)
    mask[95:106, 90:155] = 200   # road only near A-B
    mask[95:106, 155:210] = 30   # very weak signal between B and C

    repaired, stats = repair_endpoints_a0(points, edges, mask, config)
    print_repair_stats(stats)
    assert stats['n_edges_added'] == 0, "Weak mask should block connection"
    print("  TEST 4 PASSED: weak mask connection rejected\n")


def test_edge_limit():
    """Many endpoints but global edge limit should cap additions."""
    config = make_config(REPAIR_MAX_ADDED_EDGES_PER_IMAGE=2, REPAIR_SKIP_SAME_COMPONENT=False)
    # Create 6 disconnected pairs (12 nodes), each pair within range
    points = []
    edges_list = []
    for i in range(6):
        y = 100 + i * 50
        points.append([100, y])     # left node
        points.append([140, y])     # middle node (endpoint of left pair)
        # right node is separate (not connected)
        # Actually let's make: left--middle  gap  right
        # But we need edge for left-middle
        edges_list.append([i*2, i*2+1])
    
    # Actually this creates degree-1 endpoints at middle nodes, but there's no target to connect to
    # Let me simplify: just have 3 pairs of disconnected segments
    points = np.array([
        [100,100],[130,100],  # pair 0: 0-1
        [170,100],[200,100],  # pair 1: 2-3
        [100,200],[130,200],  # pair 2: 4-5
        [170,200],[200,200],  # pair 3: 6-7
        [100,300],[130,300],  # pair 4: 8-9
        [170,300],[200,300],  # pair 5: 10-11
    ], dtype=np.float32)
    edges = np.array([[0,1],[2,3],[4,5],[6,7],[8,9],[10,11]], dtype=np.int32)
    mask = np.zeros((500, 400), dtype=np.uint8)
    mask[:] = 200  # road everywhere

    repaired, stats = repair_endpoints_a0(points, edges, mask, config)
    print_repair_stats(stats)
    assert stats['n_edges_added'] <= 2, f"Edge limit should cap at 2, got {stats['n_edges_added']}"
    print("  TEST 5 PASSED: global edge limit enforced\n")


def test_coordinate_correctness():
    """CRITICAL: verify compute_path_score uses road_mask[y, x] not road_mask[x, y].
    
    Mask has a HORIZONTAL high-score stripe ONLY at row=50 (y=50), cols 0..199.
    Two points at (x=10, y=50) and (x=90, y=50) — both ON the stripe.
    Path score should be high.
    
    Two points at (x=50, y=10) and (x=50, y=90) — vertical line that crosses
    the stripe at only one point. Path score should be low (most samples miss the stripe).
    
    If xy/rc is swapped, the first pair would score low and the second high.
    """
    from local_repair import compute_path_score
    config = make_config()
    
    mask = np.zeros((200, 200), dtype=np.uint8)
    # Horizontal stripe: rows 48-52 (y=48..52), all columns
    mask[48:53, :] = 220  # high road signal at y≈50
    
    # Test 1: horizontal pair ON the stripe (x varies, y=50)
    p1 = np.array([10.0, 50.0])   # x=10, y=50
    p2 = np.array([90.0, 50.0])   # x=90, y=50
    result_h = compute_path_score(p1, p2, mask, config)
    print(f"  Horizontal pair (on stripe): mean={result_h['mean']:.3f}, min={result_h['min']:.3f}")
    assert result_h['mean'] > 0.8, f"Horizontal pair should be on high-score stripe, got mean={result_h['mean']}"
    
    # Test 2: vertical pair CROSSING the stripe (y varies, x=50)
    p3 = np.array([50.0, 10.0])   # x=50, y=10
    p4 = np.array([50.0, 90.0])   # x=50, y=90
    result_v = compute_path_score(p3, p4, mask, config)
    print(f"  Vertical pair (crosses stripe): mean={result_v['mean']:.3f}, min={result_v['min']:.3f}")
    assert result_v['mean'] < 0.2, f"Vertical pair should mostly miss stripe, got mean={result_v['mean']}"
    
    # Test 3: full repair scenario — only the horizontal pair should connect
    points = np.array([
        [10.0, 50.0],   # 0: on stripe
        [50.0, 50.0],   # 1: on stripe (edge 0-1)
        [90.0, 50.0],   # 2: on stripe (should connect to 1)
        [50.0, 10.0],   # 3: off stripe (edge 3-4)
        [50.0, 40.0],   # 4: near stripe (should NOT connect to 2 easily)
    ], dtype=np.float32)
    edges = np.array([[0, 1], [3, 4]], dtype=np.int32)
    
    repaired, stats = repair_endpoints_a0(points, edges, mask, config)
    print_repair_stats(stats)
    # Node 1 (endpoint on stripe) should connect to node 2 (on stripe)
    has_12 = any((e[0]==1 and e[1]==2) or (e[0]==2 and e[1]==1) for e in repaired)
    assert has_12, "Nodes 1-2 (both on stripe) should be connected"
    print("  TEST 6 PASSED: coordinate correctness verified (xy/rc correct)\n")


if __name__ == "__main__":
    print("=" * 50)
    print("Running Local Repair A0 Unit Tests")
    print("=" * 50 + "\n")
    
    test_basic_reconnection()
    test_same_component_blocked()
    test_direction_cone()
    test_weak_mask_rejected()
    test_edge_limit()
    test_coordinate_correctness()
    
    print("=" * 50)
    print("ALL TESTS PASSED")
    print("=" * 50)

