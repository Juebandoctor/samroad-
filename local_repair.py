"""
Local Repair Module for SAMRoad++ (A0 Version)

Repairs broken road segments by reconnecting degree-1 endpoints
to nearby existing nodes. Does NOT add new nodes.

Inspired by GLD-Road's Local Query Decoder concept, adapted to
SAMRoad++'s post-inference pipeline.
"""

import numpy as np
import networkx as nx
import scipy.spatial
import cv2
from skimage.draw import line as bresenham_line


def get_endpoint_direction(ep_idx, points_xy, G):
    """
    Infer the extension direction of an endpoint from its single existing edge.
    
    Args:
        ep_idx: index of the endpoint node
        points_xy: [N, 2] node coordinates in (x, y) format
        G: networkx Graph
    
    Returns:
        (direction, edge_length) or (None, 0) if direction is unreliable
        direction: unit vector [2,] pointing in the extension direction
        edge_length: length of the existing edge (pixels)
    """
    neighbors = list(G.neighbors(ep_idx))
    if len(neighbors) != 1:
        return None, 0
    
    neighbor_pos = points_xy[neighbors[0]]
    ep_pos = points_xy[ep_idx]
    direction = ep_pos - neighbor_pos  # from neighbor toward endpoint = extension direction
    edge_length = np.linalg.norm(direction)
    if edge_length < 1e-8:
        return None, 0
    
    return direction / edge_length, edge_length


def in_search_cone(ep_pos, ep_dir, target_pos, cone_angle_deg):
    """
    Check if a target point is within the directional search cone.
    
    Args:
        ep_pos: [2,] endpoint position (x, y)
        ep_dir: [2,] unit direction vector of the search cone axis
        target_pos: [2,] candidate target position (x, y)
        cone_angle_deg: full cone angle in degrees
    
    Returns:
        True if target is within the cone
    """
    to_target = target_pos - ep_pos
    dist = np.linalg.norm(to_target)
    if dist < 1e-8:
        return False
    to_target = to_target / dist
    
    cos_angle = np.dot(ep_dir, to_target)
    cos_threshold = np.cos(np.radians(cone_angle_deg / 2.0))
    return cos_angle >= cos_threshold


def compute_path_score(p1_xy, p2_xy, road_mask, config):
    """
    Sample road mask values along a straight line between two points.
    Compute path quality metrics.
    
    Args:
        p1_xy: [2,] start point (x, y) format
        p2_xy: [2,] end point (x, y) format
        road_mask: [H, W] uint8 (0-255) road segmentation mask
        config: config object with REPAIR_LOW_SCORE_THRESHOLD
    
    Returns:
        dict with 'mean', 'min', 'low_ratio' or None if invalid
    """
    x0, y0 = int(round(p1_xy[0])), int(round(p1_xy[1]))
    x1, y1 = int(round(p2_xy[0])), int(round(p2_xy[1]))
    
    H, W = road_mask.shape
    
    # Bresenham line: skimage.draw.line takes (r0, c0, r1, c1) = (y0, x0, y1, x1)
    rr, cc = bresenham_line(y0, x0, y1, x1)
    
    # Clip to image bounds
    valid = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
    rr, cc = rr[valid], cc[valid]
    
    if len(rr) < 2:
        return None
    
    # Normalize mask values to [0, 1]
    scores = road_mask[rr, cc].astype(np.float32) / 255.0
    
    low_threshold = getattr(config, 'REPAIR_LOW_SCORE_THRESHOLD', 0.20)
    
    return {
        'mean': float(np.mean(scores)),
        'min': float(np.min(scores)),
        'low_ratio': float(np.mean(scores < low_threshold)),
    }


def repair_endpoints_a0(graph_points_xy, pred_edges, fused_road_mask, config):
    """
    A0 Local Repair: reconnect degree-1 endpoints to nearby existing nodes.
    No new nodes are added.
    
    Args:
        graph_points_xy: [N, 2] node coordinates in (x, y) format
                         Same indexing as pred_edges
        pred_edges: [M, 2] edge index pairs
        fused_road_mask: [H, W] uint8 (0-255) fused road segmentation mask
        config: config object with REPAIR_* parameters
    
    Returns:
        repaired_edges: [M', 2] edge index pairs (nodes unchanged)
        repair_stats: dict with repair statistics
    """
    # Load config parameters with defaults
    search_radius = getattr(config, 'REPAIR_SEARCH_RADIUS', 80)
    min_node_dist = getattr(config, 'REPAIR_MIN_NODE_DISTANCE', 12)
    min_path_mean = getattr(config, 'REPAIR_MIN_PATH_MEAN_SCORE', 0.35)
    min_path_min = getattr(config, 'REPAIR_MIN_PATH_MIN_SCORE', 0.10)
    max_low_ratio = getattr(config, 'REPAIR_MAX_LOW_SCORE_RATIO', 0.30)
    cone_angle = getattr(config, 'REPAIR_CONE_ANGLE', 100)
    min_dir_edge_len = getattr(config, 'REPAIR_MIN_DIRECTION_EDGE_LEN', 20)
    max_edges_per_ep = getattr(config, 'REPAIR_MAX_EDGES_PER_ENDPOINT', 1)
    max_added_edges = getattr(config, 'REPAIR_MAX_ADDED_EDGES_PER_IMAGE', 30)
    max_added_ratio = getattr(config, 'REPAIR_MAX_ADDED_EDGE_RATIO', 0.10)
    skip_same_comp = getattr(config, 'REPAIR_SKIP_SAME_COMPONENT', True)
    use_astar = getattr(config, 'REPAIR_USE_ASTAR', True)
    block_threshold = getattr(config, 'REPAIR_BLOCK_THRESHOLD', 200)
    max_path_len = getattr(config, 'REPAIR_MAX_PATH_LEN', 100)
    
    n_nodes = len(graph_points_xy)
    n_edges_orig = len(pred_edges)
    
    stats = {
        'n_endpoints': 0,
        'n_candidates_tested': 0,
        'n_filtered_same_component': 0,
        'n_filtered_direction': 0,
        'n_filtered_astar': 0,
        'n_filtered_path_score': 0,
        'n_edges_added': 0,
    }
    
    if n_nodes == 0 or n_edges_orig == 0:
        return pred_edges, stats
    
    # Compute global edge limit
    edge_limit = min(max_added_edges, int(n_edges_orig * max_added_ratio))
    if edge_limit <= 0:
        edge_limit = 1  # At least allow 1
    
    # ---- Step 1: Build graph and find degree-1 endpoints ----
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    for e in pred_edges:
        G.add_edge(int(e[0]), int(e[1]))
    
    endpoints = [n for n in G.nodes() if G.degree(n) == 1]
    stats['n_endpoints'] = len(endpoints)
    
    if len(endpoints) == 0:
        return pred_edges, stats
    
    # ---- Step 2: Build spatial index ----
    kdtree = scipy.spatial.KDTree(graph_points_xy)
    
    # ---- Step 3: Build A* cost field (if enabled) ----
    pathfinder = None
    cost_field = None
    if use_astar:
        import tcod
        from graph_extraction import create_cost_field_astar, is_connected_astar
        # create_cost_field_astar expects points in (x,y) format as int tuples
        pts_for_cost = graph_points_xy.astype(np.int32)
        cost_field = create_cost_field_astar(pts_for_cost, fused_road_mask,
                                             block_threshold=block_threshold)
        pathfinder = tcod.path.AStar(cost_field)
    
    # ---- Step 4: Pre-compute connected components (for same-component check) ----
    if skip_same_comp:
        node_to_component = {}
        for comp_id, component in enumerate(nx.connected_components(G)):
            for node in component:
                node_to_component[node] = comp_id
    
    # ---- Step 5: Iterate over endpoints and find repair candidates ----
    new_edges = []
    repaired_endpoints = set()  # Track which endpoints have been repaired
    
    for ep_idx in endpoints:
        if len(new_edges) >= edge_limit:
            break
        if ep_idx in repaired_endpoints:
            continue
        
        ep_pos = graph_points_xy[ep_idx]
        
        # Get endpoint direction and edge length
        ep_dir, ep_edge_len = get_endpoint_direction(ep_idx, graph_points_xy, G)
        use_direction = (ep_dir is not None and ep_edge_len >= min_dir_edge_len)
        
        # Find nearby nodes within search radius
        nearby_indices = kdtree.query_ball_point(ep_pos, r=search_radius)
        
        # Score all valid candidates
        scored_candidates = []
        
        for cand_idx in nearby_indices:
            if cand_idx == ep_idx:
                continue
            
            stats['n_candidates_tested'] += 1
            
            cand_pos = graph_points_xy[cand_idx]
            dist = np.linalg.norm(cand_pos - ep_pos)
            
            # Filter: already connected
            if G.has_edge(ep_idx, cand_idx):
                continue
            
            # Filter: too close (likely the same physical point)
            if dist < min_node_dist:
                continue
            
            # Filter: same connected component (would create shortcut)
            if skip_same_comp:
                if node_to_component.get(ep_idx) == node_to_component.get(cand_idx):
                    stats['n_filtered_same_component'] += 1
                    continue
            
            # Filter: direction cone
            if use_direction:
                if not in_search_cone(ep_pos, ep_dir, cand_pos, cone_angle):
                    stats['n_filtered_direction'] += 1
                    continue
            
            # Filter: A* reachability
            if use_astar and pathfinder is not None:
                start_xy = (int(round(ep_pos[0])), int(round(ep_pos[1])))
                end_xy = (int(round(cand_pos[0])), int(round(cand_pos[1])))
                if not is_connected_astar(pathfinder, cost_field, 
                                          start_xy, end_xy, max_path_len):
                    stats['n_filtered_astar'] += 1
                    continue
            
            # Compute path score
            path_result = compute_path_score(ep_pos, cand_pos, fused_road_mask, config)
            if path_result is None:
                stats['n_filtered_path_score'] += 1
                continue
            
            # Filter: path score thresholds
            if (path_result['mean'] < min_path_mean or
                path_result['min'] < min_path_min or
                path_result['low_ratio'] > max_low_ratio):
                stats['n_filtered_path_score'] += 1
                continue
            
            # Candidate passed all filters — record with score
            scored_candidates.append((cand_idx, path_result['mean'], dist))
        
        # Select best candidate(s) for this endpoint
        if len(scored_candidates) == 0:
            continue
        
        # Sort by path mean score (descending), then by distance (ascending) for tiebreak
        scored_candidates.sort(key=lambda x: (-x[1], x[2]))
        
        # Take top-k (default k=1)
        for i in range(min(max_edges_per_ep, len(scored_candidates))):
            if len(new_edges) >= edge_limit:
                break
            
            best_idx = scored_candidates[i][0]
            new_edges.append((ep_idx, best_idx))
            G.add_edge(ep_idx, best_idx)
            repaired_endpoints.add(ep_idx)
            
            # If candidate was also an endpoint, mark it as repaired too
            if best_idx in endpoints:
                repaired_endpoints.add(best_idx)
    
    stats['n_edges_added'] = len(new_edges)
    
    # ---- Step 6: Merge edges ----
    if len(new_edges) > 0:
        repaired_edges = np.concatenate(
            [pred_edges, np.array(new_edges, dtype=pred_edges.dtype)], axis=0)
    else:
        repaired_edges = pred_edges
    
    return repaired_edges, stats


def print_repair_stats(stats):
    """Pretty-print repair statistics."""
    print("===== Local Repair A0 Stats =====")
    print(f"  Endpoints found:           {stats['n_endpoints']}")
    print(f"  Candidates tested:         {stats['n_candidates_tested']}")
    print(f"  Filtered (same component): {stats['n_filtered_same_component']}")
    print(f"  Filtered (direction cone): {stats['n_filtered_direction']}")
    print(f"  Filtered (A* blocked):     {stats['n_filtered_astar']}")
    print(f"  Filtered (path score):     {stats['n_filtered_path_score']}")
    print(f"  Edges added:               {stats['n_edges_added']}")
    print("=================================")
