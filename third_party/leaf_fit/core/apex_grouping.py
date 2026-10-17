from collections import defaultdict, deque
from collections import defaultdict as _dd, deque as _dq

def group_apexes_by_inequality(
    found_tips,
    pathes,
    root_cahced_distance=None,   
    overlap_cut=0.8,
    triangle_cut=0.62,            
    slack_eps=1e-9,             
    max_iters=50,
    dense_solver=None
):
    """
    Group apex tips using triangle inequality and overlap criteria.
    
    Algorithm:
      - Initial: Build undirected graph using Overlap(shared/min) -> initial connected components
      - Refine: Only keep edges within groups where slack(i,j) = 2*(depth(LCA_ij)-depth(G)) > triangle_cut
      - Points without positive slack remain in residual clusters for further subdivision
      - Tips cannot be used as LCA; singleton cluster LCA = path root
      - Iterate until stable (or max_iters reached)
    
    Returns cluster_info compatible with original structure.
    """
    found_tips = [int(t) for t in found_tips]
    pathes     = [[int(n) for n in p] for p in pathes]
    tip_to_path = {found_tips[i]: pathes[i] for i in range(len(found_tips))}
    tip_to_pos  = {t: {n: idx for idx, n in enumerate(p)} for t, p in tip_to_path.items()}
    tips = found_tips[:]

    root_depth = root_cahced_distance

    def depth_of(node: int) -> float:
        return float(root_depth[node])

    def pkey(a, b):
        return (a, b) if a < b else (b, a)

    def find_lca_and_pos(path_i, path_j, pos_map_i, pos_map_j, forbid_tip=True):
        """
        Find LCA of two tip->root paths and their positions (lpos_i, lpos_j).
        
        Algorithm:
          1) Suffix alignment: scan from root end to find shared suffix start
          2) Forward scan for earliest intersection from tip end of path_i
          3) When forbid_tip=True, both positions must be > 0 or result is invalid
        """
        i, j = len(path_i) - 1, len(path_j) - 1
        lca2 = lpos_i2 = lpos_j2 = None
        while i >= 0 and j >= 0 and path_i[i] == path_j[j]:
            lca2, lpos_i2, lpos_j2 = path_i[i], i, j
            i -= 1
            j -= 1

        if lca2 is not None:
            if forbid_tip and (lpos_i2 == 0 or lpos_j2 == 0):
                lca2 = lpos_i2 = lpos_j2 = None

        start_i = 1 if forbid_tip else 0
        lca1 = lpos_i1 = lpos_j1 = None
        for idx_i in range(start_i, len(path_i)):
            node = path_i[idx_i]
            jpos = pos_map_j.get(node)
            if jpos is not None and (not forbid_tip or jpos > 0):
                lca1, lpos_i1, lpos_j1 = node, idx_i, jpos
                break

        candidates = []
        if lca1 is not None:
            candidates.append((lpos_i1, lca1, lpos_i1, lpos_j1))
        if lca2 is not None:
            candidates.append((lpos_i2, lca2, lpos_i2, lpos_j2))

        if not candidates:
            return None, None, None

        candidates.sort(key=lambda x: x[0])
        _, lca, lpos_i, lpos_j = candidates[0]

        for idx_i in range(start_i, lpos_i):
            node = path_i[idx_i]
            jpos = pos_map_j.get(node)
            if jpos is not None and (not forbid_tip or jpos > 0):
                lca, lpos_i, lpos_j = node, idx_i, jpos
                break

        return lca, lpos_i, lpos_j

    def group_lca_by_sets(sub_tips):
        rep = max(sub_tips, key=lambda t: len(tip_to_path[t]))
        rep_path = tip_to_path[rep]
        for idx in range(1, len(rep_path)):
            node = rep_path[idx]
            ok = True
            for t in sub_tips:
                if t == rep:
                    continue
                pos = tip_to_pos[t].get(node, -1)
                if pos <= 0:
                    ok = False
                    break
            if ok:
                return node
        return rep_path[-1]

    def group_lca_from_cache(sub_tips, pair_lca):
        if len(sub_tips) == 1:
            lone = sub_tips[0]
            p = tip_to_path[lone]
            return int(p[-1]) if p else int(lone)
        rep = max(sub_tips, key=lambda t: len(tip_to_path[t]))
        rep_path = tip_to_path[rep]
        lpos_rep_list = []
        for t in sub_tips:
            if t == rep:
                continue
            a, b = pkey(rep, t)
            lca_node, lpos_a, lpos_b = pair_lca.get((a, b), (None, None, None))
            if lca_node is None:
                return group_lca_by_sets(sub_tips)
            lpos_rep = lpos_a if rep == a else lpos_b
            lpos_oth = lpos_b if rep == a else lpos_a
            if lpos_rep is None or lpos_oth is None or lpos_rep <= 0 or lpos_oth <= 0:
                return group_lca_by_sets(sub_tips)
            lpos_rep_list.append(lpos_rep)
        if not lpos_rep_list:
            return group_lca_by_sets(sub_tips)
        group_lpos = max(lpos_rep_list)
        return int(rep_path[group_lpos])

    def overlap_shared_min(ti, tj, pair_lca):
        """
        Calculate overlapping coefficient: shared_suffix_len / min(len_i, len_j)
        """
        a, b = pkey(ti, tj)
        lca, lpos_a, lpos_b = pair_lca.get((a, b), (None, None, None))
        if lca is None or lpos_a is None or lpos_b is None:
            return 0.0
        la = len(tip_to_path[a])
        lb = len(tip_to_path[b])
        shared = max(0, min(la - lpos_a, lb - lpos_b))
        denom  = max(min(la, lb), 1)
        return shared / denom

    pair_lca   = {}
    same_graph = defaultdict(set)

    for i in range(len(tips)):
        ti = tips[i]; pi = tip_to_path[ti]; posi = tip_to_pos[ti]; li = len(pi)
        if li == 0:
            continue
        for j in range(i+1, len(tips)):
            tj = tips[j]; pj = tip_to_path[tj]; posj = tip_to_pos[tj]; lj = len(pj)
            if lj == 0:
                continue

            lca, lpos_i, lpos_j = find_lca_and_pos(pi, pj, posi, posj, forbid_tip=False)

            a, b = pkey(ti, tj)
            if a == ti:
                pair_lca[(a,b)] = (lca, lpos_i, lpos_j)
            else:
                pair_lca[(a,b)] = (lca, lpos_j, lpos_i)

            ov = overlap_shared_min(ti, tj, pair_lca) if (lca is not None) else 0.0
            kept = (ov >= overlap_cut)
            if kept:
                same_graph[ti].add(tj)
                same_graph[tj].add(ti)

    initial_clusters = []
    visited = set()

    for t in tips:
        if t in visited:
            continue
        dq = deque([t]); comp = []
        while dq:
            u = dq.popleft()
            if u in visited:
                continue
            visited.add(u)
            comp.append(u)
            for v in same_graph[u]:
                if v not in visited:
                    dq.append(v)
        comp_sorted = sorted(comp)

        if len(comp_sorted) == 1:
            lone = comp_sorted[0]
            p = tip_to_path[lone]
            lca_node = int(p[-1]) if p else int(lone)
        else:
            lca_node = group_lca_from_cache(comp_sorted, pair_lca)

        initial_clusters.append({"tips": comp_sorted, "lca": lca_node})
        
    def _choose_group_lca_safe(tips_in_c):
        G = group_lca_from_cache(tips_in_c, pair_lca) if len(tips_in_c) > 1 else (
            int(tip_to_path[tips_in_c[0]][-1]) if tip_to_path[tips_in_c[0]] else int(tips_in_c[0])
        )
        missing = [t for t in tips_in_c if G not in tip_to_pos[t]]
        if missing:
            G = group_lca_by_sets(tips_in_c)
        return G

    def _positive_slack_edges(tips_in_c, G):
        """Return list of edges with positive slack: [(i,j,slack), ...]"""
        edges = []
        depth_G = depth_of(G)
        ts = sorted(tips_in_c)
        for a_idx in range(len(ts)):
            ti = ts[a_idx]
            for b_idx in range(a_idx+1, len(ts)):
                tj = ts[b_idx]
                a, b = pkey(ti, tj)
                lca_ij, _, _ = pair_lca.get((a,b), (None, None, None))
                if lca_ij is None:
                    continue
                s = 2.0 * (depth_of(lca_ij) - depth_G)

                if s > (triangle_cut + slack_eps):  
                    edges.append((ti, tj, s))

        return edges

    def _subdivide_once(tips_in_c, G):
        """Subdivide cluster based on transitive connectivity of positive slack edges."""
        pos_edges = _positive_slack_edges(tips_in_c, G)
        if not pos_edges:
            return [{"tips": tips_in_c, "lca": G}], False

        g = defaultdict(set)
        for (i, j, s) in pos_edges:
            g[i].add(j); g[j].add(i)

        comps = []
        seen = set()
        nodes_with_pos_edge = sorted(set([x for e in pos_edges for x in (e[0], e[1])]))
        for u in nodes_with_pos_edge:
            if u in seen:
                continue
            dq = deque([u]); comp = []
            while dq:
                x = dq.popleft()
                if x in seen:
                    continue
                seen.add(x)
                comp.append(x)
                for v in g[x]:
                    if v not in seen:
                        dq.append(v)
            if len(comp) >= 2:
                comps.append(sorted(comp))

        new_clusters = []
        for comp in comps:
            if len(comp) == 1:
                lone = comp[0]
                p = tip_to_path[lone]
                sub_lca = int(p[-1]) if p else int(lone)
            else:
                sub_lca = group_lca_from_cache(comp, pair_lca)
            new_clusters.append({"tips": comp, "lca": sub_lca})

        assigned = set(x for comp in comps for x in comp)
        leftovers = [t for t in tips_in_c if t not in assigned]
        if leftovers:
            if len(leftovers) == 1:
                lone = leftovers[0]
                p = tip_to_path[lone]
                lca_rest = int(p[-1]) if p else int(lone)
            else:
                lca_rest = group_lca_from_cache(leftovers, pair_lca)
            new_clusters.append({"tips": sorted(leftovers), "lca": lca_rest})

        changed = len(new_clusters) != 1 or new_clusters[0]["tips"] != tips_in_c
        return new_clusters, changed

    def _signature(clusters_):
        return tuple(sorted(tuple(sorted(c["tips"])) for c in clusters_))

    current = []
    for idx, c in enumerate(initial_clusters):
        tips_in_c = c["tips"]
        G = _choose_group_lca_safe(tips_in_c)
        current.append({"tips": tips_in_c, "lca": G})

    it = 0
    while True:
        it += 1
        if it > max_iters:
            break

        next_clusters = []
        changed_any = False
        for cid, c in enumerate(current):
            tips_in_c = c["tips"]
            G = _choose_group_lca_safe(tips_in_c)
            subcs, changed = _subdivide_once(tips_in_c, G)
            changed_any = changed_any or changed
            next_clusters.extend(subcs)

        if (not changed_any) or (_signature(current) == _signature(next_clusters)):
            refined_clusters = next_clusters
            break
        current = next_clusters

    _geo_solver_available = (dense_solver is not None)

    if _geo_solver_available:
        GEO_MAX_ITERS = 10

        _geo_cache = {}
        def _ensure_geo(t):
            if t not in _geo_cache:
                _geo_cache[t] = dense_solver.compute_distance(int(t))

        def _choose_G_for_geo(tips_in_c):
            if len(tips_in_c) == 1:
                lone = tips_in_c[0]
                p = tip_to_path[lone]
                return int(p[-1]) if p else int(lone)
            return _choose_group_lca_safe(tips_in_c)

        def _geo_keep_edges(tips_in_c, G):
            if len(tips_in_c) < 2:
                return []
            for t in tips_in_c:
                _ensure_geo(t)
            d_to_G = {t: float(_geo_cache[t][G]) for t in tips_in_c}

            ts = sorted(tips_in_c)
            keep_edges = []
            for i in range(len(ts)):
                ti = ts[i]
                for j in range(i+1, len(ts)):
                    tj = ts[j]
                    dij_fwd = float(_geo_cache[ti][tj])
                    dij_bwd = float(_geo_cache[tj][ti])
                    dij = dij_fwd if dij_fwd > dij_bwd else dij_bwd
                    s = d_to_G[ti] + d_to_G[tj] - dij
                    if s >= triangle_cut:
                        keep_edges.append((ti, tj, s))
            return keep_edges

        
        def _comps_from_edges(tips_in_c, edges):
            g = _dd(set)
            for (u, v, s) in edges:
                g[u].add(v); g[v].add(u)
            comps, seen = [], set()
            nodes = sorted(set([x for e in edges for x in (e[0], e[1])]))
            for u in nodes:
                if u in seen: continue
                q=_dq([u]); comp=[]
                while q:
                    x=q.popleft()
                    if x in seen: continue
                    seen.add(x); comp.append(x)
                    for w in g[x]:
                        if w not in seen: q.append(w)
                if len(comp) >= 2:
                    comps.append(sorted(comp))
            assigned = set(x for c in comps for x in c)
            leftovers = [t for t in tips_in_c if t not in assigned]
            return comps, leftovers

        def _sig(clusters_):
            return tuple(sorted(tuple(sorted(c["tips"])) for c in clusters_))

        geo_cur = [{"tips": c["tips"], "lca": _choose_G_for_geo(c["tips"])} for c in refined_clusters]
        for it_geo in range(1, GEO_MAX_ITERS+1):
            geo_next = []
            for cid, c in enumerate(geo_cur):
                tips_in_c, G = c["tips"], c["lca"]
                if len(tips_in_c) == 1:
                    geo_next.append({"tips": tips_in_c, "lca": G})
                    continue

                keep_edges = _geo_keep_edges(tips_in_c, G)
                if not keep_edges:
                    for t in sorted(tips_in_c):
                        sub_G = _choose_G_for_geo([t])
                        geo_next.append({"tips": [t], "lca": sub_G})
                    continue

                comps, leftovers = _comps_from_edges(tips_in_c, keep_edges)

                for comp in comps:
                    sub_G = _choose_G_for_geo(comp)
                    geo_next.append({"tips": comp, "lca": sub_G})

                if leftovers:
                    for t in sorted(leftovers):
                        sub_G = _choose_G_for_geo([t])
                        geo_next.append({"tips": [t], "lca": sub_G})

            if _sig(geo_cur) == _sig(geo_next):
                break
            geo_cur = geo_next

        refined_clusters = geo_cur

    cluster_info = []
    for idx, rc in enumerate(refined_clusters):
        tips_c = rc["tips"]
        lca_c  = rc["lca"]
        dist_to_lca = {t: (depth_of(t) - depth_of(lca_c)) for t in tips_c}
        final_tip = max(tips_c, key=lambda t: dist_to_lca[t])

        cluster_info.append({
            "tips": tips_c,
            "tips_dist_to_lca": dist_to_lca,
            "final_tip": int(final_tip),
            "lca": int(lca_c),
            "potential_base_idx": tip_to_path[int(final_tip)],
            "pathes": [tip_to_path[t] for t in tips_c],
            "path": tip_to_path[int(final_tip)],
            "type": "multi_tips" if len(tips_c) > 1 else "single_tip"
        })

    return cluster_info

