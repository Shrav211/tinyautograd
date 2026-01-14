# tinygrad/viz.py
from collections import defaultdict

# Tune this once you know your actual op names.
TRIVIAL_OPS = {
    "broadcast", "unbroadcast", "reshape", "expand_dims", "transpose",
}

def trace(root):
    nodes, edges = set(), set()

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for p in getattr(v, "_prev", []):
                edges.add((p, v))
                build(p)

    build(root)
    return nodes, edges

def _fmt_data(x, max_chars=40):
    s = str(getattr(x, "data", ""))
    s = s.replace("\n", " ")
    if len(s) > max_chars:
        s = s[:max_chars] + "..."
    return s

def _node_id(n):
    return f"node{id(n)}"

def _should_show(n, root, mode="full", hide_const=False):
    """
    Decide if a node should be drawn.
    """
    op  = getattr(n, "_op", "") or ""
    req = bool(getattr(n, "requires_grad", False))
    name = getattr(n, "_name", None)

    is_root = (n is root)
    is_leaf = (len(getattr(n, "_prev", [])) == 0)

    # optional: hide constant leaves (inputs/targets) to reduce clutter
    if hide_const and is_leaf and (not req) and (not is_root) and (name is None):
        return False

    if mode == "full":
        return True

    if mode == "prune_trivial":
        # keep root always
        if is_root:
            return True
        return op not in TRIVIAL_OPS

    if mode == "ops_only":
        # show ops + root; also show named leaves (params) if you name them
        if is_root:
            return True
        if name is not None:
            return True
        return bool(op)

    if mode == "params_only":
        # show only named tensors (params) + ops that connect them + root
        # (works best once you name parameters)
        if is_root:
            return True
        if name is not None:
            return True
        return bool(op)

    raise ValueError(f"Unknown mode: {mode}")

def _fold_edges(nodes, edges, visible):
    """
    If a node is hidden, reconnect its visible ancestors to visible descendants.
    This preserves connectivity while removing clutter.
    """
    parents = defaultdict(list)
    children = defaultdict(list)
    for a, b in edges:
        children[a].append(b)
        parents[b].append(a)

    def vis_ancestors(v):
        out = set()
        stack = list(parents[v])
        seen = set()
        while stack:
            u = stack.pop()
            if u in seen:
                continue
            seen.add(u)
            if visible[u]:
                out.add(u)
            else:
                stack.extend(parents[u])
        return out

    def vis_descendants(v):
        out = set()
        stack = list(children[v])
        seen = set()
        while stack:
            u = stack.pop()
            if u in seen:
                continue
            seen.add(u)
            if visible[u]:
                out.add(u)
            else:
                stack.extend(children[u])
        return out

    new_edges = set()

    # keep edges between visible nodes
    for a, b in edges:
        if visible[a] and visible[b]:
            new_edges.add((a, b))

    # fold hidden nodes
    for v in nodes:
        if visible[v]:
            continue
        A = vis_ancestors(v)
        B = vis_descendants(v)
        for a in A:
            for b in B:
                if a != b:
                    new_edges.add((a, b))

    new_nodes = {n for n in nodes if visible[n]}
    return new_nodes, new_edges

def to_dot(root, max_data_chars=40, mode="full", hide_const=False):
    nodes, edges = trace(root)

    # decide visibility
    visible = {n: _should_show(n, root, mode=mode, hide_const=hide_const) for n in nodes}

    # fold graph (reconnect around hidden nodes)
    nodes2, edges2 = _fold_edges(nodes, edges, visible)

    lines = []
    lines.append("digraph G {")
    lines.append("rankdir=LR;")
    lines.append('node [fontsize=10];')

    # nodes
    for n in nodes2:
        op   = getattr(n, "_op", "") or ""
        name = getattr(n, "_name", None)
        req  = bool(getattr(n, "requires_grad", False))

        data = _fmt_data(n, max_chars=max_data_chars)
        shp  = getattr(getattr(n, "data", None), "shape", None)
        g    = getattr(n, "grad", None)

        # build a compact label
        parts = []
        if name:
            parts.append(str(name))
        if op:
            parts.append(f"op={op}")
        if shp is not None:
            parts.append(f"shape={tuple(shp)}")
        parts.append(f"req_grad={req}")
        parts.append(f"data={data}")

        # optional grad summary
        if g is not None:
            try:
                import numpy as np
                gg = np.array(g)
                parts.append(f"grad_mean={gg.mean():.3g}")
                parts.append(f"grad_std={gg.std():.3g}")
            except Exception:
                parts.append(f"grad={str(g)[:max_data_chars]}")

        label = "\\n".join(parts)

        nid = _node_id(n)

        # style: make root stand out
        if n is root:
            lines.append(f'{nid} [label="{label}", shape=box, style="filled", fillcolor="lightgray"];')
        else:
            lines.append(f'{nid} [label="{label}", shape=box];')

    # edges
    for a, b in edges2:
        lines.append(f"{_node_id(a)} -> {_node_id(b)};")

    lines.append("}")
    return "\n".join(lines)

def save_dot(root, path="graph.dot", **kwargs):
    dot = to_dot(root, **kwargs)
    with open(path, "w") as f:
        f.write(dot)
    return path
