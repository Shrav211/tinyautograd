# tinygrad/viz.py
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

def to_dot(root, max_data_chars=40):
    nodes, edges = trace(root)

    def fmt_data(x):
        s = str(getattr(x, "data", ""))
        s = s.replace("\n", " ")
        if len(s) > max_data_chars:
            s = s[:max_data_chars] + "..."
        return s

    lines = []
    lines.append("digraph G {")
    lines.append("rankdir=LR;")

    # nodes
    for n in nodes:
        op = getattr(n, "_op", "")
        name = getattr(n, "_name", None)
        data = fmt_data(n)
        req = getattr(n, "requires_grad", False)
        label = ""
        if name:
            label += f"{name}\\n"
        if op:
            label += f"op={op}\\n"
        label += f"requires_grad={req}\\n"
        label += f"data={data}"
        nid = f"node{id(n)}"
        lines.append(f'{nid} [label="{label}", shape=box];')

    # edges
    for a, b in edges:
        aid = f"node{id(a)}"
        bid = f"node{id(b)}"
        lines.append(f"{aid} -> {bid};")

    lines.append("}")
    return "\n".join(lines)

def save_dot(root, path="graph.dot"):
    dot = to_dot(root)
    with open(path, "w") as f:
        f.write(dot)
    return path
