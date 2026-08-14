#!/usr/bin/env python3
"""Render the pipeline dependency figure as PDF / PNG / SVG on a transparent background.

Chain: depgraph.json -> filter -> layered layout -> TikZ -> pdflatex
       -> pdftocairo -> PNG (300 dpi, transparent) + SVG.

The layout is computed here (longest-path layering, then barycentre sweeps to
reduce crossings, then dummy nodes so long edges get their own corridor). TikZ
only draws at the coordinates it is given, so Graphviz is not required.

Usage:
    python tools/depgraph.py            # first, to produce depgraph.json
    python tools/render_figures.py
    python tools/render_figures.py --outdir "D:\\elsewhere" --formats pdf png
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
GRAPH_JSON = REPO / "tools" / "out" / "depgraph.json"
DEFAULT_OUT = Path(r"C:\Users\maxen\Pictures\Screenshots\Graph")

MIKTEX = Path.home() / "AppData/Local/Programs/MiKTeX/miktex/bin/x64"


# --------------------------------------------------------------------------- #
# Filtering rules
# --------------------------------------------------------------------------- #

# Hidden files: noise when reading the pipeline.
HIDE = {
    "Software/OT_NN/Pytorch_NN/verify_rotation_consistency.py",
    "Software/OT_NN/Pytorch_NN/tune_Unet.py",
    # Shadowed by OT_Software/SolveFE.m, so never executed. Keeping it would
    # pull in VectorF_Surface.m, which only the shadowed copy ever calls.
    "Software/OT_Functions/SolveFE.m",
}

# The benchmark scripts share one dependency graph: a single node stands for all.
COLLAPSE = {
    "Software/OT_NN/Pytorch_NN/TopOpt_benchmark_architecture.py":    "TopOpt_benchmark_*.py",
    "Software/OT_NN/Pytorch_NN/TopOpt_benchmark_hybrid_strategy.py": "TopOpt_benchmark_*.py",
    "Software/OT_NN/Pytorch_NN/TopOpt_benchmark_model.py":           "TopOpt_benchmark_*.py",
    "Software/OT_NN/Pytorch_NN/TopOpt_benchmark_data_augment.py":    "TopOpt_benchmark_*.py",
    "Software/OT_NN/Pytorch_NN/TopOpt_benchmark_seed.py":            "TopOpt_benchmark_*.py",
}

# Entry points the figure is built from.
ROOTS = [
    "TopOpt_benchmark_*.py",
    "Software/OT_NN/Pytorch_NN/main_training_UNet.py",
    "Software/OT_Software/generate_dataset.m",
    "Software/OT_Software/compute_energy_first_image.m",
]


def in_scope(node: str) -> bool:
    """Stay inside Software/, and drop figure scripts and legacy standalone code."""
    if node in HIDE:
        return False
    if not node.startswith("Software/"):
        return False
    if "/illustrations/" in node:
        return False
    if node in ("Software/OT_NN/Main.m", "Software/OT_NN/Main_Stress.m"):
        return False
    return True


# --------------------------------------------------------------------------- #
# Palette: inks that stay legible on light and dark backgrounds alike, since
# the PNG is transparent and may land on any surface.
# --------------------------------------------------------------------------- #

TIKZ_PREAMBLE = r"""
\documentclass[border=6pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{tikz}
\usetikzlibrary{arrows.meta, positioning, calc}

\definecolor{cPython}{HTML}{1F6076}
\definecolor{cMatlab}{HTML}{B5561F}
\definecolor{cViolet}{HTML}{7452A8}
\definecolor{cEdge}{HTML}{5A626C}

\tikzset{
  box/.style  = {rectangle, rounded corners=1.6pt, draw=none,
                 text=white, font=\ttfamily\fontsize{7}{8.4}\selectfont,
                 inner xsep=4pt, inner ysep=3.2pt, minimum height=5.4mm,
                 align=center},
  py/.style   = {box, fill=cPython},
  mat/.style  = {box, fill=cMatlab},
  ed/.style   = {draw=cEdge, line width=0.85pt, opacity=0.85,
                 -{Straight Barb[length=2.0mm, width=2.0mm]}},
  % Edge colour follows the (source, target) pair.
  edMM/.style = {ed, draw=cMatlab},
  edPP/.style = {ed, draw=cPython},
  edPM/.style = {ed, draw=cViolet, dash pattern=on 2.2pt off 1.6pt},
}
"""


# --------------------------------------------------------------------------- #
# Loading and filtering
# --------------------------------------------------------------------------- #

def load_graph() -> dict[str, set[str]]:
    if not GRAPH_JSON.exists():
        sys.exit(f"{GRAPH_JSON} missing - run this first: python tools/depgraph.py")
    raw = json.loads(GRAPH_JSON.read_text(encoding="utf-8"))["edges"]
    return {k: set(v) for k, v in raw.items()}


def label_of(node: str) -> str:
    return COLLAPSE.get(node, Path(node).name)


def filtered(graph: dict[str, set[str]]):
    """Apply HIDE / COLLAPSE / in_scope.

    Nodes are keyed by PATH, never by file name: two same-named .m files are
    distinct files, and merging them would make the survivor inherit the
    shadowed copy's calls. Only COLLAPSE merges, and it does so on purpose.
    Returns (graph, kind, labels).
    """
    def key(n: str) -> str:
        return COLLAPSE.get(n, n)

    keep = {n for n in graph if in_scope(n)}
    out: dict[str, set[str]] = defaultdict(set)
    kind: dict[str, str] = {}
    labels: dict[str, str] = {}

    for n in keep:
        k = key(n)
        kind[k] = "py" if n.endswith(".py") else "mat"
        labels[k] = label_of(n)
        out.setdefault(k, set())
        for t in graph[n]:
            if t in keep:
                tk = key(t)
                if tk != k:
                    out[k].add(tk)
    return dict(out), kind, labels


def reachable_from(graph: dict[str, set[str]], roots: list[str]) -> set[str]:
    seen, stack = set(), list(roots)
    while stack:
        n = stack.pop()
        if n in seen or n not in graph:
            continue
        seen.add(n)
        stack.extend(graph[n])
    return seen


def subgraph(graph: dict[str, set[str]], nodes: set[str]) -> dict[str, set[str]]:
    return {n: {t for t in graph.get(n, ()) if t in nodes} for n in nodes}


# --------------------------------------------------------------------------- #
# Layered layout
# --------------------------------------------------------------------------- #

def layer_nodes(g: dict[str, set[str]]) -> dict[str, int]:
    """Layer = longest path from a root, so every edge points forward."""
    indeg = {n: 0 for n in g}
    for s, ds in g.items():
        for d in ds:
            indeg[d] += 1

    layer = {n: 0 for n in g}
    # Topological order (Kahn). On a cycle, the leftovers keep their layer.
    ready = [n for n, d in indeg.items() if d == 0]
    order, deg = [], dict(indeg)
    while ready:
        n = ready.pop()
        order.append(n)
        for d in g[n]:
            deg[d] -= 1
            if deg[d] == 0:
                ready.append(d)
    for n in order:
        for d in g[n]:
            layer[d] = max(layer[d], layer[n] + 1)
    return layer


def insert_dummies(g: dict[str, set[str]], layer: dict[str, int]):
    """Split every long edge into single-layer segments.

    Dummy nodes take a slot in their layer, which pushes the real boxes apart
    and reserves a corridor for the edge. Without them, an edge that skips
    layers runs straight through the boxes in between.
    Returns (expanded adjacency, chains[(s, d)] -> [dummies]).
    """
    adj: dict[str, set[str]] = {n: set() for n in g}
    chains: dict[tuple[str, str], list[str]] = {}
    k = 0
    for s in sorted(g):
        for d in sorted(g[s]):
            if layer[d] - layer[s] <= 1:
                adj[s].add(d)
                chains[(s, d)] = []
                continue
            prev, mids = s, []
            for lv in range(layer[s] + 1, layer[d]):
                k += 1
                dm = f"__dummy{k}"
                layer[dm] = lv
                adj.setdefault(dm, set())
                adj[prev].add(dm)
                mids.append(dm)
                prev = dm
            adj[prev].add(d)
            chains[(s, d)] = mids
    return adj, chains


def order_layers(g: dict[str, set[str]], layer: dict[str, int],
                 sweeps: int = 24) -> dict[int, list[str]]:
    """Barycentre heuristic: alternate forward and backward passes to cut crossings."""
    cols: dict[int, list[str]] = defaultdict(list)
    for n, lv in sorted(layer.items(), key=lambda kv: (kv[1], kv[0])):
        cols[lv].append(n)

    pred: dict[str, list[str]] = defaultdict(list)
    for s, ds in g.items():
        for d in ds:
            pred[d].append(s)

    pos = {n: i for lv in cols for i, n in enumerate(cols[lv])}

    for it in range(sweeps):
        forward = it % 2 == 0
        keys = sorted(cols) if forward else sorted(cols, reverse=True)
        for lv in keys:
            nbrs = pred if forward else {n: sorted(g[n]) for n in g}

            def bary(n: str) -> float:
                ns = [x for x in nbrs.get(n, []) if x in pos]
                return sum(pos[x] for x in ns) / len(ns) if ns else pos[n]

            cols[lv].sort(key=lambda n: (bary(n), n))
            for i, n in enumerate(cols[lv]):
                pos[n] = i
    return dict(cols)


def coordinates(cols: dict[int, list[str]], labels: dict[str, str]):
    """Node positions in mm. Column width follows its longest label."""
    CHAR_MM, PAD_MM, GAP_MM, ROW_MM, DUMMY_MM = 1.34, 8.0, 22.0, 7.6, 2.6

    # Dummy nodes carry no label, so they do not drive column width.
    width = {lv: max([len(labels[n]) for n in ns if n in labels] or [0]) * CHAR_MM + PAD_MM
             for lv, ns in cols.items()}

    x, cursor = {}, 0.0
    for lv in sorted(cols):
        x[lv] = cursor + width[lv] / 2
        cursor += width[lv] + GAP_MM

    # Variable row pitch: an edge corridor does not need a full box height,
    # otherwise a column full of dummies stretches the whole figure.
    xy = {}
    for lv, ns in cols.items():
        pitch = [ROW_MM if n in labels else DUMMY_MM for n in ns]
        y = sum(pitch) / 2
        for n, p in zip(ns, pitch):
            xy[n] = (x[lv], y - p / 2)
            y -= p
    return xy


# --------------------------------------------------------------------------- #
# TikZ emission
# --------------------------------------------------------------------------- #

# py -> py blue, mat -> mat orange, py -> mat violet (the matlab.engine bridge).
EDGE_STYLE = {("py", "py"): "edPP", ("mat", "mat"): "edMM",
              ("py", "mat"): "edPM", ("mat", "py"): "edPM"}


def node_id(n: str) -> str:
    return "n" + str(abs(hash(n)) % (10 ** 12))


def esc(s: str) -> str:
    return s.replace("_", r"\_").replace("*", r"{*}").replace("&", r"\&")


def tikz_graph(g, kind, labels, xy, chains) -> str:
    L = [TIKZ_PREAMBLE, r"\begin{document}", r"\begin{tikzpicture}"]
    ids = {n: node_id(n) for n in g}

    for n in sorted(g):
        L.append(rf"  \node[{kind.get(n, 'mat')}] ({ids[n]}) "
                 rf"at ({xy[n][0]:.2f}mm,{xy[n][1]:.2f}mm) {{{esc(labels[n])}}};")

    for (s, d), mids in sorted(chains.items()):
        style = EDGE_STYLE.get((kind.get(s), kind.get(d)), "ed")
        hops = "".join(f" to[out=0,in=180] ({xy[m][0]:.2f}mm,{xy[m][1]:.2f}mm)"
                       for m in mids)
        L.append(rf"  \draw[{style}] ({ids[s]}.east){hops} "
                 rf"to[out=0,in=180] ({ids[d]}.west);")

    L += [r"\end{tikzpicture}", r"\end{document}"]
    return "\n".join(L)


# --------------------------------------------------------------------------- #
# Compilation
# --------------------------------------------------------------------------- #

def tool(name: str) -> str:
    p = MIKTEX / f"{name}.exe"
    if p.exists():
        return str(p)
    found = shutil.which(name)
    if found:
        return found
    sys.exit(f"{name} not found")


def build(tex: str, stem: str, outdir: Path, formats: list[str]) -> list[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    made: list[Path] = []

    if "tex" in formats:
        p = outdir / f"{stem}.tex"
        p.write_text(tex, encoding="utf-8")
        made.append(p)

    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        (tdp / f"{stem}.tex").write_text(tex, encoding="utf-8")
        r = subprocess.run(
            [tool("pdflatex"), "-interaction=nonstopmode", "-halt-on-error",
             f"-output-directory={tdp}", f"{stem}.tex"],
            cwd=tdp, capture_output=True, text=True)
        pdf = tdp / f"{stem}.pdf"
        if not pdf.exists():
            log = tdp / f"{stem}.log"
            tail = (log.read_text(encoding="utf-8", errors="replace")[-2500:]
                    if log.exists() else r.stdout[-2500:])
            sys.exit(f"\n[pdflatex failed on {stem}]\n{tail}")

        if "pdf" in formats:
            dst = outdir / f"{stem}.pdf"
            shutil.copy2(pdf, dst)
            made.append(dst)

        if "png" in formats:
            subprocess.run([tool("pdftocairo"), "-png", "-r", "300", "-transp",
                            "-singlefile", str(pdf), str(outdir / stem)],
                           check=True, capture_output=True)
            made.append(outdir / f"{stem}.png")

        if "svg" in formats:
            dst = outdir / f"{stem}.svg"
            subprocess.run([tool("pdftocairo"), "-svg", str(pdf), str(dst)],
                           check=True, capture_output=True)
            made.append(dst)

    return made


# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--formats", nargs="+", default=["pdf", "png", "svg", "tex"],
                    choices=["pdf", "png", "svg", "tex"])
    args = ap.parse_args()

    g, kind, labels = filtered(load_graph())
    pipeline = subgraph(g, reachable_from(g, [r for r in ROOTS if r in g]))

    layer = layer_nodes(pipeline)
    adj, chains = insert_dummies(pipeline, layer)   # layer gains the dummies
    cols = order_layers(adj, layer)                 # dummies ordered alongside
    xy = coordinates(cols, labels)

    made = build(tikz_graph(pipeline, kind, labels, xy, chains),
                 "topopt_pipeline", args.outdir, args.formats)

    print(f"topopt_pipeline: {len(pipeline)} nodes, "
          f"{sum(len(v) for v in pipeline.values())} edges")
    print(f"{len(made)} files written to {args.outdir}")
    for p in made:
        print(f"  {p.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
