#!/usr/bin/env python3
"""Build the dependency graph of the repository's .py and .m files.

Outputs (in tools/out/):
    depgraph.json     raw graph: edges, reachability, shadowed files
    depgraph.txt      ASCII call trees from the requested roots

depgraph.json is what tools/render_figures.py consumes.

Usage:
    python tools/depgraph.py
    python tools/depgraph.py --roots main_training_UNet.py generate_dataset.m
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict, deque
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "out"

# Directories scanned. The rest of the repo (data, figures, .git) is ignored.
SCAN_DIRS = ["Software", "HeavyFiles", "Bibliography", "PlotNeuralNet"]

# Third-party Python modules: never nodes of the graph.
THIRD_PARTY = {
    "torch", "torchvision", "numpy", "np", "matplotlib", "scipy", "pandas",
    "PIL", "tqdm", "sklearn", "optuna", "seaborn", "h5py", "joblib", "imageio",
    "imageio_ffmpeg", "matlab", "tkinter", "cv2", "plotly", "yaml",
}

# Every script does `addpath OT_Functions` then `addpath OT_Software`.
# addpath prepends, so the one added LAST wins when two .m files share a name.
MATLAB_PATH_PRIORITY = ["OT_Software", "OT_Functions"]

DEFAULT_ROOTS = [
    "Software/OT_NN/Pytorch_NN/TopOpt_benchmark_architecture.py",
    "Software/OT_NN/Pytorch_NN/main_training_UNet.py",
    "Software/OT_Software/compute_energy_first_image.m",
    "Software/OT_Software/generate_dataset.m",
]


# --------------------------------------------------------------------------- #
# File collection
# --------------------------------------------------------------------------- #

def collect_files() -> list[Path]:
    files: list[Path] = []
    for d in SCAN_DIRS:
        base = REPO / d
        if not base.exists():
            continue
        files += sorted(base.rglob("*.py"))
        files += sorted(base.rglob("*.m"))
    return [f for f in files if ".git" not in f.parts]


def rel(p: Path) -> str:
    return p.relative_to(REPO).as_posix()


# --------------------------------------------------------------------------- #
# Comment stripping: a call inside a comment is not a call
# --------------------------------------------------------------------------- #

def strip_matlab_comments(src: str) -> str:
    """Drop %{ %} blocks, whole-line % comments and trailing % comments.

    The apostrophe is ambiguous in MATLAB (string vs transpose). An apostrophe
    preceded by an identifier, a closing paren or a closing bracket is treated
    as a transpose, not as the start of a string.
    """
    out_lines = []
    in_block = False
    for line in src.splitlines():
        stripped = line.strip()
        if in_block:
            if stripped.startswith("%}"):
                in_block = False
            continue
        if stripped.startswith("%{"):
            in_block = True
            continue
        if stripped.startswith("%"):
            continue

        cleaned, in_str, prev = [], False, ""
        for ch in line:
            if ch == "'" and not (prev.isalnum() or prev in "_)]}.'"):
                in_str = not in_str
            elif ch == "'" and in_str:
                in_str = False
            elif ch == "%" and not in_str:
                break
            cleaned.append(ch)
            prev = ch
        out_lines.append("".join(cleaned))
    return "\n".join(out_lines)


def strip_python_comments(src: str) -> str:
    """Drop triple-quoted docstrings and # comments."""
    src = re.sub(r'"""(?:.|\n)*?"""', "", src)
    src = re.sub(r"'''(?:.|\n)*?'''", "", src)
    return re.sub(r"(?m)#.*$", "", src)


# --------------------------------------------------------------------------- #
# Python edges: imports of local modules
# --------------------------------------------------------------------------- #

PY_IMPORT = re.compile(
    r"(?m)^[ \t]*(?:from[ \t]+(\.*[A-Za-z_][\w.]*)[ \t]+import|import[ \t]+([A-Za-z_][\w.]*))"
)


def _resolve_module(dotted: str, origin: Path, py_by_stem) -> Path | None:
    """Resolve an imported module to a .py file, starting from the caller's folder.

    Handles plain modules (dataset), packages (pycore -> pycore/__init__.py)
    and dotted paths (pycore.tikzeng -> pycore/tikzeng.py).
    """
    # Relative import: each leading dot climbs one folder.
    up = len(dotted) - len(dotted.lstrip("."))
    parts = [p for p in dotted[up:].split(".") if p]
    if up:
        base = origin.parent
        for _ in range(up - 1):
            base = base.parent
        cand = base.joinpath(*parts)
        if cand.with_suffix(".py").exists():
            return cand.with_suffix(".py")
        if (cand / "__init__.py").exists():
            return cand / "__init__.py"
        return None

    # Absolute import: try the file's folder, then each ancestor, because the
    # scripts call sys.path.insert(0, ROOT) before importing.
    bases = [origin.parent]
    p = origin.parent
    while p != REPO and REPO in p.parents:
        p = p.parent
        bases.append(p)
    for base in bases:
        cand = base.joinpath(*parts)
        if cand.with_suffix(".py").exists():
            return cand.with_suffix(".py")
        if (cand / "__init__.py").exists():
            return cand / "__init__.py"

    head = parts[0]
    if head in py_by_stem and len(py_by_stem[head]) == 1:
        return py_by_stem[head][0]
    return None


def python_edges(path: Path, py_by_stem: dict[str, list[Path]]) -> set[str]:
    src = strip_python_comments(path.read_text(encoding="utf-8", errors="replace"))
    edges: set[str] = set()
    for m in PY_IMPORT.finditer(src):
        dotted = m.group(1) or m.group(2)
        if dotted.split(".")[0] in THIRD_PARTY:
            continue
        target = _resolve_module(dotted, path, py_by_stem)
        if target is not None:
            edges.add(rel(target))
    edges.discard(rel(path))
    return edges


# --------------------------------------------------------------------------- #
# Python -> MATLAB edges: what the code hands to matlab.engine
# --------------------------------------------------------------------------- #

def python_matlab_edges(path: Path, m_names: dict[str, Path],
                        py_stems: set[str]) -> set[str]:
    """Look for .m function names inside the strings passed to eng.eval().

    A name that is also a Python function defined in the file is skipped:
    GenTopology is reimplemented in Python inside topology_utils.py and never
    reaches GenTopology.m.
    """
    src = path.read_text(encoding="utf-8", errors="replace")
    local_defs = set(re.findall(r"(?m)^[ \t]*def[ \t]+(\w+)", src))

    # Triple-quoted blocks carry most of the MATLAB code, as in
    # eng.eval(rf"""..."""). Harvest them BEFORE stripping, or the docstring
    # removal would wipe them out.
    blocks = re.findall(r'"""((?:.|\n)*?)"""', src) + re.findall(r"'''((?:.|\n)*?)'''", src)

    rest = strip_python_comments(src)
    literals = re.findall(r'"([^"\n]*)"', rest) + re.findall(r"'([^'\n]*)'", rest)
    blob = "\n".join(blocks + literals)

    edges: set[str] = set()
    for name, target in m_names.items():
        if name in local_defs or name in py_stems:
            continue
        if re.search(rf"(?<![\w.]){re.escape(name)}\s*\(", blob):
            edges.add(rel(target))
    return edges


# --------------------------------------------------------------------------- #
# MATLAB edges: a call to a function defined in another .m
# --------------------------------------------------------------------------- #

def matlab_edges(path: Path, m_names: dict[str, Path]) -> set[str]:
    src = strip_matlab_comments(path.read_text(encoding="utf-8", errors="replace"))
    edges: set[str] = set()
    for name, target in m_names.items():
        # Same-named files are never a cross-call: only one .m per name sits on
        # the MATLAB path, and the "function X(...)" header looks like a call to X.
        if target == path or target.stem == path.stem:
            continue
        # Not preceded by a word character, followed by a paren: that is a call.
        if re.search(rf"(?<![\w.]){re.escape(name)}\s*\(", src):
            edges.add(rel(target))
    return edges


def resolve_matlab_shadowing(files: list[Path]):
    """One .m wins per function name. Returns (winners, shadowed)."""
    by_stem: dict[str, list[Path]] = defaultdict(list)
    for f in files:
        if f.suffix == ".m":
            by_stem[f.stem].append(f)

    winners: dict[str, Path] = {}
    shadowed: dict[str, list[Path]] = {}
    for stem, paths in by_stem.items():
        if len(paths) == 1:
            winners[stem] = paths[0]
            continue

        def rank(p: Path) -> int:
            for i, d in enumerate(MATLAB_PATH_PRIORITY):
                if d in p.parts:
                    return i
            return len(MATLAB_PATH_PRIORITY)

        ordered = sorted(paths, key=rank)
        winners[stem] = ordered[0]
        shadowed[stem] = ordered[1:]
    return winners, shadowed


# --------------------------------------------------------------------------- #
# Graph assembly
# --------------------------------------------------------------------------- #

def build_graph():
    files = collect_files()
    m_names, shadowed = resolve_matlab_shadowing(files)
    py_by_stem: dict[str, list[Path]] = defaultdict(list)
    for f in files:
        if f.suffix == ".py":
            py_by_stem[f.stem].append(f)
    py_stems = set(py_by_stem)

    graph: dict[str, set[str]] = {rel(f): set() for f in files}
    for f in files:
        if f.suffix == ".py":
            graph[rel(f)] |= python_edges(f, py_by_stem)
            graph[rel(f)] |= python_matlab_edges(f, m_names, py_stems)
        else:
            graph[rel(f)] |= matlab_edges(f, m_names)

    return graph, {k: [rel(p) for p in v] for k, v in shadowed.items()}


def reachable(graph: dict[str, set[str]], roots: list[str]) -> set[str]:
    seen, queue = set(), deque(roots)
    while queue:
        n = queue.popleft()
        if n in seen or n not in graph:
            continue
        seen.add(n)
        queue.extend(graph[n])
    return seen


def indegree(graph: dict[str, set[str]]) -> dict[str, int]:
    deg = {n: 0 for n in graph}
    for dsts in graph.values():
        for d in dsts:
            if d in deg:
                deg[d] += 1
    return deg


def to_tree(graph, roots: list[str]) -> str:
    lines = []
    for root in roots:
        if root not in graph:
            lines.append(f"[not found] {root}\n")
            continue
        lines.append(root)

        def walk(n: str, prefix: str, stack: set[str]):
            kids = sorted(graph.get(n, ()))
            for i, k in enumerate(kids):
                last = i == len(kids) - 1
                elbow = "`-- " if last else "|-- "
                if k in stack:
                    lines.append(f"{prefix}{elbow}{k}  (cycle)")
                    continue
                lines.append(f"{prefix}{elbow}{k}")
                walk(k, prefix + ("    " if last else "|   "), stack | {k})

        walk(root, "", {root})
        lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="*", default=DEFAULT_ROOTS,
                    help="repo-relative paths, or plain file names")
    args = ap.parse_args()

    graph, shadowed = build_graph()

    # A root given as a bare file name is resolved against the graph.
    roots = []
    for r in args.roots:
        if r in graph:
            roots.append(r)
            continue
        hits = [n for n in graph if Path(n).name == r]
        if len(hits) == 1:
            roots.append(hits[0])
        elif hits:
            print(f"[ambiguous] {r} -> {hits}", file=sys.stderr)
        else:
            print(f"[not found] {r}", file=sys.stderr)

    live = reachable(graph, roots)
    deg = indegree(graph)

    OUT.mkdir(exist_ok=True)
    (OUT / "depgraph.txt").write_text(to_tree(graph, roots), encoding="utf-8")
    (OUT / "depgraph.json").write_text(json.dumps(
        {"edges": {k: sorted(v) for k, v in graph.items()},
         "roots": roots,
         "reachable": sorted(live),
         "shadowed": shadowed},
        indent=2), encoding="utf-8")

    orphans = sorted(n for n in graph if deg[n] == 0 and n not in roots)

    print(f"{len(graph)} files, {sum(len(v) for v in graph.values())} edges")
    print(f"{len(live)} reachable from {len(roots)} roots\n")
    if shadowed:
        print("Same-named .m files (first one wins, per addpath order):")
        for stem, losers in sorted(shadowed.items()):
            print(f"  {stem}: shadowed -> {', '.join(losers)}")
        print()
    print(f"No caller (indegree 0, roots excluded): {len(orphans)}")
    for n in orphans:
        print(f"  {n}")
    print(f"\nOutside the root trees: {len(graph) - len(live)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
