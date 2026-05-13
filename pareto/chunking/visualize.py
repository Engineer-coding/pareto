"""
Tree visualization helpers.

Two outputs are supported:

    * rich-based terminal tree: human-friendly inspection during development
      and on the demo stage.

    * graphviz-based image export: a PNG/SVG you can paste in the README,
      slides, or share with reviewers.

Both go through the same ChunkTree, no transformations needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console
from rich.tree import Tree

if TYPE_CHECKING:
    from pareto.chunking.models import ChunkTree


# ── color palette per node kind ──────────────────────────────────────────
_KIND_STYLE = {
    "root": "bold magenta",
    "section": "bold cyan",
    "paragraph": "green",
    "page": "yellow",
}


def _label_for(node) -> str:
    """Compact, one-line label for a node."""
    kind = node.kind.value
    style = _KIND_STYLE.get(kind, "white")

    head = f"[{style}]{kind}[/{style}]"
    if node.level:
        head += f" [dim]L{node.level}[/dim]"

    if node.title:
        text = node.title
    elif node.content:
        text = node.content[:60].replace("\n", " ")
        if len(node.content) > 60:
            text += "…"
    else:
        text = "(empty)"

    return f"{head}  [white]{text}[/white]  [dim](len={node.length})[/dim]"


def render_rich_tree(chunk_tree: "ChunkTree", console: Console | None = None) -> None:
    """
    Print a colored, hierarchical tree to the terminal.

    Uses rich.Tree, which draws nice unicode connectors and supports
    inline color styling.
    """
    console = console or Console()
    root_node = chunk_tree.root
    rich_root = Tree(_label_for(root_node))
    _expand(chunk_tree, root_node.id, rich_root)
    console.print(rich_root)


def _expand(chunk_tree: "ChunkTree", node_id: str, parent: Tree) -> None:
    """Recursively attach rich Tree branches to mirror the ChunkTree."""
    for child_id in chunk_tree.get(node_id).children_ids:
        child = chunk_tree.get(child_id)
        branch = parent.add(_label_for(child))
        _expand(chunk_tree, child_id, branch)


# ── graphviz image export ────────────────────────────────────────────────

def render_graphviz(
    chunk_tree: "ChunkTree",
    output_path: str | Path,
    format: str = "png",
) -> Path:
    """
    Export the tree as an image via graphviz.

    Requires the graphviz binary (`dot`) to be installed and on PATH.
    Falls back with a clear error message if it isn't.

    Args:
        chunk_tree: the ChunkTree to render.
        output_path: file path WITHOUT extension. Graphviz appends it.
        format: "png", "svg", "pdf", etc.

    Returns:
        The path to the rendered file.
    """
    try:
        import graphviz
    except ImportError as e:
        raise ImportError(
            "graphviz Python package is missing. Install with: uv pip install graphviz"
        ) from e

    output_path = Path(output_path)
    dot = graphviz.Digraph(
        name=f"chunk_tree_{chunk_tree.document_id[:8]}",
        format=format,
        graph_attr={"rankdir": "TB", "splines": "ortho", "bgcolor": "white"},
        node_attr={"fontname": "Helvetica", "fontsize": "10", "style": "filled"},
        edge_attr={"color": "#888888"},
    )

    # Add nodes
    for node in chunk_tree.nodes.values():
        label, fillcolor, shape = _graphviz_attrs(node)
        dot.node(node.id, label=label, fillcolor=fillcolor, shape=shape)

    # Add edges
    for node in chunk_tree.nodes.values():
        for child_id in node.children_ids:
            dot.edge(node.id, child_id)

    # Render. graphviz appends the format extension; remove ours if present.
    output_stem = output_path.with_suffix("")
    try:
        rendered = dot.render(filename=str(output_stem), cleanup=True)
    except graphviz.ExecutableNotFound as e:
        raise RuntimeError(
            "The 'dot' executable was not found on PATH. Install graphviz: "
            "`winget install graphviz` on Windows, then restart your terminal."
        ) from e

    return Path(rendered)


def _graphviz_attrs(node) -> tuple[str, str, str]:
    """Return (label, fillcolor, shape) for a node based on its kind."""
    kind = node.kind.value

    palette = {
        "root":      ("#fde2e4", "doubleoctagon"),
        "section":   ("#cde7f0", "box"),
        "paragraph": ("#d4edda", "note"),
        "page":      ("#fff3cd", "box"),
    }
    fillcolor, shape = palette.get(kind, ("#eeeeee", "box"))

    head = f"[{kind}"
    if node.level:
        head += f" L{node.level}"
    head += "]"

    if node.title:
        body = _escape(node.title[:60])
    elif node.content:
        body = _escape(node.content[:60].replace("\n", " "))
        if len(node.content) > 60:
            body += "…"
    else:
        body = "(empty)"

    label = f"{head}\\n{body}\\nlen={node.length}"
    return label, fillcolor, shape


def _escape(text: str) -> str:
    """Escape characters that graphviz treats specially in node labels."""
    return text.replace("\\", "\\\\").replace('"', '\\"')