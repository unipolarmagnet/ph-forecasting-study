"""Compose preprocessing nodes into a pipeline: apply(df, steps) -> df."""
from . import nodes


def apply(df, steps, target="Close"):
    """Run an ordered list of (node_name, params) on a copy of df; always end with cleanup."""
    out = df.copy()
    for name, params in steps:
        out = nodes.NODES[name](out, target=target, **params)
    if not steps or steps[-1][0] != "cleanup":
        out = nodes.cleanup(out, target=target)
    return out
