"""Finding the functions to check and the order to check them in (docs/plan.md, step 4, calls inside the project)."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path

from pcc.core import FunctionRef
from pcc.project.calls import call_graph, subset_edges
from pcc.project.diff import changed
from pcc.project.discover import discover


def affected(repo_root: Path, changed: Sequence[FunctionRef]) -> tuple[FunctionRef, ...]:
    """Return ``changed`` plus all transitive callers whose proof may depend on them.

    Callers are found among every function ``discover`` reports under ``repo_root``,
    through ``call_graph``. The result holds each function once, ordered by path, then
    by ``lineno``.
    """
    if not changed:
        return ()
    universe = discover(repo_root, ["."])
    graph = call_graph(repo_root, universe)
    callers: dict[FunctionRef, set[FunctionRef]] = {}
    for caller, callees in graph.items():
        for callee in callees:
            callers.setdefault(callee, set()).add(caller)
    seen = set(changed)
    queue = list(changed)
    while queue:
        for caller in callers.get(queue.pop(), ()):
            if caller not in seen:
                seen.add(caller)
                queue.append(caller)
    return tuple(sorted(seen, key=_sort_key))


def call_order(functions: Sequence[FunctionRef]) -> tuple[tuple[FunctionRef, ...], ...]:
    """Group ``functions`` into levels: callees before callers; functions in one level can run in parallel.

    Mutually recursive functions share a level and use each other's spec as an assumption.
    The edges come from the functions' sources (see ``subset_edges``); the level of an
    SCC is one more than the highest level of the SCCs it calls, so a recursive
    function stays on the level of its other callees. Levels run from callees (first)
    to callers (last); within a level, functions are sorted by path, then by ``lineno``.
    """
    nodes = sorted(set(functions), key=_sort_key)
    if not nodes:
        return ()
    edges = subset_edges(nodes)
    sccs = _tarjan(nodes, edges)
    scc_of = {node: pos for pos, scc in enumerate(sccs) for node in scc}
    children = [set[int]() for _ in sccs]
    for pos, scc in enumerate(sccs):
        for node in scc:
            for callee in edges[node]:
                other = scc_of[callee]
                if other != pos:
                    children[pos].add(other)
    level = [0] * len(sccs)
    for pos in _post_order(children):
        if children[pos]:
            level[pos] = 1 + max(level[child] for child in children[pos])
    by_level: dict[int, list[FunctionRef]] = {}
    for pos, scc in enumerate(sccs):
        by_level.setdefault(level[pos], []).extend(scc)
    return tuple(tuple(sorted(members, key=_sort_key)) for _, members in sorted(by_level.items()))


def _sort_key(ref: FunctionRef) -> tuple[str, int]:
    return (ref.path, ref.lineno)


def _tarjan(nodes: list[FunctionRef], edges: Mapping[FunctionRef, frozenset[FunctionRef]]) -> list[list[FunctionRef]]:
    """The strongly connected components of ``nodes`` in deterministic order."""
    index_of: dict[FunctionRef, int] = {}
    low: dict[FunctionRef, int] = {}
    on_stack: set[FunctionRef] = set()
    stack: list[FunctionRef] = []
    sccs: list[list[FunctionRef]] = []
    counter = 0
    for root in nodes:
        if root in index_of:
            continue
        index_of[root] = low[root] = counter
        counter += 1
        stack.append(root)
        on_stack.add(root)
        call_stack: list[tuple[FunctionRef, Iterator[FunctionRef]]] = [(root, _successors(root, edges))]
        while call_stack:
            node, successors = call_stack[-1]
            descended = False
            for successor in successors:
                if successor not in index_of:
                    index_of[successor] = low[successor] = counter
                    counter += 1
                    stack.append(successor)
                    on_stack.add(successor)
                    call_stack.append((successor, _successors(successor, edges)))
                    descended = True
                    break
                if successor in on_stack:
                    low[node] = min(low[node], index_of[successor])
            if descended:
                continue
            call_stack.pop()
            if call_stack:
                parent = call_stack[-1][0]
                low[parent] = min(low[parent], low[node])
            if low[node] == index_of[node]:
                scc: list[FunctionRef] = []
                while True:
                    member = stack.pop()
                    on_stack.discard(member)
                    scc.append(member)
                    if member == node:
                        break
                sccs.append(scc)
    return sccs


def _successors(node: FunctionRef, edges: Mapping[FunctionRef, frozenset[FunctionRef]]) -> Iterator[FunctionRef]:
    return iter(sorted(edges.get(node, frozenset()), key=_sort_key))


def _post_order(children: list[set[int]]) -> list[int]:
    """Every SCC after the SCCs it calls (reverse topological order of the SCC graph)."""
    visited = [False] * len(children)
    order: list[int] = []
    for root in range(len(children)):
        if visited[root]:
            continue
        visited[root] = True
        call_stack = [(root, iter(sorted(children[root])))]
        while call_stack:
            node, successors = call_stack[-1]
            descended = False
            for successor in successors:
                if not visited[successor]:
                    visited[successor] = True
                    call_stack.append((successor, iter(sorted(children[successor]))))
                    descended = True
                    break
            if descended:
                continue
            call_stack.pop()
            order.append(node)
    return order


__all__ = ["affected", "call_graph", "call_order", "changed", "discover"]
