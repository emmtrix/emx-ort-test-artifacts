"""Load and sort operator information from tracked ONNX artifact models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import onnx
from onnx import AttributeProto, GraphProto, ModelProto

DEFAULT_OPERATOR_DOMAIN = "ai.onnx"


@dataclass(frozen=True)
class OperatorCase:
    """One artifact test case together with the operators used by its model."""

    path: str
    operators: tuple[str, ...]


def operator_label(*, domain: str, op_type: str) -> str:
    """Return the display label for one operator."""
    normalized_domain = domain or DEFAULT_OPERATOR_DOMAIN
    if normalized_domain == DEFAULT_OPERATOR_DOMAIN:
        return op_type
    return f"{normalized_domain}::{op_type}"


def split_operator_label(operator: str) -> tuple[str, str]:
    """Split one display label into domain and operator name."""
    if "::" not in operator:
        return (DEFAULT_OPERATOR_DOMAIN, operator)
    domain, op_type = operator.split("::", maxsplit=1)
    return (domain, op_type)


def operator_sort_key(operator: str) -> tuple[int, str, str]:
    """Sort operators alphabetically by operator name, then by domain."""
    domain, op_type = split_operator_label(operator)
    return (0, op_type.casefold(), domain.casefold())


def iter_graph_operators(graph: GraphProto) -> list[str]:
    """Collect operator labels from a graph and all nested subgraphs."""
    operators: list[str] = []
    for node in graph.node:
        operators.append(operator_label(domain=node.domain, op_type=node.op_type))
        for attribute in node.attribute:
            if attribute.type == AttributeProto.GRAPH:
                operators.extend(iter_graph_operators(attribute.g))
            elif attribute.type == AttributeProto.GRAPHS:
                for subgraph in attribute.graphs:
                    operators.extend(iter_graph_operators(subgraph))
    return operators


def case_operators_from_model(model: ModelProto) -> tuple[str, ...]:
    """Return the sorted unique operators used by one model."""
    return tuple(sorted(set(iter_graph_operators(model.graph)), key=operator_sort_key))


def load_operator_cases(artifacts_root: Path) -> list[OperatorCase]:
    """Load all tracked artifact models below ``artifacts_root`` into case rows."""
    cases: list[OperatorCase] = []
    for model_path in sorted(artifacts_root.rglob("model.onnx")):
        model = onnx.load(model_path, load_external_data=False)
        cases.append(
            OperatorCase(
                path=model_path.parent.relative_to(artifacts_root).as_posix(),
                operators=case_operators_from_model(model),
            )
        )
    return cases
