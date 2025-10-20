"""Helpers for replacing numeric literals with PARAM tokens."""

from __future__ import annotations

from typing import List, Sequence, Tuple

from tree_sitter import Node

from .ast_utils import find_params
from .parser import parse_code

NumberInfo = Tuple[str, int, int]


def extract_numbers_and_create_param_code(code: str) -> Tuple[str, List[NumberInfo]]:
    """Replace numeric literals in *code* with PARAM tokens."""
    tree = parse_code(code)

    numbers = []

    def extract(node: Node) -> None:
        if node.type in {"number_literal", "integer_literal", "float_literal"}:
            numbers.append(
                {
                    "value": node.text.decode("utf8"),
                    "start": node.start_byte,
                    "end": node.end_byte,
                }
            )
        for child in node.children:
            extract(child)

    extract(tree.root_node)

    numbers.sort(key=lambda entry: entry["start"], reverse=True)

    code_bytes = code.encode("utf8")
    for entry in numbers:
        code_bytes = code_bytes[: entry["start"]] + b"PARAM" + code_bytes[entry["end"] :]

    param_code = code_bytes.decode("utf8")

    numbers.reverse()
    number_values: List[NumberInfo] = [
        (entry["value"], entry["start"], entry["end"]) for entry in numbers
    ]

    return param_code, number_values


def create_annotated_code(
    original_code: str,
    original_numbers: Sequence[NumberInfo],
    param_map,
    original_params: Sequence[Node],
) -> str:
    """Replace matched numeric literals in *original_code* with PARAM identifiers."""
    replacements = []
    for param_node, (value, start, end) in zip(original_params, original_numbers):
        param_id = param_map[id(param_node)]
        replacements.append((start, end, param_id))

    replacements.sort(key=lambda item: item[0], reverse=True)

    code_bytes = original_code.encode("utf8")
    for start, end, param_id in replacements:
        code_bytes = code_bytes[:start] + param_id.encode("utf8") + code_bytes[end:]

    return code_bytes.decode("utf8")
