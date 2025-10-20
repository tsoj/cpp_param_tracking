"""Helpers for inspecting Tree-sitter syntax trees."""

from __future__ import annotations

from typing import List

from tree_sitter import Node

PARAM_TOKEN = "PARAM"


def find_params(root: Node) -> List[Node]:
    """Return PARAM identifier nodes discovered in depth-first order."""
    params: List[Node] = []

    def traverse(node: Node) -> None:
        if node.type == "identifier":
            text = node.text.decode("utf8") if node.text else ""
            if text == PARAM_TOKEN:
                params.append(node)
        for child in node.children:
            traverse(child)

    traverse(root)
    return params


def nodes_equal(node1: Node, node2: Node, mask_params: bool = True) -> bool:
    """Recursively compare two syntax tree nodes for structural equality."""
    if node1.type != node2.type or node1.is_named != node2.is_named:
        return False

    if node1.child_count == 0 and node2.child_count == 0:
        text1 = node1.text.decode("utf8") if node1.text else ""
        text2 = node2.text.decode("utf8") if node2.text else ""

        if mask_params and text1 == PARAM_TOKEN and text2 == PARAM_TOKEN:
            return True

        if node1.type in {
            "identifier",
            "number_literal",
            "string_literal",
            "primitive_type",
            "type_identifier",
        }:
            return text1 == text2

        if not node1.is_named:
            return text1 == text2

        return True

    if node1.child_count != node2.child_count:
        return False

    return all(
        nodes_equal(child1, child2, mask_params)
        for child1, child2 in zip(node1.children, node2.children)
    )


def count_params_in_subtree(node: Node) -> int:
    """Count PARAM tokens that appear under *node*."""
    text = node.text.decode("utf8") if node.text else ""
    count = 1 if text == PARAM_TOKEN else 0
    return count + sum(count_params_in_subtree(child) for child in node.children)


def get_params_in_subtree(node: Node) -> List[Node]:
    """Collect PARAM nodes beneath *node* preserving traversal order."""
    params: List[Node] = []

    def traverse(current: Node) -> None:
        text = current.text.decode("utf8") if current.text else ""
        if text == PARAM_TOKEN:
            params.append(current)
        for child in current.children:
            traverse(child)

    traverse(node)
    return params


def has_binary_operator_or_bounded_context(node: Node) -> bool:
    """Return True if a subtree contains a binary operator or bounded context."""
    bounded_context_types = {
        "expression_statement",
        "declaration",
        "return_statement",
        "if_statement",
        "for_statement",
        "while_statement",
        "do_statement",
        "switch_statement",
        "condition_clause",
        "init_declarator",
        "assignment_expression",
    }

    if node.type in bounded_context_types:
        return True

    binary_operator_types = {
        "binary_expression",
        "+",
        "-",
        "*",
        "/",
        "%",
        "==",
        "!=",
        "<",
        ">",
        "<=",
        ">=",
        "&&",
        "||",
        "&",
        "|",
        "^",
        "<<",
        ">>",
    }

    def traverse(current: Node) -> bool:
        if current.type in binary_operator_types:
            return True
        return any(traverse(child) for child in current.children)

    return traverse(node)


def find_matching_subtrees(target: Node, search_root: Node):
    """Find nodes in *search_root* whose subtrees equal *target*."""
    matches = []

    def traverse(node: Node) -> None:
        if nodes_equal(target, node):
            matches.append(node)
        for child in node.children:
            traverse(child)

    traverse(search_root)
    return matches
