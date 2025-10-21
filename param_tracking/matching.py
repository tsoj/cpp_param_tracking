"""Matching logic for PARAM tokens across syntax trees."""

from __future__ import annotations

from typing import Optional

from tree_sitter import Node

from .ast_utils import (
    find_matching_subtrees,
    get_params_in_subtree,
    has_binary_operator_or_bounded_context,
)


def match_param(
    param_node: Node,
    snippet1_root: Node,
    snippet2_root: Node,
    *,
    max_levels: int = 10,
    verbose: bool = False,
) -> Optional[Node]:
    """Return the PARAM node in snippet2 that corresponds to *param_node*."""
    verbose_print = (
        (lambda *args, **kwargs: print(*args, **kwargs))
        if verbose
        else (lambda *args, **kwargs: None)
    )

    current_context = param_node.parent
    level = 0

    verbose_print(f"\n  Starting from PARAM at line {param_node.start_point[0] + 1}")
    verbose_print(
        "  Growing context to include a binary operator or bounded context..."
    )

    while current_context and not has_binary_operator_or_bounded_context(
        current_context
    ):
        verbose_print(f"    Level {level}: {current_context.type} (growing...)")
        current_context = current_context.parent
        level += 1
        if level >= max_levels:
            verbose_print("    ✗ Reached max levels without finding suitable context")
            return None

    if current_context:
        verbose_print(
            f"    Found suitable context at level {level}: {current_context.type}"
        )
        verbose_print(f"\n  Starting matching from level {level}")

    while current_context and level < max_levels:
        text_preview = current_context.text.decode("utf8")[:50]
        verbose_print(f"\n  Level {level}: {current_context.type}")
        verbose_print(f"    Text: {text_preview}...")

        params_in_context = get_params_in_subtree(current_context)
        verbose_print(f"    PARAMs in this subtree: {len(params_in_context)}")

        matches_in_snippet1 = find_matching_subtrees(current_context, snippet1_root)
        matches_in_snippet2 = find_matching_subtrees(current_context, snippet2_root)

        verbose_print(f"    Matches in snippet 1: {len(matches_in_snippet1)}")
        verbose_print(f"    Matches in snippet 2: {len(matches_in_snippet2)}")

        if len(matches_in_snippet1) == 1 and len(matches_in_snippet2) == 1:
            params_in_match1 = get_params_in_subtree(matches_in_snippet1[0])
            params_in_match2 = get_params_in_subtree(matches_in_snippet2[0])

            verbose_print("    Unique match found!")
            verbose_print(f"    PARAMs in match 1: {len(params_in_match1)}")
            verbose_print(f"    PARAMs in match 2: {len(params_in_match2)}")

            if len(params_in_match1) != len(params_in_match2):
                verbose_print("    ✗ Different number of PARAMs in matched subtrees!")
                return None

            try:
                param_index = params_in_match1.index(param_node)
            except ValueError:
                verbose_print(
                    "    ✗ Our PARAM not found in matched subtree (unexpected)"
                )
                return None

            matched_param = params_in_match2[param_index]
            verbose_print("    ✓ FOUND MATCH by position!")
            return matched_param

        if len(matches_in_snippet2) == 0:
            verbose_print("    ✗ No match in snippet 2 - this PARAM has no counterpart")
            return None

        verbose_print("    Ambiguous - need to grow context")

        current_context = current_context.parent
        level += 1

    verbose_print("    ✗ Reached max levels without unique match")
    return None
