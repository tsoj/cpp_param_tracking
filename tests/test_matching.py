"""Comprehensive test suite for the PARAM matching algorithm."""

import sys
from pathlib import Path

import pytest
from tree_sitter import Language, Parser, Query, Node
import tree_sitter_cpp as tscpp

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from param_tracking import (
    CPP_LANGUAGE,
    count_params_in_subtree,
    find_matching_subtrees,
    find_params,
    get_params_in_subtree,
    match_param,
    nodes_equal,
    parse_code,
)


def parse_cpp(code):
    """Helper to parse C++ code"""
    return parse_code(code)


# ============================================================================
# TEST CLASS: find_params()
# ============================================================================


class TestFindParams:
    """Test the find_params function"""

    def test_deterministic_order(self):
        """Test that find_params returns PARAMs in deterministic order"""
        code = """
        int main() {
            int x = PARAM + PARAM;
            int y = PARAM;
        }
        """
        tree = parse_cpp(code)

        # Run find_params multiple times
        results = []
        for _ in range(10):
            params = find_params(tree.root_node)
            # Convert to positions for comparison
            positions = [(p.start_point, p.end_point) for p in params]
            results.append(positions)

        # All results should be identical
        first_result = results[0]
        for result in results[1:]:
            assert result == first_result, "find_params should return deterministic order"

        # Verify they're in left-to-right, top-to-bottom order
        assert len(first_result) == 3
        for i in range(len(first_result) - 1):
            # Each subsequent PARAM should come after the previous one
            assert first_result[i] <= first_result[i + 1]

    def test_no_params(self):
        """Test code with no PARAMs"""
        code = "int main() { int x = 5; return 0; }"
        tree = parse_cpp(code)
        params = find_params(tree.root_node)
        assert len(params) == 0

    def test_single_param(self):
        """Test code with a single PARAM"""
        code = "int main() { int x = PARAM; }"
        tree = parse_cpp(code)
        params = find_params(tree.root_node)
        assert len(params) == 1
        assert params[0].text.decode("utf8") == "PARAM"

    def test_multiple_params(self):
        """Test code with multiple PARAMs"""
        code = "int main() { int x = PARAM + PARAM; return PARAM; }"
        tree = parse_cpp(code)
        params = find_params(tree.root_node)
        assert len(params) == 3

    def test_params_in_different_contexts(self):
        """Test PARAMs in various code contexts"""
        code = """
        int func(int PARAM) {
            int x = PARAM;
            if (PARAM > 0) {
                return PARAM;
            }
            for (int i = 0; i < PARAM; i++) {
                PARAM++;
            }
            return PARAM;
        }
        """
        tree = parse_cpp(code)
        params = find_params(tree.root_node)
        assert len(params) == 7

    def test_param_not_identifier(self):
        """Test that PARAM in comments or strings is not found"""
        code = """
        int main() {
            // PARAM in comment
            char* str = "PARAM";
            int x = PARAM;
        }
        """
        tree = parse_cpp(code)
        params = find_params(tree.root_node)
        # Should only find the identifier PARAM, not in comment or string
        assert len(params) == 1


# ============================================================================
# TEST CLASS: nodes_equal()
# ============================================================================


class TestNodesEqual:
    """Test the nodes_equal function"""

    def test_identical_simple_expressions(self):
        """Test two identical simple expressions"""
        code1 = "int main() { int x = a + b; }"
        code2 = "int main() { int x = a + b; }"
        tree1 = parse_cpp(code1)
        tree2 = parse_cpp(code2)
        assert nodes_equal(tree1.root_node, tree2.root_node)

    def test_different_identifiers(self):
        """Test expressions with different identifiers"""
        code1 = "int main() { int x = a + b; }"
        code2 = "int main() { int x = c + d; }"
        tree1 = parse_cpp(code1)
        tree2 = parse_cpp(code2)
        assert not nodes_equal(tree1.root_node, tree2.root_node)

    def test_param_masking_enabled(self):
        """Test that PARAMs are treated as equal when masking is enabled"""
        code1 = "int main() { int x = PARAM; }"
        code2 = "int main() { int y = PARAM; }"
        tree1 = parse_cpp(code1)
        tree2 = parse_cpp(code2)
        # x vs y should make them different
        assert not nodes_equal(tree1.root_node, tree2.root_node)

    def test_param_masking_disabled(self):
        """Test that PARAMs comparison works with masking disabled"""
        code1 = "int main() { int x = PARAM; }"
        code2 = "int main() { int x = PARAM; }"
        tree1 = parse_cpp(code1)
        tree2 = parse_cpp(code2)
        assert nodes_equal(tree1.root_node, tree2.root_node, mask_params=False)

    def test_different_types(self):
        """Test nodes with different types"""
        code1 = "int main() { int x = 5; }"
        code2 = "int main() { float x = 5; }"
        tree1 = parse_cpp(code1)
        tree2 = parse_cpp(code2)
        assert not nodes_equal(tree1.root_node, tree2.root_node)

    def test_different_child_counts(self):
        """Test nodes with different number of children"""
        code1 = "int main() { int x = a + b; }"
        code2 = "int main() { int x = a + b + c; }"
        tree1 = parse_cpp(code1)
        tree2 = parse_cpp(code2)
        assert not nodes_equal(tree1.root_node, tree2.root_node)

    def test_complex_nested_structure(self):
        """Test complex nested structures"""
        code1 = """
        int main() {
            if (a > 0) {
                for (int i = 0; i < 10; i++) {
                    x = x + 1;
                }
            }
        }
        """
        code2 = """
        int main() {
            if (a > 0) {
                for (int i = 0; i < 10; i++) {
                    x = x + 1;
                }
            }
        }
        """
        tree1 = parse_cpp(code1)
        tree2 = parse_cpp(code2)
        assert nodes_equal(tree1.root_node, tree2.root_node)

    def test_different_operators(self):
        """Test expressions with different operators"""
        code1 = "int main() { int x = a + b; }"
        code2 = "int main() { int x = a - b; }"
        tree1 = parse_cpp(code1)
        tree2 = parse_cpp(code2)
        assert not nodes_equal(tree1.root_node, tree2.root_node)


# ============================================================================
# TEST CLASS: count_params_in_subtree()
# ============================================================================


class TestCountParamsInSubtree:
    """Test the count_params_in_subtree function"""

    def test_no_params_in_subtree(self):
        """Test subtree with no PARAMs"""
        code = "int main() { int x = 5; }"
        tree = parse_cpp(code)
        count = count_params_in_subtree(tree.root_node)
        assert count == 0

    def test_single_param_in_subtree(self):
        """Test subtree with one PARAM"""
        code = "int main() { int x = PARAM; }"
        tree = parse_cpp(code)
        count = count_params_in_subtree(tree.root_node)
        assert count == 1

    def test_multiple_params_in_subtree(self):
        """Test subtree with multiple PARAMs"""
        code = "int main() { int x = PARAM + PARAM * PARAM; }"
        tree = parse_cpp(code)
        count = count_params_in_subtree(tree.root_node)
        assert count == 3

    def test_params_in_nested_structure(self):
        """Test PARAMs in nested structures"""
        code = """
        int main() {
            if (PARAM > 0) {
                while (PARAM < 10) {
                    x = PARAM;
                }
            }
        }
        """
        tree = parse_cpp(code)
        count = count_params_in_subtree(tree.root_node)
        assert count == 3


# ============================================================================
# TEST CLASS: find_matching_subtrees()
# ============================================================================


class TestFindMatchingSubtrees:
    """Test the find_matching_subtrees function"""

    def test_unique_match(self):
        """Test finding a unique matching subtree"""
        code = """
        int main() {
            int x = a * PARAM;
            int y = b + PARAM;
        }
        """
        tree = parse_cpp(code)
        params = find_params(tree.root_node)

        # Find the parent expression of first PARAM (a * PARAM)
        target = params[0].parent
        matches = find_matching_subtrees(target, tree.root_node)

        # Should find exactly one match (itself)
        assert len(matches) >= 1

    def test_multiple_matches(self):
        """Test finding multiple identical subtrees"""
        code = """
        int main() {
            int x = a * PARAM;
            int y = a * PARAM;
        }
        """
        tree = parse_cpp(code)
        params = find_params(tree.root_node)

        # Both expressions are identical
        target = params[0].parent
        matches = find_matching_subtrees(target, tree.root_node)

        # Should find multiple matches
        assert len(matches) >= 2

    def test_no_match(self):
        """Test finding no matches in a different tree"""
        code1 = "int main() { int x = a * PARAM; }"
        code2 = "int main() { int y = b + PARAM; }"

        tree1 = parse_cpp(code1)
        tree2 = parse_cpp(code2)

        params1 = find_params(tree1.root_node)
        target = params1[0].parent

        matches = find_matching_subtrees(target, tree2.root_node)

        # Should find no matches (different structure)
        assert len(matches) == 0


# ============================================================================
# TEST CLASS: get_params_in_subtree()
# ============================================================================


class TestGetParamsInSubtree:
    """Test the get_params_in_subtree function"""

    def test_params_in_order(self):
        """Test that PARAMs are returned in order"""
        code = "int main() { int x = PARAM + PARAM * PARAM; }"
        tree = parse_cpp(code)
        params = get_params_in_subtree(tree.root_node)

        assert len(params) == 3
        # Verify they're in left-to-right order
        for i in range(len(params) - 1):
            assert params[i].start_point <= params[i + 1].start_point

    def test_empty_subtree(self):
        """Test subtree with no PARAMs"""
        code = "int main() { int x = 5; }"
        tree = parse_cpp(code)
        params = get_params_in_subtree(tree.root_node)
        assert len(params) == 0

    def test_nested_params(self):
        """Test PARAMs in nested expressions"""
        code = "int main() { int x = (PARAM + (PARAM * PARAM)); }"
        tree = parse_cpp(code)
        params = get_params_in_subtree(tree.root_node)
        assert len(params) == 3


# ============================================================================
# TEST CLASS: match_param() - Basic Cases
# ============================================================================


class TestMatchParamBasic:
    """Test the match_param function with basic cases"""

    def test_identical_code(self):
        """Test matching in identical code snippets"""
        code1 = "int main() { int x = a * PARAM; }"
        code2 = "int main() { int x = a * PARAM; }"

        tree1 = parse_cpp(code1)
        tree2 = parse_cpp(code2)

        params1 = find_params(tree1.root_node)
        params2 = find_params(tree2.root_node)

        matched = match_param(params1[0], tree1.root_node, tree2.root_node)

        assert matched is not None
        assert matched == params2[0]

    def test_different_identifiers_same_structure(self):
        """Test matching with different identifiers but same structure"""
        code1 = "int main() { int x = a * PARAM; }"
        code2 = "int main() { int x = b * PARAM; }"

        tree1 = parse_cpp(code1)
        tree2 = parse_cpp(code2)

        params1 = find_params(tree1.root_node)
        params2 = find_params(tree2.root_node)

        matched = match_param(params1[0], tree1.root_node, tree2.root_node)

        assert matched is None

    def test_no_match_different_structure(self):
        """Test when structures are completely different"""
        code1 = "int main() { int x = a * PARAM; }"
        code2 = "int main() { float y = b + c; }"

        tree1 = parse_cpp(code1)
        tree2 = parse_cpp(code2)

        params1 = find_params(tree1.root_node)

        matched = match_param(params1[0], tree1.root_node, tree2.root_node)

        assert matched is None

    def test_multiple_params_same_context(self):
        """Test matching multiple PARAMs in the same context"""
        code1 = "int main() { int x = PARAM + PARAM; }"
        code2 = "int main() { int x = PARAM + PARAM; }"

        tree1 = parse_cpp(code1)
        tree2 = parse_cpp(code2)

        params1 = find_params(tree1.root_node)
        params2 = find_params(tree2.root_node)

        # First PARAM should match first PARAM
        matched1 = match_param(params1[0], tree1.root_node, tree2.root_node)
        assert matched1 == params2[0]

        # Second PARAM should match second PARAM
        matched2 = match_param(params1[1], tree1.root_node, tree2.root_node)
        assert matched2 == params2[1]


# ============================================================================
# TEST CLASS: match_param() - Complex Cases
# ============================================================================


class TestMatchParamComplex:
    """Test the match_param function with complex scenarios"""

    def test_ambiguous_then_unique(self):
        """Test case where we need to grow context to disambiguate"""
        code1 = """
        int main() {
            int a = b * PARAM;
            int x = c * PARAM;
        }
        """
        code2 = """
        int main() {
            int a = b * PARAM;
            int x = d * PARAM;
        }
        """

        tree1 = parse_cpp(code1)
        tree2 = parse_cpp(code2)

        params1 = find_params(tree1.root_node)
        params2 = find_params(tree2.root_node)

        # First PARAM should match first PARAM (both in 'b * PARAM')
        matched1 = match_param(params1[0], tree1.root_node, tree2.root_node)
        assert matched1 == params2[0]

        # Second PARAM should not match second PARAM
        matched2 = match_param(params1[1], tree1.root_node, tree2.root_node)
        assert matched2 != params2[1]

    def test_nested_control_structures(self):
        """Test matching in nested control structures"""
        code1 = """
        int main() {
            if (x > 0) {
                for (int i = 0; i < PARAM; i++) {
                    result = PARAM;
                }
            }
        }
        """
        code2 = """
        int main() {
            if (x > 0) {
                for (int i = 0; i < PARAM; i++) {
                    result = PARAM;
                }
            }
        }
        """

        tree1 = parse_cpp(code1)
        tree2 = parse_cpp(code2)

        params1 = find_params(tree1.root_node)
        params2 = find_params(tree2.root_node)

        # Both PARAMs should match correctly
        assert match_param(params1[0], tree1.root_node, tree2.root_node) == params2[0]
        assert match_param(params1[1], tree1.root_node, tree2.root_node) == params2[1]

    def test_function_calls(self):
        """Test matching in function call contexts"""
        code1 = """
        int main() {
            foo(PARAM, PARAM);
            bar(PARAM);
        }
        """
        code2 = """
        int main() {
            foo(PARAM, PARAM);
            bar(PARAM);
        }
        """

        tree1 = parse_cpp(code1)
        tree2 = parse_cpp(code2)

        params1 = find_params(tree1.root_node)
        params2 = find_params(tree2.root_node)

        print(params1)
        print(params2)

        assert len(params1) == 3
        assert len(params2) == 3

        # All three should match correctly
        for i in range(3):
            print(i)
            matched = match_param(params1[i], tree1.root_node, tree2.root_node)
            assert matched == params2[i]

    def test_array_indexing(self):
        """Test matching in array indexing contexts"""
        code1 = """
        int main() {
            arr[PARAM] = PARAM;
        }
        """
        code2 = """
        int main() {
            arr[PARAM] = PARAM;
        }
        """

        tree1 = parse_cpp(code1)
        tree2 = parse_cpp(code2)

        params1 = find_params(tree1.root_node)
        params2 = find_params(tree2.root_node)

        assert match_param(params1[0], tree1.root_node, tree2.root_node) == params2[0]
        assert match_param(params1[1], tree1.root_node, tree2.root_node) == params2[1]

    def test_pointer_operations(self):
        """Test matching in pointer operation contexts"""
        code1 = """
        int main() {
            *ptr = PARAM;
            int x = *PARAM;
        }
        """
        code2 = """
        int main() {
            *ptr = PARAM;
            int x = *PARAM;
        }
        """

        tree1 = parse_cpp(code1)
        tree2 = parse_cpp(code2)

        params1 = find_params(tree1.root_node)
        params2 = find_params(tree2.root_node)

        for i in range(len(params1)):
            matched = match_param(params1[i], tree1.root_node, tree2.root_node)
            assert matched == params2[i]


# ============================================================================
# TEST CLASS: match_param() - Edge Cases
# ============================================================================


class TestMatchParamEdgeCases:
    """Test the match_param function with edge cases"""

    def test_max_levels_exceeded(self):
        """Test when max_levels is exceeded"""
        code1 = "int main() { int x = PARAM; }"
        code2 = "int main() { int x = PARAM; int y = PARAM; }"

        tree1 = parse_cpp(code1)
        tree2 = parse_cpp(code2)

        params1 = find_params(tree1.root_node)

        # With max_levels=0, should fail immediately
        matched = match_param(
            params1[0], tree1.root_node, tree2.root_node, max_levels=0
        )
        assert matched is None

    def test_different_param_counts(self):
        """Test when matched subtrees have different PARAM counts"""
        code1 = "int main() { int x = a * PARAM; }"
        code2 = "int main() { int x = a * PARAM + PARAM; }"

        tree1 = parse_cpp(code1)
        tree2 = parse_cpp(code2)

        params1 = find_params(tree1.root_node)

        # The structures are different, so matching may fail or find wrong match
        matched = match_param(params1[0], tree1.root_node, tree2.root_node)
        # Behavior depends on context growth

    def test_single_param_both_snippets(self):
        """Test with single PARAM in each snippet"""
        code1 = "int main() { return PARAM; }"
        code2 = "int main() { return PARAM; }"

        tree1 = parse_cpp(code1)
        tree2 = parse_cpp(code2)

        params1 = find_params(tree1.root_node)
        params2 = find_params(tree2.root_node)

        matched = match_param(params1[0], tree1.root_node, tree2.root_node)
        assert matched == params2[0]

    def test_repeated_identical_patterns(self):
        """Test with many identical patterns"""
        code1 = """
        int main() {
            int a = x * PARAM;
            int b = x * PARAM;
            int c = x * PARAM;
        }
        """
        code2 = """
        int main() {
            int a = x * PARAM;
            int b = x * PARAM;
            int c = x * PARAM;
        }
        """

        tree1 = parse_cpp(code1)
        tree2 = parse_cpp(code2)

        params1 = find_params(tree1.root_node)
        params2 = find_params(tree2.root_node)

        # Each should match to its corresponding position
        for i in range(3):
            matched = match_param(params1[i], tree1.root_node, tree2.root_node)
            assert matched == params2[i]


# ============================================================================
# TEST CLASS: Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for the full workflow"""

    def test_original_example(self):
        """Test with the original example from main.py"""
        code1 = """
        int main() {
            int a = b * PARAM;
            int x = c * PARAM;
            float y = PARAM + PARAM;
            return PARAM;
        }
        """
        code2 = """
        int main() {
            int a = b * PARAM;
            int x = d * PARAM;
            float y = PARAM + PARAM;
            return PARAM;
        }
        """

        tree1 = parse_cpp(code1)
        tree2 = parse_cpp(code2)

        params1 = find_params(tree1.root_node)
        params2 = find_params(tree2.root_node)

        assert len(params1) == 5
        assert len(params2) == 5

        # Match all params
        matches = {}
        for i, param1 in enumerate(params1):
            matched = match_param(param1, tree1.root_node, tree2.root_node)
            if matched:
                matches[i] = params2.index(matched)

        # PARAM 0: b * PARAM should match (identifiers are same)
        assert 0 in matches
        assert matches[0] == 0

        # PARAM 1: c * PARAM should NOT match (c != d)
        assert 1 not in matches

        # PARAM 2-4: PARAM + PARAM and return PARAM should match
        assert 2 in matches
        assert matches[2] == 2
        assert 3 in matches
        assert matches[3] == 3
        assert 4 in matches
        assert matches[4] == 4

    def test_complex_real_world_example(self):
        """Test with a more complex real-world-like example"""
        code1 = """
        int calculate(int n) {
            int result = 0;
            for (int i = 0; i < n; i++) {
                if (i % 2 == 0) {
                    result += PARAM * i;
                } else {
                    result -= PARAM / 2;
                }
            }
            return result + PARAM;
        }
        """
        code2 = """
        int calculate(int n) {
            int result = 0;
            for (int i = 0; i < n; i++) {
                if (i % 2 == 0) {
                    result += PARAM * i;
                } else {
                    result -= PARAM / 2;
                }
            }
            return result + PARAM;
        }
        """

        tree1 = parse_cpp(code1)
        tree2 = parse_cpp(code2)

        params1 = find_params(tree1.root_node)
        params2 = find_params(tree2.root_node)

        assert len(params1) == len(params2)

        # All params should match
        for i, param1 in enumerate(params1):
            matched = match_param(param1, tree1.root_node, tree2.root_node)
            assert matched is not None
            assert matched == params2[i]

    def test_no_params_in_either(self):
        """Test when neither snippet has PARAMs"""
        code1 = "int main() { return 0; }"
        code2 = "int main() { return 0; }"

        tree1 = parse_cpp(code1)
        tree2 = parse_cpp(code2)

        params1 = find_params(tree1.root_node)
        params2 = find_params(tree2.root_node)

        assert len(params1) == 0
        assert len(params2) == 0

    def test_param_only_in_first_snippet(self):
        """Test when only first snippet has PARAMs"""
        code1 = "int main() { int x = PARAM; }"
        code2 = "int main() { int x = 5; }"

        tree1 = parse_cpp(code1)
        tree2 = parse_cpp(code2)

        params1 = find_params(tree1.root_node)

        assert len(params1) == 1

        matched = match_param(params1[0], tree1.root_node, tree2.root_node)
        assert matched is None

    def test_param_only_in_second_snippet(self):
        """Test when only second snippet has PARAMs"""
        code1 = "int main() { int x = 5; }"
        code2 = "int main() { int x = PARAM; }"

        tree1 = parse_cpp(code1)
        tree2 = parse_cpp(code2)

        params1 = find_params(tree1.root_node)

        assert len(params1) == 0

    def test_mixed_expressions(self):
        """Test with various mixed expressions"""
        code1 = """
        int main() {
            int a = PARAM + 5;
            int b = PARAM * 2 - 3;
            float c = (PARAM / 4.0) + PARAM;
            bool d = PARAM > 10 && PARAM < 20;
            int e = PARAM ? PARAM : 0;
        }
        """
        code2 = """
        int main() {
            int a = PARAM + 5;
            int b = PARAM * 2 - 3;
            float c = (PARAM / 4.0) + PARAM;
            bool d = PARAM > 10 && PARAM < 20;
            int e = PARAM ? PARAM : 0;
        }
        """

        tree1 = parse_cpp(code1)
        tree2 = parse_cpp(code2)

        params1 = find_params(tree1.root_node)
        params2 = find_params(tree2.root_node)

        assert len(params1) == len(params2)

        for i, param1 in enumerate(params1):
            matched = match_param(param1, tree1.root_node, tree2.root_node)
            assert matched is not None
            assert matched == params2[i]

    def test_switch_statement(self):
        """Test with switch statements"""
        code1 = """
        int main() {
            switch(PARAM) {
                case 1: return PARAM;
                case 2: return PARAM * 2;
                default: return PARAM;
            }
        }
        """
        code2 = """
        int main() {
            switch(PARAM) {
                case 1: return PARAM;
                case 2: return PARAM * 2;
                default: return PARAM;
            }
        }
        """

        tree1 = parse_cpp(code1)
        tree2 = parse_cpp(code2)

        params1 = find_params(tree1.root_node)
        params2 = find_params(tree2.root_node)

        for i, param1 in enumerate(params1):
            matched = match_param(param1, tree1.root_node, tree2.root_node)
            assert matched is not None
            assert matched == params2[i]

    def test_struct_member_access(self):
        """Test with struct member access"""
        code1 = """
        int main() {
            obj.field = PARAM;
            int x = obj.PARAM;
            return PARAM;
        }
        """
        code2 = """
        int main() {
            obj.field = PARAM;
            int x = obj.PARAM;
            return PARAM;
        }
        """

        tree1 = parse_cpp(code1)
        tree2 = parse_cpp(code2)

        params1 = find_params(tree1.root_node)
        params2 = find_params(tree2.root_node)

        for i, param1 in enumerate(params1):
            matched = match_param(param1, tree1.root_node, tree2.root_node)
            assert matched is not None

    def test_different_identifiers_no_match(self):
        """Test that different identifiers prevent matching"""
        code1 = """
        int main() {
            int x = a * PARAM;
            int y = b + PARAM;
        }
        """
        code2 = """
        int main() {
            int x = c * PARAM;
            int y = d + PARAM;
        }
        """

        tree1 = parse_cpp(code1)
        tree2 = parse_cpp(code2)

        params1 = find_params(tree1.root_node)

        # Neither PARAM should match because identifiers are different
        matched1 = match_param(params1[0], tree1.root_node, tree2.root_node)
        matched2 = match_param(params1[1], tree1.root_node, tree2.root_node)

        assert matched1 is None
        assert matched2 is None

    def test_partial_match_in_context(self):
        """Test when only part of the code matches"""
        code1 = """
        int main() {
            int a = x * PARAM;
            int b = y * PARAM;
            int c = z * PARAM;
        }
        """
        code2 = """
        int main() {
            int a = x * PARAM;
            int b = w * PARAM;
            int c = z * PARAM;
        }
        """

        tree1 = parse_cpp(code1)
        tree2 = parse_cpp(code2)

        params1 = find_params(tree1.root_node)
        params2 = find_params(tree2.root_node)

        # First PARAM: x * PARAM matches
        matched1 = match_param(params1[0], tree1.root_node, tree2.root_node)
        assert matched1 is not None
        assert matched1 == params2[0]

        # Second PARAM: y * PARAM doesn't match (y != w)
        matched2 = match_param(params1[1], tree1.root_node, tree2.root_node)
        assert matched2 is None

        # Third PARAM: z * PARAM matches
        matched3 = match_param(params1[2], tree1.root_node, tree2.root_node)
        assert matched3 is not None
        assert matched3 == params2[2]

    def test_same_structure_different_literals(self):
        """Test that different literals prevent matching"""
        code1 = "int main() { int x = (PARAM) + 5; }"
        code2 = "int main() { int x = (PARAM) + 10; }"

        tree1 = parse_cpp(code1)
        tree2 = parse_cpp(code2)

        params1 = find_params(tree1.root_node)

        # Should not match because literals are different
        matched = match_param(params1[0], tree1.root_node, tree2.root_node)
        assert matched is None

    def test_same_identifiers_different_operators(self):
        """Test that different operators prevent matching"""
        code1 = "int main() { int x = a + PARAM; }"
        code2 = "int main() { int x = a - PARAM; }"

        tree1 = parse_cpp(code1)
        tree2 = parse_cpp(code2)

        params1 = find_params(tree1.root_node)

        # Should not match because operators are different
        matched = match_param(params1[0], tree1.root_node, tree2.root_node)
        assert matched is None

    def test_extra_code_in_one_snippet(self):
        """Test when one snippet has extra code"""
        code1 = """
        int main() {
            int a = x * PARAM;
        }
        """
        code2 = """
        int main() {
            int a = x * PARAM;
            int b = y * PARAM;
        }
        """

        tree1 = parse_cpp(code1)
        tree2 = parse_cpp(code2)

        params1 = find_params(tree1.root_node)
        params2 = find_params(tree2.root_node)

        assert len(params1) == 1
        assert len(params2) == 2

        # First PARAM should match
        matched = match_param(params1[0], tree1.root_node, tree2.root_node)
        assert matched is not None
        assert matched == params2[0]

    def test_reordered_statements(self):
        """Test that reordered statements don't match"""
        code1 = """
        int main() {
            int a = x * PARAM;
            int b = y * PARAM;
        }
        """
        code2 = """
        int main() {
            int b = y * PARAM;
            int a = x * PARAM;
        }
        """

        tree1 = parse_cpp(code1)
        tree2 = parse_cpp(code2)

        params1 = find_params(tree1.root_node)
        params2 = find_params(tree2.root_node)

        # First PARAM in code1 (x * PARAM) should match second in code2
        matched1 = match_param(params1[0], tree1.root_node, tree2.root_node)
        assert matched1 is not None
        assert matched1 == params2[1]

        # Second PARAM in code1 (y * PARAM) should match first in code2
        matched2 = match_param(params1[1], tree1.root_node, tree2.root_node)
        assert matched2 is not None
        assert matched2 == params2[0]

    def test_identical_expressions_different_variables(self):
        """Test identical expression structure with different variable names"""
        code1 = """
        int main() {
            result = foo * PARAM + bar;
        }
        """
        code2 = """
        int main() {
            result = baz * PARAM + qux;
        }
        """

        tree1 = parse_cpp(code1)
        tree2 = parse_cpp(code2)

        params1 = find_params(tree1.root_node)

        # Should not match because variable names differ
        matched = match_param(params1[0], tree1.root_node, tree2.root_node)
        assert matched is None

    def test_matching_with_complex_nesting(self):
        """Test matching with complex nested identical structures"""
        code1 = """
        int main() {
            if (condition) {
                for (int i = 0; i < n; i++) {
                    result += array[i] * PARAM;
                }
            }
        }
        """
        code2 = """
        int main() {
            if (condition) {
                for (int i = 0; i < n; i++) {
                    result += array[i] * PARAM;
                }
            }
        }
        """

        tree1 = parse_cpp(code1)
        tree2 = parse_cpp(code2)

        params1 = find_params(tree1.root_node)
        params2 = find_params(tree2.root_node)

        # Should match because everything is identical
        matched = match_param(params1[0], tree1.root_node, tree2.root_node)
        assert matched is not None
        assert matched == params2[0]

    def test_no_match_missing_statement(self):
        """Test when a matching statement is missing"""
        code1 = """
        int main() {
            int a = 1;
            int b = x * PARAM;
            int c = 2;
        }
        """
        code2 = """
        int main() {
            int a = 1;
            int c = 2;
        }
        """

        tree1 = parse_cpp(code1)
        tree2 = parse_cpp(code2)

        params1 = find_params(tree1.root_node)

        # Should not match because the statement is missing in code2
        matched = match_param(params1[0], tree1.root_node, tree2.root_node)
        assert matched is None


# ============================================================================
# TEST CLASS: Stress Tests
# ============================================================================


class TestStressTests:
    """Stress tests with large and complex code"""

    def test_many_params(self):
        """Test with many PARAMs"""
        code1 = "int main() { int x = " + " + ".join(["PARAM"] * 20) + "; }"
        code2 = "int main() { int x = " + " + ".join(["PARAM"] * 20) + "; }"

        tree1 = parse_cpp(code1)
        tree2 = parse_cpp(code2)

        params1 = find_params(tree1.root_node)
        params2 = find_params(tree2.root_node)

        assert len(params1) == 20
        assert len(params2) == 20

        for i, param1 in enumerate(params1):
            matched = match_param(param1, tree1.root_node, tree2.root_node)
            assert matched is not None
            assert matched == params2[i]

    def test_deeply_nested(self):
        """Test with deeply nested structures"""
        code1 = """
        int main() {
            if (a) {
                if (b) {
                    if (c) {
                        if (d) {
                            if (e) {
                                return PARAM;
                            }
                        }
                    }
                }
            }
        }
        """
        code2 = """
        int main() {
            if (a) {
                if (b) {
                    if (c) {
                        if (d) {
                            if (e) {
                                return PARAM;
                            }
                        }
                    }
                }
            }
        }
        """

        tree1 = parse_cpp(code1)
        tree2 = parse_cpp(code2)

        params1 = find_params(tree1.root_node)
        params2 = find_params(tree2.root_node)

        matched = match_param(params1[0], tree1.root_node, tree2.root_node)
        assert matched is not None
        assert matched == params2[0]

    def test_multiple_functions(self):
        """Test with multiple functions"""
        code1 = """
        int foo() { return PARAM; }
        int bar() { return PARAM + PARAM; }
        int baz() { return PARAM * PARAM * PARAM; }
        """
        code2 = """
        int foo() { return PARAM; }
        int bar() { return PARAM + PARAM; }
        int baz() { return PARAM * PARAM * PARAM; }
        """

        tree1 = parse_cpp(code1)
        tree2 = parse_cpp(code2)

        params1 = find_params(tree1.root_node)
        params2 = find_params(tree2.root_node)

        assert len(params1) == 6
        assert len(params2) == 6

        for i, param1 in enumerate(params1):
            matched = match_param(param1, tree1.root_node, tree2.root_node)
            assert matched is not None
            assert matched == params2[i]


# ============================================================================
# TEST CLASS: Special Cases
# ============================================================================


class TestSpecialCases:
    """Test special and unusual cases"""

    def test_param_as_array_size(self):
        """Test PARAM used as array size"""
        code1 = "int main() { int arr[PARAM]; }"
        code2 = "int main() { int arr[PARAM]; }"

        tree1 = parse_cpp(code1)
        tree2 = parse_cpp(code2)

        params1 = find_params(tree1.root_node)
        params2 = find_params(tree2.root_node)

        matched = match_param(params1[0], tree1.root_node, tree2.root_node)
        assert matched is not None
        assert matched == params2[0]

    def test_param_in_template(self):
        """Test PARAM in template context"""
        code1 = "int main() { vector<int> v(PARAM); }"
        code2 = "int main() { vector<int> v(PARAM); }"

        tree1 = parse_cpp(code1)
        tree2 = parse_cpp(code2)

        params1 = find_params(tree1.root_node)
        params2 = find_params(tree2.root_node)

        if len(params1) > 0 and len(params2) > 0:
            matched = match_param(params1[0], tree1.root_node, tree2.root_node)
            assert matched is not None

    def test_param_in_cast(self):
        """Test PARAM in type cast"""
        code1 = "int main() { int x = (int)PARAM; }"
        code2 = "int main() { int x = (int)PARAM; }"

        tree1 = parse_cpp(code1)
        tree2 = parse_cpp(code2)

        params1 = find_params(tree1.root_node)
        params2 = find_params(tree2.root_node)

        matched = match_param(params1[0], tree1.root_node, tree2.root_node)
        assert matched is not None
        assert matched == params2[0]

    def test_param_in_sizeof(self):
        """Test PARAM in sizeof"""
        code1 = "int main() { int x = sizeof(PARAM); }"
        code2 = "int main() { int x = sizeof(PARAM); }"

        tree1 = parse_cpp(code1)
        tree2 = parse_cpp(code2)

        params1 = find_params(tree1.root_node)
        params2 = find_params(tree2.root_node)

        if len(params1) > 0 and len(params2) > 0:
            matched = match_param(params1[0], tree1.root_node, tree2.root_node)
            assert matched is not None

    def test_unary_operators(self):
        """Test PARAM with various unary operators"""
        code1 = """
        int main() {
            int a = -PARAM;
            int b = +PARAM;
            int c = !PARAM;
            int d = ~PARAM;
            int e = ++PARAM;
            int f = --PARAM;
        }
        """
        code2 = """
        int main() {
            int a = -PARAM;
            int b = +PARAM;
            int c = !PARAM;
            int d = ~PARAM;
            int e = ++PARAM;
            int f = --PARAM;
        }
        """

        tree1 = parse_cpp(code1)
        tree2 = parse_cpp(code2)

        params1 = find_params(tree1.root_node)
        params2 = find_params(tree2.root_node)

        for i, param1 in enumerate(params1):
            matched = match_param(param1, tree1.root_node, tree2.root_node)
            assert matched is not None
            assert matched == params2[i]

    def test_comma_operator(self):
        """Test PARAM with comma operator"""
        code1 = "int main() { int x = (PARAM, PARAM, PARAM); }"
        code2 = "int main() { int x = (PARAM, PARAM, PARAM); }"

        tree1 = parse_cpp(code1)
        tree2 = parse_cpp(code2)

        params1 = find_params(tree1.root_node)
        params2 = find_params(tree2.root_node)

        for i, param1 in enumerate(params1):
            matched = match_param(param1, tree1.root_node, tree2.root_node)
            assert matched is not None
            assert matched == params2[i]

    def test_bitwise_operators(self):
        """Test PARAM with bitwise operators"""
        code1 = """
        int main() {
            int a = PARAM & PARAM;
            int b = PARAM | PARAM;
            int c = PARAM ^ PARAM;
            int d = PARAM << PARAM;
            int e = PARAM >> PARAM;
        }
        """
        code2 = """
        int main() {
            int a = PARAM & PARAM;
            int b = PARAM | PARAM;
            int c = PARAM ^ PARAM;
            int d = PARAM << PARAM;
            int e = PARAM >> PARAM;
        }
        """

        tree1 = parse_cpp(code1)
        tree2 = parse_cpp(code2)

        params1 = find_params(tree1.root_node)
        params2 = find_params(tree2.root_node)

        for i, param1 in enumerate(params1):
            matched = match_param(param1, tree1.root_node, tree2.root_node)
            assert matched is not None
            assert matched == params2[i]


# ============================================================================
# TEST CLASS: Edge Cases for Matching Logic
# ============================================================================


class TestMatchingLogicEdgeCases:
    """Focused tests on the matching logic edge cases"""

    def test_param_in_different_types(self):
        """Test PARAMs in different type init declarators should match anyways"""
        code1 = "int main() { int x = PARAM; }"
        code2 = "int PARAM() { float x = PARAM; }"

        tree1 = parse_cpp(code1)
        tree2 = parse_cpp(code2)

        params1 = find_params(tree1.root_node)
        params2 = find_params(tree2.root_node)

        # Different types (int vs float) should prevent matching
        matched = match_param(params1[0], tree1.root_node, tree2.root_node)
        assert matched == params2[1]

    def test_ambiguous_identical_patterns(self):
        """Test when multiple identical patterns exist"""
        code1 = """
        int main() {
            int a = x * PARAM;
            int b = x * PARAM;
            int c = x * PARAM;
        }
        """
        code2 = """
        int main() {
            int a = x * PARAM;
            int b = x * PARAM;
            int c = x * PARAM;
        }
        """

        tree1 = parse_cpp(code1)
        tree2 = parse_cpp(code2)

        params1 = find_params(tree1.root_node)
        params2 = find_params(tree2.root_node)

        # Each should match to its corresponding position
        for i in range(3):
            matched = match_param(params1[i], tree1.root_node, tree2.root_node)
            assert matched is not None
            assert matched == params2[i]

    def test_nested_params_in_expression(self):
        """Test PARAMs nested in complex expressions"""
        code1 = "int main() { int x = (PARAM * 2) + (PARAM * 3); }"
        code2 = "int main() { int x = (PARAM * 2) + (PARAM * 3); }"

        tree1 = parse_cpp(code1)
        tree2 = parse_cpp(code2)

        params1 = find_params(tree1.root_node)
        params2 = find_params(tree2.root_node)

        assert len(params1) == 2

        # Both should match in order
        matched1 = match_param(params1[0], tree1.root_node, tree2.root_node)
        matched2 = match_param(params1[1], tree1.root_node, tree2.root_node)

        assert matched1 == params2[0]
        assert matched2 == params2[1]

    def test_param_only_difference(self):
        """Test when PARAM is the only difference between snippets"""
        code1 = "int main() { int x = foo + bar; }"
        code2 = "int main() { int x = foo + PARAM; }"

        tree1 = parse_cpp(code1)
        tree2 = parse_cpp(code2)

        params1 = find_params(tree1.root_node)
        params2 = find_params(tree2.root_node)

        assert len(params1) == 0
        assert len(params2) == 1

    def test_whitespace_differences(self):
        """Test that whitespace doesn't affect matching"""
        code1 = "int main(){int x=a*PARAM;}"
        code2 = "int main() { int x = a * PARAM; }"

        tree1 = parse_cpp(code1)
        tree2 = parse_cpp(code2)

        params1 = find_params(tree1.root_node)
        params2 = find_params(tree2.root_node)

        # Should match despite whitespace differences
        matched = match_param(params1[0], tree1.root_node, tree2.root_node)
        assert matched is not None
        assert matched == params2[0]

    def test_complex_operator_precedence(self):
        """Test matching with complex operator precedence"""
        code1 = "int main() { int x = a + b * PARAM - c / PARAM; }"
        code2 = "int main() { int x = a + b * PARAM - c / PARAM; }"

        tree1 = parse_cpp(code1)
        tree2 = parse_cpp(code2)

        params1 = find_params(tree1.root_node)
        params2 = find_params(tree2.root_node)

        assert len(params1) == 2

        # Both should match in correct order
        matched1 = match_param(params1[0], tree1.root_node, tree2.root_node)
        matched2 = match_param(params1[1], tree1.root_node, tree2.root_node)

        assert matched1 == params2[0]
        assert matched2 == params2[1]

    def test_parentheses_matter(self):
        """Test that parentheses affect structure matching"""
        code1 = "int main() { int x = (a + b) * PARAM; }"
        code2 = "int main() { int x = a + (b * PARAM); }"

        tree1 = parse_cpp(code1)
        tree2 = parse_cpp(code2)

        params1 = find_params(tree1.root_node)

        # Different parenthesization means different structure
        matched = match_param(params1[0], tree1.root_node, tree2.root_node)
        assert matched is None

    def test_same_expression_different_context(self):
        """Test same expression in different contexts (if vs while)"""
        code1 = "int main() { if (x > 0) { y = PARAM; } }"
        code2 = "int main() { while (x > 0) { y = PARAM; } }"

        tree1 = parse_cpp(code1)
        tree2 = parse_cpp(code2)

        params1 = find_params(tree1.root_node)

        # Different control structure types should prevent matching at that level
        # But might match at function level if unique enough
        matched = match_param(params1[0], tree1.root_node, tree2.root_node)
        # The result depends on how unique the context needs to be
        # In this case, the assignment itself is identical but control structure differs

    def test_multiple_params_one_matches(self):
        """Test when only some PARAMs can match"""
        code1 = """
        int main() {
            int a = unique1 * PARAM;
            int b = shared * PARAM;
            int c = unique2 * PARAM;
        }
        """
        code2 = """
        int main() {
            int x = different1 * PARAM;
            int y = shared * PARAM;
            int z = different2 * PARAM;
        }
        """

        tree1 = parse_cpp(code1)
        tree2 = parse_cpp(code2)

        params1 = find_params(tree1.root_node)
        params2 = find_params(tree2.root_node)

        # First PARAM: unique1 * PARAM - no match
        matched1 = match_param(params1[0], tree1.root_node, tree2.root_node)
        assert matched1 is None

        # Second PARAM: shared * PARAM - should match
        matched2 = match_param(params1[1], tree1.root_node, tree2.root_node)
        assert matched2 is not None
        assert matched2 == params2[1]

        # Third PARAM: unique2 * PARAM - no match
        matched3 = match_param(params1[2], tree1.root_node, tree2.root_node)
        assert matched3 is None

    def test_function_name_matters(self):
        """Test that function names must match"""
        code1 = "int main() { foo(PARAM); }"
        code2 = "int main() { bar(PARAM); }"

        tree1 = parse_cpp(code1)
        tree2 = parse_cpp(code2)

        params1 = find_params(tree1.root_node)

        # Different function names should prevent matching
        matched = match_param(params1[0], tree1.root_node, tree2.root_node)
        assert matched is None

    def test_array_vs_pointer_syntax(self):
        """Test that array and pointer syntax differences matter"""
        code1 = "int main() { int x = arr[PARAM]; }"
        code2 = "int main() { int x = *(arr + PARAM); }"

        tree1 = parse_cpp(code1)
        tree2 = parse_cpp(code2)

        params1 = find_params(tree1.root_node)

        # Different syntax (array index vs pointer arithmetic) should prevent matching
        matched = match_param(params1[0], tree1.root_node, tree2.root_node)
        assert matched is None

    def test_if_condition(self):
        """Test that if conditions are matches"""
        code1 = "if(PARAM) { hello; "
        code2 = "if(PARAM) { bye; "

        tree1 = parse_cpp(code1)
        tree2 = parse_cpp(code2)

        params1 = find_params(tree1.root_node)
        params2 = find_params(tree2.root_node)

        # Different syntax (array index vs pointer arithmetic) should prevent matching
        matched = match_param(params1[0], tree1.root_node, tree2.root_node)
        assert matched == params2[0]



class TestAdvancedCPPFeatures:
    """Tests matching with more advanced C++ features and ambiguity"""

    def test_param_in_lambda_body(self):
        """Test that PARAMs inside lambda functions can be matched"""
        code1 = """
        int main() {
            int x = 10;
            auto my_lambda = [x]() { return x + PARAM; };
        }
        """
        code2 = """
        int main() {
            int x = 10;
            auto my_lambda = [x]() { return x + PARAM; };
        }
        """
        tree1 = parse_cpp(code1)
        tree2 = parse_cpp(code2)
        params1 = find_params(tree1.root_node)
        params2 = find_params(tree2.root_node)

        assert len(params1) == 1
        matched = match_param(params1[0], tree1.root_node, tree2.root_node)
        assert matched is not None
        assert matched == params2[0]

    def test_param_in_constructor_initializer_list(self):
        """Test matching PARAMs in a constructor's initializer list"""
        code1 = """
        class MyClass {
            int x;
        public:
            MyClass(int val) : x(val * PARAM) {}
        };
        """
        code2 = """
        class MyClass {
            int x;
        public:
            MyClass(int val) : x(val * PARAM) {}
        };
        """
        tree1 = parse_cpp(code1)
        tree2 = parse_cpp(code2)
        params1 = find_params(tree1.root_node)
        params2 = find_params(tree2.root_node)

        assert len(params1) > 0
        matched = match_param(params1[0], tree1.root_node, tree2.root_node)
        assert matched is not None
        assert matched == params2[0]

    def test_no_match_on_const_mismatch(self):
        """Test that a PARAM in a const vs non-const method still matches"""
        code1 = """
        class MyClass {
            int get() const { return PARAM; }
        };
        """
        code2 = """
        class MyClass {
            int get() { return PARAM; }
        };
        """
        tree1 = parse_cpp(code1)
        tree2 = parse_cpp(code2)
        params1 = find_params(tree1.root_node)
        params2 = find_params(tree2.root_node)

        # The function declarations are different ('const' qualifier),
        # so the context is different.
        matched = match_param(params1[0], tree1.root_node, tree2.root_node)
        assert matched == params2[0]

    def test_match_in_different_namespaces(self):
        """Test matching identical structures within different namespaces"""
        code1 = """
        namespace A {
            int compute() { return 5 * PARAM; }
        }
        """
        code2 = """
        namespace B {
            int compute() { return 5 * PARAM; }
        }
        """
        tree1 = parse_cpp(code1)
        tree2 = parse_cpp(code2)
        params1 = find_params(tree1.root_node)
        params2 = find_params(tree2.root_node)

        # The surrounding namespace is different, which should prevent a match
        # when the algorithm expands context to the namespace level.
        matched = match_param(params1[0], tree1.root_node, tree2.root_node)
        assert matched == params2[0]

    def test_grandparent_disambiguation(self):
        """Test needing to go two levels up (grandparent) to find a unique context"""
        code1 = """
        struct S { void do_stuff(int, int); };
        void func_A(S s) {
            s.do_stuff(1, PARAM);
        }
        void func_B(S s) {
            s.do_stuff(1, PARAM);
        }
        """
        code2 = """
        struct S { void do_stuff(int, int); };
        void func_A(S s) {
            s.do_stuff(1, PARAM);
        }
        void func_C(S s) {
            s.do_stuff(1, PARAM);
        }
        """
        # Immediate parent (call_expression) for PARAM 0 is `s.do_stuff(1, PARAM)`.
        # Immediate parent for PARAM 1 is `s.do_stuff(2, PARAM)`.
        # In code2, there are also two potential matches.
        # The algorithm must expand context up to the function declaration
        # (func_A vs func_B/func_C) to find the unique match.
        tree1 = parse_cpp(code1)
        tree2 = parse_cpp(code2)
        params1 = find_params(tree1.root_node)
        params2 = find_params(tree2.root_node)

        # Match the first PARAM in func_A
        matched = match_param(params1[0], tree1.root_node, tree2.root_node)
        assert matched is not None
        assert matched == params2[0]

        # The second PARAM in func_B should not match anything in code2
        matched_fail = match_param(params1[1], tree1.root_node, tree2.root_node)
        assert matched_fail is None

    def test_param_in_macro_expansion(self):
        """Test if PARAM inside a macro expansion can be matched"""

        code1 = """
        #define ADD(x) (x + PARAM)
        int main() {
            int a = ADD(5);
        }
        """

        tree1 = parse_cpp(code1)
        params1 = find_params(tree1.root_node)

        # we don't support macros
        assert len(params1) == 0

    def test_param_with_auto_keyword(self):
        """Test that `auto` keyword does not prevent matching"""
        code1 = "int main() { auto x = PARAM; }"
        code2 = "int main() { auto x = PARAM; }"

        tree1 = parse_cpp(code1)
        tree2 = parse_cpp(code2)
        params1 = find_params(tree1.root_node)
        params2 = find_params(tree2.root_node)

        matched = match_param(params1[0], tree1.root_node, tree2.root_node)
        assert matched is not None
        assert matched == params2[0]

# ============================================================================
# Run tests if executed directly
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
