"""
Step 3: Full PARAM Matching Algorithm
Matches PARAMs between two code snippets using growing subtrees
"""

from tree_sitter import Language, Parser, Query
import tree_sitter_cpp as tscpp

# Setup parser
CPP_LANGUAGE = Language(tscpp.language())
parser = Parser(CPP_LANGUAGE)


# ============================================================================
# REUSABLE FUNCTIONS
# ============================================================================


def find_params(root):
    """Find all PARAM nodes by walking the tree in deterministic order"""
    params = []

    def traverse(node):
        """Depth-first traversal to find PARAMs in order"""
        # Check if this node is a PARAM identifier
        if node.type == "identifier":
            text = node.text.decode("utf8") if node.text else ""
            if text == "PARAM":
                params.append(node)

        # Traverse children in order
        for child in node.children:
            traverse(child)

    traverse(root)
    return params


def nodes_equal(node1, node2, mask_params=True):
    """
    Recursively compare two nodes for structural and textual equality
    """
    # Check if types match
    if node1.type != node2.type:
        return False

    # Check if named status matches
    if node1.is_named != node2.is_named:
        return False

    # For leaf nodes, compare text
    if node1.child_count == 0 and node2.child_count == 0:
        text1 = node1.text.decode("utf8") if node1.text else ""
        text2 = node2.text.decode("utf8") if node2.text else ""

        # If masking PARAMs, treat all PARAMs as equal
        if mask_params and text1 == "PARAM" and text2 == "PARAM":
            return True

        # For identifiers and literals, text must match
        if node1.type in [
            "identifier",
            "number_literal",
            "string_literal",
            "primitive_type",
            "type_identifier",
        ]:
            return text1 == text2

        # For other leaf nodes (operators, keywords), type match is enough
        # or exact text match for anonymous nodes
        if not node1.is_named:  # anonymous nodes like operators
            return text1 == text2

        return True

    # Check if child counts match
    if node1.child_count != node2.child_count:
        return False

    # Recursively compare all children
    for child1, child2 in zip(node1.children, node2.children):
        if not nodes_equal(child1, child2, mask_params):
            return False

    return True


def count_params_in_subtree(node):
    """Count how many PARAMs are in a subtree"""
    text = node.text.decode("utf8") if node.text else ""
    count = 1 if text == "PARAM" else 0

    for child in node.children:
        count += count_params_in_subtree(child)

    return count


def find_matching_subtrees(target_node, search_root):
    """
    Find all nodes in search_root that match target_node's structure
    """
    matches = []

    def traverse(node):
        if nodes_equal(target_node, node):
            matches.append(node)

        for child in node.children:
            traverse(child)

    traverse(search_root)
    return matches


print("\nHelper functions defined:")
print("  - count_params_in_subtree()")
print("  - find_matching_subtrees()")

print("\n" + "=" * 60)
print("STEP 4: Implement the matching algorithm")
print("=" * 60)


def get_params_in_subtree(node):
    """Get all PARAM nodes in a subtree, in order"""
    params = []

    def traverse(n):
        text = n.text.decode("utf8") if n.text else ""
        if text == "PARAM":
            params.append(n)

        for child in n.children:
            traverse(child)

    traverse(node)
    return params


def has_binary_operator_or_bounded_context(node):
    """
    Check if a subtree contains at least one binary operator OR
    represents a bounded/limited context (statement, condition, etc.)
    """
    # Bounded contexts that are "complete" units
    bounded_context_types = [
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
    ]

    # Check if this node itself is a bounded context
    if node.type in bounded_context_types:
        return True

    # Check for binary operators
    binary_operator_types = [
        "binary_expression",
        "+", "-", "*", "/", "%",
        "==", "!=", "<", ">", "<=", ">=",
        "&&", "||", "&", "|", "^",
        "<<", ">>",
    ]

    def traverse(n):
        if n.type in binary_operator_types:
            return True
        for child in n.children:
            if traverse(child):
                return True
        return False

    return traverse(node)


def match_param(param_node, snippet1_root, snippet2_root, max_levels=10):
    """
    Find the corresponding PARAM in snippet2 for a given PARAM in snippet1

    Returns:
        - matching PARAM node from snippet2 if found
        - None if no match found
    """
    current_context = param_node.parent
    level = 0

    # First, increase the initial level until we include at least one binary operator
    # or reach a bounded context (like a statement)
    print(f"\n  Starting from PARAM at line {param_node.start_point[0] + 1}")
    print(f"  Growing context to include a binary operator or bounded context...")

    while current_context and not has_binary_operator_or_bounded_context(current_context):
        print(f"    Level {level}: {current_context.type} (growing...)")
        current_context = current_context.parent
        level += 1
        if level >= max_levels:
            print(f"    ✗ Reached max levels without finding suitable context")
            return None

    if current_context:
        print(f"    Found suitable context at level {level}: {current_context.type}")

    print(f"\n  Starting matching from level {level}")

    while current_context and level < max_levels:
        print(f"\n  Level {level}: {current_context.type}")
        print(f"    Text: {current_context.text.decode('utf8')[:50]}...")

        # Count PARAMs in current context
        params_in_context = get_params_in_subtree(current_context)
        print(f"    PARAMs in this subtree: {len(params_in_context)}")

        # Find matches in snippet1 (should be unique if we grew enough)
        matches_in_snippet1 = find_matching_subtrees(current_context, snippet1_root)
        print(f"    Matches in snippet 1: {len(matches_in_snippet1)}")

        # Find matches in snippet2
        matches_in_snippet2 = find_matching_subtrees(current_context, snippet2_root)
        print(f"    Matches in snippet 2: {len(matches_in_snippet2)}")

        # Check if we have unique match in both snippets
        if len(matches_in_snippet1) == 1 and len(matches_in_snippet2) == 1:
            # We have unique matching subtrees!
            # Find PARAMs in both subtrees
            params_in_match1 = get_params_in_subtree(matches_in_snippet1[0])
            params_in_match2 = get_params_in_subtree(matches_in_snippet2[0])

            print(f"    Unique match found!")
            print(f"    PARAMs in match 1: {len(params_in_match1)}")
            print(f"    PARAMs in match 2: {len(params_in_match2)}")

            # Check that both have the same number of PARAMs
            if len(params_in_match1) != len(params_in_match2):
                print(f"    ✗ Different number of PARAMs in matched subtrees!")
                return None

            # Find the position of our param_node in the match1 PARAMs
            try:
                param_index = params_in_match1.index(param_node)
                print(
                    f"    Our PARAM is at position {param_index} in the matched subtree"
                )

                # Return the PARAM at the same position in match2
                matched_param = params_in_match2[param_index]
                print(f"    ✓ FOUND MATCH by position!")
                return matched_param
            except ValueError:
                print(
                    f"    ✗ Our PARAM not found in matched subtree (shouldn't happen)"
                )
                return None

        elif len(matches_in_snippet2) == 0:
            print(f"    ✗ No match in snippet 2 - this PARAM has no counterpart")
            return None

        else:
            print(f"    Ambiguous - need to grow context")

        # Grow the context
        current_context = current_context.parent
        level += 1

    # If we reached max levels without finding a match
    print(f"    ✗ Reached max levels without unique match")
    return None


# ============================================================================
# DEMO CODE (runs when script is executed directly)
# ============================================================================


def main():
    """Main demonstration function"""
    # Two code snippets with anonymized parameters
    cpp_snippet_1 = """
int main() {
    int a = b * PARAM;
    int x = c * PARAM;
    float y = PARAM + PARAM;
    return PARAM;
}
"""

    cpp_snippet_2 = """
int main() {
    int a = b * PARAM;
    int x = d * PARAM;
    float y = PARAM + PARAM;
    return PARAM;
}
"""

    print("=" * 60)
    print("CODE SNIPPET 1:")
    print("=" * 60)
    print(cpp_snippet_1)

    print("=" * 60)
    print("CODE SNIPPET 2:")
    print("=" * 60)
    print(cpp_snippet_2)

    # Parse both snippets
    tree1 = parser.parse(bytes(cpp_snippet_1, "utf8"))
    tree2 = parser.parse(bytes(cpp_snippet_2, "utf8"))

    print("=" * 60)
    print("STEP 1: Find all PARAMs in both snippets")
    print("=" * 60)

    params1 = find_params(tree1.root_node)
    params2 = find_params(tree2.root_node)

    print(f"\nSnippet 1: Found {len(params1)} PARAMs")
    for i, param in enumerate(params1):
        context = param.parent.text.decode("utf8") if param.parent else "?"
        print(f"  PARAM {i + 1}: line {param.start_point[0] + 1}, context: {context}")

    print(f"\nSnippet 2: Found {len(params2)} PARAMs")
    for i, param in enumerate(params2):
        context = param.parent.text.decode("utf8") if param.parent else "?"
        print(f"  PARAM {i + 1}: line {param.start_point[0] + 1}, context: {context}")

    print("\n" + "=" * 60)
    print("STEP 2: Define node comparison function")
    print("=" * 60)
    print("\nNode comparison function defined ✓")

    print("\n" + "=" * 60)
    print("STEP 3: Define helper functions")
    print("=" * 60)
    print("\nHelper functions defined:")
    print("  - count_params_in_subtree()")
    print("  - find_matching_subtrees()")

    print("\n" + "=" * 60)
    print("STEP 4: Implement the matching algorithm")
    print("=" * 60)
    print("\nMatching algorithm defined ✓")

    print("\n" + "=" * 60)
    print("STEP 5: Match each PARAM from snippet 1 to snippet 2")
    print("=" * 60)

    matches = {}

    for i, param1 in enumerate(params1):
        print(f"\n{'=' * 60}")
        print(f"Processing PARAM {i + 1} from snippet 1:")
        print(f"{'=' * 60}")

        matched_param = match_param(param1, tree1.root_node, tree2.root_node)

        if matched_param:
            matches[i] = matched_param
            # Find its index in params2
            match_index = (
                params2.index(matched_param) if matched_param in params2 else -1
            )
            print(f"\n✓ RESULT: Matched to PARAM {match_index + 1} in snippet 2")
        else:
            print(f"\n✗ RESULT: No match found")

    print("\n" + "=" * 60)
    print("FINAL RESULTS:")
    print("=" * 60)

    print("\nPARAM Mapping (Snippet 1 → Snippet 2):")
    for i in range(len(params1)):
        if i in matches:
            match_index = params2.index(matches[i])
            context1 = params1[i].parent.text.decode("utf8")
            context2 = params2[match_index].parent.text.decode("utf8")
            print(f"  PARAM {i + 1} → PARAM {match_index + 1}")
            print(f"    Context 1: {context1}")
            print(f"    Context 2: {context2}")
        else:
            context1 = params1[i].parent.text.decode("utf8")
            print(f"  PARAM {i + 1} → NO MATCH")
            print(f"    Context 1: {context1}")


if __name__ == "__main__":
    main()
