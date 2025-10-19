"""
Step 1: Basic Tree-sitter C++ Parser
This script demonstrates how to parse C++ code and print the AST
"""

from tree_sitter import Language, Parser
import tree_sitter_cpp as tscpp

# Step 1: Create a parser
print("Step 1: Creating parser...")
CPP_LANGUAGE = Language(tscpp.language())
parser = Parser(CPP_LANGUAGE)

# Step 3: Define some C++ code to parse
cpp_code = """
int main() {
    int x = 5 * 3;
    float y = x + 2.5;
    return 0;
}
"""

print("\n" + "="*50)
print("Parsing this C++ code:")
print("="*50)
print(cpp_code)
print("="*50)

# Step 4: Parse the code
# tree-sitter expects bytes, not strings
print("\nStep 4: Parsing...")
tree = parser.parse(bytes(cpp_code, "utf8"))

# Step 5: Get the root node
root_node = tree.root_node

print("\nStep 5: Getting root node...")
print(f"Root node type: {root_node.type}")
print(f"Root node has {root_node.child_count} children")

# Step 6: Print the S-expression (tree structure)
print("\n" + "="*50)
print("Full S-expression (AST structure):")
print("="*50)
print(str(root_node))

# Step 7: Let's walk through the tree manually
print("\n" + "="*50)
print("Walking through the tree:")
print("="*50)

def print_tree(node, indent=0):
    """Recursively print the tree structure"""
    # Print current node
    indent_str = "  " * indent
    node_text = node.text.decode('utf8') if node.text else ""

    # Truncate long text for readability
    if len(node_text) > 50:
        node_text = node_text[:50] + "..."

    # Replace newlines with space for display
    node_text = node_text.replace('\n', ' ')

    print(f"{indent_str}{node.type}", end="")
    if node_text:
        print(f' "{node_text}"', end="")

    # Show if it's a named node
    if not node.is_named:
        print(" [anonymous]", end="")

    print()  # newline

    # Print children
    for child in node.children:
        print_tree(child, indent + 1)

print_tree(root_node)

# Step 8: Let's examine a specific interesting node
print("\n" + "="*50)
print("Examining specific nodes:")
print("="*50)

# Find the function definition
function_def = root_node.children[0]  # Should be function_definition
print(f"\nFirst child type: {function_def.type}")
print(f"First child text: {function_def.text.decode('utf8')[:100]}...")

# Get the function body
body = None
for child in function_def.children:
    if child.type == "compound_statement":
        body = child
        break

if body:
    print(f"\nFunction body type: {body.type}")
    print(f"Function body has {body.child_count} children")

    # Look at statements in the body
    print("\nStatements in function body:")
    for i, child in enumerate(body.children):
        if child.is_named:  # Skip braces
            print(f"  Statement {i}: {child.type}")
            print(f"    Text: {child.text.decode('utf8')}")
