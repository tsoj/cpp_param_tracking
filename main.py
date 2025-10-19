"""
Step 2: Finding Specific Nodes
This script demonstrates how to find all PARAMs in C++ code
"""

from tree_sitter import Language, Parser, Query
import tree_sitter_cpp as tscpp

# Setup parser
CPP_LANGUAGE = Language(tscpp.language())
parser = Parser(CPP_LANGUAGE)

# C++ code with PARAM placeholders (simulating anonymized parameters)
cpp_code = """
int main() {
    int x = PARAM * PARAM;
    float y = x + PARAM;
    int z = PARAM;
    return PARAM;
}
"""

print("Parsing this C++ code:")
print("="*50)
print(cpp_code)
print("="*50)

# Parse the code
tree = parser.parse(bytes(cpp_code, "utf8"))
root_node = tree.root_node

print("\n" + "="*50)
print("METHOD 1: Manual traversal to find PARAMs")
print("="*50)

def find_params_manual(node, params_list):
    """Recursively find all PARAM nodes"""
    # Check if this node's text is "PARAM"
    if node.text and node.text.decode('utf8') == 'PARAM':
        params_list.append(node)

    # Recurse into children
    for child in node.children:
        find_params_manual(child, params_list)

params_manual = []
find_params_manual(root_node, params_manual)

print(f"\nFound {len(params_manual)} PARAMs using manual traversal:")
for i, param in enumerate(params_manual):
    print(f"\n  PARAM #{i+1}:")
    print(f"    Type: {param.type}")
    print(f"    Text: {param.text.decode('utf8')}")
    print(f"    Position: line {param.start_point[0] + 1}, column {param.start_point[1]}")
    print(f"    Parent type: {param.parent.type if param.parent else 'None'}")

print("\n" + "="*50)
print("METHOD 2: Using tree-sitter queries")
print("="*50)

# Create a query to find identifiers
# In tree-sitter, PARAM will be parsed as an identifier
query_string = """
(identifier) @param
"""

print(f"Query pattern: {query_string.strip()}")

# Compile the query
query = Query(CPP_LANGUAGE, query_string)

# Execute the query
captures = query.captures(root_node)

print(f"\nFound {len(captures)} identifier captures")

# Filter to only actual PARAMs
params_query = []

for capture_name, nodes in captures.items():
    print(f"Capture: {capture_name}")
    for node in nodes:
        if node.text.decode('utf8') == 'PARAM':
            params_query.append(node)

print(f"Filtered to {len(params_query)} actual PARAMs:")
for i, param in enumerate(params_query):
    print(f"\n  PARAM #{i+1}:")
    print(f"    Capture name: @param")
    print(f"    Type: {param.type}")
    print(f"    Text: {param.text.decode('utf8')}")

print("\n" + "="*50)
print("METHOD 3: Understanding node context")
print("="*50)

# Let's look at the context around each PARAM
for i, param in enumerate(params_manual):
    print(f"\nPARAM #{i+1} context:")
    print(f"  Node type: {param.type}")

    # Look at parent
    if param.parent:
        print(f"  Parent type: {param.parent.type}")
        print(f"  Parent text: {param.parent.text.decode('utf8')}")

    # Look at grandparent
    if param.parent and param.parent.parent:
        print(f"  Grandparent type: {param.parent.parent.type}")

    # Look at siblings
    if param.parent:
        siblings = param.parent.children
        param_index = siblings.index(param)
        print(f"  Position among siblings: {param_index} of {len(siblings)}")

        print(f"  Siblings:")
        for j, sibling in enumerate(siblings):
            marker = " <-- THIS PARAM" if sibling == param else ""
            sib_text = sibling.text.decode('utf8').replace('\n', ' ')[:30]
            print(f"    [{j}] {sibling.type}: \"{sib_text}\"{marker}")

print("\n" + "="*50)
print("METHOD 4: Getting context subtrees")
print("="*50)

# For each PARAM, show progressively larger contexts
param = params_manual[0]  # Let's focus on the first PARAM: "PARAM * PARAM"

print(f"Analyzing first PARAM: {param.text.decode('utf8')}")
print(f"Position: line {param.start_point[0] + 1}, column {param.start_point[1]}")

# Show growing context
current = param
level = 0
while current and level < 5:
    print(f"\nContext level {level} ({current.type}):")
    print(f"  Text: {current.text.decode('utf8')}")
    print(f"  S-expression:")
    # Show indented S-expression
    sexp_lines = str(current).split('\n')
    for line in sexp_lines[:10]:  # Limit lines for readability
        print(f"    {line}")
    if len(sexp_lines) > 10:
        print(f"    ... ({len(sexp_lines) - 10} more lines)")

    current = current.parent
    level += 1

print("\n" + "="*50)
print("KEY TAKEAWAYS:")
print("="*50)
print("""
1. Manual traversal: Simple but you control everything
2. Queries: More powerful, but need to filter results
3. Context: Use .parent to grow subtrees upward
4. S-expression: str(node) gives you the structure
5. Every node has: type, text, parent, children, start_point, end_point
""")
