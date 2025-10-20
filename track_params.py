#!/usr/bin/env python3
"""
Track numeric parameter changes in C++ code across git history.
"""

import argparse
import subprocess
from typing import List, Dict, Tuple
from tree_sitter import Language, Parser
import tree_sitter_cpp as tscpp

# Import matching functions from main.py
from main import find_params, match_param


def get_file_lines(
    file_path: str, start_line: int, end_line: int, commit: str = "HEAD"
) -> str:
    """Get specific lines from a file at a given commit."""
    try:
        result = subprocess.run(
            ["git", "show", f"{commit}:{file_path}"],
            capture_output=True,
            text=True,
            check=True,
        )
        lines = result.stdout.split("\n")
        # Git line numbers are 1-indexed
        return "\n".join(lines[start_line - 1 : end_line])
    except subprocess.CalledProcessError:
        return ""


def get_git_log_range(
    file_path: str, start_line: int, end_line: int, max_commits: int
) -> List[str]:
    """Get git log for a specific line range."""
    try:
        result = subprocess.run(
            [
                "git",
                "log",
                f"-L{start_line},{end_line}:{file_path}",
                f"--max-count={max_commits}",
                "--format=%H",
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        commits = []
        lines = result.stdout.split("\n")
        current_commit = None

        for line in lines:
            if (
                line
                and not line.startswith("diff")
                and not line.startswith("---")
                and not line.startswith("+++")
                and not line.startswith("@@")
                and not line.startswith("+")
                and not line.startswith("-")
                and not line.startswith(" ")
            ):
                # This is a commit hash
                current_commit = line.strip()
                if current_commit:
                    commits.append(current_commit)

        return commits
    except subprocess.CalledProcessError as e:
        print(f"Error getting git log: {e}")
        return []


def extract_numbers_and_create_param_code(
    code: str,
) -> Tuple[str, List[Tuple[str, int, int]]]:
    """
    Extract numeric literals from code and replace them with PARAM identifiers.

    Returns:
        - Code with numbers replaced by PARAM
        - List of (original_value, start_byte, end_byte) tuples in order
    """
    CPP_LANGUAGE = Language(tscpp.language())
    parser = Parser(CPP_LANGUAGE)

    tree = parser.parse(bytes(code, "utf8"))

    # Find all numeric literals
    numbers = []

    def extract_numbers_from_node(node):
        """Extract all numeric literals from a node and its children."""
        if node.type in ("number_literal", "integer_literal", "float_literal"):
            numbers.append(
                {
                    "value": node.text.decode("utf8"),
                    "start": node.start_byte,
                    "end": node.end_byte,
                }
            )

        for child in node.children:
            extract_numbers_from_node(child)

    extract_numbers_from_node(tree.root_node)

    # Sort by position (reverse order for replacement)
    numbers.sort(key=lambda x: x["start"], reverse=True)

    # Replace numbers with PARAM
    code_bytes = code.encode("utf8")
    for num in numbers:
        code_bytes = code_bytes[: num["start"]] + b"PARAM" + code_bytes[num["end"] :]

    param_code = code_bytes.decode("utf8")

    # Return in forward order for tracking
    numbers.reverse()
    number_values = [(n["value"], n["start"], n["end"]) for n in numbers]

    return param_code, number_values


def track_parameters(
    file_path: str, start_line: int, end_line: int, max_commits: int, verbose: bool = False
) -> Tuple:
    """Track parameter changes across git history using tree-based matching."""

    # Get the original snippet from HEAD
    original_code = get_file_lines(file_path, start_line, end_line, "HEAD")
    if not original_code:
        print(f"Error: Could not read lines {start_line}-{end_line} from {file_path}")
        return None

    # Extract numbers and create PARAM version
    original_param_code, original_numbers = extract_numbers_and_create_param_code(
        original_code
    )

    # Parse the PARAM version
    CPP_LANGUAGE = Language(tscpp.language())
    parser = Parser(CPP_LANGUAGE)
    original_tree = parser.parse(bytes(original_param_code, "utf8"))

    # Find all PARAM nodes in original code
    original_params = find_params(original_tree.root_node)

    if len(original_params) != len(original_numbers):
        print(
            f"Warning: Number of PARAMs ({len(original_params)}) doesn't match extracted numbers ({len(original_numbers)})"
        )

    # Get git history
    commits = get_git_log_range(file_path, start_line, end_line, max_commits)

    if not commits:
        print("No commits found in history")
        return None

    print(
        f"Tracking {len(commits)} commits with {len(original_params)} parameters...\n"
    )

    # Initialize parameter tracking
    param_history = {}
    param_map = {}  # Maps PARAM node id to PARAM_N identifier

    # Create initial mapping
    for idx, (param_node, (value, _, _)) in enumerate(
        zip(original_params, original_numbers)
    ):
        param_id = f"PARAM_{idx}"
        param_map[id(param_node)] = param_id
        param_history[param_id] = [value]  # Start with HEAD value

    # Track through history
    for commit_idx, commit in enumerate(commits[1:], 1):  # Skip first commit (HEAD)
        print(f"\n{'='*80}")
        print(f"COMMIT {commit_idx}/{len(commits)-1}: {commit[:8]}")
        print(f"{'='*80}")
        
        commit_code = get_file_lines(file_path, start_line, end_line, str(commit))
        if not commit_code:
            print(f"  File or lines don't exist at this commit, stopping tracking")
            break
        
        # Show the commit code
        print(f"\nCode in this commit:")
        print("-" * 80)
        print(commit_code)
        print("-" * 80)
        
        # Extract numbers and create PARAM version for commit
        commit_param_code, commit_numbers = extract_numbers_and_create_param_code(commit_code)
        commit_tree = parser.parse(bytes(commit_param_code, "utf8"))
        commit_params = find_params(commit_tree.root_node)
        
        print(f"\nFound {len(commit_params)} parameters in commit version")
        print(f"Commit parameter values: {[n[0] for n in commit_numbers]}")
        
        # Match each original PARAM to commit PARAM
        matched_count = 0
        param_matches = {}  # Store matches for display
        
        for orig_param_node in original_params:
            param_id = param_map[id(orig_param_node)]
            
            # Use the sophisticated matching algorithm from main.py
            matched_param = match_param(
                orig_param_node,
                original_tree.root_node,
                commit_tree.root_node,
                max_levels=10,
                verbose=verbose
            )
            
            if matched_param:
                # Find the index of matched param in commit_params
                try:
                    commit_param_idx = commit_params.index(matched_param)
                    if commit_param_idx < len(commit_numbers):
                        value = commit_numbers[commit_param_idx][0]
                        param_history[param_id].append(value)
                        param_matches[param_id] = value
                        matched_count += 1
                    else:
                        print(f"  Warning: {param_id} matched but index out of range")
                        param_matches[param_id] = "ERROR: index out of range"
                except ValueError:
                    print(f"  Warning: {param_id} matched param not in commit_params list")
                    param_matches[param_id] = "ERROR: not in list"
            else:
                param_matches[param_id] = "NOT MATCHED"
        
        # Display matching results
        print(f"\nParameter Matching Results:")
        print("-" * 80)
        for param_id in sorted(param_matches.keys(), key=lambda x: int(x.split('_')[1])):
            match_result = param_matches[param_id]
            if match_result == "NOT MATCHED":
                print(f"  {param_id}: ✗ NOT MATCHED")
            else:
                print(f"  {param_id}: ✓ matched to value {match_result}")
        
        print(f"\nSummary: Successfully matched {matched_count}/{len(original_params)} parameters")

    return (
        original_code,
        original_param_code,
        original_params,
        original_numbers,
        param_map,
        param_history,
    )


def create_annotated_code(
    original_code: str,
    original_numbers: List[Tuple[str, int, int]],
    param_map: Dict,
    original_params: List,
) -> str:
    """Create version of code with parameters replaced by identifiers."""
    # Sort by position (reverse order for replacement to maintain byte positions)
    replacements = []
    for param_node, (value, start, end) in zip(original_params, original_numbers):
        param_id = param_map[id(param_node)]
        replacements.append((start, end, value, param_id))

    replacements.sort(key=lambda x: x[0], reverse=True)

    # Replace numbers with PARAM_N identifiers
    code_bytes = original_code.encode("utf8")
    for start, end, value, param_id in replacements:
        code_bytes = code_bytes[:start] + param_id.encode("utf8") + code_bytes[end:]

    return code_bytes.decode("utf8")


def main():
    parser = argparse.ArgumentParser(
        description="Track numeric parameter changes in C++ code across git history"
    )
    parser.add_argument("--file", required=True, help="Path to the C++ file")
    parser.add_argument(
        "--start-line", type=int, required=True, help="Start line number (1-indexed)"
    )
    parser.add_argument(
        "--end-line", type=int, required=True, help="End line number (inclusive)"
    )
    parser.add_argument(
        "--max-commits", type=int, default=10, help="Maximum number of commits to track"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Show detailed matching debug output"
    )

    args = parser.parse_args()

    result = track_parameters(
        args.file, args.start_line, args.end_line, args.max_commits, args.verbose
    )

    if result:
        (
            original_code,
            original_param_code,
            original_params,
            original_numbers,
            param_map,
            param_history,
        ) = result

        # Create annotated version
        annotated = create_annotated_code(
            original_code, original_numbers, param_map, original_params
        )

        print("\n" + "=" * 80)
        print("ORIGINAL CODE:")
        print("=" * 80)
        print(original_code)
        print()

        print("=" * 80)
        print("ANNOTATED CODE (parameters replaced with identifiers):")
        print("=" * 80)
        print(annotated)
        print()

        print("=" * 80)
        print("PARAMETER HISTORY (newest to oldest):")
        print("=" * 80)
        for param_id in sorted(
            param_history.keys(), key=lambda x: int(x.split("_")[1])
        ):
            values = param_history[param_id]
            print(f"{param_id}: {values}")
        print()


if __name__ == "__main__":
    main()
