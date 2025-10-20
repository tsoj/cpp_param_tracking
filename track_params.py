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
) -> List[Tuple[str, int, int]]:
    """
    Get git log for a specific line range, tracking how line numbers change.
    
    Returns:
        List of (commit_hash, start_line, end_line) tuples
    """
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

        commits_with_ranges = []
        lines = result.stdout.split("\n")
        current_commit = None
        current_start = start_line
        current_end = end_line
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Check if this is a commit hash line
            if line and not any(line.startswith(prefix) for prefix in 
                               ["diff", "---", "+++", "@@", "+", "-", " ", "\\", "index"]):
                # This is a commit hash
                current_commit = line.strip()
                if current_commit:
                    commits_with_ranges.append((current_commit, current_start, current_end))
            
            # Check for @@ markers to track line range changes
            elif line.startswith("@@"):
                # Parse the hunk header: @@ -old_start,old_count +new_start,new_count @@
                # The +new_start,new_count tells us where the lines are in this commit
                import re
                match = re.search(r'@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@', line)
                if match:
                    new_start = int(match.group(1))
                    new_count = int(match.group(2)) if match.group(2) else 1
                    
                    # Update the range for the next (older) commit
                    # We read forward to see how many context/removed lines there are
                    context_and_removed = 0
                    j = i + 1
                    while j < len(lines) and not lines[j].startswith("@@") and not lines[j].startswith("diff"):
                        if lines[j].startswith(" ") or lines[j].startswith("-"):
                            context_and_removed += 1
                        j += 1
                    
                    # For the previous commit, the lines are at the + position
                    current_start = new_start
                    current_end = new_start + new_count - 1
            
            i += 1
        
        return commits_with_ranges
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


def is_valid_parameter_change(old_value: str, new_value: str) -> bool:
    """
    Check if a parameter change is valid based on:
    - Absolute change <= 2 (for integers)
    - OR relative change <= 20%

    Returns True if the change is acceptable, False otherwise.
    """
    try:
        # Try to parse as float to handle both int and float
        old_num = float(old_value)
        new_num = float(new_value)

        # Check if both are integers (no decimal point in original strings)
        is_integer = "." not in old_value and "." not in new_value

        if is_integer:
            # For integers, check absolute difference
            abs_diff = abs(new_num - old_num)
            if abs_diff <= 2:
                return True

        # Check relative change (for all numeric types)
        if old_num != 0:
            relative_change = abs((new_num - old_num) / old_num)
            if relative_change <= 0.20:  # 20%
                return True

        return False
    except (ValueError, ZeroDivisionError):
        # If we can't parse as numbers, reject the match
        return False


def track_parameters(
    file_path: str,
    start_line: int,
    end_line: int,
    max_commits: int,
    verbose: bool = False,
):
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

    # Get git history with adjusted line ranges
    commits_with_ranges = get_git_log_range(file_path, start_line, end_line, max_commits)

    if not commits_with_ranges:
        print("No commits found in history")
        return None

    print(
        f"Tracking {len(commits_with_ranges)} commits with {len(original_params)} parameters...\n"
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
    for commit_idx, (commit, commit_start, commit_end) in enumerate(commits_with_ranges[1:], 1):  # Skip first commit (HEAD)
        print(f"\n{'=' * 80}")
        print(f"COMMIT {commit_idx}/{len(commits_with_ranges) - 1}: {commit[:8]}")
        print(f"  Lines: {commit_start}-{commit_end} (adjusted for this commit)")
        print(f"{'=' * 80}")

        commit_code = get_file_lines(file_path, commit_start, commit_end, str(commit))
        if not commit_code:
            print(f"  File or lines don't exist at this commit, stopping tracking")
            break

        # Show the commit code
        print(f"\nCode in this commit:")
        print("-" * 80)
        print(commit_code)
        print("-" * 80)

        # Extract numbers and create PARAM version for commit
        commit_param_code, commit_numbers = extract_numbers_and_create_param_code(
            commit_code
        )
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
                verbose=verbose,
            )

            if matched_param:
                # Find the index of matched param in commit_params
                try:
                    commit_param_idx = commit_params.index(matched_param)
                    if commit_param_idx < len(commit_numbers):
                        new_value = commit_numbers[commit_param_idx][0]

                        # Get the previous value (most recent in history)
                        old_value = param_history[param_id][-1]

                        # Validate the change
                        if is_valid_parameter_change(old_value, new_value):
                            param_history[param_id].append(new_value)
                            param_matches[param_id] = new_value
                            matched_count += 1
                        else:
                            param_matches[param_id] = (
                                f"REJECTED: change too large ({old_value} -> {new_value})"
                            )
                            if verbose:
                                print(
                                    f"  {param_id}: Rejected match due to large change: {old_value} -> {new_value}"
                                )
                    else:
                        print(f"  Warning: {param_id} matched but index out of range")
                        param_matches[param_id] = "ERROR: index out of range"
                except ValueError:
                    print(
                        f"  Warning: {param_id} matched param not in commit_params list"
                    )
                    param_matches[param_id] = "ERROR: not in list"
            else:
                param_matches[param_id] = "NOT MATCHED"

        # Display matching results
        print(f"\nParameter Matching Results:")
        print("-" * 80)
        for param_id in sorted(
            param_matches.keys(), key=lambda x: int(x.split("_")[1])
        ):
            match_result = param_matches[param_id]
            if match_result == "NOT MATCHED":
                print(f"  {param_id}: ✗ NOT MATCHED")
            elif isinstance(match_result, str) and match_result.startswith("REJECTED"):
                print(f"  {param_id}: ✗ {match_result}")
            else:
                print(f"  {param_id}: ✓ matched to value {match_result}")

        print(
            f"\nSummary: Successfully matched {matched_count}/{len(original_params)} parameters"
        )

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
    param_map,
    original_params,
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
    parser.add_argument(
        "--show-all",
        action="store_true",
        help="Show all parameters including those that don't change",
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

        # Filter parameters by default (unless --show-all is specified)
        if not args.show_all:
            # Only keep parameters that have more than one unique value
            param_history_filtered = {
                param_id: values
                for param_id, values in param_history.items()
                if len(set(values)) > 1
            }

            # Update param_map to only include filtered parameters
            param_map_filtered = {
                node_id: param_id
                for node_id, param_id in param_map.items()
                if param_id in param_history_filtered
            }

            # Update original_params and original_numbers to match
            filtered_indices = [
                idx
                for idx, (param_node, _) in enumerate(
                    zip(original_params, original_numbers)
                )
                if param_map[id(param_node)] in param_history_filtered
            ]
            original_params_filtered = [original_params[i] for i in filtered_indices]
            original_numbers_filtered = [original_numbers[i] for i in filtered_indices]

            display_param_history = param_history_filtered
            display_param_map = param_map_filtered
            display_original_params = original_params_filtered
            display_original_numbers = original_numbers_filtered
        else:
            display_param_history = param_history
            display_param_map = param_map
            display_original_params = original_params
            display_original_numbers = original_numbers

        # Create annotated version (only with displayed parameters)
        annotated = create_annotated_code(
            original_code,
            display_original_numbers,
            display_param_map,
            display_original_params,
        )

        print("\n" + "=" * 80)
        print("ORIGINAL CODE:")
        print("=" * 80)
        print(original_code)
        print()

        print("=" * 80)
        if args.show_all:
            print("ANNOTATED CODE (parameters replaced with identifiers):")
        else:
            print("ANNOTATED CODE (only changed parameters shown):")
        print("=" * 80)
        print(annotated)
        print()

        print("=" * 80)
        if args.show_all:
            print("PARAMETER HISTORY (newest to oldest):")
        else:
            print("PARAMETER HISTORY (only changed parameters, newest to oldest):")
        print("=" * 80)
        for param_id in sorted(
            display_param_history.keys(), key=lambda x: int(x.split("_")[1])
        ):
            values = display_param_history[param_id]
            print(f"{param_id}: {values}")
        print()


if __name__ == "__main__":
    main()
