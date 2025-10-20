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

def get_changed_lines_from_diff() -> Tuple[str, int, int]:
    """
    Parse git diff to automatically determine the changed file and line range.

    Returns:
        Tuple of (file_path, start_line, end_line)

    Raises:
        ValueError: If no changes found, multiple files changed, or cannot parse diff
    """
    try:
        # Get the diff with line numbers
        result = subprocess.run(
            ["git", "diff", "HEAD", "--unified=0"],
            capture_output=True,
            text=True,
            check=True,
        )

        diff_output = result.stdout
        if not diff_output.strip():
            raise ValueError("No changes found in git diff. Please commit or stage your changes first, or provide --file, --start-line, and --end-line explicitly.")

        # Parse the diff output
        files_changed = {}
        current_file = None

        for line in diff_output.split('\n'):
            # Look for file headers
            if line.startswith('--- a/'):
                current_file = line[6:]  # Remove '--- a/' prefix
            elif line.startswith('+++ b/'):
                new_file = line[6:]  # Remove '+++ b/' prefix
                if new_file != '/dev/null':
                    current_file = new_file
            # Look for hunk headers like @@ -10,5 +10,6 @@
            elif line.startswith('@@') and current_file:
                # Extract the new file line numbers (the second pair)
                parts = line.split('@@')[1].strip().split()
                if len(parts) >= 2:
                    # Parse +start,count format
                    new_range = parts[1] if parts[1].startswith('+') else parts[0]
                    if new_range.startswith('+'):
                        new_range = new_range[1:]  # Remove '+' prefix

                        if ',' in new_range:
                            start, count = map(int, new_range.split(','))
                            end = start + count - 1
                        else:
                            # Single line change
                            start = int(new_range)
                            end = start

                        if current_file not in files_changed:
                            files_changed[current_file] = []
                        files_changed[current_file].append((start, end))

        if not files_changed:
            raise ValueError("Could not parse any changes from git diff. Please provide --file, --start-line, and --end-line explicitly.")

        if len(files_changed) > 1:
            file_list = ', '.join(files_changed.keys())
            raise ValueError(f"Multiple files changed: {file_list}. Please provide --file, --start-line, and --end-line explicitly to specify which file to analyze.")

        # Get the single file and its changes
        file_path = list(files_changed.keys())[0]
        line_ranges = files_changed[file_path]

        if not line_ranges:
            raise ValueError(f"No line changes found in {file_path}")

        # Merge all line ranges into a single range (min start to max end)
        min_start = min(start for start, _ in line_ranges)
        max_end = max(end for _, end in line_ranges)

        return file_path, min_start, max_end

    except subprocess.CalledProcessError as e:
        raise ValueError(f"Failed to run git diff: {e}")


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
    parser.add_argument("--file", required=False, help="Path to the C++ file (auto-detected from git diff if not provided)")
    parser.add_argument(
        "--start-line", type=int, required=False, help="Start line number (1-indexed, auto-detected from git diff if not provided)"
    )
    parser.add_argument(
        "--end-line", type=int, required=False, help="End line number (inclusive, auto-detected from git diff if not provided)"
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

    # Auto-detect file and lines from git diff if not provided
    file_path = args.file
    start_line = args.start_line
    end_line = args.end_line

    # Check if all three are provided or none are provided
    provided_args = [file_path is not None, start_line is not None, end_line is not None]
    
    if not any(provided_args):
        # None provided - auto-detect from git diff
        try:
            file_path, start_line, end_line = get_changed_lines_from_diff()
            print(f"Auto-detected from git diff: {file_path} lines {start_line}-{end_line}")
        except ValueError as e:
            print(f"Error: {e}")
            return
    elif not all(provided_args):
        # Some but not all provided - error
        print("Error: Either provide all of --file, --start-line, and --end-line, or provide none of them to auto-detect from git diff.")
        return

    result = track_parameters(
        file_path, start_line, end_line, args.max_commits, args.verbose
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
