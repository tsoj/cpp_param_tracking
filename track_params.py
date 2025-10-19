#!/usr/bin/env python3
"""
Track numeric parameter changes in C++ code across git history.
"""

import argparse
import subprocess
import re
from typing import List, Dict, Tuple, Optional
from tree_sitter import Language, Parser
import tree_sitter_cpp as tscpp


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
        return None


def get_git_log_range(
    file_path: str, start_line: int, end_line: int, max_commits: int
) -> List[Tuple[str, str]]:
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


def parse_cpp_code(code: str) -> List[Dict]:
    """Parse C++ code and extract statements with their numeric literals."""
    CPP_LANGUAGE = Language(tscpp.language())
    parser = Parser(CPP_LANGUAGE)

    tree = parser.parse(bytes(code, "utf8"))

    statements = []

    def extract_numbers_from_node(node):
        """Extract all numeric literals from a node and its children."""
        numbers = []

        if node.type in ("number_literal", "integer_literal", "float_literal"):
            numbers.append(
                {
                    "value": node.text.decode("utf8"),
                    "start": node.start_byte,
                    "end": node.end_byte,
                }
            )

        for child in node.children:
            numbers.extend(extract_numbers_from_node(child))

        return numbers

    def get_statement_nodes(node):
        """Get top-level statement nodes."""
        statement_types = {
            "expression_statement",
            "return_statement",
            "declaration",
            "if_statement",
            "while_statement",
            "for_statement",
            "compound_statement",
            "function_definition",
        }

        result = []

        if node.type in statement_types:
            result.append(node)
        else:
            for child in node.children:
                result.extend(get_statement_nodes(child))

        return result

    statement_nodes = get_statement_nodes(tree.root_node)

    for stmt_node in statement_nodes:
        stmt_text = stmt_node.text.decode("utf8")
        numbers = extract_numbers_from_node(stmt_node)

        # Sort numbers by position
        numbers.sort(key=lambda x: x["start"])

        statements.append(
            {
                "text": stmt_text,
                "numbers": [n["value"] for n in numbers],
                "node": stmt_node,
            }
        )

    return statements


def normalize_code(code: str) -> str:
    """Normalize code by collapsing whitespace and replacing numbers with placeholder."""
    # Replace all numbers with a placeholder
    normalized = re.sub(r"\b\d+\.?\d*[fFlLuU]?\b", "NUM", code)
    # Collapse whitespace
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def match_statement(target_stmt: Dict, candidate_stmts: List[Dict]) -> Optional[Dict]:
    """Try to match a target statement against candidates."""
    target_normalized = normalize_code(target_stmt["text"])

    for candidate in candidate_stmts:
        candidate_normalized = normalize_code(candidate["text"])
        if target_normalized == candidate_normalized:
            # Check if they have the same number of numeric literals
            if len(target_stmt["numbers"]) == len(candidate["numbers"]):
                return candidate

    return None


def track_parameters(
    file_path: str, start_line: int, end_line: int, max_commits: int
) -> Dict:
    """Track parameter changes across git history."""

    # Get the original snippet from HEAD
    original_code = get_file_lines(file_path, start_line, end_line, "HEAD")
    if not original_code:
        print(f"Error: Could not read lines {start_line}-{end_line} from {file_path}")
        return None

    # Parse the original code
    original_statements = parse_cpp_code(original_code)

    # Get git history
    commits = get_git_log_range(file_path, start_line, end_line, max_commits)

    if not commits:
        print("No commits found in history")
        return None

    print(f"Tracking {len(commits)} commits...\n")

    # Initialize parameter tracking
    param_history = {}
    param_counter = 0
    param_map = {}  # Maps (statement_idx, param_idx) to PARAM_N identifier

    # Process each statement in the original code
    for stmt_idx, stmt in enumerate(original_statements):
        for param_idx, param_value in enumerate(stmt["numbers"]):
            key = (stmt_idx, param_idx)
            param_id = f"PARAM_{param_counter}"
            param_counter += 1
            param_map[key] = param_id
            param_history[param_id] = [param_value]  # Start with HEAD value

    # Track through history
    for commit in commits[1:]:  # Skip first commit (HEAD)
        commit_code = get_file_lines(file_path, start_line, end_line, commit)
        if not commit_code:
            # File or lines don't exist at this commit, stop tracking
            break

        commit_statements = parse_cpp_code(commit_code)

        # Try to match each original statement
        for stmt_idx, orig_stmt in enumerate(original_statements):
            matched = match_statement(orig_stmt, commit_statements)

            if matched:
                # Add parameter values to history
                for param_idx, param_value in enumerate(matched["numbers"]):
                    key = (stmt_idx, param_idx)
                    if key in param_map:
                        param_id = param_map[key]
                        param_history[param_id].append(param_value)
            else:
                # Statement not found, stop tracking its parameters
                for param_idx in range(len(orig_stmt["numbers"])):
                    key = (stmt_idx, param_idx)
                    if key in param_map:
                        param_id = param_map[key]
                        # Mark as not found (could append None, but we'll just stop)
                        pass

    return original_code, original_statements, param_map, param_history


def create_annotated_code(
    original_code: str, statements: List[Dict], param_map: Dict
) -> str:
    """Create version of code with parameters replaced by identifiers."""
    annotated = original_code

    # We need to replace numbers in reverse order to maintain positions
    replacements = []

    for stmt_idx, stmt in enumerate(statements):
        stmt_text = stmt["text"]
        for param_idx, number in enumerate(stmt["numbers"]):
            key = (stmt_idx, param_idx)
            if key in param_map:
                param_id = param_map[key]
                replacements.append((stmt_text, number, param_id))

    # Build annotated version by processing each statement
    for stmt_idx, stmt in enumerate(statements):
        stmt_text = stmt["text"]
        annotated_stmt = stmt_text

        # Replace numbers in reverse order to maintain positions
        offset = 0
        for param_idx, number in enumerate(stmt["numbers"]):
            key = (stmt_idx, param_idx)
            if key in param_map:
                param_id = param_map[key]
                # Find the number in the statement
                pattern = r"\b" + re.escape(number) + r"\b"
                match = re.search(pattern, annotated_stmt[offset:])
                if match:
                    pos = offset + match.start()
                    annotated_stmt = (
                        annotated_stmt[:pos]
                        + param_id
                        + annotated_stmt[pos + len(number) :]
                    )
                    offset = pos + len(param_id)

        annotated = annotated.replace(stmt_text, annotated_stmt, 1)

    return annotated


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

    args = parser.parse_args()

    result = track_parameters(
        args.file, args.start_line, args.end_line, args.max_commits
    )

    if result:
        original_code, statements, param_map, param_history = result

        # Create annotated version
        annotated = create_annotated_code(original_code, statements, param_map)

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
