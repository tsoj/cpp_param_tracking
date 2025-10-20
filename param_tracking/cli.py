"""Command-line interface for cpp_param_tracking."""

from __future__ import annotations

import argparse
from typing import Iterable

from .tracking import CommitSnapshot, ParamMatch, auto_detect_range, track_parameters


def _format_param_match(param_id: str, match: ParamMatch) -> str:
    if match.status == "matched" and match.value is not None:
        return f"{param_id}: ✓ matched to value {match.value}"
    if match.status == "rejected" and match.detail:
        return f"{param_id}: ✗ {match.detail}"
    if match.status == "error" and match.detail:
        return f"{param_id}: ⚠ {match.detail}"
    return f"{param_id}: ✗ {match.status.replace('_', ' ')}"


def _print_commit(snapshot: CommitSnapshot, param_order: Iterable[str]) -> None:
    print("\n" + "=" * 80)
    print(f"COMMIT: {snapshot.commit[:8]}")
    print(f"Lines: {snapshot.start_line}-{snapshot.end_line}")
    print("=" * 80)
    print("\nCode in this commit:")
    print("-" * 80)
    print(snapshot.code)
    print("-" * 80)

    print("\nParameter Matching Results:")
    print("-" * 80)
    for param_id in param_order:
        match = snapshot.matches.get(param_id)
        if match:
            print(f"  {_format_param_match(param_id, match)}")

    print(
        f"\nSummary: Successfully matched {snapshot.matched_count}/"
        f"{len(snapshot.matches)} parameters"
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Track numeric parameter changes in C++ code across git history",
    )
    parser.add_argument("--file", help="Path to the C++ file")
    parser.add_argument("--start-line", type=int, help="Start line number (1-indexed)")
    parser.add_argument("--end-line", type=int, help="End line number (inclusive)")
    parser.add_argument(
        "--max-commits",
        type=int,
        default=10,
        help="Maximum number of commits to track",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed matching debug output",
    )
    parser.add_argument(
        "--show-all",
        action="store_true",
        help="Show parameters even if they do not change across history",
    )

    args = parser.parse_args(argv)

    provided = [args.file is not None, args.start_line is not None, args.end_line is not None]
    if all(not flag for flag in provided):
        file_path, start_line, end_line = auto_detect_range()
        print(f"Auto-detected from git diff: {file_path} lines {start_line}-{end_line}")
    elif all(provided):
        file_path = args.file
        start_line = args.start_line
        end_line = args.end_line
    else:
        raise ValueError(
            "Either provide all of --file/--start-line/--end-line or none to auto-detect."
        )

    result = track_parameters(
        file_path,
        start_line,
        end_line,
        args.max_commits,
        verbose=args.verbose,
    )

    display_history = result.filtered_history(show_all=args.show_all)
    param_order = sorted(
        display_history.keys(),
        key=lambda pid: int(pid.split("_")[1]),
    )

    print("\n" + "=" * 80)
    print("ORIGINAL CODE:")
    print("=" * 80)
    print(result.original_code)

    print("\n" + "=" * 80)
    header = (
        "ANNOTATED CODE (parameters replaced with identifiers)"
        if args.show_all
        else "ANNOTATED CODE (only changed parameters shown)"
    )
    print(header)
    print("=" * 80)
    print(result.annotated_code(show_all=args.show_all))

    print("\n" + "=" * 80)
    history_header = (
        "PARAMETER HISTORY (newest to oldest)"
        if args.show_all
        else "PARAMETER HISTORY (only changed parameters, newest to oldest)"
    )
    print(history_header)
    print("=" * 80)
    for param_id in param_order:
        history = display_history.get(param_id)
        if history:
            print(f"{param_id}: {history}")

    for snapshot in result.commits:
        _print_commit(snapshot, param_order)


if __name__ == "__main__":
    main()
