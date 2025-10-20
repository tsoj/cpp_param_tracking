"""Git helpers for tracking parameter changes."""

from __future__ import annotations

import re
import subprocess
from typing import Dict, Iterable, List, Tuple


def get_file_lines(file_path: str, start_line: int, end_line: int, commit: str = "HEAD") -> str:
    """Return specific lines from *file_path* at *commit*."""
    if start_line < 1 or end_line < start_line:
        raise ValueError("Invalid line range")

    result = subprocess.run(
        ["git", "show", f"{commit}:{file_path}"],
        capture_output=True,
        text=True,
        check=True,
    )
    lines = result.stdout.split("\n")
    return "\n".join(lines[start_line - 1 : end_line])


def _collect_diff_ranges(diff_output: str) -> Dict[str, List[Tuple[int, int]]]:
    files_changed: Dict[str, List[Tuple[int, int]]] = {}
    current_file: str | None = None

    for line in diff_output.splitlines():
        if line.startswith("+++ b/"):
            candidate = line[6:]
            if candidate != "/dev/null":
                current_file = candidate
        elif line.startswith("@@") and current_file:
            match = re.search(r"\+([0-9]+)(?:,([0-9]+))?", line)
            if not match:
                continue
            start = int(match.group(1))
            count = int(match.group(2)) if match.group(2) else 1
            end = start + count - 1
            files_changed.setdefault(current_file, []).append((start, end))

    return files_changed


def get_changed_lines_from_diff() -> Tuple[str, int, int]:
    """Infer the changed file and aggregated line range from ``git diff``."""
    result = subprocess.run(
        ["git", "diff", "HEAD", "--unified=0"],
        capture_output=True,
        text=True,
        check=True,
    )

    diff_output = result.stdout
    if not diff_output.strip():
        raise ValueError(
            "No changes found in git diff. Provide --file/--start-line/--end-line explicitly."
        )

    files_changed = _collect_diff_ranges(diff_output)

    if not files_changed:
        raise ValueError("Could not parse any changes from git diff")

    if len(files_changed) > 1:
        file_list = ", ".join(files_changed)
        raise ValueError(
            f"Multiple files changed: {file_list}. Please specify --file, --start-line, and --end-line."
        )

    file_path, ranges = next(iter(files_changed.items()))
    if not ranges:
        raise ValueError(f"No line changes found in {file_path}")

    start = min(start for start, _ in ranges)
    end = max(end for _, end in ranges)
    return file_path, start, end


def get_git_log_range(
    file_path: str, start_line: int, end_line: int, max_commits: int
) -> List[Tuple[str, int, int]]:
    """Return ``git log -L`` output describing line history for a range."""
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

    commits_with_ranges: List[Tuple[str, int, int]] = []
    current_commit: str | None = None
    current_start = start_line
    current_end = end_line

    lines = result.stdout.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]

        if line and not any(
            line.startswith(prefix)
            for prefix in ("diff", "---", "+++", "@@", "+", "-", " ", "\\", "index")
        ):
            current_commit = line.strip()
            if current_commit:
                commits_with_ranges.append((current_commit, current_start, current_end))

        elif line.startswith("@@"):
            match = re.search(r"@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", line)
            if match:
                new_start = int(match.group(1))
                new_count = int(match.group(2)) if match.group(2) else 1
                current_start = new_start
                current_end = new_start + new_count - 1

        i += 1

    return commits_with_ranges
