"""Core workflow for tracking numeric parameters across git history."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Tuple

from tree_sitter import Node

from .ast_utils import find_params
from .extraction import (
    NumberInfo,
    create_annotated_code,
    extract_numbers_and_create_param_code,
)
from .git_support import get_changed_lines_from_diff, get_file_lines, get_git_log_range
from .matching import match_param
from .parser import parse_code


@dataclass
class ParamMatch:
    status: str
    value: Optional[str] = None
    detail: Optional[str] = None


@dataclass
class CommitSnapshot:
    commit: str
    start_line: int
    end_line: int
    code: str
    numbers: List[NumberInfo]
    params: List[Node]
    matches: Dict[str, ParamMatch]

    @property
    def matched_count(self) -> int:
        return sum(1 for match in self.matches.values() if match.status == "matched")


@dataclass
class TrackingResult:
    original_code: str
    original_param_code: str
    original_params: List[Node]
    original_numbers: List[NumberInfo]
    param_map: Dict[int, str]
    param_history: Dict[str, List[str]]
    commits: List[CommitSnapshot]

    def annotated_code(self, *, show_all: bool = True) -> str:
        params = self.original_params
        numbers = self.original_numbers
        param_map = self.param_map

        if not show_all:
            visible_ids = {
                param_id
                for param_id, values in self.param_history.items()
                if len(set(values)) > 1
            }
            params = [p for p in self.original_params if self.param_map[id(p)] in visible_ids]
            numbers = [
                n
                for p, n in zip(self.original_params, self.original_numbers)
                if self.param_map[id(p)] in visible_ids
            ]
            param_map = {node_id: pid for node_id, pid in self.param_map.items() if pid in visible_ids}

        return create_annotated_code(self.original_code, numbers, param_map, params)

    def filtered_history(self, *, show_all: bool = True) -> Mapping[str, List[str]]:
        if show_all:
            return self.param_history
        return {
            param_id: values
            for param_id, values in self.param_history.items()
            if len(set(values)) > 1
        }


def is_valid_parameter_change(old_value: str, new_value: str) -> bool:
    """Validate a parameter change by absolute/relative thresholds."""
    try:
        old_num = float(old_value)
        new_num = float(new_value)
    except ValueError:
        return False

    is_integer = "." not in old_value and "." not in new_value

    if is_integer and abs(new_num - old_num) <= 2:
        return True

    if old_num == 0:
        return False

    relative_change = abs((new_num - old_num) / old_num)
    return relative_change <= 0.20


def auto_detect_range() -> Tuple[str, int, int]:
    """Return file path and line range inferred from ``git diff``."""
    return get_changed_lines_from_diff()


def track_parameters(
    file_path: str,
    start_line: int,
    end_line: int,
    max_commits: int,
    *,
    verbose: bool = False,
) -> TrackingResult:
    """Track numeric parameter changes across git history."""
    original_code = get_file_lines(file_path, start_line, end_line, "HEAD")
    if not original_code:
        raise ValueError(
            f"Could not read lines {start_line}-{end_line} from {file_path} at HEAD"
        )

    original_param_code, original_numbers = extract_numbers_and_create_param_code(
        original_code
    )
    original_tree = parse_code(original_param_code)
    original_params = find_params(original_tree.root_node)

    if len(original_params) != len(original_numbers):
        raise ValueError(
            "Mismatch between PARAM nodes and extracted numeric literals in original code"
        )

    commits_with_ranges = get_git_log_range(file_path, start_line, end_line, max_commits)
    if not commits_with_ranges:
        raise ValueError("No commits found for the requested range")

    param_map: Dict[int, str] = {}
    param_history: Dict[str, List[str]] = {}

    for index, (param_node, (value, _, _)) in enumerate(
        zip(original_params, original_numbers)
    ):
        param_id = f"PARAM_{index}"
        param_map[id(param_node)] = param_id
        param_history[param_id] = [value]

    commits: List[CommitSnapshot] = []

    for commit, commit_start, commit_end in commits_with_ranges[1:]:
        commit_code = get_file_lines(file_path, commit_start, commit_end, commit)
        commit_param_code, commit_numbers = extract_numbers_and_create_param_code(
            commit_code
        )
        commit_tree = parse_code(commit_param_code)
        commit_params = find_params(commit_tree.root_node)

        param_matches: Dict[str, ParamMatch] = {}

        for original_param in original_params:
            param_id = param_map[id(original_param)]
            matched = match_param(
                original_param,
                original_tree.root_node,
                commit_tree.root_node,
                max_levels=10,
                verbose=verbose,
            )

            if matched is None:
                param_matches[param_id] = ParamMatch("not_matched")
                continue

            try:
                commit_index = commit_params.index(matched)
            except ValueError:
                param_matches[param_id] = ParamMatch(
                    "error", detail="Matched node missing from commit params"
                )
                continue

            if commit_index >= len(commit_numbers):
                param_matches[param_id] = ParamMatch(
                    "error", detail="Matched param index outside numeric literal list"
                )
                continue

            new_value = commit_numbers[commit_index][0]
            old_value = param_history[param_id][-1]

            if is_valid_parameter_change(old_value, new_value):
                param_history[param_id].append(new_value)
                param_matches[param_id] = ParamMatch("matched", value=new_value)
            else:
                param_matches[param_id] = ParamMatch(
                    "rejected",
                    value=new_value,
                    detail=f"change too large ({old_value} -> {new_value})",
                )

        commits.append(
            CommitSnapshot(
                commit=commit,
                start_line=commit_start,
                end_line=commit_end,
                code=commit_code,
                numbers=commit_numbers,
                params=commit_params,
                matches=param_matches,
            )
        )

    return TrackingResult(
        original_code=original_code,
        original_param_code=original_param_code,
        original_params=original_params,
        original_numbers=original_numbers,
        param_map=param_map,
        param_history=param_history,
        commits=commits,
    )
