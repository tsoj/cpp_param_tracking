"""Public interface for the cpp_param_tracking package."""

from .parser import CPP_LANGUAGE, DEFAULT_PARSER, create_parser, get_parser, parse_code
from .ast_utils import (
    count_params_in_subtree,
    find_matching_subtrees,
    find_params,
    get_params_in_subtree,
    has_binary_operator_or_bounded_context,
    nodes_equal,
)
from .matching import match_param
from .extraction import create_annotated_code, extract_numbers_and_create_param_code
from .tracking import (
    CommitSnapshot,
    ParamMatch,
    TrackingResult,
    auto_detect_range,
    is_valid_parameter_change,
    track_parameters,
)

__all__ = [
    "CPP_LANGUAGE",
    "DEFAULT_PARSER",
    "create_parser",
    "get_parser",
    "parse_code",
    "count_params_in_subtree",
    "find_matching_subtrees",
    "find_params",
    "get_params_in_subtree",
    "has_binary_operator_or_bounded_context",
    "nodes_equal",
    "match_param",
    "create_annotated_code",
    "extract_numbers_and_create_param_code",
    "CommitSnapshot",
    "ParamMatch",
    "TrackingResult",
    "auto_detect_range",
    "is_valid_parameter_change",
    "track_parameters",
]
