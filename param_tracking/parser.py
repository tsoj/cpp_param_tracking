"""Utilities for configuring the Tree-sitter C++ parser."""

from __future__ import annotations

from typing import Optional

from tree_sitter import Language, Parser
import tree_sitter_cpp as tscpp

CPP_LANGUAGE: Language = Language(tscpp.language())


def create_parser() -> Parser:
    """Build a new Tree-sitter parser configured for C++."""
    return Parser(CPP_LANGUAGE)


DEFAULT_PARSER: Parser = create_parser()


def get_parser() -> Parser:
    """Return a parser configured for C++."""
    return create_parser()


def parse_code(code: str, parser: Optional[Parser] = None):
    """Parse C++ code into a syntax tree."""
    parser = parser or DEFAULT_PARSER
    return parser.parse(bytes(code, "utf8"))
