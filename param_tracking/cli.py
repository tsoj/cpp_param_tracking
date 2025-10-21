"""Command-line interface for cpp_param_tracking."""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

from .tracking import CommitSnapshot, ParamMatch, auto_detect_range, track_parameters

ANSI_RESET = "\x1b[0m"
KEYWORD_COLOR = "\x1b[94m"
STRING_COLOR = "\x1b[95m"
NUMBER_COLOR = "\x1b[96m"
COMMENT_COLOR = "\x1b[90m"
OPERATOR_COLOR = "\x1b[93m"
PREPROCESSOR_COLOR = "\x1b[36m"
DEFAULT_PARAM_COLOR = "\x1b[1;3m"
DIM_SEPARATOR = "\x1b[2m-\x1b[0m"

ANSI_COLOR_HEX = {
    KEYWORD_COLOR: "#4c8bf5",
    STRING_COLOR: "#d17aff",
    NUMBER_COLOR: "#33c3c1",
    COMMENT_COLOR: "#6f6f6f",
    OPERATOR_COLOR: "#8c6a00",
    PREPROCESSOR_COLOR: "#2aa1b3",
    DEFAULT_PARAM_COLOR: "#333333",
}

CPP_KEYWORDS = {
    "alignas",
    "alignof",
    "and",
    "and_eq",
    "asm",
    "auto",
    "bitand",
    "bitor",
    "bool",
    "break",
    "case",
    "catch",
    "char",
    "char16_t",
    "char32_t",
    "class",
    "compl",
    "concept",
    "const",
    "consteval",
    "constexpr",
    "constinit",
    "const_cast",
    "continue",
    "co_await",
    "co_return",
    "co_yield",
    "decltype",
    "default",
    "delete",
    "do",
    "double",
    "dynamic_cast",
    "else",
    "enum",
    "explicit",
    "export",
    "extern",
    "false",
    "float",
    "for",
    "friend",
    "goto",
    "if",
    "inline",
    "int",
    "long",
    "mutable",
    "namespace",
    "new",
    "noexcept",
    "not",
    "not_eq",
    "nullptr",
    "operator",
    "or",
    "or_eq",
    "private",
    "protected",
    "public",
    "register",
    "reinterpret_cast",
    "requires",
    "return",
    "short",
    "signed",
    "sizeof",
    "static",
    "static_assert",
    "static_cast",
    "struct",
    "switch",
    "template",
    "this",
    "thread_local",
    "throw",
    "true",
    "try",
    "typedef",
    "typeid",
    "typename",
    "union",
    "unsigned",
    "using",
    "virtual",
    "void",
    "volatile",
    "wchar_t",
    "while",
    "xor",
    "xor_eq",
}

NUMBER_RE = re.compile(r"0[xX][0-9a-fA-F]+|\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")
PARAM_RE = re.compile(r"PARAM_\d+")
IDENTIFIER_RE = re.compile(r"[A-Za-z_]\w*")
OPERATOR_RE = re.compile(
    r"::|->|\+\+|--|<=|>=|==|!=|&&|\|\||<<|>>|[-+*/%&|^~!=<>?:]=?|[.,;()\[\]{}]"
)
WHITESPACE_RE = re.compile(r"\s+")

PALETTE = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


@dataclass(frozen=True)
class ParamColor:
    hex_value: str
    ansi_code: str
    identifier_ansi_code: str
    dim_ansi_code: str


@dataclass
class HighlightSegment:
    text: str
    ansi_code: str | None
    hex_color: str | None
    italic: bool = False
    bold: bool = False


def _rgb_to_ansi256(r: int, g: int, b: int) -> int:
    """Approximate an RGB color with an ANSI-256 color code."""
    if r == g == b:
        if r < 8:
            return 16
        if r > 248:
            return 231
        return int(round(((r - 8) / 247) * 24)) + 232

    def _channel_to_index(channel: int) -> int:
        return int(round(channel / 255 * 5))

    return (
        16
        + (36 * _channel_to_index(r))
        + (6 * _channel_to_index(g))
        + _channel_to_index(b)
    )


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    hex_digits = hex_color.lstrip("#")
    return tuple(int(hex_digits[i : i + 2], 16) for i in (0, 2, 4))


def _assign_param_colors(param_ids: Iterable[str]) -> Dict[str, ParamColor]:
    colors: Dict[str, ParamColor] = {}
    for index, param_id in enumerate(param_ids):
        hex_color = PALETTE[index % len(PALETTE)]
        r, g, b = _hex_to_rgb(hex_color)
        color_index = _rgb_to_ansi256(r, g, b)
        base_code = f"\x1b[38;5;{color_index}m"
        identifier_code = f"\x1b[1;3;38;5;{color_index}m"
        dim_code = f"\x1b[2;38;5;{color_index}m"
        colors[param_id] = ParamColor(
            hex_value=hex_color,
            ansi_code=base_code,
            identifier_ansi_code=identifier_code,
            dim_ansi_code=dim_code,
        )
    return colors


def _syntax_highlight_segments(
    code: str, param_colors: Dict[str, ParamColor]
) -> List[HighlightSegment]:
    """Return syntax-highlighted segments for *code*."""
    if not code:
        return []

    segments: List[HighlightSegment] = []
    length = len(code)
    index = 0

    def _append(
        text: str,
        ansi: str | None = None,
        hex_color: str | None = None,
        *,
        italic: bool = False,
        bold: bool = False,
    ) -> None:
        if not text:
            return
        if (
            segments
            and segments[-1].ansi_code == ansi
            and segments[-1].hex_color == hex_color
            and segments[-1].italic == italic
            and segments[-1].bold == bold
        ):
            last = segments[-1]
            segments[-1] = HighlightSegment(
                last.text + text,
                ansi,
                hex_color,
                italic=italic,
                bold=bold,
            )
            return
        segments.append(
            HighlightSegment(
                text,
                ansi,
                hex_color,
                italic=italic,
                bold=bold,
            )
        )

    while index < length:
        char = code[index]

        if char == "\n":
            _append("\n")
            index += 1
            continue

        if char == "#" and (index == 0 or code[index - 1] == "\n"):
            line_end = code.find("\n", index)
            if line_end == -1:
                line_end = length
            text = code[index:line_end]
            _append(
                text,
                PREPROCESSOR_COLOR,
                ANSI_COLOR_HEX.get(PREPROCESSOR_COLOR),
            )
            index = line_end
            continue

        if code.startswith("//", index):
            line_end = code.find("\n", index)
            if line_end == -1:
                line_end = length
            text = code[index:line_end]
            _append(text, COMMENT_COLOR, ANSI_COLOR_HEX.get(COMMENT_COLOR))
            index = line_end
            continue

        if code.startswith("/*", index):
            end = code.find("*/", index + 2)
            if end == -1:
                end = length
            else:
                end += 2
            text = code[index:end]
            _append(text, COMMENT_COLOR, ANSI_COLOR_HEX.get(COMMENT_COLOR))
            index = end
            continue

        if char in {'"', "'"}:
            quote = char
            j = index + 1
            while j < length:
                if code[j] == "\\":
                    j += 2
                    continue
                if code[j] == quote:
                    j += 1
                    break
                j += 1
            else:
                j = length
            text = code[index:j]
            _append(text, STRING_COLOR, ANSI_COLOR_HEX.get(STRING_COLOR))
            index = j
            continue

        match = NUMBER_RE.match(code, index)
        if match:
            text = match.group()
            _append(text, NUMBER_COLOR, ANSI_COLOR_HEX.get(NUMBER_COLOR))
            index = match.end()
            continue

        match = PARAM_RE.match(code, index)
        if match:
            text = match.group()
            color = param_colors.get(text)
            if color:
                ansi = color.identifier_ansi_code
                hex_color = color.hex_value
            else:
                ansi = DEFAULT_PARAM_COLOR
                hex_color = ANSI_COLOR_HEX.get(DEFAULT_PARAM_COLOR)
            _append(text, ansi, hex_color, italic=True, bold=True)
            index = match.end()
            continue

        match = IDENTIFIER_RE.match(code, index)
        if match:
            text = match.group()
            if text in CPP_KEYWORDS:
                _append(text, KEYWORD_COLOR, ANSI_COLOR_HEX.get(KEYWORD_COLOR))
            else:
                _append(text)
            index = match.end()
            continue

        match = OPERATOR_RE.match(code, index)
        if match:
            text = match.group()
            _append(text, OPERATOR_COLOR, ANSI_COLOR_HEX.get(OPERATOR_COLOR))
            index = match.end()
            continue

        match = WHITESPACE_RE.match(code, index)
        if match:
            text = match.group()
            _append(text)
            index = match.end()
            continue

        _append(char)
        index += 1

    return segments


def _segments_to_ansi(segments: Sequence[HighlightSegment]) -> str:
    if not segments:
        return ""
    parts: List[str] = []
    for segment in segments:
        if segment.ansi_code:
            parts.append(f"{segment.ansi_code}{segment.text}{ANSI_RESET}")
        else:
            parts.append(segment.text)
    return "".join(parts)


def _syntax_highlight_cpp(code: str, param_colors: Dict[str, ParamColor]) -> str:
    """Render syntax-highlighted ANSI string."""
    return _segments_to_ansi(_syntax_highlight_segments(code, param_colors))


def _draw_highlighted_text(
    ax,
    segments: Sequence[HighlightSegment],
    *,
    fontsize: int,
) -> None:
    """Render highlighted segments inside *ax* using renderer-measured metrics."""
    ax.axis("off")

    if not segments:
        ax.text(
            0.0,
            1.0,
            "(no code available)",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontfamily="monospace",
            fontsize=fontsize,
            color="#000000",
        )
        return

    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox = ax.get_window_extent(renderer=renderer)
    axis_width_points = max(bbox.width, 1.0)
    axis_height_points = max(bbox.height, 1.0)

    base_font = FontProperties(family="monospace", size=fontsize)
    _, base_height, base_descent = renderer.get_text_width_height_descent(
        "Ag", base_font, False
    )
    line_height_points = (base_height + base_descent) or (fontsize * 1.45)
    line_height = line_height_points / axis_height_points
    top_offset = line_height * 0.3

    current_x_points = 0.0
    row = 0

    for segment in segments:
        portions = segment.text.split("\n")
        for idx, part in enumerate(portions):
            if part:
                font = FontProperties(
                    family="monospace",
                    size=fontsize,
                    style="italic" if segment.italic else "normal",
                    weight="bold" if segment.bold else "normal",
                )
                text_width_points, _, _ = renderer.get_text_width_height_descent(
                    part, font, False
                )
                x = current_x_points / axis_width_points
                y = 1.0 - row * line_height - top_offset
                ax.text(
                    x,
                    y,
                    part,
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontfamily="monospace",
                    fontsize=fontsize,
                    fontstyle="italic" if segment.italic else "normal",
                    fontweight="bold" if segment.bold else "normal",
                    color=segment.hex_color or "#000000",
                    clip_on=True,
                )
                current_x_points += text_width_points
            if idx < len(portions) - 1:
                row += 1
                current_x_points = 0.0


def _format_history_sequence(values: List[str], color: ParamColor | None) -> str:
    if not values:
        return ""

    def _dim(text: str) -> str:
        return f"\x1b[2m{text}{ANSI_RESET}"

    tokens: List[str] = []
    index = 0
    total = len(values)

    while index < total:
        value = values[index]
        run_end = index + 1
        while run_end < total and values[run_end] == value:
            run_end += 1

        run_length = run_end - index
        formatted_value = (
            f"{_dim('(')}{color.ansi_code}{value}{ANSI_RESET}{_dim(')')}"
            if color
            else value
        )

        if run_length == 1:
            tokens.append(formatted_value)
        else:
            placeholders = "<" + "-" * (run_length - 1)
            tokens.append(f"{_dim(placeholders)}{formatted_value}")

        index = run_end

    return "  ".join(tokens)


def _format_param_history_line(
    param_id: str,
    values: List[str],
    param_colors: Dict[str, ParamColor],
) -> str:
    color = param_colors.get(param_id)
    colored_id = (
        f"{color.identifier_ansi_code}{param_id}{ANSI_RESET}" if color else param_id
    )
    sequence = _format_history_sequence(values, color)
    return f"{colored_id}: {sequence}"


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


def _save_param_plots(
    param_order: Sequence[str],
    histories: Mapping[str, List[str]],
    param_colors: Dict[str, ParamColor],
    syntax_segments: Sequence[HighlightSegment],
    output_path: Path | None = None,
) -> Path | None:
    numeric_histories: Dict[str, List[float]] = {}
    string_histories: Dict[str, List[str]] = {}
    for param_id in param_order:
        raw_values = histories.get(param_id)
        if not raw_values:
            continue
        converted: List[float] = []
        for value in raw_values:
            sanitized = value.replace("_", "").replace("'", "")
            if sanitized.lower().startswith("0x"):
                try:
                    converted.append(float(int(sanitized, 0)))
                    continue
                except ValueError:
                    pass
            sanitized = re.sub(r"[fFuUlL]+$", "", sanitized)
            try:
                converted.append(float(sanitized))
            except ValueError:
                converted = []
                break
        if converted:
            numeric_histories[param_id] = list(reversed(converted))
            string_histories[param_id] = list(reversed(raw_values))

    if not numeric_histories:
        return None

    count = len(numeric_histories)
    cols = max(1, min(3, count))
    rows = math.ceil(count / cols)
    syntax_plain = (
        "".join(segment.text for segment in syntax_segments) if syntax_segments else ""
    )
    syntax_lines = syntax_plain.splitlines() or [""]
    max_line_width = max(len(line) for line in syntax_lines)
    text_font_size = 12
    char_to_inches = text_font_size * 0.85 / 78.0
    text_width_inches = max(5.0, max_line_width * char_to_inches + 1.0)
    plot_width_inches = 2.6
    plot_height_inches = 2.8
    inner_width = text_width_inches + cols * plot_width_inches
    text_height_inches = max(len(syntax_lines), 1) * (text_font_size * 1.45) / 72.0
    inner_height = max(6.0, rows * plot_height_inches, text_height_inches + 0.8)
    left_margin = 0.05
    right_margin = 0.98
    top_margin = 0.95
    bottom_margin = 0.08
    figure_width = inner_width / (right_margin - left_margin)
    figure_height = inner_height / (top_margin - bottom_margin)

    fig = plt.figure(figsize=(figure_width, figure_height))
    grid = fig.add_gridspec(
        rows,
        cols + 1,
        width_ratios=[text_width_inches] + [plot_width_inches] * cols,
        wspace=0.5,
        hspace=0.8,
    )

    syntax_ax = fig.add_subplot(grid[:, 0])
    syntax_ax.set_title("Syntax Highlighted", fontsize=10, fontweight="bold", pad=6)
    syntax_ax.axis("off")

    plot_axes: List = []
    for index in range(rows * cols):
        row_index = index // cols
        col_index = index % cols
        ax = fig.add_subplot(grid[row_index, col_index + 1])
        if index < count:
            plot_axes.append(ax)
        else:
            ax.axis("off")

    for ax, (param_id, values) in zip(plot_axes, numeric_histories.items()):
        x_values = list(range(len(values)))
        color = param_colors.get(param_id)
        hex_color = color.hex_value if color else "#000000"
        ax.plot(x_values, values, color=hex_color, linewidth=1.4)

        if values:
            x_last = x_values[-1]
            y_last = values[-1]
            if len(values) >= 2:
                x_prev = x_values[-2]
                y_prev = values[-2]
            else:
                x_prev = x_last - 0.6
                y_prev = y_last

            dx = x_last - x_prev
            dy = y_last - y_prev
            if dx == dy == 0:
                dx = 1.0
                dy = 0.0

            magnitude = math.hypot(dx, dy)
            dir_x = dx / magnitude
            dir_y = dy / magnitude
            arrow_length = max(0.7, min(2.5, len(values) * 0.05 + 0.4))

            arrow_tip_x = x_last + dir_x * arrow_length
            arrow_tip_y = y_last + dir_y * arrow_length
            arrow_base_x = x_last - dir_x * arrow_length * 0.3
            arrow_base_y = y_last - dir_y * arrow_length * 0.3

            ax.annotate(
                "",
                xy=(arrow_tip_x, arrow_tip_y),
                xytext=(arrow_base_x, arrow_base_y),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=hex_color,
                    linewidth=1.4,
                    shrinkA=0,
                    shrinkB=0,
                    mutation_scale=16,
                ),
            )

            labels = string_histories.get(param_id)
            if labels and len(labels) == len(values):
                ax.annotate(
                    labels[0],
                    xy=(x_values[0], values[0]),
                    xytext=(-6, 8),
                    textcoords="offset points",
                    fontsize=11,
                    fontfamily="monospace",
                    color="#000000",
                    ha="right",
                    va="bottom",
                )
                ax.annotate(
                    labels[-1],
                    xy=(x_last, y_last),
                    xytext=(6, -10),
                    textcoords="offset points",
                    fontsize=11,
                    fontfamily="monospace",
                    color="#000000",
                    ha="left",
                    va="top",
                )

            x_min = min(x_values[0], arrow_base_x) - 0.5
            x_max = max(arrow_tip_x, x_last) + 0.8
            y_min = min(min(values), arrow_tip_y)
            y_max = max(max(values), arrow_tip_y)
            y_pad = (y_max - y_min) * 0.2
            if y_pad == 0:
                y_pad = 0.5
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min - y_pad, y_max + y_pad)

        ax.set_title(param_id, fontsize=11, color=hex_color)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_facecolor("white")

    fig.subplots_adjust(
        left=left_margin,
        right=right_margin,
        top=top_margin,
        bottom=bottom_margin,
    )
    _draw_highlighted_text(syntax_ax, syntax_segments, fontsize=text_font_size)
    if plot_axes:
        syn_box = syntax_ax.get_position()
        first_plot_box = plot_axes[0].get_position()
        separator_x = (syn_box.x1 + first_plot_box.x0) / 2
        separator = Line2D(
            [separator_x, separator_x],
            [bottom_margin, top_margin],
            transform=fig.transFigure,
            color="#d0d0d0",
            linewidth=1.0,
            alpha=0.9,
        )
        fig.add_artist(separator)

        plot_left = min(ax.get_position().x0 for ax in plot_axes)
        plot_right = max(ax.get_position().x1 for ax in plot_axes)
        arrow_y = bottom_margin * 0.6
        label_y = arrow_y + 0.015

        fig.text(
            (plot_left + plot_right) / 2,
            label_y,
            "Time (oldest \u2192 newest)",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

        arrow = FancyArrowPatch(
            (plot_left + 0.01, arrow_y),
            (plot_right - 0.01, arrow_y),
            transform=fig.transFigure,
            arrowstyle="-|>",
            mutation_scale=24,
            linewidth=2.2,
            color="#2c3e50",
        )
        fig.add_artist(arrow)
    output = output_path or Path("param_history_plots.png")
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output


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

    provided = [
        args.file is not None,
        args.start_line is not None,
        args.end_line is not None,
    ]
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
    param_colors = _assign_param_colors(param_order)
    annotated = result.annotated_code(show_all=args.show_all)
    syntax_segments = _syntax_highlight_segments(annotated, param_colors)
    plot_path = _save_param_plots(
        param_order,
        display_history,
        param_colors,
        syntax_segments,
    )

    if args.verbose:
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
        print(annotated)

        for snapshot in result.commits:
            _print_commit(snapshot, param_order)

    print("\n" + "=" * 80)
    print("PARAMETER HISTORY PLOTS")
    print("=" * 80)
    if plot_path:
        print(f"Saved plot image to {plot_path} (x-axis: oldest \u2192 newest)")
    else:
        print(
            "Unable to generate parameter plots "
            "(requires matplotlib and numeric parameter histories)."
        )

    print("\n" + "=" * 80)
    print("SYNTAX HIGHLIGHTED SNIPPET")
    print("=" * 80)
    print(_segments_to_ansi(syntax_segments))

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
            print(_format_param_history_line(param_id, history, param_colors))


if __name__ == "__main__":
    main()
