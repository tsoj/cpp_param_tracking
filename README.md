# cpp_param_tracking

Utilities for tracking changes to numeric parameters in C++ snippets using Tree-sitter.

## Setup


```bash
conda create -n cpp_param_tracking python=3.12
conda activate cpp_param_tracking
pip install tree-sitter tree-sitter-cpp
```

## CLI

```bash
python track_params.py --file PATH --start-line N --end-line M
```

Omit the file and line arguments to auto-detect the range from `git diff`. Use `--verbose` to surface matcher diagnostics and `--show-all` to include unchanged parameters in the summary.

## Tests

Run the test suite with:

```bash
pytest
```
