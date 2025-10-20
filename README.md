# cpp_param_tracking

Utilities for tracking changes to numeric parameters in C++ snippets using Tree-sitter.

## CLI

```
python track_params.py --file PATH --start-line N --end-line M
```

Omit the file and line arguments to auto-detect the range from `git diff`. Use `--verbose` to surface matcher diagnostics and `--show-all` to include unchanged parameters in the summary.

## Tests

Run the test suite with:

```
pytest
```
