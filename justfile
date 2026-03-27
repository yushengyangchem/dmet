# Execute all recipes in strict bash mode.

set shell := ["bash", "-euo", "pipefail", "-c"]

# Python interpreter used by build/test recipes.

PYTHON := "python3"
repo_root := `git rev-parse --show-toplevel`

# Show available recipes.
default:
    just --list --unsorted

# Remove build and Python cache artifacts.
clean:
    cd {{ repo_root }}
    rm -rf dist
    rm -rf .pytest_cache
    find . -type f -name '*.pyc' -not -path './.direnv/*' -delete
    find . -type d -name '__pycache__' -not -path './.direnv/*' -prune -exec rm -rf {} +

# Build wheel/sdist with PEP 517 backend.
build:
    cd {{ repo_root }} && {{ PYTHON }} -m build

# Build package, validate, and upload to PyPI via twine.
twine:
    cd {{ repo_root }} && rm -rf dist
    cd {{ repo_root }} && {{ PYTHON }} -m build
    cd {{ repo_root }} && twine check dist/*
    cd {{ repo_root }} && twine upload dist/*
    cd {{ repo_root }} && rm -rf dist

# Run pytest; pass through extra flags, e.g. `just test -k example`.
test *flags:
    cd {{ repo_root }} && {{ PYTHON }} -m pytest tests {{ flags }} -v

# Update all flake inputs and refresh flake.lock.
update:
    cd {{ repo_root }} && nix flake update
