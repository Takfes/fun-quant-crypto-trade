# UV & Python Packaging Guide

## Distribution vs Packages

**Distribution** (what `pip list` shows):
- The installable unit managed by pip/uv
- Defined by `[project] name = "quantyx"`
- Has one version number
- Shows up in `uv pip list`

**Packages** (what you import):
- The Python modules inside the distribution
- Defined by `[tool.setuptools] packages = ["pytrade", "quanttak"]`
- What you use in `import` statements
- Can have multiple packages per distribution

```
📦 quantyx (distribution)
├── 📁 pytrade (package)
├── 📁 pyquants (package)
└── 📁 pyfolio (package)
```

**Result:**
```bash
uv pip list        # Shows: quantyx
```

```python
import pyfolio     # ✅ Works
import pyquants    # ✅ Works
import pytrade     # ✅ Works
```

```bash
python -c "import pyfolio; print(pyfolio.__file__)"
python -c "import pyquants; print(pyquants.__file__)"
python -c "import pytrade; print(pytrade.__file__)"
```

## Editable Mode (`-e`)

```bash
uv pip install -e .
```

**What it does:**
- Creates a **live link** to your source code instead of copying files
- Edits to `src/` are immediately available without reinstall
- Uses `.pth` files or egg-links pointing to your `src/` directory

**Without `-e`:**
```bash
uv pip install .
```
- Copies files to site-packages
- Requires reinstall after every code change

## PYTHONPATH

**What it is:** Environment variable telling Python where to look for importable modules.

**Python's import search order:**
1. Current working directory
2. `PYTHONPATH` directories
3. Standard library
4. site-packages (where pip/uv install things)

**Why it matters:**

Without installing packages:
```bash
python script.py
# import pytrade  ❌ ModuleNotFoundError
```

Workaround with PYTHONPATH:
```bash
export PYTHONPATH="/path/to/src:$PYTHONPATH"
python script.py
# import pytrade  ✅ Works
```

**Why installing is better:**
- Works in notebooks, IDEs, pytest without manual path manipulation
- No environment variable juggling
- Standard site-packages mechanism

## UV Sync vs Manual Process

### UV Sync (high-level, all-in-one)

```bash
uv sync                  # runtime deps + packages
uv sync --extra dev      # + dev deps (if using [project.optional-dependencies])
```

**What it does:**
1. Reads `pyproject.toml`
2. Creates/updates `.venv`
3. Installs dependencies + local packages in editable mode
4. Creates/updates `uv.lock` file

### Manual Process (step-by-step)

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -e .
uv pip install -e ".[dev]"  # if using [project.optional-dependencies]
```

**When to use manual:**
- More explicit control
- Debugging installation issues
- `uv sync` not working/available
- Always works, regardless of uv version

## Clean Setup

```bash
cd /path/to/project
rm -rf .venv
uv venv .venv
source .venv/bin/activate
uv pip install -e .

# Test
python -c "import pytrade; import quanttak; print('✅')"
```

## Key pyproject.toml Sections

```toml
[project]
name = "quantyx"        # Distribution name (pip list)
dependencies = [...]                    # Runtime deps

[project.optional-dependencies]         # Standard extras
dev = [...]

[tool.setuptools]
packages = ["pytrade", "pyquants" ,"pyfolio"]      # What you can import
package-dir = {"" = "src"}              # Where packages live
```

## Quick Reference

| Command | Purpose |
|---------|---------|
| `uv venv .venv` | Create empty virtualenv |
| `uv pip install -e .` | Install project + deps (editable) |
| `uv pip install -e ".[dev]"` | Install with dev extras |
| `uv pip list` | Show installed distributions |
| `uv pip show <name>` | Show distribution metadata |
| `python -c "import X; print(X.__file__)"` | Find package location |
