.PHONY: install
install: ## Install the virtual environment and install the pre-commit hooks
	@echo "🚀 Creating virtual environment using uv"
	@uv sync
	@uv run pre-commit install

.PHONY: lock
lock: ## Update the lock file
	@echo "🔒 Updating lock file with uv lock"
	@uv lock

.PHONY: check
check: ## Run code quality tools.
	@echo "🚀 Checking lock file consistency with 'pyproject.toml'"
	@uv lock --locked
	@echo "🚀 Linting code: Running pre-commit"
	@uv run pre-commit run -a
	@echo "🚀 Static type checking: Running mypy"
	@uv run mypy

.PHONY: venv
venv: ## Show Python version, venv path, and installed packages
	@echo "🔍 Python environment information:"
	@uv run python --version
	@uv run python -c "import sys; print('Virtual env path:', sys.prefix)"
	@echo "📦 Installed packages:"
	@uv run pip list

.PHONY: reset-venv
reset-venv: ## Reset the virtual environment (requires confirmation)
	@echo "⚠️  This will delete your .venv and reinstall everything."
	@printf "Are you sure? [y/N] " && read ans && [ "${ans:-N}" = "y" ]
	@echo "🚀 Resetting environment..."
	@rm -rf .venv
	@uv sync
	@uv run pre-commit install

.PHONY: pre-commit
pre-commit: ## Run pre-commit hooks on staged files (as if committing)
	@echo "🚀 Running pre-commit hooks on staged files"
	@uv run pre-commit run

.PHONY: ruff-check
ruff-check: ## Run ruff linter on staged files
	@echo "🐍 Running ruff check on staged files"
	@uv run pre-commit run ruff-check

.PHONY: ruff-format
ruff-format: ## Run ruff formatter on staged files
	@echo "🐍 Running ruff format on staged files"
	@uv run pre-commit run ruff-format

.PHONY: interrogate
interrogate: ## Check docstring coverage on staged files
	@echo "📚 Running interrogate on staged files"
	@uv run pre-commit run interrogate

.PHONY: deptry
deptry: ## Analyze dependencies on staged files
	@echo "📦 Running deptry on staged files"
	@uv run pre-commit run deptry

.PHONY: gitleaks
gitleaks: ## Scan for hardcoded secrets on staged files
	@echo "🔒 Running gitleaks on staged files"
	@uv run pre-commit run gitleaks

.PHONY: file-checks
file-checks: ## Run all file validation checks on staged files
	@echo "📁 Running file validation checks on staged files"
	@uv run pre-commit run check-case-conflict
	@uv run pre-commit run check-merge-conflict
	@uv run pre-commit run end-of-file-fixer
	@uv run pre-commit run trailing-whitespace
	@uv run pre-commit run check-added-large-files
	@uv run pre-commit run mixed-line-ending
	@uv run pre-commit run check-json
	@uv run pre-commit run pretty-format-json
	@uv run pre-commit run check-toml
	@uv run pre-commit run check-yaml
	@uv run pre-commit run check-xml
	@uv run pre-commit run check-executables-have-shebangs
	@uv run pre-commit run check-shebang-scripts-are-executable

.PHONY: clean-cache
clean-cache: ## Remove cache and temporary files
	@echo "🧹 Running cache cleanup"
	@uv run pre-commit run clean-cache-files

.PHONY: lint
lint: ruff-check ruff-format interrogate deptry ## Run all Python linting and analysis hooks on staged files
	@echo "✅ All Python linting checks completed"

.PHONY: validate
validate: file-checks gitleaks clean-cache ## Run all validation and security checks on staged files
	@echo "✅ All validation checks completed"

.PHONY: hooks
hooks: lint validate ## Run all pre-commit hooks on staged files (consolidated)
	@echo "✅ All pre-commit hooks completed successfully"

.PHONY: test
test: ## Test the code with pytest
	@echo "🚀 Testing code: Running pytest"
	@uv run python -m pytest --doctest-modules

.PHONY: build
build: clean-build ## Build wheel file
	@echo "🚀 Creating wheel file"
	@uvx --from build pyproject-build --installer uv

.PHONY: clean-build
clean-build: ## Clean build artifacts
	@echo "🚀 Removing build artifacts"
	@uv run python -c "import shutil; import os; shutil.rmtree('dist') if os.path.exists('dist') else None"

.PHONY: docs-test
docs-test: ## Test if documentation can be built without warnings or errors
	@uv run mkdocs build -s

.PHONY: docs
docs: ## Build and serve the documentation
	@uv run mkdocs serve

.PHONY: help
help:
	@uv run python -c "import re; \
	[[print(f'\033[36m{m[0]:<20}\033[0m {m[1]}') for m in re.findall(r'^([a-zA-Z_-]+):.*?## (.*)$$', open(makefile).read(), re.M)] for makefile in ('$(MAKEFILE_LIST)').strip().split()]"

.DEFAULT_GOAL := help
