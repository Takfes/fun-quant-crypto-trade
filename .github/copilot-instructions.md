# Project coding standards

### Project Scaffold

- Split components into appropriate modules under an src project to be hosted in the root directory.
- Assume the project environment is already setup. Don't bother with development tools unless excplicitly instructed.

## Project Context

- Adheres to PEP8 standards, utilizes type hints, and formats code with `black`.
- Use ALL_CAPS for constants

## Coding Standards

- Organize code into modular components, each not exceeding 500 lines.
- Use consistent naming conventions and relative imports within packages.
- Implement unit tests using Pytest, ensuring coverage for typical use, edge cases, and failure scenarios.

## Documentation

- Write docstrings for every function using the Google style.
- Comment non-obvious code and add inline `# Reason:` comments for complex logic.
- Regularly update `WRITEME.md` with progress and decisions.

## Error Handling

- Use try/catch blocks for async operations
- Implement proper error boundaries in React components
- Always log errors with contextual information

## AI Behavior Rules

- Avoid assumptions; ask clarifying questions when context is missing.
- Do not hallucinate libraries or functions; use only verified Python packages.
- Confirm file paths and module names before referencing them.
