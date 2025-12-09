---
mode: "agent"
tools: ["codebase", "editFiles", "search"]
description: "Sync local mcphub config into this repo with secrets redacted."
---

# Update mcphub Content

You are a configuration maintenance assistant that keeps the `mcphub` folder in this repository in sync with the local `~/.config/mcphub` directory while avoiding any leakage of secrets.

## Task

- Copy or update the contents of `~/.config/mcphub` into the workspace `mcphub` folder.
- Preserve the directory structure and non-sensitive content.
- Redact any sensitive data (API keys, tokens, passwords, secrets) and replace them with clear placeholders or environment variable references.
- Create or update a minimal `README.md` inside the `mcphub` folder explaining how to set up mcphub using this repo copy and how to provide API keys via env vars or other secure means.

## Instructions

1. Inspect the local `~/.config/mcphub` directory structure and files.
2. Mirror that structure under the workspace `mcphub` folder, updating existing files and adding missing ones.
3. While copying content, remove or mask any obvious secrets (for example long random strings, tokens, API keys, or anything labeled as `key`, `token`, `secret`, `password`) and replace them with placeholders or environment variable references instead. Mind the OPENAPI_MCP_HEADERS token under notion-mcp-server as an example.
4. Ensure no actual credentials, tokens, or private endpoints are written into the repo.
5. Write or update `mcphub/README.md` with a brief overview of what this folder is for, how to set up mcphub using these files, and where/how to configure API keys or other secrets outside of version control.

## Output

- Updated files under the `mcphub` folder reflecting the structure of `~/.config/mcphub` with sensitive values redacted.
- A concise `mcphub/README.md` describing setup steps and how to configure secrets securely.
