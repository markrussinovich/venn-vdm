# Upstream client checks

These scripts prove that unmodified third-party Decisions API clients work
against a Venn server. They are run by CI against `scripts/contract_server.py`,
which uses fake embeddings and an untrained head, so they test the wire format
and not decision quality.

| File | Client |
| --- | --- |
| `contract.mjs` | `@openrouter/sdk` and the `@effect/ai-openrouter` schemas |
| `playground.mjs` | The browser playground's JavaScript workflows, driven from Node |
| `plugin.py` | The `llm` CLI with the `llm-openrouter-decisions` plugin |

Run locally with a server on port 8000:

```bash
npm ci --ignore-scripts
node contract.mjs http://127.0.0.1:8000
node playground.mjs http://127.0.0.1:8000
python -m pip install -r requirements.txt
python plugin.py http://127.0.0.1:8000
```
