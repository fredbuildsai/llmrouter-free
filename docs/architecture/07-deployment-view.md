# 7 Deployment view

`llmrouter-free` is a library: it runs *inside* the host process. There is no server to deploy.

```mermaid
flowchart LR
    subgraph "Developer machine / CI runner"
        subgraph "Host process (Python 3.11/3.12)"
            Host[host application] --> Lib[llmrouter_free]
        end
        Env[".env / environment<br/>provider keys"] --> Host
        File[("SQLite file(s)<br/>ledger, optional metrics")]
        Lib --> File
        Oll["Ollama daemon :11434"]
    end
    Lib -- HTTPS --> Cloud[Provider APIs]
    Lib -- HTTP --> Oll
```

| Concern | Choice |
|---|---|
| Distribution | `pip install git+https://github.com/fredbuildsai/llmrouter-free.git@<tag>`; PyPI later |
| Configuration | One YAML file + environment variables; `llmrouter-free init` scaffolds it |
| Storage | SQLite file with WAL and a 30 s busy timeout (via `register_sqlite_pragmas`); any SQLAlchemy engine works |
| Default storage | Private temp-file SQLite when no engine is given (removed at exit) |
| CI | GitHub Actions: lint + tests on Python 3.11 and 3.12 |
