# 2 Constraints

## Technical
- **Python 3.11 / 3.12**, synchronous API (LiteLLM's blocking `completion`).
- **LiteLLM** is the single provider abstraction; provider quirks are handled by passing `extra_body`, not by per-provider code.
- **SQLAlchemy 2** for persistence, portable column types only (SQLite in practice, PostgreSQL-compatible).
- Installed **from GitHub** (not PyPI) for now; a plain `pip install git+https://...` must work with no extra steps.

## Organizational
- MIT licensed, public repository, developed by one maintainer alongside two sibling projects.
- Public API changes follow SemVer; consumers pin a tag.

## Conventions
- Every behaviour change ships with a test; the suite must pass before every commit.
- Secrets live only in environment variables; config files hold the *name* of the variable, never the value.
- Free-tier reality drives defaults: limits are opt-in per deployment, cooldowns are conservative.
