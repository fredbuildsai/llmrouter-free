"""`llmrouter-free` command line: inspect deployment status, send a test prompt, scaffold a config."""

import shutil
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from sqlalchemy import create_engine

from llmrouter_free.config import TEMPLATE_PATH, build_router, load_routes
from llmrouter_free.router import AllDeploymentsExhausted
from llmrouter_free.store import register_sqlite_pragmas

app = typer.Typer(help="Quota-aware LLM failover router for free-tier providers.", no_args_is_help=True)
console = Console()

CONFIG_OPTION = typer.Option(Path("llm_routes.yaml"), "--config", "-c", help="Router config (llm_routes.yaml)")
DB_OPTION = typer.Option(
    Path("llmrouter.db"), "--db", help="SQLite file holding the call ledger / response cache"
)


def _router(config_path: Path, db_path: Path):
    load_dotenv()  # provider API keys live in .env, like every other tool in this family
    engine = register_sqlite_pragmas(create_engine(f"sqlite:///{db_path.resolve()}", future=True))
    return build_router(load_routes(config_path), engine=engine)


@app.command()
def init(target: Path = typer.Argument(Path("llm_routes.yaml"), help="Where to write the starter config")) -> None:
    """Write the annotated starter `llm_routes.yaml` (deployments, routes, cooldowns, limits)."""
    if target.exists():
        console.print(f"{target}: already exists, left alone")
        raise typer.Exit(1)
    shutil.copy2(TEMPLATE_PATH, target)
    console.print(f"wrote {target} - edit the deployments/routes, then put provider keys in .env")


@app.command()
def status(config: Path = CONFIG_OPTION, db: Path = DB_OPTION) -> None:
    """Show each deployment's availability, requests used today and cooldown."""
    router = _router(config, db)
    table = Table("name", "model", "tier", "family", "enabled", "used today", "rpd", "cooldown s")
    for row in router.status():
        table.add_row(row["name"], row["model"], row["tier"], row["family"], "yes" if row["enabled"] else "no",
                      str(row["used_today"]), str(row["rpd"] or "∞"), str(row["cooldown_s"]))
    console.print(table)
    console.print(f"Paid spend today: ${router.spent_today_usd():.4f} (allow_paid={router.allow_paid})")


@app.command()
def test(
    route: str = typer.Option(..., help="Route name from the config"),
    prompt: str = typer.Option("In one sentence, what is a rate limit?", help="Prompt to send"),
    config: Path = CONFIG_OPTION,
    db: Path = DB_OPTION,
) -> None:
    """Send one prompt through a route and show which deployment answered."""
    try:
        result = _router(config, db).complete(route, [{"role": "user", "content": prompt}], use_cache=False)
    except AllDeploymentsExhausted as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    console.print(f"[bold]{result.deployment}[/bold] ({result.model})\n{result.text}")


if __name__ == "__main__":
    app()
