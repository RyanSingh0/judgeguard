"""`judgeguard` command line.

judgeguard status                      what mode am I in, what is cached
judgeguard degrade --show numeric_swap  eyeball a degradation before trusting it
judgeguard judge "some answer"          one-off judge call
judgeguard guard  "some answer"         one-off guardrail call, with latency
judgeguard findings                     the three headline findings
judgeguard serve                        run the API + demo UI
"""

from __future__ import annotations

import json

import typer
from rich.console import Console
from rich.table import Table

from judgeguard import __version__

app = typer.Typer(add_completion=False, help=__doc__)
console = Console()


@app.command()
def status() -> None:
    """Provider mode, model registry and cache state."""
    from judgeguard.config import get_settings, load_registry
    from judgeguard.providers.cache import cache_stats
    from judgeguard.providers.registry import is_simulated
    from judgeguard.store import RESULTS_DIR

    s, reg = get_settings(), load_registry()
    console.print(
        f"[bold]JudgeGuard[/bold] v{__version__}   mode=[cyan]{s.effective_mode().value}[/cyan]"
    )
    t = Table("alias", "provider", "family", "tier", "served by")
    for a in reg.panel:
        m = reg.by_alias(a)
        t.add_row(
            a,
            m.provider,
            m.family,
            m.tier,
            "[yellow]simulator[/yellow]" if is_simulated(a) else "[green]live API[/green]",
        )
    console.print(t)
    console.print(f"cache: {cache_stats()}")
    console.print(f"results: {sorted(p.stem for p in RESULTS_DIR.glob('*.json'))}")


@app.command()
def degrade(
    kind: str = typer.Option("numeric_swap", "--show", help="degradation name"),
    severity: float = 0.5,
) -> None:
    """Print a reference answer next to its degraded variant, with the audit trail."""
    from judgeguard.data.load import load_items
    from judgeguard.degrade.text import DEGRADATIONS

    if kind not in DEGRADATIONS:
        raise typer.BadParameter(f"choose from {list(DEGRADATIONS)}")
    item = load_items(3)[0]
    v = DEGRADATIONS[kind](item, severity, seed=0, neighbours=load_items(20))
    console.print(f"[bold]{item.question}[/bold]\n")
    console.print(f"[green]REFERENCE[/green]\n{item.reference}\n")
    console.print(f"[red]{kind} @ severity {severity}[/red]\n{v.text}\n")
    console.print(f"[dim]audit: {v.edit}[/dim]")
    console.print(
        f"[dim]length ratio: {v.len_ratio:.2f}  introduces_error={v.introduces_error}[/dim]"
    )


@app.command()
def judge(
    answer: str,
    question: str = "",
    context: str = "",
    model: str = typer.Option("", "--model"),
    config: str = "cot",
) -> None:
    """Score one answer with an LLM judge."""
    from judgeguard.config import load_registry
    from judgeguard.data.schema import Item
    from judgeguard.judges.parse import parse_score
    from judgeguard.judges.prompts import SYSTEM, score_prompt
    from judgeguard.providers.registry import complete

    alias = model or load_registry().panel[0]
    item = Item(id="cli", domain="cli", context=context, question=question, reference="")
    c = complete(
        alias,
        score_prompt(config, item, answer),
        system=SYSTEM,
        max_tokens=384,
        meta={"task": "score", "config": config, "uid": "cli"},
    )
    console.print(
        f"[bold]{alias}[/bold] ({config})  score=[cyan]{parse_score(c.text)}[/cyan]  "
        f"{c.latency_ms:.0f} ms  served={c.model_served}"
    )
    console.print(c.text)


@app.command()
def guard(answer: str, question: str = "") -> None:
    """Run the inline guardrail on one answer."""
    from judgeguard.distill.train import Student
    from judgeguard.store import RESULTS_DIR

    path = RESULTS_DIR / "student_model.joblib"
    if not path.exists():
        console.print("[red]No student model. Run: python experiments/09_distill.py[/red]")
        raise typer.Exit(1)
    out = Student.load(path).guard(question, answer)
    colour = "green" if out["decision"] == "allow" else "red"
    console.print(
        f"[{colour}]{out['decision'].upper()}[/{colour}]  "
        f"p={out['probability_good']:.3f}  threshold={out['threshold']:.2f}  "
        f"{out['latency_ms']:.2f} ms"
    )


@app.command()
def findings() -> None:
    """The headline findings, computed from results/ rather than typed by hand."""
    from judgeguard.serve.app import summary

    for f in summary()["findings"]:
        console.print(
            f"[bold cyan]{f['id']}[/bold cyan]\n  {f['headline']}\n  [dim]{f['why']}[/dim]\n"
        )


@app.command()
def report(experiment: str = typer.Argument("01_discrimination")) -> None:
    """Dump one experiment's result JSON."""
    from judgeguard.store import load

    d = load(experiment)
    d.pop("records", None)
    console.print_json(json.dumps(d)[:20000])


@app.command()
def serve(host: str = "0.0.0.0", port: int = 8080, reload: bool = False) -> None:
    """Run the API and demo UI."""
    import uvicorn

    uvicorn.run("judgeguard.serve.app:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    app()
