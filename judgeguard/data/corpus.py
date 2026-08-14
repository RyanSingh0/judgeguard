"""A procedurally generated reference corpus with explicit fact structure.

Why not just download SQuAD or CNN/DailyMail?
--------------------------------------------
Because the method needs *composable* gold answers. Degradation-based gold
labelling is only defensible if you can state exactly what a perturbation
destroyed. With a free-text gold answer you can delete a sentence, but you
cannot enumerate which required facts went with it, and you certainly cannot
regenerate a fluent answer that is identical except for one corrupted number.

Here every reference answer is the join of per-fact clauses. Dropping a fact and
recomposing yields a fluent, grammatical answer that is *provably* missing
exactly one piece of required information. That is the whole trick.

A real-corpus loader is provided in ``load.py`` for the live setting; this
generator is what makes the pipeline runnable, seeded and reproducible with no
network access.
"""

from __future__ import annotations

import random
from functools import lru_cache
from typing import Any

from judgeguard.data.schema import Fact, FactKind, Item

DOMAINS: list[str] = [
    "clinical_trial",
    "quarterly_earnings",
    "materials_experiment",
    "transit_operations",
    "climate_station",
    "product_benchmark",
    "epidemiology_survey",
    "energy_grid",
    "agronomy_field_trial",
    "network_incident",
]

_DRUGS = ["Verazolimab", "Ostenacin", "Palviritide", "Kanterol", "Emrisidine", "Torvaxen"]
_CONDITIONS = [
    "treatment-resistant hypertension",
    "moderate plaque psoriasis",
    "stage II diabetic nephropathy",
    "chronic migraine",
    "early rheumatoid arthritis",
]
_COMPANIES = [
    "Northwind Logistics",
    "Arclight Semiconductor",
    "Belmont Foods",
    "Verity Health",
    "Kestrel Robotics",
]
_ALLOYS = [
    "A-712 titanium alloy",
    "CX-9 nickel superalloy",
    "M-40 magnesium composite",
    "S-221 steel laminate",
]
_CITIES = ["Sacramento", "Rotterdam", "Fukuoka", "Belo Horizonte", "Gothenburg", "Pune"]
_LINES = ["the Orange Line", "the Riverside branch", "the C4 express corridor", "the Harbour loop"]
_CHIPS = ["Halcyon M2", "Tessera 400", "Orbit V9", "Northstar A1"]
_CROPS = ["winter wheat", "soybean", "sorghum", "durum wheat"]
_SERVICES = ["the checkout API", "the identity service", "the media pipeline", "the search tier"]


def _pct(rng: random.Random, lo: float, hi: float) -> str:
    return f"{rng.uniform(lo, hi):.1f}"


def _int(rng: random.Random, lo: int, hi: int) -> str:
    return str(rng.randint(lo, hi))


def _build(
    idx: int,
    domain: str,
    context: str,
    question: str,
    facts: list[Fact],
) -> Item:
    """Compose the item, embedding the fact VALUES in the source context.

    This is load-bearing and was found the hard way on the first live run. If the
    context only names which quantities were reported, a reference-free judge has
    no way to verify any number -- and it does not fail quietly. Asked to score
    the *correct* reference, qwen3.6-27b returned:

        "SCORE: 2 -- The candidate answer fabricates specific numerical values
         for all trial metrics that are entirely absent from the provided source
         context."

    The judge was right. Under that design the reference and a numeric_swap
    variant are equally unsupported, discrimination accuracy collapses to chance
    for the wrong reason, and the experiment measures the corpus rather than the
    judge. Putting the values in the record makes numeric substitution *checkable
    from the prompt*, which is what turns "the judge missed it" into a finding
    rather than an artefact.
    """
    record = "\n".join(f"  - {f.key.replace('_', ' ')}: {f.render()}" for f in facts)
    context = f"{context}\n\nSOURCE RECORD\n{record}"
    reference = " ".join(f.sentence() for f in facts)
    return Item(
        id=f"{domain[:4]}-{idx:04d}",
        domain=domain,
        context=context,
        question=question,
        reference=reference,
        facts=facts,
    )


def _clinical_trial(rng: random.Random, idx: int) -> Item:
    drug, cond = rng.choice(_DRUGS), rng.choice(_CONDITIONS)
    n, sites = _int(rng, 180, 4200), _int(rng, 4, 62)
    weeks = _int(rng, 12, 78)
    red, p, adv = _pct(rng, 8, 41), f"{rng.uniform(0.0005, 0.043):.4f}", _pct(rng, 2, 19)
    facts = [
        Fact(
            key="enrolled",
            value=n,
            kind=FactKind.NUMERIC,
            unit="participants",
            clause="The trial enrolled {value}.",
        ),
        Fact(
            key="sites",
            value=sites,
            kind=FactKind.NUMERIC,
            unit="sites",
            clause="Recruitment ran across {value}.",
        ),
        Fact(
            key="duration",
            value=weeks,
            kind=FactKind.NUMERIC,
            unit="weeks",
            clause="The treatment period lasted {value}.",
        ),
        Fact(
            key="effect",
            value=red,
            kind=FactKind.NUMERIC,
            unit="percent",
            clause="The primary endpoint improved by {value} relative to placebo.",
        ),
        Fact(
            key="p_value",
            value=p,
            kind=FactKind.NUMERIC,
            clause="The effect was statistically significant at p = {value}.",
        ),
        Fact(
            key="adverse",
            value=adv,
            kind=FactKind.NUMERIC,
            unit="percent",
            clause="Serious adverse events occurred in {value} of the treatment arm.",
        ),
    ]
    ctx = (
        f"Registry record for a randomised placebo-controlled trial of {drug} in adults with "
        f"{cond}. The sponsor reported enrolment, site count, treatment duration, the primary "
        f"efficacy endpoint, its significance level, and the serious adverse event rate."
    )
    q = f"Summarise the primary results of the {drug} trial in {cond}."
    return _build(idx, "clinical_trial", ctx, q, facts)


def _quarterly_earnings(rng: random.Random, idx: int) -> Item:
    co = rng.choice(_COMPANIES)
    rev, growth = f"{rng.uniform(41, 980):.1f}", _pct(rng, -12, 34)
    margin, eps = _pct(rng, 4, 38), f"{rng.uniform(0.11, 4.2):.2f}"
    heads, guide = _int(rng, 300, 21000), _pct(rng, -6, 22)
    facts = [
        Fact(
            key="revenue",
            value=rev,
            kind=FactKind.NUMERIC,
            unit="million USD",
            clause="Quarterly revenue was {value}.",
        ),
        Fact(
            key="growth",
            value=growth,
            kind=FactKind.NUMERIC,
            unit="percent",
            clause="That represents year-over-year growth of {value}.",
        ),
        Fact(
            key="margin",
            value=margin,
            kind=FactKind.NUMERIC,
            unit="percent",
            clause="Gross margin came in at {value}.",
        ),
        Fact(
            key="eps",
            value=eps,
            kind=FactKind.NUMERIC,
            unit="USD",
            clause="Diluted earnings per share were {value}.",
        ),
        Fact(
            key="headcount",
            value=heads,
            kind=FactKind.NUMERIC,
            unit="employees",
            clause="The company ended the quarter with {value}.",
        ),
        Fact(
            key="guidance",
            value=guide,
            kind=FactKind.NUMERIC,
            unit="percent",
            clause="Management guided to next-quarter revenue growth of {value}.",
        ),
    ]
    ctx = (
        f"{co} filed its quarterly report covering revenue, year-over-year growth, gross margin, "
        f"diluted EPS, closing headcount and forward guidance."
    )
    return _build(idx, "quarterly_earnings", ctx, f"What did {co} report this quarter?", facts)


def _materials_experiment(rng: random.Random, idx: int) -> Item:
    alloy = rng.choice(_ALLOYS)
    temp, strength = _int(rng, 380, 1240), f"{rng.uniform(210, 1450):.0f}"
    elong, cycles = _pct(rng, 2, 27), _int(rng, 12000, 940000)
    dens, cost = f"{rng.uniform(1.7, 8.9):.2f}", f"{rng.uniform(11, 240):.1f}"
    facts = [
        Fact(
            key="temperature",
            value=temp,
            kind=FactKind.NUMERIC,
            unit="degrees Celsius",
            clause="Specimens were tested at {value}.",
        ),
        Fact(
            key="strength",
            value=strength,
            kind=FactKind.NUMERIC,
            unit="MPa",
            clause="Ultimate tensile strength measured {value}.",
        ),
        Fact(
            key="elongation",
            value=elong,
            kind=FactKind.NUMERIC,
            unit="percent",
            clause="Elongation at failure was {value}.",
        ),
        Fact(
            key="cycles",
            value=cycles,
            kind=FactKind.NUMERIC,
            unit="cycles",
            clause="Fatigue life reached {value} before crack initiation.",
        ),
        Fact(
            key="density",
            value=dens,
            kind=FactKind.NUMERIC,
            unit="grams per cubic centimetre",
            clause="Measured density was {value}.",
        ),
        Fact(
            key="cost",
            value=cost,
            kind=FactKind.NUMERIC,
            unit="USD per kilogram",
            clause="Estimated material cost was {value}.",
        ),
    ]
    ctx = (
        f"A mechanical characterisation study of {alloy} recorded test temperature, ultimate "
        f"tensile strength, elongation at failure, fatigue life, density and material cost."
    )
    return _build(
        idx, "materials_experiment", ctx, f"Report the characterisation results for {alloy}.", facts
    )


def _transit_operations(rng: random.Random, idx: int) -> Item:
    line, city = rng.choice(_LINES), rng.choice(_CITIES)
    riders, otp = _int(rng, 4200, 310000), _pct(rng, 61, 97)
    headway, delay = _int(rng, 3, 22), f"{rng.uniform(0.4, 11.5):.1f}"
    stops, cost = _int(rng, 8, 44), f"{rng.uniform(1.1, 6.8):.2f}"
    facts = [
        Fact(
            key="ridership",
            value=riders,
            kind=FactKind.NUMERIC,
            unit="average weekday boardings",
            clause="The line carried {value}.",
        ),
        Fact(
            key="otp",
            value=otp,
            kind=FactKind.NUMERIC,
            unit="percent",
            clause="On-time performance was {value}.",
        ),
        Fact(
            key="headway",
            value=headway,
            kind=FactKind.NUMERIC,
            unit="minutes",
            clause="Peak headway was {value}.",
        ),
        Fact(
            key="delay",
            value=delay,
            kind=FactKind.NUMERIC,
            unit="minutes",
            clause="Mean passenger delay was {value}.",
        ),
        Fact(
            key="stops",
            value=stops,
            kind=FactKind.NUMERIC,
            unit="stops",
            clause="The route serves {value}.",
        ),
        Fact(
            key="fare",
            value=cost,
            kind=FactKind.NUMERIC,
            unit="USD",
            clause="The base fare was {value}.",
        ),
    ]
    ctx = (
        f"Monthly operations summary for {line} in {city}, covering ridership, on-time "
        f"performance, peak headway, mean delay, stop count and base fare."
    )
    return _build(idx, "transit_operations", ctx, f"How did {line} perform last month?", facts)


def _climate_station(rng: random.Random, idx: int) -> Item:
    city = rng.choice(_CITIES)
    mean_t, anom = f"{rng.uniform(-4, 31):.1f}", f"{rng.uniform(-1.8, 2.9):+.2f}"
    rain, days = _int(rng, 2, 410), _int(rng, 1, 28)
    wind, humid = f"{rng.uniform(1.2, 19.4):.1f}", _pct(rng, 21, 94)
    facts = [
        Fact(
            key="mean_temp",
            value=mean_t,
            kind=FactKind.NUMERIC,
            unit="degrees Celsius",
            clause="The monthly mean temperature was {value}.",
        ),
        Fact(
            key="anomaly",
            value=anom,
            kind=FactKind.NUMERIC,
            unit="degrees Celsius",
            clause="That is an anomaly of {value} against the 1991-2020 baseline.",
        ),
        Fact(
            key="precipitation",
            value=rain,
            kind=FactKind.NUMERIC,
            unit="millimetres",
            clause="Total precipitation reached {value}.",
        ),
        Fact(
            key="wet_days",
            value=days,
            kind=FactKind.NUMERIC,
            unit="days",
            clause="Measurable rain fell on {value}.",
        ),
        Fact(
            key="wind",
            value=wind,
            kind=FactKind.NUMERIC,
            unit="metres per second",
            clause="Mean wind speed was {value}.",
        ),
        Fact(
            key="humidity",
            value=humid,
            kind=FactKind.NUMERIC,
            unit="percent",
            clause="Mean relative humidity was {value}.",
        ),
    ]
    ctx = (
        f"Monthly climate summary from the {city} reference station: mean temperature and its "
        f"anomaly, total precipitation, wet-day count, mean wind speed and relative humidity."
    )
    return _build(
        idx,
        "climate_station",
        ctx,
        f"Summarise last month's conditions at the {city} station.",
        facts,
    )


def _product_benchmark(rng: random.Random, idx: int) -> Item:
    chip = rng.choice(_CHIPS)
    thr, lat = _int(rng, 120, 9800), f"{rng.uniform(0.8, 74.0):.1f}"
    watts, price = _int(rng, 12, 410), _int(rng, 180, 5400)
    mem, uplift = _int(rng, 8, 192), _pct(rng, 3, 62)
    facts = [
        Fact(
            key="throughput",
            value=thr,
            kind=FactKind.NUMERIC,
            unit="tokens per second",
            clause="Sustained throughput measured {value}.",
        ),
        Fact(
            key="latency",
            value=lat,
            kind=FactKind.NUMERIC,
            unit="milliseconds",
            clause="Median request latency was {value}.",
        ),
        Fact(
            key="power",
            value=watts,
            kind=FactKind.NUMERIC,
            unit="watts",
            clause="Package power draw averaged {value}.",
        ),
        Fact(
            key="price",
            value=price,
            kind=FactKind.NUMERIC,
            unit="USD",
            clause="List price is {value}.",
        ),
        Fact(
            key="memory",
            value=mem,
            kind=FactKind.NUMERIC,
            unit="gigabytes",
            clause="On-board memory is {value}.",
        ),
        Fact(
            key="uplift",
            value=uplift,
            kind=FactKind.NUMERIC,
            unit="percent",
            clause="It is faster than the previous generation by {value}.",
        ),
    ]
    ctx = (
        f"An independent lab benchmarked the {chip} accelerator for throughput, median latency, "
        f"power draw, list price, on-board memory and generational uplift."
    )
    return _build(idx, "product_benchmark", ctx, f"What were the {chip} benchmark results?", facts)


def _epidemiology_survey(rng: random.Random, idx: int) -> Item:
    city = rng.choice(_CITIES)
    n, prev = _int(rng, 900, 42000), _pct(rng, 1, 29)
    ci_w, resp = _pct(rng, 0.4, 4.2), _pct(rng, 32, 88)
    age, ratio = f"{rng.uniform(31, 62):.1f}", f"{rng.uniform(0.7, 2.4):.2f}"
    facts = [
        Fact(
            key="sample",
            value=n,
            kind=FactKind.NUMERIC,
            unit="respondents",
            clause="The survey sampled {value}.",
        ),
        Fact(
            key="prevalence",
            value=prev,
            kind=FactKind.NUMERIC,
            unit="percent",
            clause="Weighted prevalence was {value}.",
        ),
        Fact(
            key="ci_width",
            value=ci_w,
            kind=FactKind.NUMERIC,
            unit="percentage points",
            clause="The 95 percent confidence interval spanned {value}.",
        ),
        Fact(
            key="response_rate",
            value=resp,
            kind=FactKind.NUMERIC,
            unit="percent",
            clause="The response rate was {value}.",
        ),
        Fact(
            key="median_age",
            value=age,
            kind=FactKind.NUMERIC,
            unit="years",
            clause="Median respondent age was {value}.",
        ),
        Fact(
            key="sex_ratio",
            value=ratio,
            kind=FactKind.NUMERIC,
            clause="The female-to-male prevalence ratio was {value}.",
        ),
    ]
    ctx = (
        f"A cross-sectional health survey in {city} reported sample size, weighted prevalence, "
        f"confidence interval width, response rate, median age and the sex prevalence ratio."
    )
    return _build(
        idx, "epidemiology_survey", ctx, f"What did the {city} prevalence survey find?", facts
    )


def _energy_grid(rng: random.Random, idx: int) -> Item:
    city = rng.choice(_CITIES)
    peak, ren = _int(rng, 400, 18000), _pct(rng, 9, 78)
    curtail, price = _pct(rng, 0.2, 12.0), f"{rng.uniform(11, 240):.1f}"
    outage, storage = _int(rng, 0, 340), _int(rng, 5, 1900)
    facts = [
        Fact(
            key="peak_demand",
            value=peak,
            kind=FactKind.NUMERIC,
            unit="megawatts",
            clause="Peak demand reached {value}.",
        ),
        Fact(
            key="renewable_share",
            value=ren,
            kind=FactKind.NUMERIC,
            unit="percent",
            clause="Renewables supplied {value} of generation.",
        ),
        Fact(
            key="curtailment",
            value=curtail,
            kind=FactKind.NUMERIC,
            unit="percent",
            clause="Curtailment amounted to {value} of available renewable output.",
        ),
        Fact(
            key="price",
            value=price,
            kind=FactKind.NUMERIC,
            unit="USD per megawatt hour",
            clause="The average wholesale price was {value}.",
        ),
        Fact(
            key="outage_minutes",
            value=outage,
            kind=FactKind.NUMERIC,
            unit="customer-minutes",
            clause="Unplanned outages totalled {value}.",
        ),
        Fact(
            key="storage",
            value=storage,
            kind=FactKind.NUMERIC,
            unit="megawatt hours",
            clause="Installed battery storage stood at {value}.",
        ),
    ]
    ctx = (
        f"The {city} balancing authority published peak demand, renewable share, curtailment, "
        f"average wholesale price, unplanned outage minutes and installed storage."
    )
    return _build(
        idx, "energy_grid", ctx, f"Summarise grid operations for the {city} balancing area.", facts
    )


def _agronomy_field_trial(rng: random.Random, idx: int) -> Item:
    crop = rng.choice(_CROPS)
    yld, gain = f"{rng.uniform(1.9, 9.8):.2f}", _pct(rng, 2, 31)
    n_rate, water = _int(rng, 40, 260), _int(rng, 90, 720)
    plots, protein = _int(rng, 6, 96), _pct(rng, 8, 17)
    facts = [
        Fact(
            key="yield",
            value=yld,
            kind=FactKind.NUMERIC,
            unit="tonnes per hectare",
            clause="Mean yield was {value}.",
        ),
        Fact(
            key="gain",
            value=gain,
            kind=FactKind.NUMERIC,
            unit="percent",
            clause="That is a gain over the control of {value}.",
        ),
        Fact(
            key="nitrogen",
            value=n_rate,
            kind=FactKind.NUMERIC,
            unit="kilograms per hectare",
            clause="The nitrogen application rate was {value}.",
        ),
        Fact(
            key="irrigation",
            value=water,
            kind=FactKind.NUMERIC,
            unit="millimetres",
            clause="Seasonal irrigation totalled {value}.",
        ),
        Fact(
            key="plots",
            value=plots,
            kind=FactKind.NUMERIC,
            unit="replicate plots",
            clause="The design used {value}.",
        ),
        Fact(
            key="protein",
            value=protein,
            kind=FactKind.NUMERIC,
            unit="percent",
            clause="Grain protein content was {value}.",
        ),
    ]
    ctx = (
        f"A replicated {crop} field trial reported mean yield, gain over control, nitrogen rate, "
        f"seasonal irrigation, replicate count and grain protein content."
    )
    return _build(
        idx, "agronomy_field_trial", ctx, f"What were the results of the {crop} field trial?", facts
    )


def _network_incident(rng: random.Random, idx: int) -> Item:
    svc = rng.choice(_SERVICES)
    mins, users = _int(rng, 3, 480), _int(rng, 900, 2400000)
    err, ttd = _pct(rng, 0.4, 64.0), _int(rng, 1, 95)
    ttr, reqs = _int(rng, 4, 300), _int(rng, 10000, 88000000)
    facts = [
        Fact(
            key="duration",
            value=mins,
            kind=FactKind.NUMERIC,
            unit="minutes",
            clause="The incident lasted {value}.",
        ),
        Fact(
            key="users_affected",
            value=users,
            kind=FactKind.NUMERIC,
            unit="users",
            clause="It affected {value}.",
        ),
        Fact(
            key="error_rate",
            value=err,
            kind=FactKind.NUMERIC,
            unit="percent",
            clause="Peak error rate reached {value}.",
        ),
        Fact(
            key="time_to_detect",
            value=ttd,
            kind=FactKind.NUMERIC,
            unit="minutes",
            clause="Time to detection was {value}.",
        ),
        Fact(
            key="time_to_recover",
            value=ttr,
            kind=FactKind.NUMERIC,
            unit="minutes",
            clause="Time to full recovery was {value}.",
        ),
        Fact(
            key="requests",
            value=reqs,
            kind=FactKind.NUMERIC,
            unit="requests",
            clause="The blast radius covered {value}.",
        ),
    ]
    ctx = (
        f"A postmortem for a production incident in {svc} recorded duration, users affected, "
        f"peak error rate, time to detection, time to recovery and total requests in scope."
    )
    return _build(idx, "network_incident", ctx, f"Summarise the {svc} incident postmortem.", facts)


_GENERATORS = {
    "clinical_trial": _clinical_trial,
    "quarterly_earnings": _quarterly_earnings,
    "materials_experiment": _materials_experiment,
    "transit_operations": _transit_operations,
    "climate_station": _climate_station,
    "product_benchmark": _product_benchmark,
    "epidemiology_survey": _epidemiology_survey,
    "energy_grid": _energy_grid,
    "agronomy_field_trial": _agronomy_field_trial,
    "network_incident": _network_incident,
}


#: Size of the fixed item pool every experiment draws a prefix of.
POOL_SIZE = 1000


@lru_cache(maxsize=4)
def _pool(seed: int) -> tuple[Item, ...]:
    """The corpus, built once per seed.

    Round-robin across domains rather than domain-by-domain, so every prefix
    stays domain-balanced. That's what makes generate_corpus nested.
    """
    rng = random.Random(seed)
    items: list[Item] = []
    for _ in range(POOL_SIZE // len(DOMAINS)):
        for domain in DOMAINS:
            items.append(_GENERATORS[domain](rng, len(items)))
    return tuple(items)


def generate_corpus(n: int = 500, seed: int = 20260731) -> list[Item]:
    """Deterministic reference corpus, balanced across the ten domains.

    Nested by construction: generate_corpus(80) is a prefix of
    generate_corpus(150). The old version rebuilt the corpus for each n, and
    since `per = n // 10` changed how the RNG stream got consumed, an 80-item
    run and a 150-item run used different items that happened to share ids.

    That caused two problems. Comparisons across experiments weren't on the same
    items even though the write-up assumes they are. And almost everything was a
    cache miss: a full battery shared 527 hits out of ~64,000 calls. Nesting
    turns those into hits, which on a free tier decides whether this finishes.
    """
    if n > POOL_SIZE:
        raise ValueError(f"corpus pool holds {POOL_SIZE} items; asked for {n}")
    return list(_pool(seed)[:n])


def corpus_stats(items: list[Item]) -> dict[str, Any]:
    lens = [len(i.reference.split()) for i in items]
    return {
        "n_items": len(items),
        "domains": sorted({i.domain for i in items}),
        "facts_per_item": len(items[0].facts) if items else 0,
        "mean_reference_words": round(sum(lens) / max(len(lens), 1), 1),
        "min_reference_words": min(lens) if lens else 0,
        "max_reference_words": max(lens) if lens else 0,
    }


__all__ = ["DOMAINS", "corpus_stats", "generate_corpus"]
