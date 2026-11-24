# Contributing

The most valuable contribution is a **bug report with topic + seed** (`csat-forge gen --topic <name> --seed <n>`).

## Add a generator
A generator is one function returning a record (see `gen_conditional_prob` in `csat_forge/problem_engine.py`):

1. Sample parameters (keep them small and valid).
2. Compute the answer with SymPy — never hard-code or estimate it.
3. Build distractors with `make_mc_tagged`; tag each with the mistake that produces it.
4. Return `_rec(...)` and register it in `GENERATORS`.
5. Add a regression test that re-solves the *rendered text* where feasible.

```bash
pip install -e '.[figures,dev]'
pytest -q
```

Do not add past-exam text or images (copyright). Problems must be generated from scratch.
