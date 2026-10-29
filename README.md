# Proof-Carrying Code

Muse Spark and Muse Glimmer turn Python code into formally verified Dafny code, together with its proof. Glimmer is
trained with the proof checker as its reward.

Built for the [Meta Global AI Developer Hackathon 2026](https://dev.meta.ai/events/global-hackathon).

## Built during the hackathon

Everything in this repository was created after the official hackathon kickoff on October 26, 2026, 12:00 PM ET.
No code, data or assets were prepared before the event.

## Status

Work in progress. The package skeleton is in place; the pipeline steps are being built.

## Development

```bash
uv sync
uv run pytest
```

Needs Python 3.12, [uv](https://docs.astral.sh/uv/) and [Dafny](https://dafny.org) 4.11. See
[docs/plan.md](docs/plan.md) for the plan, [docs/architecture.md](docs/architecture.md) for the code layout and
[AGENTS.md](AGENTS.md) for the rules every contributor, human or agent, follows. All work is tracked as GitHub issues
and follows the process in AGENTS.md.

## License

[AGPL-3.0](LICENSE).

Third-party datasets, demo sources and models are listed in [THIRD_PARTY.md](THIRD_PARTY.md).
