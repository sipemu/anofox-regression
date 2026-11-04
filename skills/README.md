# Agent skills

`anofox-regression-statistician/` is a Claude Code skill: it teaches an agent
to analyse data in Rust with this crate the way an applied statistician
would (Harrell, *Regression Modeling Strategies*). It lives here, next to the
code it documents, so that it changes with the API.

- `SKILL.md`: the workflow and the decisions the agent makes.
- `references/rust_api.md`: the verified API reference (names, options,
  defaults, conventions, R equivalents, gotchas).
- `references/statistical_workflow.md`, `references/report_standard.md`:
  methodology and the reporting standard.
- `assets/analysis-template/`: a Cargo project the agent copies to start an
  analysis.
- `evals/`: `claude plugin eval` cases (prompt, data, LLM graders).

## Keeping it current

CI (`Statistician skill` job, required for publishing) fails when:

1. the skill names a Rust item, `Type::member` or builder option that the
   crate does not have (`scripts/skill_api_check.py`; names that are not
   crate items, e.g. R or sklearn functions, go in `api-allowlist.txt` with a
   reason);
2. `references/rust_api.md` does not state the current crate `major.minor`
   in its header (`**0.11.0**`): every release bump fails until the skill
   has been reviewed for that release;
3. the analysis template does not build and run against the commit.

So, in the PR that changes the public API: update `references/rust_api.md`
(and `SKILL.md` if the workflow or the gap list changes). In the release PR:
review the CHANGELOG section against the skill and bump the version in the
reference header.

Run the checks locally:

```bash
cargo +nightly rustdoc --lib --all-features --target-dir target/skill-pub -- \
    -Z unstable-options --output-format json
cargo +nightly rustdoc --lib --all-features --target-dir target/skill -- \
    -Z unstable-options --output-format json --document-private-items
python3 scripts/skill_api_check.py
cd skills/anofox-regression-statistician/assets/analysis-template && \
    RAYON_NUM_THREADS=1 cargo run --release \
    --config 'patch.crates-io.anofox-regression.path="../../../.."'
```

Run the evals after larger changes (each case is a full agent run; delete the
template's `target/` first, the eval runner rejects its hard links):

```bash
claude plugin eval skills/anofox-regression-statistician [--case <name>] -j 3
```

## Installing

Link the skill into Claude Code so it always uses this checkout:

```bash
ln -s "$PWD/skills/anofox-regression-statistician" ~/.claude/skills/anofox-regression-statistician
```
