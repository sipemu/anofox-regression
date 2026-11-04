# House reporting standard for statistical analyses

Every analysis ends with a report (`answer.md`, or the final message) that follows this standard. It exists so that every report reads the same way: a decision-maker finds the answer in the first lines, and a statistician can audit the work without re-running it.

## Required structure (use these headings, in this order)

```
## Bottom line
## Question and estimand
## Data
## Methods
## Results
## Checks and diagnostics
## Limitations
## Reproducibility
```

### 1. Bottom line
**Exactly three sentences** of plain language, readable by a non-statistician:
1. **Answer:** the direct answer to the question asked, or why it cannot be answered as asked.
2. **Estimate:** the headline number with its 95 % CI, in the units the reader cares about. For planning tasks, give the required sample size or duration and its key assumption.
3. **Implication:** what to do or decide.

**Counting rule:** every string ending in `.`, `?` or `!` is a sentence, including a standalone "No." or "Yes.". Fold a yes/no answer into sentence 1 ("No — the data do not show ..."). Do not use bullet points, tables or a second paragraph in this section.

*Example:* "After correcting for when users adopted it, Teams adoption shows no measurable effect on churn. The hazard ratio is 1.00 (95 % CI 0.90 to 1.12), which rules out a churn reduction larger than about 10 %. Do not put a churn-reduction number in the deck; a randomised rollout would be needed to claim one."

Put everything else (secondary results, caveats, method notes) in later sections. If you are tempted to write a fourth sentence, it belongs in Results or Limitations. Do not include a p-value here unless it *is* the answer.

### 2. Question and estimand
- The goal type: **effect estimation**, **hypothesis testing**, **prediction / forecasting**, or **design / planning**.
- The estimand in one sentence, adapted to the goal type:
  - **Effect estimation or testing:** population, contrast, outcome scale, and time horizon. For example: "average difference in 30-day revenue per customer, discount vs no discount, among all randomised customers".
  - **Prediction:** the target, the horizon, the population it will be applied to, and the accuracy metric.
  - **Planning:** the quantity to be determined, and the effect size, error rates and design it is computed for.
- If you changed the user's question (for example, refusing to adjust for a mediator), say so here and give the reason.

### 3. Data
- n analysed. Exclusions, each with a count and a reason.
- Effective sample size: events for binary or survival outcomes, clusters for clustered data.
- Missing data: the amount per relevant variable, and how it was handled.
- Design facts that matter, such as randomised or observational, time ordering, clustering, and censoring.

### 4. Methods
- The model or test, with its a priori justification: why this family, and how many d.f. it spends against the effective sample size.
- **Covariates:**
  - *Effect estimation:* for each variable, say whether it was adjusted for, and **why**, by its causal role: confounder, mediator, collider, or precision variable. If there are no candidate covariates, say so and name the likely unmeasured confounders.
  - *Prediction:* explain why each predictor is included (prespecified, available at prediction time, no leakage). Causal roles are not needed.
  - *Planning:* list the design assumptions.
- How uncertainty was obtained (model-based, robust, bootstrap) and how validation was done if prediction was the goal.
- Any deviation from what the user asked for.

### 5. Results
- A table of effects. Each row gives the **estimate, 95 % CI and p-value** on an interpretable scale, with units: differences, ratios (OR / RR / HR / time ratio), or predicted values. Raw spline or dummy coefficients do not count.
- Multi-d.f. or pooled tests where predictors span several parameters.
- For prediction models: discrimination, calibration and overall accuracy, both apparent and optimism-corrected.
- Absolute effects next to relative effects whenever a baseline risk is available.

### 6. Checks and diagnostics
A bulleted list: each check performed, what it showed, and what was done about it. Examples are influence, functional form, dispersion, assumptions specific to the model, randomisation balance, sensitivity analyses, and alternative specifications.

### 7. Limitations
- Threats to validity: confounding, selection, measurement, and generalisability.
- How strongly the design supports causal claims, and where it does not.
- What would resolve each limitation (more data, an experiment, external validation).

### 8. Reproducibility
- The paths of the code / SQL files and how to run them.
- Software and versions (crate / extension version, DuckDB / Rust version), random seeds, and the number of resamples.

## Language rules (apply throughout)

1. **Never use "significant" alone.** Every claim of an effect, or of no effect, carries an estimate and a CI.
2. **Absence of evidence ≠ evidence of absence.** For **every** null or non-significant result, give the range of effects the data still allow ("compatible with anything from −4 % to +2 %"). This includes secondary and nuisance tests such as interactions, nonlinearity, heterogeneity and balance checks. For a multi-d.f. test without a single estimate, give a bound on the largest plausible effect, or the CI of the most relevant contrast.
3. **Causal verbs only for causal designs.** With observational data, write "associated with", not "drives", "reduces" or "causes", unless a causal identification argument is given in Methods.
4. **Report p-values as numbers** (p = 0.03, p < 0.001), never as stars, and never as "p = 0.000".
5. **Round consistently**: estimates to a precision that reflects the CI width, and percentages to at most one decimal place.
6. **No unexplained jargon in the Bottom line.** Technical terms belong in Methods.
