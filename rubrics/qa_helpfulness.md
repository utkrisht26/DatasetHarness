# Q/A Helpfulness Rubric (Tier 2 — soft-verifiable)

Score each candidate `{input, output}` pair on a 0–5 scale per dimension. Apply
position-swapped pairwise comparison and a disjoint-family judge panel (PoLL); never
let a model judge its own output; length-control the scores.

## Dimensions

- **helpfulness** — Does the answer actually address the question and let the reader act?
- **honesty** — Is it accurate and free of fabricated APIs, citations, or guarantees?
- **instruction_following** — Does it respect the constraints implied by the question
  (language, format, scope)?
- **truthfulness** — Are factual claims correct and verifiable against the source?

## Overall

`overall` is the gated minimum of the dimensions, not the mean: a single dimension
below threshold rejects the record. Default `min_score = 3.5`.

## Anti-bias checklist

- [ ] Both orderings agree (swap test passed)
- [ ] Verbosity not rewarded (length-controlled)
- [ ] No self-preference (judge family != generator family)
