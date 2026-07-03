# DESIGN.md — ADH Console Design System

> The design contract for the ADH frontend (see [docs/design/frontend-plan.md](docs/design/frontend-plan.md)).
> Source of truth for aesthetic, tokens, components, and screen patterns. Deviations
> from this doc are higher-severity findings in design review. `frontend/src/styles/tokens.css`
> (an F0 deliverable) implements the token tables here; this doc governs it.

**Classification:** APP UI (operator console, data-dense, task-focused). App-UI rules
apply throughout: calm surface hierarchy, strong typography, few colors, minimal chrome,
cards only when the card *is* the interaction.

**Two decisions, decided (override if you disagree):**
- **Type:** IBM Plex Sans (UI/prose) + IBM Plex Mono (machine data). Free, self-hostable,
  characterful, honest to the "instrument" voice. Upgrade path: Söhne + Berkeley Mono.
- **Dark mode: from day one.** Operators watch long crawls; dark reduces fatigue. Both
  themes are first-class and authored together (retrofitting later is expensive).

---

## 1. What this product is, and who watches it

Plain-English goal in, verified dataset out, with a human approving sources in the
middle (GOAL.md). The console is watched by ML/data engineers during long-running
crawls and dataset builds. Two mechanics are inherently visual and drive every choice:

1. **The approval gate** (default ON): a human reviews a ranked source list with
   reasoning and says go. The single most important screen the platform will ship.
2. **Observability of intelligence:** why the system kept, dropped, pruned, or stopped;
   saturation curves; budgets; token/$ cost. Trust comes from *seeing why*.

The v1 crawl console is the smallest slice that exercises both muscles against a real
running service.

## 2. Design principles

1. **Verdicts are the product.** The app decides: kept / dropped / pruned / saturated /
   failed. So color *is* status, and numerals *are* the display type. That is the
   personality, and it is honest to what the system does.
2. **Don't make me think.** Every screen self-evident. Status carries a text label, not
   color alone. One obvious primary action per screen.
3. **Omit, then omit again.** No intro paragraphs, no happy talk, no instructions that
   need reading. If a screen needs explaining, redesign it.
4. **Density with legibility.** This is an operator console, not a marketing page. Pack
   information, keep it scannable. White space is structure, not theater.
5. **Structure from type and hairlines, not shadows.** Hierarchy comes from the type
   scale, borders, and density. The design must feel intentional with every decorative
   shadow removed.
6. **Cards earn their existence.** No card grids. Cards appear only where a card *is* the
   unit of interaction (the approval-gate source, saved-preset chips). Everything else is
   tables, panels, and the log.

## 3. Aesthetic direction — "Lab console"

Warm-paper light surface (real dark twin), hairline structure, monospace machine-data as
texture, one teal accent, status color as the entire palette, and large tabular numerals
as the display type. The live frontier feed is the hero. Reference points: Linear's
restraint, Prefect/Datadog density, a terminal's honesty. Explicitly NOT: SaaS gradients,
card mosaics, icon-in-circle rows, centered hero copy.

The signature move: the big counts (pages kept, tokens used, saturation confidence)
render large, in mono, tabular. Numbers are the headline; prose is the caption.

---

## 4. Tokens

Values are OKLCH. Both themes authored together. `color-scheme` set per theme so native
controls match. Reviewers verify contrast against §8.

### 4.1 Color — neutrals & accent

| Token | Light | Dark | Use |
|---|---|---|---|
| `--paper` | `oklch(98.5% 0.004 90)` | `oklch(20% 0.008 90)` | app background (warm, never pure white/black) |
| `--surface` | `oklch(99.5% 0.003 90)` | `oklch(24% 0.008 90)` | panels, table body |
| `--surface-2` | `oklch(97% 0.004 90)` | `oklch(28% 0.008 90)` | elevated / hover (elevation = lightness step) |
| `--ink` | `oklch(22% 0.01 80)` | `oklch(92% 0.006 90)` | primary text (off-white in dark, not pure white) |
| `--ink-muted` | `oklch(46% 0.01 80)` | `oklch(68% 0.01 90)` | meta, timestamps, secondary |
| `--line` | `oklch(90% 0.005 80)` | `oklch(34% 0.01 90)` | hairline borders, separators |
| `--accent` | `oklch(60% 0.12 210)` | `oklch(66% 0.10 210)` | primary action + "system is working" (deep teal, NOT purple) |
| `--accent-weak` | `oklch(60% 0.12 210 / 12%)` | `oklch(66% 0.10 210 / 18%)` | accent fills / selection |

### 4.2 Color — status (this is the palette)

Every status has a text label and shape, never color alone (§8). `-weak` is the row/pill
tint; the base is the text/glyph.

| Token | Light | Dark | Meaning |
|---|---|---|---|
| `--kept` | `oklch(62% 0.15 150)` | `oklch(70% 0.14 150)` | page kept / job succeeded |
| `--dropped` | `oklch(70% 0.15 75)` | `oklch(78% 0.14 75)` | judge rejected (off-topic) |
| `--pruned` | `oklch(58% 0.02 80)` | `oklch(62% 0.02 90)` | filter skipped (domain/locale/dup) — neutral, NOT an error |
| `--failed` | `oklch(58% 0.20 25)` | `oklch(66% 0.18 25)` | crawl error / job failed |
| `--saturated` | `oklch(55% 0.10 280)` | `oklch(66% 0.10 280)` | coverage saturated / stop — reserved indigo, used only here |
| `--running` | = `--accent` | = `--accent` | in-flight (pulsing) |

Discipline: no decorative color anywhere. Color appears only as status or the single
accent. This is what dodges the AI-slop purple-gradient signal.

### 4.3 Typography

Families: `--font-sans: "IBM Plex Sans"`, `--font-mono: "IBM Plex Mono"`. Optional
`--font-serif: "Newsreader"` for large section intros / empty-state headlines only (skip
for zero decoration). Self-hosted, `font-display: swap`, preload the two critical weights.

Scale (major third, ~1.25):

| Token | px | Use |
|---|---|---|
| `--text-caption` | 12 | meta, table sub-labels, timestamps |
| `--text-sm` | 14 | table cells, dense body (default in data views) |
| `--text-base` | 16 | prose, form labels (never below 16 for body) |
| `--text-lg` | 20 | panel titles |
| `--text-xl` | 26 | screen section titles |
| `--text-2xl` | 33 | screen title |
| `--text-metric` | 44 | hero numerals (mono, tabular) |

Rules:
- Line-height: body 1.5, headings 1.2, mono data 1.4, metric 1.0.
- `font-variant-numeric: tabular-nums` on every numeric column and every live-updating
  number (scores, tokens, durations, counts) so digits do not jitter on update.
- Machine data (URL, id, sha, score, token count, markdown) is always mono.
- Weights: 400 body, 500 labels/emphasis, 600 titles. Two-weight minimum for hierarchy.
- Curly quotes and real ellipsis `…`; loading strings end in `…` ("Crawling…").

### 4.4 Space, radius, elevation, motion, z-index

- **Space** (4px base): `2 4 8 12 16 24 32 48 64`. Feed and table rows target ~30px, not
  airy 56px SaaS rows. Prose max-width ~72ch; the feed and tables are full-bleed.
- **Radius:** `--r-cell 4` (inputs/cells), `--r-panel 8` (panels/cards), `--r-pill 999`
  (status pills only), `--r-flush 0` (log lines, table rows — flush, terminal-like). Not
  uniform bubbly radius on everything.
- **Elevation:** light uses hairline `--line` + at most `--shadow-1: 0 1px 2px
  oklch(0% 0 0 / 6%)`. Dark uses `--surface`/`--surface-2` lightness steps + border, near
  zero shadow. Hierarchy is type + density + line, not soft shadows.
- **Motion:** `--dur-fast 120ms`, `--dur 180ms`, `--dur-slow 260ms`; `--ease-out
  cubic-bezier(0.16,1,0.3,1)`. Animate transform/opacity only. `prefers-reduced-motion`
  → crossfade or none.
- **Z-index:** base 0, sticky header 10, feed "jump to latest" 20, dropdown 100, modal
  1000, toast 1100.

---

## 5. Components

### Status pill
Rounded (`--r-pill`), `-weak` background tint + base-color text, always a text label
("kept", "dropped", "pruned", "saturated", "failed", "running"), optional 8px glyph.
Never color-only. `running` pulses (opacity, reduced-motion → static).

### Buttons
Primary (accent fill, `--ink`-on-accent or white per contrast), secondary (surface +
`--line` border), ghost (text only), destructive (failed color). Every button has hover,
`:active` (subtle depth/shift), `focus-visible` (accent ring, never `outline:none`
without replacement), and disabled (reduced opacity + `not-allowed`). Min target 44px.
Labels are specific: "Start crawl", "Approve source", "Cancel job" — never "Submit".

### Table / data grid
30px rows, sticky header, sortable columns, hairline row separators (no zebra striping),
row hover = `--surface-2`. Numeric columns right-aligned, mono, tabular. Truncate long
URLs with middle ellipsis so the path tail stays visible. This is the default for job
lists, page tables, and (recommended) the approval gate.

### Frontier feed (the hero)
Append-only log. Each line: timestamp (mono, muted) · status pill · URL (mono,
middle-truncated) · score (mono tabular) · one-line reason. Row tinted with the status
`-weak`. Virtualized; pinned to bottom; a "jump to latest" control appears when the
operator scrolls up (terminal convention). New rows fade + 4px slide-in at `--dur`. Do
not use a 3px colored left-border-on-cards treatment; use the pill + row tint.

### Panels
Section container: title that names the area or action ("Active crawls", "Frontier
decisions", "Extracted pages", "Token usage"), hairline border, `--r-panel`, 16–24
padding. No panel without a naming title.

### Cards — restricted
Allowed ONLY for: (a) the approval-gate source unit, (b) saved-preset chips on the submit
form. Anywhere else, use a table or panel. A card grid as a screen's first impression is
an instant-fail.

### Forms
Label always visible above the field (never placeholder-as-label). 44px controls,
`focus-visible` accent ring, inline validation next to the source, correct input types.
Keyword entry is a chip input. No intro paragraph. Submit label states the action.

### States (design all four, every data surface)
- **Loading:** skeletons that match the real shape (log lines, table rows), subtle
  shimmer, reduced-motion → static.
- **Empty:** warm one-liner + primary action ("No crawls yet. Start one."), not "No items."
- **Error:** what happened + why + the next step; specific, near the source.
- **Partial/streaming:** the feed shows progress live; never a blank screen while working.

### Data-visualization (part of the system, not a bolt-on)
Saturation confidence sparkline (confidence over pages), per-URL relevance score bars
(0–1 with a threshold marker), budget gauges (fetched/kept vs caps). All use the status
tokens, tabular nums, one consistent axis/gridline treatment. No default rainbow palette.

---

## 6. Layout

App shell: persistent **left rail** (surfaces nav — crawl, spec, discovery, runs,
curation; future ones marked "soon"; current section visually indicated for wayfinding),
a quiet **top strip** (breadcrumb + health pill + token-usage readout — never three big
stat cards with icon-circles), and the **main pane**. Breakpoints: 375 / 768 / 1024 /
1440. The console is desktop-first but must not break or horizontally scroll on tablet;
touch targets stay ≥44px regardless.

---

## 7. Screen patterns

- **Crawl list:** anchor = an "active crawls" strip (live status + counts) above a dense
  job table (status pill, seed URL, kept/fetched, duration, saturated badge).
- **Submit:** single-column form, no intro. Judge picker (none/bm25/llm + model +
  threshold); when llm is selected, show a token/$ cost estimate before enabling.
  Presets as chips.
- **Job detail:** header (status pill + Cancel) · frontier feed (hero) · extracted-pages
  table (sortable by score/tokens/depth) · provenance strip.
- **Page viewer:** rendered-markdown ↔ raw toggle, provenance strip (final URL, license
  hint, sha), copy/download.
- **Approval gate (F2, signature screen):** ranked source table; each row expands to show
  reasoning; columns = score, license, freshness, est. yield. Keyboard-driven (j/k move,
  a approve, x exclude) for power operators, every key unambiguous. A running summary bar:
  "approved N · est. yield M records · est. cost $X". Table-first; a card layout is the
  alternative only if sources are few and each is content-rich.
- **Run monitor (F3):** per-source timeline, saturation sparkline per aspect,
  circuit-breaker/limit gauges, corpus chunk browser with the full provenance chain.

## 8. Accessibility (WCAG 2.2 AA)

- Contrast: body text ≥ 4.5:1, large text (≥18px or 14px bold) ≥ 3:1, UI components and
  status indicators ≥ 3:1. Tokens in §4 are chosen to clear this in both themes; verify.
- Status is never color-only: pill shape + text label (+ optional glyph). Covers red-green
  color deficiency.
- `focus-visible` ring on every interactive element; never remove focus without a
  replacement. Keyboard-complete flows, the approval gate especially.
- `prefers-reduced-motion`: feed and transitions degrade to crossfade/none.
- Touch targets ≥ 44px. Semantic landmarks (`header`/`nav`/`main`). `color-scheme` set
  per theme. Body text never below 16px.

## 9. Anti-slop guardrails (the blacklist, applied here)

Reject on sight in review:
- The submit form rendered as three centered cards. It is one column.
- Health/usage as three big stat cards with icons-in-circles. It is a quiet top strip.
- Any purple/violet/indigo gradient, or accent drift toward SaaS purple. Accent is teal;
  indigo is reserved for the saturation stop only.
- Uniform bubbly radius on everything. Radius follows §4.4.
- Emoji as status or bullets. Status is a pill + label.
- IBM Plex is the type; `system-ui`/Inter/Roboto as the primary face is an instant flag.
- Card grids anywhere outside §5's allowlist.
- Centered everything. Data views are left-aligned; numbers right-aligned.

## 10. Reviewer litmus (for /plan-design-review and the F1 /design-review gate)

Answer YES for a pass:
1. Product unmistakable on the first screen (wordmark + the active-crawls anchor)?
2. One strong visual anchor per screen (the feed on detail, the active strip on list)?
3. Screen understandable by scanning headings alone (panels name their area)?
4. Each section one job?
5. Are the only cards the approval-gate source and preset chips?
6. Does every motion clarify state (feed-in, status transition, saturation pulse)?
7. Does it feel intentional with all decorative shadows removed?

App-UI hard rule to enforce: no stacked-card mosaic standing in for a real layout.

## 11. Governance

This file is the source of truth. `frontend/src/styles/tokens.css` implements §4 (F0).
Any rendered deviation from these tokens, component rules, or anti-slop guardrails is a
higher-severity design-review finding than a generic issue. The live `/design-review`
audit becomes the F1 exit gate once the crawl console renders.
