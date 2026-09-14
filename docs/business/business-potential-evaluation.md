# Dart Vision — Business Potential Evaluation

> **Status: external analysis, not a Wayfinder decision.** This record evaluates Dart Vision as a
> commercial venture rather than as an engineering problem. It does not resolve any ticket on maps
> #1 or #12, and it does not override a decision on either. It exists because a market event on
> 2026-09-10 changed the premise both maps were planned under.

Date: 2026-09-14 · Full memo: <https://claude.ai/artifact/5cjJe9KD3XyZsX1f4G2yMt>

---

## 1. The finding that governs everything else

On **10 September 2026** — after map #1 reached decision-complete and four days before this record —
**Winmau and Autodarts launched LENS**: point any phone camera at any standard dartboard, on-device
autoscoring, free limited tier, £5.99/month unlimited, shipping globally on web, iOS and Android in
English, German and Dutch.

That is this repository's product specification, shipped by the manufacturer whose boards a large
share of the target customer already owns, attached to **200,000 registered users** and a
40,000-member Discord.

It is not the only one. The single-camera, no-hardware segment already contained:

| Product | Price | Note |
| --- | --- | --- |
| **Autodarts LENS** | Free tier; £5.99/mo | Winmau retail + 200K users; launched 2026-09-10 |
| **Dartsmind** | ~$15 one-time | On-device, no calibration, dual-device mode, good reviews. Same cluster-occlusion limit #2 identified |
| **DeepDarts** (MWM) | App Store | The #13 paper's own authors, commercialised |
| Autodarts X / Vantage | £449.99 / €299 | 4 cameras + lighting, 99.5%+ |
| Scolia Home 2 / Pro | ~$300–600 | 99.8% on Pro; venue standard |
| DartConnect | $24/yr | Manual entry; the statistics standard |

**Consequence for the maps:** the engineering risk the Wayfinder process has been methodically
retiring was never the binding constraint. Distribution is. No decision on #1 or #12 is wrong
because of this; the *sequencing* is. Map #12's remaining tasks (#25, #26, #19, #20) cost roughly
1,150 founder hours, and none of them tests the assumption that now carries the most risk.

## 2. Headline numbers

| | |
| --- | --- |
| Overall business potential | **3.9 / 10** (weighted) |
| Recommendation | **3 — Validate cheaply**, default to abandoning the consumer app |
| P(commercial success, ≥$100K cumulative profit) | 12% |
| P(failure — never sustainable) | 70% |
| P(>$1M annual revenue) | 3% |
| Most likely 3-year revenue / profit | $70K / $45K |
| High-case 3-year revenue / profit | $350K / $210K |
| Capital required | $5.2K lean · $22K recommended |
| Initial human hours / AI hours | 1,190 / 1,560 |
| Ongoing human / AI hours per month | 60 / 96 |
| AI-automatable | ~50% of effort, ~25% of the critical path |
| Risk · Scalability · Founder leverage · Capital efficiency · Moat | 8 · 7 · 6 · 8 · 3 (/10) |

Expected values at year 3: revenue **$93.6K**, profit **$48.8K**, enterprise value **$300K** — of
which **67% comes from a 3% tail**. The median outcome is $0 and 1,190 hours spent.

## 3. What the market can actually pay

Sized bottom-up from players, not from the $6.16B "darts market" figure, which measures equipment
and is not the addressable market for software.

- **TAM** (global home steel-tip players who track scores): 6–12M people → $150M–$300M/yr
- **SAM** (EN/DE/NL, NPU-class phone, board mountable, autoscoring-aware): 1.5–3M → $40M–$75M/yr
- **SOM** (solo entrant, 3 years, post-LENS): 2K–15K subscribers → $40K–$375K/yr

At $24.99/yr, the revenue ladder is:

| Target | Subscribers | % of SAM |
| --- | --- | --- |
| $100K/yr | 4,000 | 0.18% |
| $500K/yr | 20,000 | 0.9% |
| $1M/yr | 40,000 | 1.8% |
| $5M/yr | 200,000 | 9.1% — **equals Autodarts' entire registered base** |
| $10M/yr | 400,000 | 18.2% — **twice it** |

The top two rows are outside the category. **The realistic ceiling is ~$1M of annual revenue.**

## 4. Where the economics are genuinely good

The **#4 decision to run inference on-device is worth more commercially than technically.** There is
no per-frame inference bill and no bandwidth cost, so:

- Gross margin **82–84%**
- Marginal cost ~$0.30/user/yr
- **Cash break-even at ~300 paying subscribers**

This business will not fail on unit economics. Paid acquisition is nonetheless unviable — $25–$60
CAC against a ~$30 LTV — so the only viable channels are organic, which is exactly the constraint
LENS does not have.

## 5. The asymmetric upside the maps have not noticed

Issue #7 concluded that no authorized API exists for a third-party autoscorer to push throws into
another app, because *each scoring app is paired with one autoscoring vendor* — recorded as a
constraint. It is also an opportunity, and the Winmau deal sharpened it:

**Winmau taking Autodarts exclusive leaves Target, Unicorn, Red Dragon, Mission, Shot, Viper and
Bull's without an autoscoring answer while their largest competitor has one.** They have retail
distribution. Dart Vision would have the only engine anyone is building that a competitor of Winmau
is permitted to buy — because of #8, whose licence register makes the IP position clean enough to
sell.

That reframes the venture from a 40,000-customer consumer problem into a 5–15-customer B2B one, and
it converts the memo's worst score (distribution, 2/10) into its best. It is also the cheapest thing
on this list to test: five emails.

Second-order: the synthetic-render → keypoint pipeline in #25 transfers directly to archery, air
pistol and rifle targets, axe throwing, shuffleboard and cornhole — planar, dimensionally
standardised targets, most with no autoscoring product and no incumbent.

## 6. Recommended change to the plan

**Insert a Phase 0 ahead of #25 and #26.** Four weeks, under $400, ~30 founder hours:

1. Buy one month of LENS and a Dartsmind licence. Stress-test both on a real board. Document exactly
   where they fail — low-end Android, no light ring, clusters.
2. 15 structured interviews and a 100-response survey in r/Darts and DartsNutz.
3. **Five outbound emails to non-Winmau dart brands** asking whether they want a licensed engine.

**Go/no-go:** ≥25% switch intent with a *named concrete reason*, **or** ≥1 brand replies with real
interest. Fail both → stop the consumer app.

Nothing else on map #12 should start until this returns. It is the only step whose result can save
the other 1,150 hours.

## 7. Kill criteria to adopt

| | Criterion |
| --- | --- |
| K1 | Week 4 — Phase 0 fails both limbs above → stop the consumer app |
| K2 | Month 3 — #25 Arm C fails to beat Arm A by ≥3pp at a 25% real-data budget, and Arm B's sim-to-real tip gap >15mm p95 → full capture bill applies; re-budget before continuing |
| K3 | Month 9 — Gate R missed (per-dart <96% or tip p95 >12mm) *and* #14's margin cross-tab shows errors are not predominantly small-margin → stop or commit to dual-camera |
| K4 | Month 12 — error concentration <60% at a 15% flag rate → the confidence signal is uninformative, and #21 establishes gating *is* the product. Stop |
| K5 | Month 14 — <60% of 50 beta users complete ≥10 legs in week 2, or median wrong-scores-per-leg >0.5 → stop |
| K6 | Month 18 — blended CAC >$12 or organic installs <300/mo after three sustained months → stop |
| K7 | Hard caps — cumulative cash >$25,000 or founder hours >900 with zero paying customers → stop |

K7 exists because of the quality of the planning in this repository. Work this good creates real
reluctance to conclude the market moved underneath it.

## 8. Two gaps in an otherwise exemplary record

1. **Freedom to operate is unexamined.** The licence register (#8) covers copyright thoroughly and
   covers patents not at all. `EP 3311100 B1` ("Automatic Dartboard Scoring System") and
   `EP 4029581 A1` exist in this exact space. A $300–800 preliminary FTO search is overdue.
2. **Capture is the bottleneck, and users can do it.** A guided in-app capture mode — "help train the
   model on your board, get a year free" — converts the most expensive human bottleneck (#26, #19)
   into a community asset and builds the proprietary-data moat at the same time. It is in real
   tension with #4's promise that frames never leave the device, so it must be explicit, opt-in and
   separately consented. It is the highest-leverage unbuilt idea in either map.

## 9. Verdict

> If I were the founder, I would **validate for four weeks and then, on the evidence as it stands,
> abandon the consumer app while actively pursuing the licensing pivot** — because the engineering
> plan is genuinely excellent and the market position is indefensible: Winmau shipped this exact
> product on 10 September with 200,000 users, a free tier and the retail shelf, and no amount of
> model quality closes a distribution gap that large.

**What would change this assessment:**

1. A letter of intent from a non-Winmau dart brand. Moves the score from 3.9 to ~6.5.
2. A measured, material gap in LENS coverage — mid-range Android, or normally-lit rooms without a
   light ring, which Autodarts' own documentation concedes is a weakness.
3. A steep Arm D curve from #25 (synthetic buys back ≥75% of real capture), which halves
   time-to-model and makes the pipeline itself the transferable asset.

## Assumptions

Founder context was inferred from this repository, not supplied: solo technical founder, MacBook Pro
plus hourly cloud GPUs (#16), strong systems judgment, no existing darts audience, no shipped model,
no disclosed capital constraint. **If an existing darts-community audience or a distribution-capable
co-founder exists, the distribution score and the overall verdict change materially.** All financial
figures are estimates with stated ranges; internal technical facts are cited to their issue numbers
and are this project's own research.
