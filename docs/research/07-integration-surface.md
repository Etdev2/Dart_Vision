# The authorized integration surface

> Research record for **[Wayfinder Research] Define the authorized integration surface for other dart apps** (issue #7), parent map #1.

## 1. Recommendation

**Build the export surface. Do not build platform adapters in the MVP.**

Concretely:

1. **Document and version the `ThrowEvent` stream** as the integration surface. It already exists (#9) and is already the boundary every internal consumer uses; making it a published contract costs almost nothing.
2. **Give the player an export of their own match data** — JSON and CSV. This needs nobody's permission, serves a real user need, and is the only integration available today that is unambiguously ours to build.
3. **Ship no adapter to DartConnect, DartCounter, Autodarts or Scolia.** §3 explains why each is either closed, forbidden, or does not do the thing we would need.
4. **Treat partnership as business development, not engineering.** The thing that makes that conversation possible is a clean event contract and a *measured* accuracy number (#21) — nobody integrates an unmeasured autoscorer.

## 2. A caveat about this record's evidence

This environment's network proxy blocked `scoliadarts.com` and `public-api.dartconnect.com`, so **the Scolia API page was never read and DartConnect's API documentation was never read.** What follows about them is assembled from search results and endpoint names, not from their terms.

That is enough to decide *not to build adapters now*. It is **not** enough to act on if that decision is ever revisited. Before any integration work begins, someone must read the actual terms of service and API documentation for the platform in question. This record should not be cited as having done so.

## 3. What the ecosystem actually looks like

The striking thing is not that integration is hard. It is the **shape** of the market.

| Platform | Integration surface | Status for us |
| --- | --- | --- |
| **DartCounter** | Auto-scoring integrates *exclusively* with Target's Omni hardware over Wi-Fi | **Closed by exclusivity.** Not a technical obstacle — a commercial one |
| **Autodarts** | No official public API specification. The Python bindings in the wild describe themselves as unofficial; community projects document endpoints by observing traffic | **Forbidden by our own rule.** Reverse-engineered private endpoints are exactly what `AGENTS.md` rules out |
| **Scolia** | Publishes a business-facing API page inviting commercial enquiries | Plausible, and **they sell the hardware we replace** |
| **DartConnect** | A public API exists; its documented endpoints are tournaments, organisation group members and player registration | Real and documented, but it appears to be **league and tournament administration**, not live throw ingestion |

Read down that column and the pattern is clear: **each scoring app is paired with one autoscoring vendor** — DartCounter with Omni, Scolia with Scolia, Autodarts with Autodarts. The pairing is the business model.

The consequence for #7 is sharp: **there is no authorized API anywhere for a third-party autoscorer to push live throws into someone else's scoring app.** That is not an oversight in their documentation. It is the market structure. The surfaces that do exist are for administering leagues, or for a vendor's own hardware.

## 4. What that means

**It removes the adapter work from the MVP entirely.** Not deferred on grounds of effort — there is nothing available to build against on authorized terms. An adapter to Autodarts would be built on reverse-engineered endpoints, which we will not do. An adapter to DartConnect would be built against an API that does not appear to accept live throws. An adapter to DartCounter would be built against an exclusivity we are not party to.

**It strengthens the case for our own scoring loop.** #10 is not duplicating what an integration could have provided. It is the only route to a usable product, because feeding someone else's app is not on offer.

**And it makes the export surface the whole answer for now.** A documented event stream plus a user-owned data export is what a future partner would need anyway, and it is useful on its own the day it ships.

## 5. The line, stated concretely

`AGENTS.md` sets the rule; this is what it forbids in practice, so that a future contributor does not have to re-derive it:

- **No scraping.** Not of match pages, leaderboards, player profiles, or tournament results.
- **No credential automation.** Never ask a user for their DartConnect or Autodarts password, and never drive a login on their behalf.
- **No private endpoints.** An endpoint discovered by watching a web app's network traffic is not a public API, however well a community has documented it. Community documentation of an undocumented endpoint does not make it authorized — it makes it *known*.
- **No ToS bypass**, including rate-limit evasion, user-agent spoofing, or using a personal account for machine access.

The test is simple and worth keeping: **would the platform's own documentation describe what we are doing?** If the answer requires explaining, it is not authorized.

The reason to be strict here is the same reason the Brain trains on synthetic and our own data (#8, #24, and the data-strategy decision): an unbounded dependency on somebody else's goodwill, at the base of a product that might be commercial, is worse than a larger amount of work we control. A scraper is a dependency on a page not changing and on nobody minding. Both fail silently and at the worst time.

## 6. Should we publish an API of our own?

Eventually, and not yet.

The `ThrowEvent` contract is the right shape for it — that was #9's whole point, and the separation has held through #3, #6 and #10 without strain. What is missing is not design but **a number**. An autoscoring feed is worth integrating only if its accuracy is known, and #21's gates do not have values against them until the training campaign runs.

So the order is: measure first (#20, #21), publish the contract with the measurement attached (#22), and only then approach anyone. A partner's first question will be "how often is it wrong, and how do you know" — and the honest answer to that is the actual deliverable.

## 7. What is worth doing now

Small, and none of it blocked:

- **Version the `ThrowEvent` schema** and treat changes to it as breaking. It is already the internal boundary; naming a version costs one field.
- **User-facing export**, JSON and CSV, of the player's own matches. Theirs, so no permission needed.
- **A webhook or local websocket** emitting `ThrowEvent`s as they happen — the thing a hobbyist or a partner would actually ask for first, and the same surface the app already consumes internally.

Each of those is a few hours' work, needs no agreement from anyone, and is what makes an integration conversation possible later.

## 8. Sources

Search results, September 2026. Neither vendor documentation site could be retrieved — see §2.

- [DartConnect API Documentation](https://public-api.dartconnect.com/) — *blocked; endpoint names taken from search results*
- [Scolia — Need an API for your business?](https://scoliadarts.com/api/) — *blocked; not read*
- [Target Omni Auto Scoring System — exclusively for DartCounter](https://www.target-darts.co.uk/omni-auto-scoring-system)
- [python-autodarts — "unofficial python binding for the autodarts web api"](https://github.com/belese/python-autodarts)
- [Autodarts community API capability notes](https://github.com/thomasasen/autodarts_local_tournament/blob/main/docs/autodarts-api-capabilities.md)
