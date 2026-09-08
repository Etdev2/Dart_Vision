# Issue Tracker

Tracker: **GitHub Issues** in `Etdev2/Dart_Vision`.

## Wayfinding operations

### Map

The canonical map is one GitHub issue titled with the prefix `[Wayfinder Map]`. For the current effort it is issue #1, **Dart Vision — Phone Camera Auto-Scoring MVP**.

A map contains:

- `## Destination`
- `## Notes`
- `## Decisions so far`
- `## Not yet specified`
- `## Out of scope`

### Decision tickets

Each decision ticket is a GitHub issue whose title starts with one of:

- `[Wayfinder Research]`
- `[Wayfinder Prototype]`
- `[Wayfinder Grilling]`
- `[Wayfinder Task]`

Its body begins with:

- `Parent map: #<map>`
- `Type: wayfinder:<type>`
- `Blocked by: ...`

and contains exactly one `## Question` describing the decision or investigation.

### Child relationships

GitHub supports native sub-issues, but the currently connected GitHub tool does not expose sub-issue mutation. Until native relationships are wired manually, `Parent map: #1` is the authoritative fallback parent relation.

### Blocking

GitHub supports dependency relationships, but the currently connected GitHub tool does not expose dependency mutation. Until those edges are wired manually, `Blocked by: #A, #B` is the authoritative fallback.

A ticket is on the **frontier** when:

1. it is open;
2. every issue listed in `Blocked by` is closed (or it says `none`); and
3. it is unassigned.

### Claiming

Claim a frontier ticket by assigning it to the agent/developer driving that decision **before** doing work. An open, unassigned, unblocked ticket is available.

### Resolving

1. Post the decision as a resolution comment on the ticket.
2. Close the ticket.
3. Add one linked one-line gist under the map's `## Decisions so far` section.
4. Create newly visible decision tickets and update blockers/fog as needed.
5. Do not copy the full resolution into the map; the ticket remains the source of detail.

### Research tickets

Research findings should cite primary sources where possible and clearly separate facts, assumptions, benchmarks, and recommendations. Existing dart-scoring projects may be studied, but code/model/data reuse requires a licensing decision first.

### Prototypes

Prototype branches should use `prototype/<short-name>`. Prototype code exists to answer the ticket's question and may be discarded. It is not automatically production code.

## Pull requests

PRs are an implementation/review surface, not the Wayfinder decision tracker. Planning bootstrap documents may use a PR, but decision resolutions live on their issues.