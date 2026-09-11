# The match and correction loop

> Prototype record for **[Wayfinder Prototype] Define the core mobile match and correction UX** (issue #10), parent map #1.
>
> Prototype: `src/dartvision/match/`. This is the decision logic — the state machine, the confirmation queue and the correction contract — not the screen. What the app still owns is in §6.

## 1. The two decisions that carry it

Everything else here is bookkeeping.

### 1.1 Confirmations wait for the retrieval pause

#5 sets the budget: roughly **two confirmation taps a leg**, and **300 ms** from impact to displayed score. The obvious reading is a prompt the moment an uncertain dart lands.

That cannot work. Three darts take about ten seconds. By the time a prompt for dart one has rendered, the player is already mid-throw with dart two. Interrupting a throw is a worse outcome than scoring it wrong, and it arrives during the only part of the game that demands concentration.

But **the pause already exists.** Walking to the board and pulling three darts is ten to fifteen seconds in which the player's hands are free, their attention is on the board rather than the throw, and nothing is being scored.

So: an uncertain dart is **scored provisionally the moment it lands** — which is what holds the 300 ms budget — and the question is **queued and asked when the visit ends**. Two taps a leg then cost nothing at all, because they are spent in time the game was already wasting.

This is what makes #5's budget affordable rather than merely small. A budget of two interruptions per leg is intrusive; two taps during dart retrieval is not a budget at all.

### 1.2 Correcting is not a state

#10's suggested state list includes *correcting*, and this prototype deliberately does not have one. A modal correction screen is exactly the interruption the ticket asks to design out: it blocks the next throw to fix the last one.

Correction is an **action**, available on any throw of the current or previous visit, at any time and from any state. It emits a new event whose `corrects` points at the one it replaces. The original stays in history, so:

- a leg replays to the same numbers however many times it is recomputed,
- a correction can itself be corrected, and
- the UI never mutates what was recorded, which #9 requires.

`effective_history()` is the view with supersessions resolved — what the game engine scores.

## 2. States

Five, and the state is **derived from the facts, never stored**. Storing it was the first attempt and it drifted immediately: every input had to remember to update it, and the combinations that matter are exactly the ones a hand-maintained field gets wrong — calibration lost while a question is queued, a two-dart checkout, darts pulled before the question is answered.

| State | When | Screen says |
| --- | --- | --- |
| `not_calibrated` | fewer than 4 landmarks | "Board not visible" + whether the visit is safe |
| `ready` | calibrated, board empty | "Ready" |
| `scoring` | darts in the board, visit not closed | the last dart's notation, running total |
| `confirming` | visit closed, questions queued | "Was that T20?" + why it is asking |
| `awaiting_removal` | visit closed, darts still in | the visit total, "collect your darts" |

**`not_calibrated` is a first-class state, not degraded scoring.** #13 established the cliff: below four landmarks *nothing* scores. Showing that as a frozen or blank scoreboard is the worst failure available, because the player cannot tell whether to keep throwing. The screen says the board is not visible, and says explicitly that darts already thrown are safe.

Which leads to a rule worth stating on its own: **losing calibration never discards the visit.** Those darts were scored under a calibration that was valid when they landed. Throwing them away because the camera got bumped afterwards would be a worse failure than the one being reported.

## 3. Why the question is asked at all

`ConfirmationPolicy` (from #5, in `events.py`) routes on **confidence *and* margin-to-boundary**, and the second half is the part that is easy to drop.

Confidence answers *"how sure am I where the tip is."* It does not answer *"does that uncertainty change the score."* Those come apart constantly:

- 2 mm of uncertainty, 20 mm from any wire — certain, whatever the model says.
- The same 2 mm, 0.2 mm from the treble wire — a coin flip.

So the prompt does not merely ask; it says **why**: *"0.2 mm from the outer treble wire."* A player who knows the dart is on the wire can answer from memory in one glance. A bare "was that T20?" makes them reconstruct the throw.

## 4. A dart the camera never saw

#3 established that a dart leaving the board produces no tip and no event — there is nothing to detect. It is entered by hand, producing a `Source.MANUAL` event, and it is never queued for confirmation: the player entered it, and asking them to confirm their own entry is noise.

One detail that looks like a triviality and is not: a miss is stored at a board coordinate that **genuinely is off the board**, not at the origin. Anything re-deriving the score from the coordinate then reads `MISS`. Storing it at `(0, 0)` would make it re-derive as a double bull — a trap that would sit dormant until the first time someone recomputed a leg from stored positions.

## 5. What this deliberately does not do

No rules. No bust, no checkout, no turn order, no remaining score. `AGENTS.md`'s guardrail is that vision never decides game rules, and a test asserts that none of that vocabulary appears anywhere in what the session produces.

The session says *"this dart was a treble 20."* Whether that busted the leg is the game engine's to say, and keeping them apart is what lets scoring be tested without a rulebook and rules tested without a camera.

## 6. What the UI still owns

- **The single screen.** This produces one headline, one detail line and a list of available actions at a time, in #6's discipline — but the layout, the board overlay and the animation are the app's.
- **Debouncing `ready`.** A verdict that flickers while the player is still walking back is worse than one that settles a beat late.
- **The correction gesture.** The contract is "a throw id and a board position"; whether that is a tap on a board diagram, a segment picker, or a drag on the live overlay is a design question this record does not settle. The board diagram is the obvious first try, since the position is what the model got wrong.
- **Whether corrected throws are uploaded** as training data, with consent (#4 §7). A corrected throw is a labelled hard case from a board the project has never seen — more valuable than most of what #26 will spend hours capturing. This belongs to the privacy reviewer as much as to the UX lead.
- **Two players.** The session models one scoring stream; whose visit it is belongs to the game engine, and the app swaps the player between visits.
