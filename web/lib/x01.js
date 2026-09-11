/**
 * X01 rules: 501, 301, double-out.
 *
 * A port of `src/dartvision/game/x01.py`, checked against it by
 * `web/test/parity.test.mjs`. Knows nothing about cameras: vision says *this
 * dart is a treble 20*, this says whether that won the leg.
 *
 * **The engine is a fold, not a mutable scoreboard.** `replay` is the whole
 * rule set for a leg and `play` folds a match out of visits. That shape is
 * what makes corrections work: a correction never mutates history, so applying
 * one means folding the corrected sequence again. A mutable scoreboard would
 * have to *undo* a dart, and undoing a bust correctly -- restoring the
 * pre-visit total, un-ending the turn, un-switching the player -- is the kind
 * of thing that is wrong for a year before anyone notices.
 */

export const DARTS_PER_VISIT = 3;

export const SCORED = 'scored';
export const BUST = 'bust';
export const WON = 'won';
export const IGNORED = 'ignored';

/** @typedef {{start:number, doubleOut:boolean, doubleIn:boolean}} X01Rules */

/** @returns {X01Rules} */
export function rules({ start = 501, doubleOut = true, doubleIn = false } = {}) {
  if (start < 2) throw new Error('start must be at least 2');
  if (doubleOut && start % 2 && start < 4) {
    throw new Error('an odd start below 4 cannot be finished on a double');
  }
  return { start, doubleOut, doubleIn };
}

/** @typedef {{segment:number, multiplier:number}} Dart */

/** A dart, validated. Not a ThrowEvent: the rules must work on paper. */
export function dart(segment, multiplier) {
  if (![0, 1, 2, 3].includes(multiplier)) throw new Error('multiplier must be 0, 1, 2 or 3');
  if (multiplier === 0 && segment !== 0) throw new Error('a miss has segment 0');
  if (segment === 25 && multiplier === 3) throw new Error('there is no treble bull');
  if (segment !== 0 && segment !== 25 && !(segment >= 1 && segment <= 20)) {
    throw new Error(`no such segment: ${segment}`);
  }
  return { segment, multiplier };
}

export const score = (d) => d.segment * d.multiplier;
/** The double bull counts as a double, which is why 50 finishes a leg. */
export const isDouble = (d) => d.multiplier === 2;

export function notation(d) {
  if (d.multiplier === 0) return 'MISS';
  if (d.segment === 25) return d.multiplier === 2 ? 'DB' : 'SB';
  return `${'SDT'[d.multiplier - 1]}${d.segment}`;
}

/**
 * @typedef {{dart: Dart, outcome: string, remaining: number, reason: string}} VisitResult
 * @typedef {{rules: X01Rules, remaining: number, dartsThrown: number, visitDarts: number,
 *   visitStartRemaining: number, opened: boolean, winner: boolean,
 *   results: VisitResult[]}} LegState
 */

/** @param {X01Rules} r @returns {LegState} */
const startLeg = (r) => ({
  rules: r,
  remaining: r.start,
  dartsThrown: 0,
  visitDarts: 0,
  visitStartRemaining: r.start,
  opened: !r.doubleIn,
  winner: false,
  results: [],
});

/** @param {LegState} leg */
export const visitScore = (leg) => leg.visitStartRemaining - leg.remaining;
/** @param {LegState} leg */
export const threeDartAverage = (leg) =>
  (leg.dartsThrown ? ((leg.rules.start - leg.remaining) / leg.dartsThrown) * DARTS_PER_VISIT : 0);

/**
 * Fold one dart into a leg. The entire rule set lives here.
 * @param {LegState} leg @param {Dart} d @returns {LegState}
 */
function applyDart(leg, d) {
  const record = (outcome, remaining, reason = '') => ({
    ...leg,
    remaining,
    dartsThrown: leg.dartsThrown + 1,
    visitDarts: leg.visitDarts + 1,
    results: [...leg.results, { dart: d, outcome, remaining, reason }],
  });

  if (leg.winner) {
    return {
      ...leg,
      results: [...leg.results,
        { dart: d, outcome: IGNORED, remaining: leg.remaining, reason: 'the leg is already won' }],
    };
  }

  if (!leg.opened) {
    // Double-in: nothing counts until a double lands.
    if (!isDouble(d)) return record(IGNORED, leg.remaining, 'not opened yet');
    return { ...record(SCORED, leg.remaining - score(d)), opened: true };
  }

  const remaining = leg.remaining - score(d);
  if (remaining === 0 && (isDouble(d) || !leg.rules.doubleOut)) {
    return { ...record(WON, 0), winner: true };
  }

  // Three ways to bust, and they are one rule: the leg must stay finishable.
  let reason = null;
  if (remaining < 0) reason = 'went below zero';
  else if (remaining === 0) reason = 'finished on a single';
  else if (remaining === 1 && leg.rules.doubleOut) reason = 'left 1, which no double can finish';

  if (reason) {
    // A bust voids the whole visit, not just the offending dart.
    return { ...record(BUST, leg.visitStartRemaining, reason), visitDarts: DARTS_PER_VISIT };
  }
  return record(SCORED, remaining);
}

const endVisit = (leg) => ({ ...leg, visitDarts: 0, visitStartRemaining: leg.remaining });

/** Fold a sequence of darts into a leg. A correction is this, called again. */
/** @param {Dart[]} darts @param {X01Rules} [r] @returns {LegState} */
export function replay(darts, r = rules()) {
  let leg = startLeg(r);
  for (const d of darts) {
    if (!leg.winner && leg.visitDarts >= DARTS_PER_VISIT) leg = endVisit(leg);
    leg = applyDart(leg, d);
  }
  return leg;
}

/**
 * Fold a match from its visits.
 *
 * A **visit** is one player's darts before they collect them -- one to three,
 * fewer when they check out or stand down. Taking visits rather than a flat
 * list removes the one ambiguity a flat list has: a player who throws two and
 * walks up is indistinguishable from one mid-visit. The match session knows
 * where that boundary is, because the board clearing is what tells it.
 *
 * @param {Dart[][]} visits
 * @param {{rules?: X01Rules, players?: string[], legsToWin?: number}} [options]
 */
export function play(visits, { rules: r = rules(), players = ['Player 1', 'Player 2'], legsToWin = 1 } = {}) {
  const names = [...players];
  if (names.length < 1) throw new Error('a game needs at least one player');
  if (new Set(names).size !== names.length) throw new Error('player names must be distinct');
  if (legsToWin < 1) throw new Error('legsToWin must be at least 1');

  let legs = names.map(() => startLeg(r));
  const won = names.map(() => 0);
  // Who throws first alternates leg by leg, so the advantage does not sit with
  // one player for the whole match.
  let legStarter = 0;
  let toThrow = 0;
  let legNumber = 1;

  for (const visit of visits) {
    if (won.some((count) => count >= legsToWin)) break;

    const index = toThrow;
    let leg = legs[index];
    for (const d of visit) {
      if (leg.visitDarts >= DARTS_PER_VISIT || leg.winner) break;
      leg = applyDart(leg, d);
    }
    legs[index] = leg;

    if (leg.winner) {
      won[index] += 1;
      legs = names.map(() => startLeg(r));
      legNumber += 1;
      legStarter = (legStarter + 1) % names.length;
      toThrow = legStarter;
      continue;
    }
    legs[index] = endVisit(leg);
    toThrow = (toThrow + 1) % names.length;
  }

  return {
    rules: r,
    players: names,
    legsToWin,
    legs,
    legsWon: won,
    toThrow,
    legNumber,
    currentPlayer: names[toThrow],
    remaining: Object.fromEntries(names.map((p, i) => [p, legs[i].remaining])),
    wonLegs: Object.fromEntries(names.map((p, i) => [p, won[i]])),
    winner: names.find((_, i) => won[i] >= legsToWin) ?? null,
    legFor: (player) => legs[names.indexOf(player)],
  };
}
