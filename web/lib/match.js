/**
 * The match loop: what the screen says, and what a tap does.
 *
 * A port of `src/dartvision/match/session.py`. Two decisions carry it.
 *
 * **Confirmations wait for the retrieval pause.** Three darts take about ten
 * seconds, so a prompt when an uncertain dart lands arrives while the player is
 * already mid-throw. Instead the dart scores provisionally on landing -- which
 * holds the 300 ms impact-to-score budget -- and the question is asked when the
 * visit ends, during the ten to fifteen seconds of pulling darts out. Two taps
 * a leg then cost nothing, because they are spent in time the game was wasting.
 *
 * **Correcting is not a state.** A modal correction screen blocks the next
 * throw to fix the last one. It is an action, available any time, emitting a
 * new event that supersedes rather than mutates.
 *
 * Knows nothing about rules. No bust, no checkout, no turn order -- that is the
 * game engine's, and the separation is what lets scoring be tested without a
 * rulebook.
 */

import { BDO_BOARD, nearestBoundary, scoreAt } from './board.js';

export const DARTS_PER_VISIT = 3;

export const NOT_CALIBRATED = 'not_calibrated';
export const READY = 'ready';
export const SCORING = 'scoring';
export const CONFIRMING = 'confirming';
export const AWAITING_REMOVAL = 'awaiting_removal';

export const CAMERA = 'camera';
export const MANUAL = 'manual';
export const CORRECTION = 'correction';

/**
 * Which throws get asked about.
 *
 * Confidence answers "how sure am I where the tip is". It does not answer
 * "does that uncertainty change the score", and the two come apart constantly:
 * 2 mm of error 20 mm from any wire is certain; the same 2 mm a fifth of a
 * millimetre from the treble wire is a coin flip. So the policy reads both.
 */
export const DEFAULT_POLICY = Object.freeze({
  minConfidence: 0.9,
  minMarginMm: 2.0,
  highConfidence: 0.99,
});

/** @param {object} event @param {typeof DEFAULT_POLICY} policy */
export function shouldConfirm(event, policy = DEFAULT_POLICY) {
  const { confidence, marginMm } = event.detected;
  if (confidence < policy.minConfidence) return true;
  // Near a boundary, only a very confident call stands on its own.
  if (marginMm < policy.minMarginMm) return confidence < policy.highConfidence;
  return false;
}

let counter = 0;
const nextId = () => `t${(counter += 1)}`;

export class MatchSession {
  /** @param {{policy?: object, board?: object, newId?: () => string}} [options] */
  constructor(options = {}) {
    this.policy = options.policy ?? DEFAULT_POLICY;
    this.board = options.board ?? BDO_BOARD;
    this.newId = options.newId ?? nextId;

    this.calibration = null;
    this.visit = [];
    this.history = [];
    this.unconfirmed = [];
    this.dartsInBoard = false;
    this.visitClosed = false;
  }

  setCalibration(status) {
    // Losing calibration does not discard the visit. Those darts were scored
    // under a calibration that was valid when they landed; throwing them away
    // would be a worse failure than the one being reported.
    this.calibration = status;
    return this.prompt();
  }

  /** Build an event from a board coordinate, scoring it deterministically. */
  event(x, y, confidence, source, extra = {}) {
    const hit = scoreAt(x, y, this.board);
    const boundary = nearestBoundary(x, y, this.board);
    return {
      id: this.newId(),
      capturedAt: new Date().toISOString(),
      source,
      boardXMm: x,
      boardYMm: y,
      detected: {
        segment: hit.segment,
        multiplier: hit.multiplier,
        score: hit.score,
        notation: hit.notation,
        confidence,
        marginMm: boundary.distanceMm,
        boundary: boundary.name,
      },
      ...extra,
    };
  }

  recordThrow(x, y, confidence) {
    if (!this.calibration?.scoringPossible) {
      throw new Error('no valid calibration; a throw cannot be scored and must not be invented');
    }
    if (this.visit.length >= DARTS_PER_VISIT) {
      throw new Error(`a visit holds at most ${DARTS_PER_VISIT} darts`);
    }
    const event = this.event(x, y, confidence, CAMERA, { sequence: this.visit.length + 1 });
    this.visit.push(event);
    this.history.push(event);
    if (shouldConfirm(event, this.policy)) this.unconfirmed.push(event.id);
    this.dartsInBoard = true;
    if (this.visit.length === DARTS_PER_VISIT) this.visitClosed = true;
    return event;
  }

  /**
   * A dart that left the board and was never detected.
   *
   * Placed at a coordinate that genuinely is off the board rather than at the
   * origin, so anything re-deriving the score from the position still reads
   * MISS instead of a double bull. Never queued: the player entered it.
   */
  recordMiss() {
    const event = this.event(0, this.board.rBoard + 10, 1, MANUAL,
      { sequence: this.visit.length + 1 });
    this.visit.push(event);
    this.history.push(event);
    if (this.visit.length === DARTS_PER_VISIT) this.visitClosed = true;
    return event;
  }

  /** The darts came out. Ends the visit however many were thrown. */
  boardCleared() {
    this.dartsInBoard = false;
    this.visitClosed = true;
    // Queued questions survive this: they are about the score, not about what
    // is still stuck in the board.
    if (!this.unconfirmed.length) this.startVisit();
    return this.prompt();
  }

  confirm(throwId) {
    this.resolve(throwId);
    return this.prompt();
  }

  /** Replace a throw's score without rewriting what was recorded. */
  correct(throwId, x, y) {
    const original = this.find(throwId);
    if (!original) throw new Error(`no throw ${throwId}`);

    const replacement = this.event(x, y, 1, CORRECTION, {
      corrects: original.id,
      sequence: original.sequence,
    });
    this.history.push(replacement);
    this.visit = this.visit.map((t) => (t.id === original.id ? replacement : t));
    this.resolve(throwId);
    return replacement;
  }

  resolve(throwId) {
    this.unconfirmed = this.unconfirmed.filter((id) => id !== throwId);
    if (this.visitClosed && !this.unconfirmed.length && !this.dartsInBoard) this.startVisit();
  }

  startVisit() {
    this.visit = [];
    this.visitClosed = false;
  }

  find(throwId) {
    return this.history.find((t) => t.id === throwId) ?? null;
  }

  get pending() {
    return this.visit.filter((t) => this.unconfirmed.includes(t.id));
  }

  get visitScore() {
    return this.visit.reduce((total, t) => total + t.detected.score, 0);
  }

  /** History with corrections applied -- what a game engine would score. */
  effectiveHistory() {
    const superseded = new Set(this.history.filter((t) => t.corrects).map((t) => t.corrects));
    return this.history.filter((t) => !superseded.has(t.id));
  }

  /** Derived, never stored. Storing it drifts the moment two inputs interact. */
  get state() {
    if (!this.calibration?.scoringPossible) return NOT_CALIBRATED;
    if (this.visitClosed && this.unconfirmed.length) return CONFIRMING;
    if (this.visitClosed && this.dartsInBoard) return AWAITING_REMOVAL;
    if (this.visit.length) return SCORING;
    return READY;
  }

  /** The single thing the screen should say, and what can be tapped. */
  prompt() {
    const state = this.state;

    if (state === NOT_CALIBRATED) {
      // Never show this as a frozen or blank scoreboard -- the player cannot
      // tell whether to keep throwing.
      return {
        state,
        headline: 'Board not visible',
        detail: this.visit.length
          ? 'Scoring is paused. Darts already thrown this visit are safe.'
          : 'Point the phone at the board to start scoring.',
        actions: ['recalibrate', 'enter_manually'],
        throwId: null,
      };
    }

    if (state === CONFIRMING) {
      const throwEvent = this.pending[0];
      const { notation, marginMm, boundary } = throwEvent.detected;
      return {
        state,
        headline: `Was that ${notation}?`,
        detail: boundary ? `${marginMm.toFixed(1)} mm from the ${boundary}` : 'Close to a boundary.',
        actions: ['confirm', 'correct'],
        throwId: throwEvent.id,
      };
    }

    if (state === AWAITING_REMOVAL) {
      return {
        state,
        headline: `${this.visitScore}`,
        detail: 'Collect your darts.',
        actions: ['correct', 'undo_visit'],
        throwId: null,
      };
    }

    if (state === SCORING) {
      const last = this.visit[this.visit.length - 1];
      return {
        state,
        headline: last.detected.notation,
        detail: `${this.visitScore} so far · dart ${this.visit.length} of 3`,
        actions: ['correct', 'record_miss'],
        throwId: last.id,
      };
    }

    return {
      state: READY,
      headline: 'Ready',
      detail: 'Throw when you are.',
      actions: ['record_miss'],
      throwId: null,
    };
  }
}
