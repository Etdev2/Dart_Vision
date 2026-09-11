'use client';

/**
 * The match loop: the vision side and the rules side, joined.
 *
 * Two engines, deliberately kept apart. `MatchSession` owns the visit -- what
 * the screen says, which darts need confirming, how a correction is recorded.
 * The X01 engine owns the game -- remaining, busts, checkouts, whose turn.
 * Neither knows about the other, and this component is the only place they
 * meet: a completed visit goes from one to the other when the darts come out.
 *
 * That is `AGENTS.md`'s guardrail made concrete. Vision never decides a rule,
 * and the rules never see a confidence.
 *
 * Tapping the board stands in for a dart landing until there is a model; a real
 * build swaps `onThrow` for the stream layer's throw events.
 */

import { useState } from 'react';
import { CONFIRMING, MatchSession } from '@/lib/match.js';
import { BUST, WON, dart, play, replay } from '@/lib/x01.js';
import Dartboard from '@/components/Dartboard';
import Scoreboard, { type Provisional } from '@/components/Scoreboard';

const PLAYERS = ['You', 'Them'];
const LEGS_TO_WIN = 2;

/** A visit as the rules see it: segments and multipliers, no confidence. */
type Visit = { segment: number; multiplier: number }[];

function newSession() {
  const session = new MatchSession();
  session.setCalibration({ scoringPossible: true, calibrationId: 'demo' });
  return session;
}

export default function MatchLoop() {
  const [session, setSession] = useState(newSession);
  const [visits, setVisits] = useState<Visit[]>([]);
  const [confidence, setConfidence] = useState(0.97);
  const [, setTick] = useState(0);
  const render = () => setTick((n) => n + 1);

  const game = play(visits, { players: PLAYERS, legsToWin: LEGS_TO_WIN });

  // The running score, folded from the current player's leg plus the darts
  // already in the board. Same fold, so it cannot disagree with the value that
  // gets recorded when the visit settles.
  const provisional: Provisional | null = session.visit.length
    ? (() => {
        const leg = game.legFor(game.currentPlayer);
        const thrown = leg.results.map((r) => r.dart);
        const inFlight: Visit = session.visit.map((t) => ({
          segment: t.detected.segment, multiplier: t.detected.multiplier,
        }));
        const live = replay([...thrown, ...inFlight], game.rules);
        const last = live.results[live.results.length - 1];
        return {
          remaining: live.remaining,
          busted: last?.outcome === BUST,
          checkedOut: last?.outcome === WON,
        };
      })()
    : null;
  const prompt = session.prompt();
  const confirming = prompt.state === CONFIRMING;
  const over = game.winner !== null;

  const onThrow = (x: number, y: number) => {
    if (over) return;
    try {
      if (confirming && prompt.throwId) session.correct(prompt.throwId, x, y);
      else session.recordThrow(x, y, confidence);
    } catch {
      // Visit full, or no calibration. The prompt already says so.
    }
    render();
  };

  /**
   * The darts come out: hand the visit to the rules.
   *
   * This is the only crossing point. The vision side has finished deciding
   * what each dart was -- including any correction -- so the rules receive a
   * settled visit rather than a stream it would have to revise.
   */
  const collect = () => {
    const visit: Visit = session.visit.map((t) => ({
      segment: t.detected.segment,
      multiplier: t.detected.multiplier,
    }));
    visit.forEach((d) => dart(d.segment, d.multiplier));   // reject the impossible early
    setVisits((current) => [...current, visit]);
    session.boardCleared();
    render();
  };

  const restart = () => {
    setSession(newSession());
    setVisits([]);
  };

  return (
    <>
      {over && (
        <div className="banner">
          <div className="headline">{game.winner} won the match</div>
          <button onClick={restart}>New match</button>
        </div>
      )}

      <Scoreboard game={game} visitDarts={session.visit.length} provisional={provisional} />

      <div className="card">
        <div className="prompt-block">
          <div className="big">{prompt.headline}</div>
          <div className="detail">{prompt.detail}</div>
        </div>

        <div className="throws">
          {session.visit.map((thrown) => {
            const unsure = session.pending.includes(thrown);
            const corrected = thrown.source === 'correction';
            const className = corrected ? 'throw corrected' : unsure ? 'throw unsure' : 'throw';
            return (
              <span key={thrown.id} className={className}>
                {thrown.detected.notation}{corrected ? ' ✎' : unsure ? ' ?' : ''}
              </span>
            );
          })}
        </div>

        <div className="row" style={{ marginTop: 12 }}>
          {confirming && prompt.throwId ? (
            <button className="primary" onClick={() => { session.confirm(prompt.throwId!); render(); }}>
              Yes, that’s right
            </button>
          ) : (
            <>
              {session.visit.length > 0 && (
                <button className="primary" onClick={collect} disabled={over}>
                  Darts collected
                </button>
              )}
              {prompt.actions.includes('record_miss') && !over && (
                <button onClick={() => { try { session.recordMiss(); } catch { /* full */ } render(); }}>
                  Missed the board
                </button>
              )}
            </>
          )}
        </div>
      </div>

      <div className="card">
        <h2>
          {over ? 'Match over' : confirming ? 'Tap the board to correct this dart' : `Tap the board — ${game.currentPlayer} to throw`}
        </h2>
        <Dartboard
          darts={session.visit.map((d) => ({ x: d.boardXMm, y: d.boardYMm }))}
          onThrow={onThrow}
        />
        <div className="row" style={{ marginTop: 10 }}>
          <label style={{ fontSize: 13, color: 'var(--muted)' }}>
            Model confidence{' '}
            <input
              type="range" min={0.5} max={1} step={0.01} value={confidence}
              onChange={(e) => setConfidence(Number(e.target.value))}
              style={{ verticalAlign: 'middle' }}
            />{' '}
            <b style={{ fontVariantNumeric: 'tabular-nums' }}>{confidence.toFixed(2)}</b>
          </label>
        </div>
        <p className="note" style={{ margin: '10px 0 0' }}>
          Uncertain darts still score instantly — the question waits until the visit
          ends, which is when you would be pulling darts out anyway. Drop the
          confidence, or throw near a wire, to see it happen.
        </p>
      </div>
    </>
  );
}
