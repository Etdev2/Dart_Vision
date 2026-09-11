'use client';

/**
 * The match loop, driven by the real `MatchSession`.
 *
 * The session is the port of `src/dartvision/match/session.py`, so this
 * component holds no scoring or state-machine logic of its own -- it renders
 * whatever `session.prompt()` says and turns taps into calls. That is the
 * point of the boundary: the rules of the loop are tested in Python, checked
 * in the port, and the React here is a view.
 *
 * A real build swaps `onThrow` for the stream layer's `throw` events. Tapping
 * stands in for a dart landing until there is a model.
 */

import { useRef, useState } from 'react';
import { CONFIRMING, MatchSession } from '@/lib/match.js';
import Dartboard from '@/components/Dartboard';

export default function MatchLoop() {
  const sessionRef = useRef<MatchSession | null>(null);
  if (sessionRef.current === null) {
    const session = new MatchSession();
    session.setCalibration({ scoringPossible: true, calibrationId: 'demo' });
    sessionRef.current = session;
  }
  const session = sessionRef.current;

  const [, setTick] = useState(0);
  const render = () => setTick((n) => n + 1);
  const [confidence, setConfidence] = useState(0.97);

  const prompt = session.prompt();
  const confirming = prompt.state === CONFIRMING;

  const onThrow = (x: number, y: number) => {
    try {
      if (confirming && prompt.throwId) session.correct(prompt.throwId, x, y);
      else session.recordThrow(x, y, confidence);
    } catch {
      // The visit is full, or there is no calibration. The prompt already
      // says so; a thrown error is not the player's problem.
    }
    render();
  };

  const act = (fn: () => void) => () => { fn(); render(); };

  return (
    <>
      <div className="card">
        <div className="prompt-block">
          <div className="big">{prompt.headline}</div>
          <div className="detail">{prompt.detail}</div>
        </div>

        <div className="throws">
          {session.visit.map((dart) => {
            const unsure = session.pending.includes(dart);
            const corrected = dart.source === 'correction';
            const className = corrected ? 'throw corrected' : unsure ? 'throw unsure' : 'throw';
            return (
              <span key={dart.id} className={className}>
                {dart.detected.notation}{corrected ? ' ✎' : unsure ? ' ?' : ''}
              </span>
            );
          })}
        </div>

        <div className="row" style={{ marginTop: 12 }}>
          {confirming && prompt.throwId ? (
            <button className="primary" onClick={act(() => session.confirm(prompt.throwId!))}>
              Yes, that’s right
            </button>
          ) : (
            <>
              {session.visit.length > 0 && (
                <button className="primary" onClick={act(() => session.boardCleared())}>
                  Darts collected
                </button>
              )}
              {prompt.actions.includes('record_miss') && (
                <button onClick={act(() => { try { session.recordMiss(); } catch { /* full */ } })}>
                  Missed the board
                </button>
              )}
            </>
          )}
        </div>
      </div>

      <div className="card">
        <h2>{confirming ? 'Tap the board to correct this dart' : 'Tap the board to throw'}</h2>
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
