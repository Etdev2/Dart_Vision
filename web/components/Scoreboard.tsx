'use client';

/**
 * The game state, folded from every visit thrown.
 *
 * Reads from the X01 engine, never from the vision side. The scorer says what
 * a dart was; this says what it did to the leg.
 *
 * The active player's number is *provisional* -- folded from their leg so far
 * plus the darts currently in the board. A player needs to watch the score come
 * down as they throw, and the same fold that settles the visit produces the
 * running value, so the preview cannot drift from what will be recorded. It
 * also means a bust shows the instant it happens, which is exactly when the
 * player needs to know to stop throwing.
 */


type Game = {
  players: string[];
  remaining: Record<string, number>;
  wonLegs: Record<string, number>;
  currentPlayer: string;
  legNumber: number;
  legsToWin: number;
  winner: string | null;
  legFor: (player: string) => { dartsThrown: number };
};

export type Provisional = { remaining: number; busted: boolean; checkedOut: boolean };

export default function Scoreboard({
  game, visitDarts, provisional,
}: { game: Game; visitDarts: number; provisional: Provisional | null }) {
  return (
    <div className="card">
      <h2>Leg {game.legNumber} · first to {game.legsToWin}</h2>
      <div className="scoreboard">
        {game.players.map((player) => {
          const active = player === game.currentPlayer && !game.winner;
          const live = active && provisional ? provisional : null;
          const remaining = live ? live.remaining : game.remaining[player];

          let state = `${game.legFor(player).dartsThrown} darts this leg`;
          if (live?.checkedOut) state = 'checked out';
          else if (live?.busted) state = 'bust — score stands';
          // The prompt already owns "dart n of 3" for the dart that just
          // landed. Counting the next one here put two different numbers on
          // the same screen.
          else if (active) {
            state = visitDarts ? `${visitDarts} dart${visitDarts === 1 ? '' : 's'} in` : 'to throw';
          }

          return (
            <div
              key={player}
              className={`player${active ? ' active' : ''}${live?.busted ? ' busted' : ''}`}
            >
              <div className="who">
                {player}
                {game.legsToWin > 1 && <span className="legs"> · {game.wonLegs[player]}</span>}
              </div>
              <b>{remaining}</b>
              <div className="detail">{state}</div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
