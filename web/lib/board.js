/**
 * Deterministic dartboard geometry and scoring.
 *
 * A direct port of `src/dartvision/geometry/board.py`. The Python is the
 * reference implementation and its test suite is the specification;
 * `web/test/parity.test.mjs` checks the two agree on fixtures generated from
 * the Python itself, because a port that silently disagrees is worse than no
 * port at all.
 *
 * Plain ES modules rather than TypeScript on purpose: they run in a browser
 * today with no build step, and Next.js imports them unchanged. The JSDoc
 * types are real types under `checkJs`.
 *
 * Board coordinates are millimetres, origin at the bull, +x right and +y up.
 * Angles run counter-clockwise from +x. The 20 bed is centred on +y.
 */

/** Sector numbers clockwise from 20, which sits at the top. */
export const SECTORS_CLOCKWISE_FROM_20 = Object.freeze([
  20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5,
]);

const SECTOR_ARC_DEG = 360 / 20;
/** The 20 bed is centred at 90 degrees, so its counter-clockwise edge is 99. */
const FIRST_BOUNDARY_DEG = 90 + SECTOR_ARC_DEG / 2;

const DEG = 180 / Math.PI;

/**
 * Distance from the bull.
 *
 * A note on portability, because it was measured rather than assumed. Python's
 * `math.hypot` is correctly rounded; neither `Math.hypot` nor `sqrt(x*x+y*y)`
 * reproduces it bit-for-bit, and the two JS forms disagree with each other too.
 * Across the parity fixtures every such disagreement lands **within 1e-9 mm of
 * a boundary radius** and none anywhere else -- so scoring is exactly portable
 * except for a point sitting on a wire to within a picometre, where the float
 * representation decides and no implementation is canonical.
 *
 * That is a measure-zero case a physical dart never reaches, and
 * `parity.test.mjs` pins the property rather than pretending it away.
 *
 * @param {number} x @param {number} y
 */
const radius = (x, y) => Math.hypot(x, y);

/**
 * @typedef {object} BoardSpec
 * @property {number} rBoard        full radius, including the number ring
 * @property {number} rDouble       outer edge of the double ring
 * @property {number} rTreble       outer edge of the treble ring
 * @property {number} rOuterBull    outer edge of the 25 ring
 * @property {number} rInnerBull    outer edge of the bullseye
 * @property {number} ringWidth     radial width of the double and treble rings
 */

/** BDO standard measurements. @type {BoardSpec} */
export const BDO_BOARD = Object.freeze({
  rBoard: 225.5,
  rDouble: 170.0,
  rTreble: 107.4,
  rOuterBull: 15.9,
  rInnerBull: 6.35,
  ringWidth: 10.0,
});

/** @param {BoardSpec} b */
export const rDoubleInner = (b) => b.rDouble - b.ringWidth;
/** @param {BoardSpec} b */
export const rTrebleInner = (b) => b.rTreble - b.ringWidth;

/**
 * @typedef {object} Hit
 * @property {number} segment     bed number, or 25 for either bull
 * @property {number} multiplier  0 miss, 1 single, 2 double, 3 treble
 * @property {number} score
 * @property {string} notation
 */

/** @param {number} angleDeg */
function sectorIndex(angleDeg) {
  // Beds advance clockwise, i.e. as the angle decreases.
  const raw = Math.floor((FIRST_BOUNDARY_DEG - angleDeg) / SECTOR_ARC_DEG);
  return ((raw % 20) + 20) % 20;
}

/**
 * Score a board-coordinate point, in millimetres.
 *
 * Boundaries belong to the higher-scoring region: a point exactly on the outer
 * treble wire is a treble. It only matters for measure-zero cases, but fixing
 * the convention keeps the function total and the tests exact.
 *
 * @param {number} x @param {number} y @param {BoardSpec} [board]
 * @returns {Hit}
 */
export function scoreAt(x, y, board = BDO_BOARD) {
  const r = radius(x, y);

  if (r > board.rDouble) return { segment: 0, multiplier: 0, score: 0, notation: 'MISS' };
  if (r <= board.rInnerBull) return { segment: 25, multiplier: 2, score: 50, notation: 'DB' };
  if (r <= board.rOuterBull) return { segment: 25, multiplier: 1, score: 25, notation: 'SB' };

  const segment = SECTORS_CLOCKWISE_FROM_20[sectorIndex(Math.atan2(y, x) * DEG)];

  let multiplier, prefix;
  if (r > rDoubleInner(board)) {
    [multiplier, prefix] = [2, 'D'];
  } else if (rTrebleInner(board) < r && r <= board.rTreble) {
    [multiplier, prefix] = [3, 'T'];
  } else {
    [multiplier, prefix] = [1, 'S'];
  }
  return { segment, multiplier, score: segment * multiplier, notation: `${prefix}${segment}` };
}

/**
 * @typedef {object} Boundary
 * @property {'radial'|'sector'} kind
 * @property {string} name
 * @property {number} distanceMm
 * @property {boolean} inside
 */

/** @param {BoardSpec} b */
const radialBoundaries = (b) => [
  [b.rInnerBull, 'bullseye ring'],
  [b.rOuterBull, 'outer bull ring'],
  [rTrebleInner(b), 'inner treble wire'],
  [b.rTreble, 'outer treble wire'],
  [rDoubleInner(b), 'inner double wire'],
  [b.rDouble, 'outer double wire'],
];

/**
 * The nearest score-changing boundary, named.
 *
 * Sector wires are ignored inside the outer bull, where the bed number does
 * not affect the score.
 *
 * @param {number} x @param {number} y @param {BoardSpec} [board]
 * @returns {Boundary}
 */
export function nearestBoundary(x, y, board = BDO_BOARD) {
  const r = radius(x, y);

  let best = null;
  for (const [radius, label] of radialBoundaries(board)) {
    const distance = Math.abs(r - Number(radius));
    if (best === null || distance < best.distanceMm) {
      best = { kind: 'radial', name: String(label), distanceMm: distance, inside: r < radius };
    }
  }
  if (r <= board.rOuterBull) return /** @type {Boundary} */ (best);

  const angle = Math.atan2(y, x) * DEG;
  const offset = (((FIRST_BOUNDARY_DEG - angle) % SECTOR_ARC_DEG) + SECTOR_ARC_DEG) % SECTOR_ARC_DEG;
  const angularMm = (r * Math.min(offset, SECTOR_ARC_DEG - offset)) / DEG;
  if (angularMm >= /** @type {Boundary} */ (best).distanceMm) return /** @type {Boundary} */ (best);

  const index = Math.floor((FIRST_BOUNDARY_DEG - angle) / SECTOR_ARC_DEG);
  const wire = offset <= SECTOR_ARC_DEG / 2 ? index : index + 1;
  const ccw = SECTORS_CLOCKWISE_FROM_20[(((wire - 1) % 20) + 20) % 20];
  const cw = SECTORS_CLOCKWISE_FROM_20[(((wire % 20) + 20) % 20)];
  return { kind: 'sector', name: `${ccw}|${cw} wire`, distanceMm: angularMm, inside: false };
}

/**
 * Millimetres to the nearest boundary that changes the score.
 *
 * The quantity that decides whether a localization error matters at all: an
 * error smaller than the margin cannot change the score.
 *
 * @param {number} x @param {number} y @param {BoardSpec} [board]
 */
export function marginToNearestBoundary(x, y, board = BDO_BOARD) {
  return nearestBoundary(x, y, board).distanceMm;
}
