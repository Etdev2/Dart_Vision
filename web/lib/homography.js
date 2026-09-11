/**
 * Image-to-board homography by normalized DLT.
 *
 * A port of `src/dartvision/geometry/calibration.py`, with one substitution.
 * The Python takes the smallest singular vector of the 2n x 9 design matrix via
 * NumPy's SVD; a browser has no NumPy, so this takes the smallest *eigen*vector
 * of A-transpose-A (9 x 9, symmetric) by cyclic Jacobi rotation instead.
 *
 * The two are the same vector. Forming A-transpose-A squares the condition
 * number, which is normally the reason to prefer SVD -- but the isotropic
 * normalization step exists precisely to keep that condition number small, and
 * parity against the Python is checked on real camera poses rather than
 * assumed.
 */

/** @typedef {[number, number]} Point */

const DEG = 180 / Math.PI;

/** Sector-boundary wires 90 degrees apart, in the fixed landmark order. */
export const CALIBRATION_ANGLES_DEG = Object.freeze([9.0, 99.0, 189.0, 279.0]);

/** @param {number} radius */
const ringPoints = (radius) =>
  CALIBRATION_ANGLES_DEG.map((a) => [
    radius * Math.cos(a / DEG),
    radius * Math.sin(a / DEG),
  ]);

/**
 * Board coordinates for the standard landmark counts.
 * Four: the outer double wire. Eight: those plus the outer treble wire.
 * @param {number} count
 * @param {{rDouble:number, rTreble:number}} board
 * @returns {Point[]}
 */
export function calibrationPoints(count, board) {
  if (count === 4) return /** @type {Point[]} */ (ringPoints(board.rDouble));
  if (count === 8) {
    return /** @type {Point[]} */ ([...ringPoints(board.rDouble), ...ringPoints(board.rTreble)]);
  }
  throw new Error(`no canonical board points for ${count} landmarks; 4 and 8 are standard`);
}

/** Indices of the landmarks actually found. Slot identifies a point, not position. */
export function visibleLandmarks(points) {
  const out = [];
  points.forEach((p, i) => { if (p) out.push(i); });
  return out;
}

/** Isotropic normalization: centroid at origin, mean distance sqrt(2). */
function normalize(points) {
  const n = points.length;
  const cx = points.reduce((s, p) => s + p[0], 0) / n;
  const cy = points.reduce((s, p) => s + p[1], 0) / n;
  const centred = points.map(([x, y]) => [x - cx, y - cy]);
  const mean = centred.reduce((s, [x, y]) => s + Math.hypot(x, y), 0) / n;
  if (mean < 1e-12) throw new Error('degenerate point set: all points coincide');
  const s = Math.SQRT2 / mean;
  return {
    points: centred.map(([x, y]) => [x * s, y * s]),
    t: [[s, 0, -s * cx], [0, s, -s * cy], [0, 0, 1]],
  };
}

/**
 * Eigenvector of the smallest eigenvalue of a symmetric matrix, by cyclic
 * Jacobi rotation. Converges quadratically and needs no pivoting strategy
 * beyond sweeping every off-diagonal, which for a 9x9 run once per frame is
 * far below the cost of anything else on the page.
 */
function smallestEigenvector(matrix) {
  const n = matrix.length;
  const a = matrix.map((row) => row.slice());
  let v = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));

  for (let sweep = 0; sweep < 60; sweep += 1) {
    let off = 0;
    for (let p = 0; p < n; p += 1) for (let q = p + 1; q < n; q += 1) off += a[p][q] ** 2;
    if (off < 1e-24) break;

    for (let p = 0; p < n; p += 1) {
      for (let q = p + 1; q < n; q += 1) {
        if (Math.abs(a[p][q]) < 1e-18) continue;
        const theta = (a[q][q] - a[p][p]) / (2 * a[p][q]);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1);
        const s = t * c;
        for (let k = 0; k < n; k += 1) {
          const akp = a[k][p], akq = a[k][q];
          a[k][p] = c * akp - s * akq;
          a[k][q] = s * akp + c * akq;
        }
        for (let k = 0; k < n; k += 1) {
          const apk = a[p][k], aqk = a[q][k];
          a[p][k] = c * apk - s * aqk;
          a[q][k] = s * apk + c * aqk;
          const vkp = v[k][p], vkq = v[k][q];
          v[k][p] = c * vkp - s * vkq;
          v[k][q] = s * vkp + c * vkq;
        }
      }
    }
  }

  let best = 0;
  for (let i = 1; i < n; i += 1) if (a[i][i] < a[best][best]) best = i;
  return v.map((row) => row[best]);
}

/** @param {number[][]} a @param {number[][]} b */
export function matMul(a, b) {
  return a.map((row, i) =>
    b[0].map((_, j) => row.reduce((s, _v, k) => s + a[i][k] * b[k][j], 0)));
}

/** @param {number[][]} m */
export function matInv3(m) {
  const [[a, b, c], [d, e, f], [g, h, i]] = m;
  const det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
  if (Math.abs(det) < 1e-18) throw new Error('singular matrix');
  return [
    [(e * i - f * h) / det, (c * h - b * i) / det, (b * f - c * e) / det],
    [(f * g - d * i) / det, (a * i - c * g) / det, (c * d - a * f) / det],
    [(d * h - e * g) / det, (b * g - a * h) / det, (a * e - b * d) / det],
  ];
}

/** Apply a 3x3 projective map to a point. @param {number[][]} m */
export function apply(m, [x, y]) {
  const w = m[2][0] * x + m[2][1] * y + m[2][2];
  if (Math.abs(w) < 1e-12) throw new Error('point maps to the line at infinity');
  return [
    (m[0][0] * x + m[0][1] * y + m[0][2]) / w,
    (m[1][0] * x + m[1][1] * y + m[1][2]) / w,
  ];
}

/**
 * Estimate the image-to-board homography.
 *
 * `imagePoints` are in the fixed landmark order and may contain `null` for
 * landmarks not found; those are dropped together with their board partners,
 * so a partial set still works as long as four survive. Pairing is by slot --
 * compacting the image points alone would match landmark 3 to landmark 2's
 * board coordinate and produce a plausible, wrong homography.
 *
 * @param {(Point|null)[]} imagePoints
 * @param {{rDouble:number, rTreble:number}} board
 * @param {Point[]} [boardPoints] one entry per slot, including missing ones
 * @returns {number[][]} 3x3, image -> board millimetres
 */
export function estimateHomography(imagePoints, board, boardPoints) {
  const kept = visibleLandmarks(imagePoints);
  if (kept.length < 4) {
    throw new Error(`a homography needs at least 4 correspondences, got ${kept.length}`);
  }
  const dst = boardPoints ?? calibrationPoints(imagePoints.length, board);
  if (dst.length !== imagePoints.length) {
    throw new Error(`got ${imagePoints.length} image points but ${dst.length} board points`);
  }

  const src = normalize(kept.map((i) => imagePoints[i]));
  const target = normalize(kept.map((i) => dst[i]));

  const rows = [];
  src.points.forEach(([x, y], i) => {
    const [u, v] = target.points[i];
    rows.push([-x, -y, -1, 0, 0, 0, u * x, u * y, u]);
    rows.push([0, 0, 0, -x, -y, -1, v * x, v * y, v]);
  });

  const ata = Array.from({ length: 9 }, (_, i) =>
    Array.from({ length: 9 }, (_, j) => rows.reduce((s, r) => s + r[i] * r[j], 0)));
  const h = smallestEigenvector(ata);
  const hn = [h.slice(0, 3), h.slice(3, 6), h.slice(6, 9)];

  const m = matMul(matMul(matInv3(target.t), hn), src.t);
  if (Math.abs(m[2][2]) < 1e-12) throw new Error('degenerate landmark configuration');
  return m.map((row) => row.map((value) => value / m[2][2]));
}

/**
 * Whether the first four visible landmarks form a convex quadrilateral in
 * order. Any four points in general position admit an exact homography, so a
 * detector reporting four points along one wire produces confident nonsense
 * that nothing downstream would complain about. This is where it gets caught.
 *
 * @param {(Point|null)[]} imagePoints
 */
export function convexAndOrdered(imagePoints) {
  const quad = visibleLandmarks(imagePoints).slice(0, 4).map((i) => imagePoints[i]);
  if (quad.length < 4) return false;

  let sign = 0;
  for (let i = 0; i < 4; i += 1) {
    const [ax, ay] = quad[i];
    const [bx, by] = quad[(i + 1) % 4];
    const [cx, cy] = quad[(i + 2) % 4];
    const cross = (bx - ax) * (cy - by) - (by - ay) * (cx - bx);
    if (Math.abs(cross) < 1e-12) return false;
    const current = Math.sign(cross);
    if (sign === 0) sign = current;
    else if (current !== sign) return false;
  }
  return true;
}
