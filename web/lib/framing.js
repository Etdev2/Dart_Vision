/**
 * Setup-time framing guidance.
 *
 * A port of `src/dartvision/calibrate/framing.py`. The screen never asks the
 * player for an angle -- it measures one, from nothing but the eight
 * landmarks.
 *
 * A plane seen face-on scales equally in every direction; tilted by `el` out of
 * the image plane, it compresses along the tilt by sin(el). So the two singular
 * values of the homography's local Jacobian at the bull stand in ratio
 * 1/sin(el), and asin recovers the angle -- with no focal length, sensor size
 * or distance, none of which a browser can be trusted to report.
 */

import { BDO_BOARD } from './board.js';
import { apply, convexAndOrdered, estimateHomography, matInv3, visibleLandmarks } from './homography.js';

const DEG = 180 / Math.PI;

export const READY = 'ready';
export const MARGINAL = 'marginal';
export const UNUSABLE = 'unusable';

/**
 * Where the bands come from, so they can be argued with rather than guessed.
 *
 * `minElevationDeg` is the floor from #2: below it, the 4 mm gate needs
 * sub-pixel precision the representation cannot deliver. `maxElevationDeg` is
 * not about accuracy at all -- face-on is the best view measured. It is where
 * the darts are.
 *
 * `minRingPx` is the precision budget from #15, measured at the model's input
 * size and through the player's actual perspective. A board tilted out of the
 * image plane has narrower rings than the face-on arithmetic suggests, by
 * roughly sin(elevation), which is why the default input is 768 rather than
 * the 640 a face-on calculation would justify.
 */
export const DEFAULT_LIMITS = Object.freeze({
  minElevationDeg: 30,
  idealElevationDeg: [45, 65],
  maxElevationDeg: 70,
  minRingPx: 10,
  idealRingPx: 12,
  modelInputPx: 768,
});

/** Local board-to-image scale at a board point, in pixels per millimetre. */
function jacobian(boardToImage, x, y, step = 0.5) {
  const col = (dx, dy) => {
    const ahead = apply(boardToImage, [x + dx, y + dy]);
    const behind = apply(boardToImage, [x - dx, y - dy]);
    return [(ahead[0] - behind[0]) / (2 * step), (ahead[1] - behind[1]) / (2 * step)];
  };
  const [a, c] = col(step, 0);
  const [b, d] = col(0, step);
  return [a, b, c, d];
}

/** Singular values of a 2x2, largest first. Exact, via the eigenvalues of MtM. */
function singularValues([a, b, c, d]) {
  const trace = a * a + b * b + c * c + d * d;
  const det = (a * d - b * c) ** 2;
  const disc = Math.sqrt(Math.max(0, trace * trace - 4 * det));
  return [Math.sqrt((trace + disc) / 2), Math.sqrt(Math.max(0, (trace - disc) / 2))];
}

/** Angle above the board plane, from the anisotropy at the bull. */
function elevationDeg(boardToImage) {
  const [largest, smallest] = singularValues(jacobian(boardToImage, 0, 0));
  if (largest <= 0) throw new Error('degenerate homography');
  return Math.asin(Math.min(1, smallest / largest)) * DEG;
}

/**
 * Median and worst pixels-per-mm around the double ring, plus the board's
 * extent in pixels. Both statistics are taken in the most compressed direction
 * at each point: one squashed axis is enough to lose a score however good the
 * other looks. The median is what the guidance judges, since the gates are
 * distributional over darts and darts land all round the board.
 */
function ringScale(boardToImage, board, samples = 180) {
  const scales = [];
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;

  for (let i = 0; i < samples; i += 1) {
    const angle = (2 * Math.PI * i) / samples;
    const cos = Math.cos(angle), sin = Math.sin(angle);
    scales.push(singularValues(jacobian(boardToImage, board.rDouble * cos, board.rDouble * sin))[1]);

    const [px, py] = apply(boardToImage, [board.rBoard * cos, board.rBoard * sin]);
    minX = Math.min(minX, px); maxX = Math.max(maxX, px);
    minY = Math.min(minY, py); maxY = Math.max(maxY, py);
  }

  const sorted = [...scales].sort((p, q) => p - q);
  const mid = sorted.length >> 1;
  const median = sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
  return { median, worst: sorted[0], extent: Math.max(maxX - minX, maxY - minY) };
}

/**
 * Judge one frame's framing and say what to do about it.
 *
 * @param {([number,number]|null)[]} landmarks normalized image coordinates, fixed order
 * @param {{height:number, width:number}} imageSize pixels
 * @param {object} [limits]
 * @param {object} [board]
 */
export function assessFraming(landmarks, imageSize, limits = DEFAULT_LIMITS, board = BDO_BOARD) {
  const found = visibleLandmarks(landmarks).length;
  const refuse = (guidance) => ({
    verdict: UNUSABLE, guidance, landmarksFound: found, ready: false,
    elevationDeg: null, boardFill: null, ringPx: null, ringPxWorst: null, mmPerPxWorst: null,
  });

  if (found < 4) {
    return refuse(found === 0
      ? 'Point the phone at the board — not enough of it is visible yet.'
      : 'Part of the board is out of frame or hidden. Show the whole board.');
  }

  const pixels = landmarks.map((p) =>
    p ? [p[0] * imageSize.width, p[1] * imageSize.height] : null);

  if (!convexAndOrdered(pixels)) {
    return refuse('The board’s outline does not look right. Reposition and try again.');
  }

  let boardToImage, elevation, scale;
  try {
    boardToImage = matInv3(estimateHomography(pixels, board));
    elevation = elevationDeg(boardToImage);
    scale = ringScale(boardToImage, board);
  } catch {
    return refuse('The board’s outline does not look right. Reposition and try again.');
  }

  const shorter = Math.min(imageSize.height, imageSize.width);
  // What the model will actually see: the frame is resized to its input, and
  // that resize is where the resolution is lost.
  const resize = limits.modelInputPx / shorter;
  const ringPx = board.ringWidth * scale.median * resize;

  const { verdict, guidance } = judge(elevation, ringPx, limits);
  return {
    verdict, guidance, ready: verdict === READY, landmarksFound: found,
    elevationDeg: elevation,
    boardFill: scale.extent / shorter,
    ringPx,
    ringPxWorst: board.ringWidth * scale.worst * resize,
    mmPerPxWorst: 1 / scale.worst,
  };
}

/** One verdict and the single most useful next action. */
function judge(elevation, ringPx, limits) {
  const [low, high] = limits.idealElevationDeg;

  // Being in the flight path outranks everything: that one costs a phone
  // rather than a score.
  if (elevation > limits.maxElevationDeg) {
    return { verdict: UNUSABLE, guidance:
      'The phone is nearly in line with the board — that is where the darts fly. '
      + 'Move it further to one side, or higher up.' };
  }
  if (elevation < limits.minElevationDeg) {
    return { verdict: UNUSABLE, guidance:
      'The view is too side-on to score accurately. Move the phone around toward '
      + 'the front of the board.' };
  }

  if (ringPx < limits.minRingPx) {
    // Two causes look identical in this number. An oblique view compresses the
    // rings by roughly sin(elevation), so a player already filling the frame at
    // 35 degrees cannot fix it by moving closer -- and telling them to is the
    // most annoying advice available.
    if (elevation < low) {
      return { verdict: UNUSABLE, guidance:
        'Too side-on for this distance — the scoring rings are squashed. Move the '
        + 'phone around toward the front of the board.' };
    }
    return { verdict: UNUSABLE, guidance:
      'The board is too small in the frame. Move the phone closer until the board '
      + 'nearly fills it.' };
  }

  if (elevation < low) {
    return { verdict: MARGINAL, guidance:
      'Good enough to score. A little more toward the front of the board would be better.' };
  }
  if (ringPx < limits.idealRingPx) {
    return { verdict: MARGINAL, guidance:
      'This will work, but scores near the wires will be less certain. Move a little '
      + 'closer if you can.' };
  }
  if (elevation > high) {
    return { verdict: MARGINAL, guidance:
      'Good enough to score, but close to the throwing line. Further to the side would '
      + 'be safer for the phone.' };
  }
  return { verdict: READY, guidance: 'Ready to throw.' };
}
