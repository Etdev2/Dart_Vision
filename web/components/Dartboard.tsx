'use client';

/**
 * The board, drawn from the same BDO measurements the scorer uses.
 *
 * Not decorative: a tap is converted straight back to board millimetres and
 * scored by `scoreAt`, so what you see and what gets scored cannot drift.
 */

import { useCallback, useEffect, useRef } from 'react';
import { BDO_BOARD, SECTORS_CLOCKWISE_FROM_20 } from '@/lib/board.js';

const DEG = Math.PI / 180;
const B = BDO_BOARD;
const SIZE = 560;
const SCALE = (SIZE / 2 - 8) / B.rBoard;

export type Dart = { x: number; y: number };

export default function Dartboard({
  darts, onThrow,
}: { darts: Dart[]; onThrow: (x: number, y: number) => void }) {
  const ref = useRef<HTMLCanvasElement>(null);

  const draw = useCallback(() => {
    const ctx = ref.current?.getContext('2d');
    if (!ctx) return;
    ctx.clearRect(0, 0, SIZE, SIZE);

    const disc = (radius: number, fill: string) => {
      ctx.fillStyle = fill;
      ctx.beginPath();
      ctx.arc(SIZE / 2, SIZE / 2, radius * SCALE, 0, Math.PI * 2);
      ctx.fill();
    };
    const wedge = (inner: number, outer: number, from: number, to: number, fill: string) => {
      ctx.beginPath();
      ctx.arc(SIZE / 2, SIZE / 2, outer * SCALE, -to * DEG, -from * DEG);
      ctx.arc(SIZE / 2, SIZE / 2, inner * SCALE, -from * DEG, -to * DEG, true);
      ctx.closePath();
      ctx.fillStyle = fill;
      ctx.fill();
    };

    disc(B.rBoard, '#18181b');
    SECTORS_CLOCKWISE_FROM_20.forEach((_: number, index: number) => {
      const end = 99 - index * 18;
      const start = end - 18;
      const dark = index % 2 === 1;
      const bed = dark ? '#1c1917' : '#e7e0cf';
      const ring = dark ? '#15803d' : '#b91c1c';
      wedge(B.rOuterBull, B.rTreble - B.ringWidth, start, end, bed);
      wedge(B.rTreble - B.ringWidth, B.rTreble, start, end, ring);
      wedge(B.rTreble, B.rDouble - B.ringWidth, start, end, bed);
      wedge(B.rDouble - B.ringWidth, B.rDouble, start, end, ring);
    });
    disc(B.rOuterBull, '#15803d');
    disc(B.rInnerBull, '#b91c1c');

    ctx.fillStyle = '#e7e0cf';
    ctx.font = `600 ${Math.round(22 * SCALE)}px ui-sans-serif, system-ui, sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    const labelRadius = (B.rBoard + B.rDouble) / 2;
    SECTORS_CLOCKWISE_FROM_20.forEach((number: number, index: number) => {
      const angle = (90 - index * 18) * DEG;
      ctx.fillText(String(number),
        SIZE / 2 + labelRadius * Math.cos(angle) * SCALE,
        SIZE / 2 - labelRadius * Math.sin(angle) * SCALE);
    });

    darts.forEach(({ x, y }) => {
      ctx.fillStyle = '#fff';
      ctx.strokeStyle = '#000';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(SIZE / 2 + x * SCALE, SIZE / 2 - y * SCALE, 7, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    });
  }, [darts]);

  useEffect(draw, [draw]);

  const onClick = (event: React.MouseEvent<HTMLCanvasElement>) => {
    const box = event.currentTarget.getBoundingClientRect();
    const x = (((event.clientX - box.left) / box.width) * SIZE - SIZE / 2) / SCALE;
    const y = -(((event.clientY - box.top) / box.height) * SIZE - SIZE / 2) / SCALE;
    if (Math.hypot(x, y) > B.rBoard) return;
    onThrow(x, y);
  };

  return (
    <canvas
      ref={ref}
      width={SIZE}
      height={SIZE}
      onClick={onClick}
      style={{ width: '100%', height: 'auto', touchAction: 'manipulation' }}
    />
  );
}
