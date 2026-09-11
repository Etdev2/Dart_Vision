'use client';

/**
 * "Where should the phone go?"
 *
 * Takes a photo from the intended mounting position, has the player tap the
 * eight landmarks, and reports the mounting angle. The angle is measured, not
 * asked for: it falls out of the homography's local anisotropy at the bull, so
 * it needs no focal length, sensor size or distance -- none of which a browser
 * reports reliably.
 *
 * The camera path deliberately leads with a file input rather than
 * `getUserMedia`. A phone opening this over http://192.168.x.x has no secure
 * context and would be refused the camera, while "Take a photo" opens the same
 * camera and works anywhere. Live preview is the enhancement, not the route.
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import { BDO_BOARD } from '@/lib/board.js';
import { CALIBRATION_ANGLES_DEG } from '@/lib/homography.js';
import { assessFraming } from '@/lib/framing.js';

const DEG = Math.PI / 180;
const B = BDO_BOARD;

// The landmark order is fixed: four sector-boundary wires 90 degrees apart on
// the outer double wire, then the same four on the outer treble wire.
const WIRES = ['6 | 13', '20 | 5', '11 | 8', '3 | 17'];
const LANDMARKS = [
  ...WIRES.map((wire) => ({ wire, ring: 'double' as const })),
  ...WIRES.map((wire) => ({ wire, ring: 'treble' as const })),
];

type Mark = [number, number];
type Report = ReturnType<typeof assessFraming>;

const VERDICT_HEADLINE: Record<string, string> = {
  ready: 'Good spot for the phone',
  marginal: 'Workable',
  unusable: 'Not usable here',
};

export default function MountCheck() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const frameRef = useRef<ImageBitmap | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const [marks, setMarks] = useState<Mark[]>([]);
  const [hasFrame, setHasFrame] = useState(false);
  const [live, setLive] = useState(false);
  const [hint, setHint] = useState('Take a photo of your board to begin.');
  const [report, setReport] = useState<Report | null>(null);

  const draw = useCallback((current: Mark[]) => {
    const canvas = canvasRef.current;
    const frame = frameRef.current;
    if (!canvas || !frame) return;
    canvas.width = frame.width;
    canvas.height = frame.height;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.drawImage(frame, 0, 0);

    const r = Math.max(canvas.width, canvas.height) / 110;
    current.forEach(([nx, ny], index) => {
      const x = nx * canvas.width;
      const y = ny * canvas.height;
      ctx.strokeStyle = '#dc2626';
      ctx.lineWidth = r / 3;
      ctx.beginPath(); ctx.arc(x, y, r, 0, 7); ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(x - r * 2, y); ctx.lineTo(x + r * 2, y);
      ctx.moveTo(x, y - r * 2); ctx.lineTo(x, y + r * 2);
      ctx.stroke();
      ctx.font = `700 ${r * 2.4}px system-ui`;
      ctx.lineWidth = r / 4;
      ctx.strokeStyle = '#000'; ctx.fillStyle = '#fff';
      ctx.strokeText(String(index + 1), x + r * 1.6, y - r * 1.4);
      ctx.fillText(String(index + 1), x + r * 1.6, y - r * 1.4);
    });
  }, []);

  useEffect(() => {
    draw(marks);
    const canvas = canvasRef.current;
    if (!canvas || marks.length < 4) { setReport(null); return; }
    const landmarks = Array.from({ length: 8 }, (_, i) => marks[i] ?? null);
    setReport(assessFraming(landmarks, { height: canvas.height, width: canvas.width }));
  }, [marks, draw]);

  const stopCamera = () => {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    setLive(false);
  };
  useEffect(() => stopCamera, []);

  const useFrame = (bitmap: ImageBitmap) => {
    frameRef.current = bitmap;
    stopCamera();
    setHasFrame(true);
    setMarks([]);
  };

  const onPhoto = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) useFrame(await createImageBitmap(file));
  };

  const startCamera = async () => {
    if (!navigator.mediaDevices?.getUserMedia) {
      setHint('Live camera needs a secure page (https or localhost). “Take a photo” '
        + 'works anywhere and is the same measurement.');
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: 'environment' } }, audio: false,
      });
      streamRef.current = stream;
      if (videoRef.current) videoRef.current.srcObject = stream;
      setLive(true);
      setHasFrame(false);
    } catch (error) {
      const name = error instanceof Error ? error.name : 'unknown';
      setHint(`Camera unavailable (${name}). “Take a photo” works without a secure page.`);
    }
  };

  const freeze = async () => {
    const video = videoRef.current;
    if (video?.videoWidth) useFrame(await createImageBitmap(video));
  };

  const onCanvasClick = (event: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas || !frameRef.current || marks.length >= 8) return;
    const box = canvas.getBoundingClientRect();
    // The canvas is `object-fit: contain`, so the image is letterboxed inside
    // the element and a click has to be mapped through that box, not the box.
    const scale = Math.min(box.width / canvas.width, box.height / canvas.height);
    const drawnW = canvas.width * scale;
    const drawnH = canvas.height * scale;
    const x = (event.clientX - box.left - (box.width - drawnW) / 2) / drawnW;
    const y = (event.clientY - box.top - (box.height - drawnH) / 2) / drawnH;
    if (x < 0 || x > 1 || y < 0 || y > 1) return;
    setMarks((current) => [...current, [x, y]]);
  };

  const next = marks.length < 8 ? LANDMARKS[marks.length] : null;

  return (
    <>
      <div className="card">
        <h2>Where should the phone go?</h2>
        <p className="note" style={{ margin: '0 0 12px' }}>
          Take a photo from where you plan to mount the phone, then tap the eight
          landmarks. This measures your mounting angle from the photo alone — no
          focal length, no distance, no protractor.
        </p>
        <div className="row">
          <label className="file">
            Take a photo
            <input type="file" accept="image/*" capture="environment" onChange={onPhoto} />
          </label>
          <button onClick={startCamera}>Use live camera</button>
          <button onClick={() => setMarks([])} disabled={!marks.length}>
            Start marks again
          </button>
        </div>
      </div>

      <div className="stage">
        {!hasFrame && !live && <div className="hint">{hint}</div>}
        <video ref={videoRef} playsInline autoPlay muted hidden={!live} />
        <canvas ref={canvasRef} hidden={!hasFrame} onClick={onCanvasClick} />
      </div>
      {live && (
        <div className="row" style={{ marginTop: 10 }}>
          <button className="primary" onClick={freeze}>Freeze frame</button>
        </div>
      )}

      {next && hasFrame && (
        <div className="card" style={{ marginTop: 12 }}>
          <div className="target">
            <TargetMap index={marks.length} />
            <div>
              <div className="headline">{marks.length + 1} of 8 — the {next.wire} wire</div>
              <div className="detail">
                {next.ring === 'double'
                  ? 'where it meets the OUTER edge of the double ring (the board’s rim)'
                  : 'where it meets the OUTER edge of the treble ring'}
              </div>
            </div>
          </div>
        </div>
      )}

      {report && <Verdict report={report} />}
    </>
  );
}

/** A thumbnail of the board with the landmark being asked for marked. */
function TargetMap({ index }: { index: number }) {
  const radius = index < 4 ? B.rDouble : B.rTreble;
  const angle = CALIBRATION_ANGLES_DEG[index % 4] * DEG;
  return (
    <svg width={86} height={86} viewBox="-196 -196 392 392">
      <circle cx={0} cy={0} r={B.rDouble} fill="none" stroke="currentColor"
              strokeOpacity={0.35} strokeWidth={4} />
      <circle cx={0} cy={0} r={B.rTreble} fill="none" stroke="currentColor"
              strokeOpacity={0.35} strokeWidth={4} />
      {CALIBRATION_ANGLES_DEG.map((a: number) => (
        <line key={a} x1={0} y1={0}
              x2={B.rDouble * Math.cos(a * DEG)} y2={-B.rDouble * Math.sin(a * DEG)}
              stroke="currentColor" strokeOpacity={0.2} strokeWidth={3} />
      ))}
      <circle cx={radius * Math.cos(angle)} cy={-radius * Math.sin(angle)} r={20} fill="#dc2626" />
    </svg>
  );
}

function Verdict({ report }: { report: Report }) {
  const value = (n: number | null, digits: number, suffix = '') =>
    (n === null ? '—' : `${n.toFixed(digits)}${suffix}`);
  return (
    <div className="card">
      <div className={`verdict ${report.verdict}`}>
        <div className="dot" />
        <div>
          <div className="headline">{VERDICT_HEADLINE[report.verdict]}</div>
          <div className="detail">{report.guidance}</div>
        </div>
      </div>
      <div className="stats">
        <div className="stat"><b>{value(report.elevationDeg, 0, '°')}</b><span>Elevation</span></div>
        <div className="stat"><b>{value(report.ringPx, 1)}</b><span>Ring px @768</span></div>
        <div className="stat">
          <b>{report.boardFill === null ? '—' : `${(report.boardFill * 100).toFixed(0)}%`}</b>
          <span>Board fill</span>
        </div>
        <div className="stat"><b>{value(report.mmPerPxWorst, 2)}</b><span>mm / px worst</span></div>
      </div>
      <p className="note" style={{ margin: '12px 0 0' }}>
        45–65° is the band. Below ~30° the rings are too squashed to score
        accurately; above ~70° the phone is in the flight path.
      </p>
    </div>
  );
}
