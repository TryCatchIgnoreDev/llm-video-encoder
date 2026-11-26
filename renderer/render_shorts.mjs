#!/usr/bin/env node
/**
 * True-vertical 1080x1920 60fps vertical Short renderer for llm-video-encoder.
 *
 * Layout (mobile-safe zones, zero phone rotation needed):
 *   y =   0..125 : Top Shorts UI safe zone
 *   y = 125..305 : Crisp Hook Headline + Accent Metric Pill
 *   y = 320..928 : 1080x608 Video window (optional custom crop)
 *   y = 958..1435: Giant live Arithmetic Coding overlay (~988x470px, scale ~1.78)
 *   y = 1465..1520: Subtle lossless tech footer (above Shorts bottom caption bar)
 *   y = 1520..1920: Clean Shorts bottom UI safe zone
 *
 * Usage:
 *   node render_shorts.mjs --job-dir <dir> --start 30 --duration 35 --headline "..." --subhead "..." [--crop "..."] [--out <mp4>] [--still-only]
 *   (Also accepts explicit --source, --transcript, --compression, --palette-json for external projects.)
 */

import { createHash } from 'node:crypto';
import { mkdirSync, readFileSync, writeFileSync, existsSync } from 'node:fs';
import http from 'node:http';
import { spawn, spawnSync } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright-core';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const FONTS_ROOT = path.join(__dirname, 'fonts');

function detectVideoEncoderArgs(cq = 18) {
  const probe = spawnSync('ffmpeg', ['-hide_banner', '-encoders'], { encoding: 'utf8' });
  const hasNvenc = probe.status === 0 && /\bh264_nvenc\b/.test(probe.stdout || '');
  if (hasNvenc && process.env.LLM_VIDEO_FORCE_CPU !== '1') {
    const test = spawnSync('ffmpeg', [
      '-loglevel', 'error', '-f', 'lavfi', '-i', 'color=c=black:s=64x64:d=0.1',
      '-c:v', 'h264_nvenc', '-f', 'null', '-',
    ]);
    if (test.status === 0) {
      return ['-c:v', 'h264_nvenc', '-preset', 'p5', '-rc', 'vbr', '-cq', String(cq), '-pix_fmt', 'yuv420p'];
    }
  }
  return ['-c:v', 'libx264', '-preset', 'medium', '-crf', String(cq), '-pix_fmt', 'yuv420p'];
}

const argv = process.argv.slice(2);
const hasFlag = (k) => argv.includes(`--${k}`);
const opt = (k, d = null) => {
  const i = argv.indexOf(`--${k}`);
  return i >= 0 ? argv[i + 1] : d;
};

const JOB_DIR = opt('job-dir') ? path.resolve(opt('job-dir')) : null;
const JOB_CFG = JOB_DIR && existsSync(path.join(JOB_DIR, 'job.json'))
  ? JSON.parse(readFileSync(path.join(JOB_DIR, 'job.json'), 'utf8'))
  : {};

const SOURCE_VIDEO = path.resolve(opt('source') || JOB_CFG.source_video || path.join(JOB_DIR, 'work', 'source.mp4'));
const defaultTranscript = JOB_DIR && existsSync(path.join(JOB_DIR, 'data', 'transcript.json'))
  ? path.join(JOB_DIR, 'data', 'transcript.json')
  : (JOB_DIR ? path.join(JOB_DIR, 'data', 'lyrics.json') : '');
const TRANSCRIPT_PATH = path.resolve(opt('transcript') || opt('lyrics') || defaultTranscript);
const COMP_PATH = path.resolve(opt('compression') || path.join(JOB_DIR, 'data', 'compression.json'));

const START_SEC = Number(opt('start', '0'));
const DUR_SEC = Number(opt('duration', '35'));
const FPS = Number(opt('fps', '60'));
const WORKERS = Number(opt('workers', '2'));
const SCALE = Number(opt('scale', '1.78'));
const CROP_EXPR = opt('crop', '');
const HEADLINE = opt('headline', JOB_CFG.title || 'Compressing Video Speech with an LLM');
const SUBHEAD = opt('subhead', 'LLM Arithmetic Coding vs Traditional ZIP');
const FOOTER = opt('footer', 'Exact Softmax CDF  ·  62-bit Arithmetic Coding  ·  100% Lossless (SHA-256)');
const THEME = opt('theme', JOB_CFG.theme || 'dark-slate');
const THEME_PALETTES = {
  'dark-slate': { bg: '#090C0F', signal: '#E8452C', accent: '#FF9642', fg: '#F1F5F8', ash: '#95A7B5' },
  'dark-forest': { bg: '#0C110E', signal: '#E94B43', accent: '#F5C15C', fg: '#F7F1E5', ash: '#B3B9A8' },
  'warm-cinema': { bg: '#110C0A', signal: '#E63950', accent: '#F7B84B', fg: '#FAF2EA', ash: '#C3AEA2' },
  'neon-pop': { bg: '#0A0916', signal: '#FF2E7E', accent: '#FFB830', fg: '#F5F4FF', ash: '#A9A5CC' },
};
const PALETTE = opt('palette-json')
  ? JSON.parse(opt('palette-json'))
  : { ...(THEME_PALETTES[THEME] || {}), ...(JOB_CFG.palette || {}) };

const OUT_MP4 = path.resolve(
  opt('out') || path.join(JOB_DIR || path.dirname(SOURCE_VIDEO), 'out', `${JOB_CFG.slug || 'video'}-vertical-short-1080x1920.mp4`)
);
mkdirSync(path.dirname(OUT_MP4), { recursive: true });

// Virtual job.json served to overlay.html with giant scale for vertical Shorts
const VIRTUAL_JOB = {
  ...JOB_CFG,
  theme: THEME,
  palette: PALETTE,
  scale: SCALE,
};

function startStaticServer() {
  const mime = { '.html': 'text/html; charset=utf-8', '.json': 'application/json; charset=utf-8', '.ttf': 'font/ttf' };
  return http.createServer((req, res) => {
    const reqPath = new URL(req.url, 'http://127.0.0.1').pathname;
    if (reqPath === '/job.json') {
      res.writeHead(200, { 'Content-Type': 'application/json; charset=utf-8' });
      res.end(JSON.stringify(VIRTUAL_JOB));
      return;
    }
    let filePath = null;
    if (reqPath === '/' || reqPath === '/overlay.html') filePath = path.join(__dirname, 'overlay.html');
    else if (reqPath === '/data/transcript.json' || reqPath === '/data/lyrics.json') filePath = TRANSCRIPT_PATH;
    else if (reqPath === '/data/compression.json') filePath = COMP_PATH;
    else if (reqPath.startsWith('/fonts/')) filePath = path.join(FONTS_ROOT, reqPath.slice('/fonts/'.length).replace(/^\/+/, ''));
    if (!filePath || !existsSync(filePath)) {
      res.writeHead(404); res.end('Not found'); return;
    }
    const ext = path.extname(filePath).toLowerCase();
    res.writeHead(200, { 'Content-Type': mime[ext] ?? 'application/octet-stream' });
    res.end(readFileSync(filePath));
  });
}

function attachWs(server, onBinaryFrame, onError) {
  server.on('upgrade', (req, socket) => {
    const k = Number(new URL(req.url, 'http://127.0.0.1').searchParams.get('w') ?? 0);
    const accept = createHash('sha1').update(req.headers['sec-websocket-key'] + '258EAFA5-E914-47DA-95CA-C5AB0DC85B11').digest('base64');
    socket.write(`HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: ${accept}\r\n\r\n`);

    const sendText = (str) => {
      if (socket.destroyed) return;
      const p = Buffer.from(str, 'utf8');
      const hdr = p.length < 126 ? Buffer.from([0x81, p.length]) : Buffer.from([0x81, 126, (p.length >> 8) & 0xff, p.length & 0xff]);
      socket.write(Buffer.concat([hdr, p]));
    };
    const wsProxy = { send: sendText, close: () => { try { socket.destroy(); } catch {} } };

    let buf = Buffer.alloc(0), fragChunks = [], fragOp = 0;
    socket.on('data', (chunk) => {
      buf = Buffer.concat([buf, chunk]);
      while (buf.length >= 2) {
        const b0 = buf[0], b1 = buf[1], fin = (b0 & 0x80) !== 0, opcode = b0 & 0x0f, masked = (b1 & 0x80) !== 0;
        let len = b1 & 0x7f, off = 2;
        if (len === 126) { if (buf.length < off + 2) return; len = buf.readUInt16BE(off); off += 2; }
        else if (len === 127) { if (buf.length < off + 8) return; len = buf.readUInt32BE(off) * 4294967296 + buf.readUInt32BE(off + 4); off += 8; }
        if (masked && buf.length < off + 4) return;
        const maskOff = off; if (masked) off += 4;
        if (buf.length < off + len) return;
        const payload = Buffer.allocUnsafe(len);
        buf.copy(payload, 0, off, off + len);
        if (masked) {
          const m0 = buf[maskOff], m1 = buf[maskOff + 1], m2 = buf[maskOff + 2], m3 = buf[maskOff + 3];
          for (let i = 0; i < len; i++) { const m = i & 3; payload[i] ^= m === 0 ? m0 : m === 1 ? m1 : m === 2 ? m2 : m3; }
        }
        buf = buf.subarray(off + len);
        if (opcode === 0x8) { socket.end(); return; }
        if (opcode === 0x9) { socket.write(Buffer.concat([Buffer.from([0x8a, payload.length]), payload])); continue; }
        if (opcode !== 0) { fragOp = opcode; fragChunks = [payload]; } else fragChunks.push(payload);
        if (fin) {
          const full = fragChunks.length === 1 ? fragChunks[0] : Buffer.concat(fragChunks);
          fragChunks = [];
          if (fragOp === 0x2) onBinaryFrame(k, full, wsProxy);
        }
      }
    });
    socket.on('error', (err) => { if (err?.code !== 'ECONNRESET') onError(err); });
  });
}

async function renderBackdropPng(page, backdropPath) {
  const b64 = await page.evaluate(({ headline, subhead, footer, pal }) => {
    const W = 1080, H = 1920;
    const c = document.createElement('canvas');
    c.width = W; c.height = H;
    const ctx = c.getContext('2d');

    const basePal = (window.__studio?.palette) || {};
    const eff = { ...basePal, ...(pal || {}) };
    const bg = eff.bg || '#070B11';
    const accent = eff.accent || '#38D9FF';
    const signal = eff.signal || '#FF4D2E';
    const fg = eff.fg || '#F4F8FF';
    const ash = eff.ash || '#9BB0C7';

    // Base deep gradient
    const gBg = ctx.createLinearGradient(0, 0, 0, H);
    gBg.addColorStop(0, '#05070B');
    gBg.addColorStop(0.22, bg);
    gBg.addColorStop(0.82, bg);
    gBg.addColorStop(1, '#040508');
    ctx.fillStyle = gBg;
    ctx.fillRect(0, 0, W, H);

    // Subtle radial glow behind top hook banner
    const rad = ctx.createRadialGradient(W / 2, 220, 40, W / 2, 220, 520);
    rad.addColorStop(0, accent + '24');
    rad.addColorStop(0.55, signal + '10');
    rad.addColorStop(1, 'rgba(0,0,0,0)');
    ctx.fillStyle = rad;
    ctx.fillRect(0, 0, W, 620);

    // Subtle tech grid texture
    ctx.strokeStyle = 'rgba(255,255,255,0.028)';
    ctx.lineWidth = 1;
    for (let x = 0; x < W; x += 60) {
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
    }
    for (let y = 0; y < H; y += 60) {
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
    }

    // Small top pill badge (y = 112)
    const badge = 'LIVE LLM ARITHMETIC CODING';
    ctx.font = '500 22px "Plex-500", Consolas, monospace';
    const bw = ctx.measureText(badge).width + 44;
    const bx = (W - bw) / 2, by = 104, bh = 42;
    ctx.fillStyle = accent + '1C';
    ctx.strokeStyle = accent + '78';
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    ctx.roundRect(bx, by, bw, bh, 21);
    ctx.fill(); ctx.stroke();
    ctx.fillStyle = accent;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(badge, W / 2, by + bh / 2 + 1);

    // Bold hook headline (auto-fit font size so 1 or 2 crisp lines)
    ctx.fillStyle = fg;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'alphabetic';
    let fontSize = 50;
    ctx.font = `700 ${fontSize}px "Archivo-1000-700", system-ui, sans-serif`;
    while (fontSize > 34 && ctx.measureText(headline).width > 980) {
      fontSize -= 2;
      ctx.font = `700 ${fontSize}px "Archivo-1000-700", system-ui, sans-serif`;
    }
    ctx.fillText(headline, W / 2, 216);

    // Accent metric subheadline
    ctx.font = '500 30px "Plex-500", Consolas, monospace';
    ctx.fillStyle = accent;
    const niceSub = subhead.replace(/->/g, '→').replace(/(\d)x\b/g, '$1×');
    ctx.fillText(niceSub, W / 2, 272);

    // Thin accent divider lines framing the 1080x608 video window (y = 314 .. 928)
    const lineGrad = ctx.createLinearGradient(0, 0, W, 0);
    lineGrad.addColorStop(0, 'rgba(255,255,255,0.04)');
    lineGrad.addColorStop(0.25, accent + '99');
    lineGrad.addColorStop(0.75, signal + '99');
    lineGrad.addColorStop(1, 'rgba(255,255,255,0.04)');
    ctx.fillStyle = lineGrad;
    ctx.fillRect(0, 314, W, 3);
    ctx.fillRect(0, 927, W, 3);

    // Subtle lossless footer safely ABOVE YouTube Shorts bottom bar (y = 1498)
    ctx.font = '400 22px "Plex-400", Consolas, monospace';
    ctx.fillStyle = ash + 'C8';
    ctx.textAlign = 'center';
    ctx.fillText(footer, W / 2, 1502);

    return c.toDataURL('image/png').slice('data:image/png;base64,'.length);
  }, {
    headline: HEADLINE,
    subhead: SUBHEAD,
    footer: FOOTER,
    pal: PALETTE,
  });
  writeFileSync(backdropPath, Buffer.from(b64, 'base64'));
}

function buildVideoFilter(forStill = false) {
  const base = CROP_EXPR
    ? CROP_EXPR
    : 'scale=1080:608:force_original_aspect_ratio=increase:flags=lanczos,crop=1080:608';
  return forStill ? `setpts=PTS-STARTPTS,${base}` : `setpts=PTS-STARTPTS,${base},fps=${FPS}`;
}

async function main() {
  const server = startStaticServer();
  await new Promise((r) => server.listen(0, '127.0.0.1', r));
  const port = server.address().port;
  const baseUrl = `http://127.0.0.1:${port}`;

  const backdropPath = OUT_MP4.replace(/\.mp4$/i, '_backdrop.png');
  const stillPath = OUT_MP4.replace(/\.mp4$/i, '_preview.jpg');

  const browsers = [];
  const pages = [];
  let ovW = 0, ovH = 0;

  try {
    const nPages = hasFlag('still-only') ? 1 : WORKERS;
    for (let k = 0; k < nPages; k++) {
      const b = await chromium.launch({
        headless: true,
        args: ['--disable-background-timer-throttling', '--disable-renderer-backgrounding', '--disable-backgrounding-occluded-windows'],
      });
      const p = await b.newPage({ viewport: { width: 1200, height: 640 }, deviceScaleFactor: 1 });
      await p.goto(`${baseUrl}/overlay.html`);
      await p.waitForFunction(() => window.__studio?.ready || window.__studio?.error, null, { timeout: 60000 });
      const err = await p.evaluate(() => window.__studio?.error);
      if (err) throw new Error(`Overlay boot failed: ${err}`);
      if (k === 0) {
        [ovW, ovH] = await p.evaluate(() => [window.__studio.width, window.__studio.height]);
        await renderBackdropPng(p, backdropPath);
      }
      browsers.push(b);
      pages.push(p);
    }

    const ox = Math.round((1080 - ovW) / 2);
    const oy = 960;
    const vidFilterStill = buildVideoFilter(true);
    const vidFilter = buildVideoFilter(false);

    // Always write 1 QA still at mid-clip so we can visually inspect!
    const tStill = +(START_SEC + DUR_SEC * 0.40).toFixed(2);
    const ovStillPng = OUT_MP4.replace(/\.mp4$/i, '_ov_still.png');
    const b64Still = await pages[0].evaluate((tv) => window.__studio.pngAt(tv), tStill);
    writeFileSync(ovStillPng, Buffer.from(b64Still, 'base64'));
    const ffStill = spawnSync('ffmpeg', [
      '-y', '-loglevel', 'error',
      '-i', backdropPath,
      '-ss', String(tStill), '-i', SOURCE_VIDEO,
      '-i', ovStillPng,
      '-frames:v', '1',
      '-filter_complex', `[1:v]${vidFilterStill}[vid];[0:v][vid]overlay=0:317[tmp];[tmp][2:v]overlay=${ox}:${oy}:format=auto`,
      '-q:v', '2', stillPath,
    ], { encoding: 'utf8' });
    if (ffStill.status !== 0) throw new Error(`still ffmpeg failed: ${ffStill.stderr}`);
    console.log(`Wrote preview still: ${stillPath} (overlay ${ovW}x${ovH} at (${ox},${oy}))`);

    if (hasFlag('still-only')) return;

    const totalFrames = Math.ceil(DUR_SEC * FPS);
    const f0 = Math.round(START_SEC * FPS);
    const f1 = f0 + totalFrames;
    console.log(`Rendering vertical Short: ${totalFrames} frames (${DUR_SEC}s @ ${FPS}fps, frames ${f0}..${f1}), overlay ${ovW}x${ovH} at (${ox},${oy})`);

    const fadeOutStart = Math.max(0.5, DUR_SEC - 0.65).toFixed(2);
    const ff = spawn('ffmpeg', [
      '-y', '-loglevel', 'warning',
      '-loop', '1', '-framerate', String(FPS), '-i', backdropPath,
      '-ss', String(START_SEC), '-t', String(DUR_SEC), '-i', SOURCE_VIDEO,
      '-f', 'rawvideo', '-pix_fmt', 'rgba', '-s', `${ovW}x${ovH}`, '-r', String(FPS), '-i', 'pipe:0',
      '-filter_complex', `[1:v]${vidFilter}[vid];[0:v][vid]overlay=0:317:shortest=1[tmp];[tmp][2:v]overlay=${ox}:${oy}:format=auto:shortest=1[outv]`,
      '-map', '[outv]', '-map', '1:a:0?',
      '-af', `afade=t=in:st=0:d=0.25,afade=t=out:st=${fadeOutStart}:d=0.6`,
      ...detectVideoEncoderArgs(18),
      '-c:a', 'aac', '-b:a', '192k', '-movflags', '+faststart',
      '-t', String(DUR_SEC),
      OUT_MP4,
    ], { stdio: ['pipe', 'inherit', 'inherit'] });

    const frameBytes = ovW * ovH * 4;
    const chunk = 60;
    const queues = Array.from({ length: nPages }, () => []);
    const sockets = new Array(nPages).fill(null);
    const acked = new Array(nPages).fill(0);
    const waiting = new Map(), owed = new Map();
    const cap = 140;
    let next = f0, written = 0, cursor = f0, ending = false, failure = null, draining = false;
    const t0 = performance.now();

    let resolveAll, rejectFail;
    const allWritten = new Promise((r) => { resolveAll = r; });
    const failed = new Promise((_, rej) => { rejectFail = rej; });
    failed.catch(() => {});

    const fail = (e) => {
      if (failure) return;
      failure = e instanceof Error ? e : new Error(String(e));
      for (const s of sockets) s?.close();
      rejectFail(failure);
    };
    const ack = (k) => { acked[k]++; sockets[k]?.send(String(acked[k])); };
    ff.on('exit', (code) => { if (!ending && code !== 0) fail(new Error(`ffmpeg exited early (${code})`)); });
    ff.stdin.on('error', (err) => { if (!ending) fail(err); });

    const writeToFfmpeg = (buf) => new Promise((res) => { if (ff.stdin.write(buf)) res(); else ff.stdin.once('drain', res); });
    const drain = async () => {
      if (draining || failure) return;
      draining = true;
      try {
        while (!failure && waiting.has(next)) {
          const buf = waiting.get(next); waiting.delete(next);
          await writeToFfmpeg(buf);
          const okK = owed.get(next);
          if (okK !== undefined) { owed.delete(next); ack(okK); }
          next++; written++;
          for (const [m, kk] of owed) { if (waiting.size >= cap) break; owed.delete(m); ack(kk); }
          if (written % 500 === 0 || written === totalFrames) {
            const el = (performance.now() - t0) / 1000, rate = written / Math.max(0.01, el);
            console.log(`${written}/${totalFrames} frames (${((written / totalFrames) * 100).toFixed(1)}%)  ${rate.toFixed(1)} fps  eta ${((totalFrames - written) / rate).toFixed(0)}s`);
          }
          if (written === totalFrames) resolveAll();
        }
      } catch (e) { fail(e); } finally { draining = false; }
    };

    attachWs(server, (k, buf, wsProxy) => {
      if (failure) return;
      sockets[k] = wsProxy;
      const n = queues[k].shift();
      if (n === undefined) return fail(new Error(`worker ${k} unexpected frame`));
      if (buf.length !== frameBytes) return fail(new Error(`worker ${k} frame ${n} bad size`));
      waiting.set(n, buf);
      if (waiting.size < cap || n === next) ack(k); else owed.set(n, k);
      void drain();
    }, (err) => fail(err));

    const take = () => { if (cursor >= f1) return null; const a = cursor; cursor = Math.min(f1, a + chunk); return [a, cursor]; };

    const rendering = Promise.all(pages.map(async (page, k) => {
      const wsUrl = `ws://127.0.0.1:${port}/?w=${k}`;
      for (let r = take(); r && !failure; r = take()) {
        for (let n = r[0]; n < r[1]; n++) queues[k].push(n);
        await page.evaluate((o) => window.__studio.streamChunk(o), { fromFrame: r[0], toFrame: r[1], fps: FPS, wsUrl });
      }
      if (!failure) await page.evaluate((u) => window.__studio.flushWs(u), wsUrl);
    }));
    rendering.catch(() => {});
    await Promise.race([Promise.all([rendering, allWritten]), failed]);
    ending = true;
    ff.stdin.end();
    const code = await new Promise((res) => ff.on('exit', res));
    if (code !== 0) throw new Error(`ffmpeg exited with code ${code}`);
    for (const s of sockets) s?.close();

    const elapsed = (performance.now() - t0) / 1000;
    console.log(`Wrote ${OUT_MP4} (${totalFrames} frames @ ${FPS}fps in ${elapsed.toFixed(1)}s)`);
  } finally {
    await Promise.all(browsers.map((b) => b.close().catch(() => {})));
    server.close();
  }
}

await main();
