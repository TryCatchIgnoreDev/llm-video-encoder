#!/usr/bin/env node
/**
 * Modular multi-worker RGBA overlay renderer + ffmpeg compositor for llm-video-encoder.
 *
 * Usage:
 *   node render.mjs stills --job-dir <dir> [--t 10,40,90,150]
 *   node render.mjs video  --job-dir <dir> [--fps 60 --workers 2 --out <mp4>]
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
      return ['-c:v', 'h264_nvenc', '-preset', 'p5', '-rc', 'vbr', '-cq', String(cq), '-b:v', '0', '-pix_fmt', 'yuv420p'];
    }
  }
  return ['-c:v', 'libx264', '-preset', 'medium', '-crf', String(cq), '-pix_fmt', 'yuv420p'];
}

const argv = process.argv.slice(2);
const mode = argv[0] ?? 'stills';
const opt = (k, d) => {
  const i = argv.indexOf(`--${k}`);
  return i >= 0 ? argv[i + 1] : d;
};

const JOB_DIR = path.resolve(opt('job-dir', '.'));
const JOB_CFG = JSON.parse(readFileSync(path.join(JOB_DIR, 'job.json'), 'utf8'));
const SOURCE_VIDEO = path.resolve(JOB_CFG.source_video || path.join(JOB_DIR, 'work', 'source.mp4'));

function resolveOverlayXY(ovW, ovH) {
  if (typeof JOB_CFG.x === 'number' && typeof JOB_CFG.y === 'number') {
    return [JOB_CFG.x, JOB_CFG.y];
  }
  const pos = String(JOB_CFG.position || 'bottom-left').toLowerCase();
  const mx = Number(JOB_CFG.margin_x ?? 40);
  const my = Number(JOB_CFG.margin_y ?? 32);
  const xMap = {
    left: mx,
    center: Math.round((1920 - ovW) / 2),
    right: 1920 - ovW - mx,
  };
  const yMap = {
    top: my,
    bottom: 1080 - ovH - my,
  };
  const [vPart, hPart] = pos.split('-');
  return [xMap[hPart] ?? mx, yMap[vPart] ?? yMap.bottom];
}

function getVideoDuration(vidPath) {
  const r = spawnSync('ffprobe', [
    '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', vidPath,
  ], { encoding: 'utf8' });
  if (r.status !== 0) throw new Error(`ffprobe failed: ${r.stderr}`);
  return parseFloat(r.stdout.trim());
}

function startStaticServer() {
  const mime = { '.html': 'text/html; charset=utf-8', '.json': 'application/json; charset=utf-8', '.ttf': 'font/ttf' };
  return http.createServer((req, res) => {
    const reqPath = new URL(req.url, 'http://127.0.0.1').pathname;
    let filePath = null;
    if (reqPath === '/' || reqPath === '/overlay.html') filePath = path.join(__dirname, 'overlay.html');
    else if (reqPath === '/job.json') filePath = path.join(JOB_DIR, 'job.json');
    else if (reqPath.startsWith('/data/')) filePath = path.join(JOB_DIR, 'data', path.basename(reqPath));
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

async function openOverlayPage(baseUrl) {
  const browser = await chromium.launch({
    headless: true,
    args: ['--disable-background-timer-throttling', '--disable-renderer-backgrounding', '--disable-backgrounding-occluded-windows'],
  });
  const page = await browser.newPage({ viewport: { width: 1000, height: 520 }, deviceScaleFactor: 1 });
  await page.goto(`${baseUrl}/overlay.html`);
  await page.waitForFunction(() => window.__studio?.ready || window.__studio?.error, null, { timeout: 60000 });
  const err = await page.evaluate(() => window.__studio?.error);
  if (err) throw new Error(`Overlay boot failed: ${err}`);
  const [width, height] = await page.evaluate(() => [window.__studio.width, window.__studio.height]);
  return { browser, page, width, height };
}

async function runStills(baseUrl) {
  const outDir = path.resolve(opt('out', path.join(JOB_DIR, 'out', 'stills')));
  mkdirSync(outDir, { recursive: true });
  const dur = getVideoDuration(SOURCE_VIDEO);
  const defTimes = [0.12, 0.28, 0.48, 0.68, 0.85, 0.94].map((f) => +(dur * f).toFixed(1));
  const times = opt('t') ? opt('t').split(',').map(Number) : defTimes;

  const { browser, page, width: ovW, height: ovH } = await openOverlayPage(baseUrl);
  const [ox, oy] = resolveOverlayXY(ovW, ovH);
  try {
    for (const t of times) {
      const b64 = await page.evaluate((tv) => window.__studio.pngAt(tv), t);
      const ovPng = path.join(outDir, `ov_${t.toFixed(1)}.png`);
      const compJpg = path.join(outDir, `comp_${t.toFixed(1)}.jpg`);
      writeFileSync(ovPng, Buffer.from(b64, 'base64'));
      const ff = spawnSync('ffmpeg', [
        '-y', '-loglevel', 'error',
        '-ss', String(t), '-i', SOURCE_VIDEO, '-i', ovPng, '-frames:v', '1',
        '-filter_complex', `[0:v]scale=1920:1080:flags=lanczos[bg];[bg][1:v]overlay=${ox}:${oy}:format=auto`,
        '-q:v', '2', compJpg,
      ], { encoding: 'utf8' });
      if (ff.status !== 0) throw new Error(`ffmpeg still failed at t=${t}: ${ff.stderr}`);
      console.log(`Wrote ${compJpg}`);
    }
  } finally {
    await browser.close();
  }
}

async function runVideo(baseUrl, server) {
  const fps = Number(opt('fps', String(JOB_CFG.fps || 60)));
  const workers = Math.max(1, Number(opt('workers', String(JOB_CFG.workers || 2))));
  const chunk = Math.max(10, Number(opt('chunk', '40')));
  const dur = Number(opt('to', String(getVideoDuration(SOURCE_VIDEO))));
  const from = Number(opt('from', '0'));
  const slug = JOB_CFG.slug || path.basename(JOB_DIR);
  const outPath = path.resolve(opt('out', path.join(JOB_DIR, 'out', `${slug}-encoder-1080p${fps}.mp4`)));
  mkdirSync(path.dirname(outPath), { recursive: true });

  const pages = await Promise.all(Array.from({ length: workers }, () => openOverlayPage(baseUrl)));
  const ovW = pages[0].width, ovH = pages[0].height;
  const [ox, oy] = resolveOverlayXY(ovW, ovH);
  const frameBytes = ovW * ovH * 4;
  const f0 = Math.round(from * fps), f1 = Math.round(dur * fps), total = f1 - f0;
  console.log(`Rendering ${slug}: ${total} frames @ ${fps}fps, overlay ${ovW}x${ovH} at (${ox},${oy}), theme='${JOB_CFG.theme}'`);

  const ffArgs = ['-y', '-loglevel', 'warning'];
  if (from > 0) ffArgs.push('-ss', String(from));
  if (opt('to')) ffArgs.push('-t', String(dur - from));
  ffArgs.push(
    '-i', SOURCE_VIDEO,
    '-f', 'rawvideo', '-pix_fmt', 'rgba', '-s', `${ovW}x${ovH}`, '-r', String(fps), '-i', 'pipe:0',
    '-filter_complex', `[0:v]fps=${fps},scale=1920:1080:flags=lanczos,format=yuv420p[bg];[bg][1:v]overlay=${ox}:${oy}:shortest=1:format=auto,format=yuv420p[v]`,
    '-map', '[v]', '-map', '0:a:0?',
    ...detectVideoEncoderArgs(JOB_CFG.cq || 18),
    '-c:a', 'copy', '-movflags', '+faststart', outPath,
  );

  const ff = spawn('ffmpeg', ffArgs, { stdio: ['pipe', 'inherit', 'inherit'] });
  const queues = Array.from({ length: workers }, () => []);
  const sockets = new Array(workers).fill(null);
  const acked = new Array(workers).fill(0);
  const waiting = new Map(), owed = new Map();
  const cap = 160;
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
        if (written % 1000 === 0 || written === total) {
          const el = (performance.now() - t0) / 1000, rate = written / Math.max(0.01, el);
          console.log(`${written}/${total} frames (${((written / total) * 100).toFixed(1)}%)  ${rate.toFixed(1)} fps  eta ${((total - written) / rate).toFixed(0)}s`);
        }
        if (written === total) resolveAll();
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

  try {
    const port = server.address().port;
    const rendering = Promise.all(pages.map(async ({ page }, k) => {
      const wsUrl = `ws://127.0.0.1:${port}/?w=${k}`;
      for (let r = take(); r && !failure; r = take()) {
        for (let n = r[0]; n < r[1]; n++) queues[k].push(n);
        await page.evaluate((o) => window.__studio.streamChunk(o), { fromFrame: r[0], toFrame: r[1], fps, wsUrl });
      }
      if (!failure) await page.evaluate((u) => window.__studio.flushWs(u), wsUrl);
    }));
    rendering.catch(() => {});
    await Promise.race([Promise.all([rendering, allWritten]), failed]);
    ending = true;
    ff.stdin.end();
    const code = await new Promise((res) => ff.on('exit', res));
    if (code !== 0) throw new Error(`ffmpeg exited with code ${code}`);
    console.log(`\nWrote ${outPath} (${written} frames at ${fps}fps in ${((performance.now() - t0) / 1000).toFixed(1)}s, ${workers} workers)`);
  } finally {
    ending = true;
    for (const p of pages) await p.browser.close().catch(() => {});
  }
}

const server = startStaticServer();
await new Promise((res) => server.listen(0, '127.0.0.1', res));
const baseUrl = `http://127.0.0.1:${server.address().port}`;
try {
  if (mode === 'video') await runVideo(baseUrl, server);
  else await runStills(baseUrl);
} finally {
  server.close();
}
