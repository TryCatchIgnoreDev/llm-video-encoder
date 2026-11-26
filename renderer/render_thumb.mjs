#!/usr/bin/env node
/**
 * High-CTR 1280x720 YouTube Thumbnail generator for llm-video-encoder.
 *
 * Usage:
 *   node render_thumb.mjs --bg <jpg> --out <png> --big "635 BYTES" --zip "1.63 KB ZIP" --badge "2.6x SMALLER" --tag "LLM ARITHMETIC CODING" --accent "#38D9FF" --signal "#FF4D2E"
 */

import { mkdirSync, readFileSync, writeFileSync, existsSync } from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright-core';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const FONTS_ROOT = path.join(__dirname, 'fonts');

const argv = process.argv.slice(2);
const opt = (k, d = '') => {
  const i = argv.indexOf(`--${k}`);
  return i >= 0 ? argv[i + 1] : d;
};

const BG_PATH = path.resolve(opt('bg'));
const OUT_PATH = path.resolve(opt('out'));
const BIG_TEXT = opt('big', '635 BYTES');
const ZIP_TEXT = opt('zip', '1.63 KB ZIP');
const BADGE_TEXT = opt('badge', '2.6x SMALLER').replace(/(\d)x\b/gi, '$1×');
const TAG_TEXT = opt('tag', 'LLM ARITHMETIC CODING');
const FOOTER_TEXT = opt('footer', '100% LOSSLESS · ARITHMETIC CODING');
const ACCENT = opt('accent', '#38D9FF');
const SIGNAL = opt('signal', '#FF4D2E');
const BG_SHIFT_X = Number(opt('bg-shift-x', '-180'));
const BG_SCALE = Number(opt('bg-scale', '1.08'));

mkdirSync(path.dirname(OUT_PATH), { recursive: true });

const server = http.createServer((req, res) => {
  const p = new URL(req.url, 'http://127.0.0.1').pathname;
  if (p === '/') {
    res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
    res.end('<!doctype html><html><body style="margin:0;background:#000"><canvas id="c" width="1280" height="720"></canvas></body></html>');
    return;
  }
  if (p === '/bg.jpg') {
    res.writeHead(200, { 'Content-Type': 'image/jpeg' });
    res.end(readFileSync(BG_PATH));
    return;
  }
  if (p.startsWith('/fonts/')) {
    const fp = path.join(FONTS_ROOT, p.slice('/fonts/'.length).replace(/^\/+/, ''));
    if (existsSync(fp)) {
      res.writeHead(200, { 'Content-Type': 'font/ttf' });
      res.end(readFileSync(fp));
      return;
    }
  }
  res.writeHead(404); res.end('Not found');
});

await new Promise((r) => server.listen(0, '127.0.0.1', r));
const baseUrl = `http://127.0.0.1:${server.address().port}`;
const browser = await chromium.launch({ headless: true });

try {
  const page = await browser.newPage({ viewport: { width: 1280, height: 720 }, deviceScaleFactor: 1 });
  await page.goto(`${baseUrl}/`);
  const b64 = await page.evaluate(async (cfg) => {
    const defs = [
      ['Archivo-1000-700', '/fonts/Archivo-w1000-700.ttf'],
      ['Plex-400', '/fonts/IBMPlexMono-Regular.ttf'],
      ['Plex-500', '/fonts/IBMPlexMono-Medium.ttf'],
    ];
    await Promise.all(defs.map(async ([fam, url]) => {
      const ff = new FontFace(fam, await (await fetch(url)).arrayBuffer());
      await ff.load();
      document.fonts.add(ff);
    }));
    await document.fonts.ready;

    const img = new Image();
    img.src = '/bg.jpg';
    await new Promise((res, rej) => { img.onload = res; img.onerror = rej; });

    const c = document.getElementById('c');
    const ctx = c.getContext('2d');
    const W = 1280, H = 720;

    // 1. Draw high-res background shifted left so subject sits prominently on left half
    ctx.fillStyle = '#05070B';
    ctx.fillRect(0, 0, W, H);
    const scale = Math.max(W / img.width, H / img.height) * cfg.bgScale;
    const dw = img.width * scale;
    const dh = img.height * scale;
    const dx = (W - dw) / 2 + cfg.bgShiftX;
    const dy = (H - dh) / 2;
    ctx.drawImage(img, dx, dy, dw, dh);

    // 2. Cinematic left-to-right gradient scrim so right side has 100% punchy contrast
    const scrim = ctx.createLinearGradient(0, 0, W, 0);
    scrim.addColorStop(0.0, 'rgba(5, 7, 12, 0.10)');
    scrim.addColorStop(0.38, 'rgba(5, 7, 12, 0.34)');
    scrim.addColorStop(0.56, 'rgba(6, 9, 15, 0.88)');
    scrim.addColorStop(0.74, 'rgba(6, 9, 15, 0.96)');
    scrim.addColorStop(1.0, 'rgba(5, 7, 12, 0.98)');
    ctx.fillStyle = scrim;
    ctx.fillRect(0, 0, W, H);

    // Subtle accent glow on the right panel
    const glow = ctx.createRadialGradient(970, 340, 30, 970, 340, 460);
    glow.addColorStop(0, cfg.accent + '2C');
    glow.addColorStop(0.55, cfg.signal + '14');
    glow.addColorStop(1, 'rgba(0,0,0,0)');
    ctx.fillStyle = glow;
    ctx.fillRect(460, 0, 820, H);

    // Subtle binary code backdrop texture on right half
    ctx.font = '400 18px "Plex-400", monospace';
    ctx.fillStyle = 'rgba(255,255,255,0.045)';
    const bits = '010011010111001010110100101011100101101001101011100101010110';
    for (let row = 0; row < 24; row++) {
      const shifted = bits.slice((row * 7) % bits.length) + bits;
      ctx.fillText(shifted.slice(0, 52), 620, 32 + row * 31);
    }

    // Diagonal glowing separator line
    const sep = ctx.createLinearGradient(580, 0, 520, H);
    sep.addColorStop(0, cfg.accent + '00');
    sep.addColorStop(0.3, cfg.accent + 'CC');
    sep.addColorStop(0.7, cfg.signal + 'CC');
    sep.addColorStop(1, cfg.signal + '00');
    ctx.strokeStyle = sep;
    ctx.lineWidth = 4;
    ctx.beginPath();
    ctx.moveTo(615, 36);
    ctx.lineTo(555, H - 36);
    ctx.stroke();

    // Right column anchor
    const RX = 935;

    // 3. Top pill: AI vs ZIP
    const pill = 'AI  VS  ZIP';
    ctx.font = '700 30px "Archivo-1000-700", sans-serif';
    const pw = ctx.measureText(pill).width + 54;
    ctx.fillStyle = cfg.signal;
    ctx.beginPath();
    ctx.roundRect(RX - pw / 2, 68, pw, 54, 12);
    ctx.fill();
    ctx.fillStyle = '#FFFFFF';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(pill, RX, 96);

    // 4. Tag line (song/method)
    ctx.font = '500 24px "Plex-500", monospace';
    ctx.fillStyle = '#B8C6D8';
    ctx.fillText(cfg.tagText, RX, 162);

    // 5. GIANT compressed size number (readable at 200px thumbnail width!)
    ctx.textBaseline = 'alphabetic';
    let bigSize = 136;
    ctx.font = `700 ${bigSize}px "Archivo-1000-700", sans-serif`;
    while (bigSize > 86 && ctx.measureText(cfg.bigText).width > 610) {
      bigSize -= 4;
      ctx.font = `700 ${bigSize}px "Archivo-1000-700", sans-serif`;
    }
    ctx.save();
    ctx.shadowColor = cfg.accent + '88';
    ctx.shadowBlur = 32;
    ctx.fillStyle = '#FFFFFF';
    ctx.fillText(cfg.bigText, RX, 326);
    ctx.restore();

    // 6. Struck-through ZIP size for instant visual comparison
    ctx.font = '700 44px "Plex-500", monospace';
    const zw = ctx.measureText(cfg.zipText).width;
    ctx.fillStyle = 'rgba(210, 222, 238, 0.62)';
    ctx.fillText(cfg.zipText, RX, 404);
    ctx.strokeStyle = cfg.signal;
    ctx.lineWidth = 5;
    ctx.beginPath();
    ctx.moveTo(RX - zw / 2 - 12, 391);
    ctx.lineTo(RX + zw / 2 + 12, 386);
    ctx.stroke();

    // 7. Prominent ratio callout badge ("2.9× SMALLER")
    ctx.font = '700 50px "Archivo-1000-700", sans-serif';
    const bw = Math.max(410, ctx.measureText(cfg.badgeText).width + 68);
    ctx.fillStyle = cfg.accent + '22';
    ctx.strokeStyle = cfg.accent;
    ctx.lineWidth = 3.5;
    ctx.beginPath();
    ctx.roundRect(RX - bw / 2, 444, bw, 84, 16);
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = cfg.accent;
    ctx.textBaseline = 'middle';
    ctx.fillText(cfg.badgeText, RX, 488);

    // 8. Visual compression bar comparison (ZIP vs LLM) at bottom-right
    const barX = 640, barW = 590, barY = 568;
    // ZIP bar (100% width)
    ctx.fillStyle = 'rgba(255,255,255,0.14)';
    ctx.beginPath(); ctx.roundRect(barX, barY, barW, 22, 6); ctx.fill();
    ctx.fillStyle = 'rgba(180,198,220,0.45)';
    ctx.beginPath(); ctx.roundRect(barX, barY, barW, 22, 6); ctx.fill();
    // LLM bar (~35% width, glowing accent)
    const llmW = Math.round(barW * 0.36);
    ctx.fillStyle = cfg.accent;
    ctx.beginPath(); ctx.roundRect(barX, barY + 32, llmW, 22, 6); ctx.fill();
    ctx.fillStyle = 'rgba(255,255,255,0.08)';
    ctx.beginPath(); ctx.roundRect(barX + llmW + 6, barY + 32, barW - llmW - 6, 22, 6); ctx.fill();

    // Bar labels
    ctx.font = '500 20px "Plex-500", monospace';
    ctx.textAlign = 'left';
    ctx.fillStyle = '#E6EEF8';
    ctx.fillText(cfg.footerText, barX, 654);

    // Subtle crisp border around entire thumbnail
    ctx.strokeStyle = cfg.accent + '55';
    ctx.lineWidth = 4;
    ctx.strokeRect(2, 2, W - 4, H - 4);

    return c.toDataURL('image/png').slice('data:image/png;base64,'.length);
  }, {
    bigText: BIG_TEXT,
    zipText: ZIP_TEXT,
    badgeText: BADGE_TEXT,
    tagText: TAG_TEXT,
    footerText: FOOTER_TEXT,
    accent: ACCENT,
    signal: SIGNAL,
    bgShiftX: BG_SHIFT_X,
    bgScale: BG_SCALE,
  });

  writeFileSync(OUT_PATH, Buffer.from(b64, 'base64'));
  console.log(`Wrote thumbnail: ${OUT_PATH}`);
} finally {
  await browser.close();
  server.close();
}
