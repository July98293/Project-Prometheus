"""
Xenarch Mk14 - Production Web Edition
======================================
Deployable to Railway, Render, Fly.io, or any cloud host.

Local dev:
    pip install -r requirements.txt
    python xenarch_mk14_web.py

Production (via gunicorn — set by Procfile automatically):
    gunicorn xenarch_mk14_web:app

Environment variables (all optional):
    PORT          — HTTP port (default 5000)
    HOST          — bind address (default 0.0.0.0)
    ALLOWED_ORIGINS — comma-separated CORS origins, e.g.
                      https://xenarch.example.com,https://app.example.com
                      Defaults to * (all origins) for easy first deployment.
"""

import os
import sys
import json
import base64
import tempfile
import threading
import time
import traceback
from pathlib import Path
from typing import List, Dict, Optional
from io import BytesIO

import numpy as np
from scipy.ndimage import gaussian_filter, generic_filter, label, zoom
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from loguru import logger
from PIL import Image

# ── optional heavy deps ──────────────────────────────────────────────────────
try:
    import rasterio
    from rasterio.windows import Window
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    logger.warning("rasterio not available — falling back to Pillow for image I/O")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
    logger.info("PyTorch available — VAE training enabled")
except ImportError:
    HAS_TORCH = False
    logger.warning("PyTorch not available — using numpy-only anomaly scoring")

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

# ─────────────────────────────────────────────────────────────────────────────
# GLOBALS
# ─────────────────────────────────────────────────────────────────────────────

CHIP_SIZE = 256
PROGRESS: Dict = {}
RESULTS:  Dict = {}

# ── CORS configuration ───────────────────────────────────────────────────────
# Set ALLOWED_ORIGINS env var to restrict in production, e.g.:
#   ALLOWED_ORIGINS=https://xenarch.yourdomain.com
_raw_origins = os.environ.get("ALLOWED_ORIGINS", "*")
CORS_ORIGINS = [o.strip() for o in _raw_origins.split(",")] if _raw_origins != "*" else "*"

app = Flask(__name__)
CORS(app, origins=CORS_ORIGINS, supports_credentials=True)

logger.remove()
logger.add(
    sys.stderr,
    level="INFO",
    format="<green>{time:HH:mm:ss}</green> | {level} | {message}",
)

# ─────────────────────────────────────────────────────────────────────────────
# EMBEDDED FRONTEND  (served at GET /)
# ─────────────────────────────────────────────────────────────────────────────

FRONTEND_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>XENARCH · Planetary Technosignature Detection</title>
<link rel="preconnect" href="https://fonts.googleapis.com" />
<link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Barlow+Condensed:wght@300;400;600;700&display=swap" rel="stylesheet" />
<style>
  /* ── reset & variables ─────────────────────────────────────────────── */
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg:        #04060d;
    --surface:   #090e1a;
    --panel:     #0d1526;
    --border:    #1a2744;
    --accent:    #00e5ff;
    --accent2:   #ff4f00;
    --dim:       #3a5080;
    --text:      #c8daf5;
    --textlo:    #4a6080;
    --mono:      'Share Tech Mono', monospace;
    --sans:      'Barlow Condensed', sans-serif;
    --glow:      0 0 18px rgba(0,229,255,.35);
    --glow2:     0 0 18px rgba(255,79,0,.35);
  }

  html, body { height: 100%; background: var(--bg); color: var(--text); font-family: var(--sans); }

  /* ── scan-line overlay ─────────────────────────────────────────────── */
  body::before {
    content: '';
    position: fixed; inset: 0; z-index: 9999; pointer-events: none;
    background: repeating-linear-gradient(
      0deg,
      transparent,
      transparent 3px,
      rgba(0,0,0,.09) 3px,
      rgba(0,0,0,.09) 4px
    );
  }

  /* ── layout ────────────────────────────────────────────────────────── */
  .shell {
    max-width: 1280px;
    margin: 0 auto;
    padding: 0 24px 80px;
  }

  /* ── header ────────────────────────────────────────────────────────── */
  header {
    display: flex;
    align-items: center;
    gap: 20px;
    padding: 32px 0 24px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 36px;
  }

  .logo-mark {
    width: 44px; height: 44px;
    border: 2px solid var(--accent);
    border-radius: 4px;
    display: grid;
    place-items: center;
    box-shadow: var(--glow);
    position: relative;
    flex-shrink: 0;
  }
  .logo-mark::after {
    content: '';
    position: absolute;
    inset: 5px;
    border: 1px solid var(--accent);
    border-radius: 2px;
    opacity: .5;
  }
  .logo-cross {
    width: 16px; height: 16px;
    background: linear-gradient(var(--accent), var(--accent)) 50% 0/2px 100%,
                linear-gradient(var(--accent), var(--accent)) 0 50%/100% 2px;
    background-color: transparent;
  }

  .logo-text h1 {
    font-family: var(--mono);
    font-size: 22px;
    letter-spacing: .18em;
    color: var(--accent);
    text-shadow: var(--glow);
  }
  .logo-text p {
    font-size: 11px;
    letter-spacing: .25em;
    color: var(--dim);
    text-transform: uppercase;
    margin-top: 2px;
  }

  .header-badge {
    margin-left: auto;
    font-family: var(--mono);
    font-size: 10px;
    color: var(--dim);
    letter-spacing: .1em;
    text-align: right;
    line-height: 1.8;
  }
  .header-badge span { color: var(--accent); }

  /* ── two-col layout ─────────────────────────────────────────────────── */
  .main-grid {
    display: grid;
    grid-template-columns: 320px 1fr;
    gap: 24px;
  }
  @media (max-width: 860px) { .main-grid { grid-template-columns: 1fr; } }

  /* ── panel ─────────────────────────────────────────────────────────── */
  .panel {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 24px;
  }
  .panel-title {
    font-family: var(--mono);
    font-size: 11px;
    letter-spacing: .18em;
    color: var(--dim);
    text-transform: uppercase;
    margin-bottom: 18px;
    padding-bottom: 10px;
    border-bottom: 1px solid var(--border);
  }

  /* ── drop zone ─────────────────────────────────────────────────────── */
  #dropzone {
    border: 2px dashed var(--border);
    border-radius: 6px;
    padding: 36px 16px;
    text-align: center;
    cursor: pointer;
    transition: border-color .2s, background .2s;
    position: relative;
  }
  #dropzone:hover, #dropzone.drag { border-color: var(--accent); background: rgba(0,229,255,.04); }
  #dropzone input { position: absolute; inset: 0; opacity: 0; cursor: pointer; width: 100%; height: 100%; }
  #dropzone .dz-icon { font-size: 28px; margin-bottom: 10px; opacity: .5; }
  #dropzone .dz-label { font-size: 13px; color: var(--textlo); line-height: 1.6; }
  #dropzone .dz-label span { color: var(--accent); }

  #file-list { margin-top: 12px; display: flex; flex-direction: column; gap: 6px; }
  .file-tag {
    background: rgba(0,229,255,.07);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 6px 10px;
    font-size: 11px;
    font-family: var(--mono);
    color: var(--text);
    display: flex; align-items: center; justify-content: space-between;
  }
  .file-tag button { background: none; border: none; color: var(--dim); cursor: pointer; font-size: 13px; }
  .file-tag button:hover { color: var(--accent2); }

  /* ── parameters ────────────────────────────────────────────────────── */
  .param-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 16px; }
  .param-block label {
    display: block;
    font-size: 10px;
    letter-spacing: .12em;
    color: var(--dim);
    margin-bottom: 5px;
    text-transform: uppercase;
  }
  .param-block input, .param-block select {
    width: 100%;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 4px;
    color: var(--text);
    font-family: var(--mono);
    font-size: 13px;
    padding: 7px 10px;
    outline: none;
    transition: border-color .15s;
  }
  .param-block input:focus, .param-block select:focus { border-color: var(--accent); }

  /* ── run button ────────────────────────────────────────────────────── */
  #run-btn {
    margin-top: 22px;
    width: 100%;
    padding: 13px;
    background: transparent;
    border: 2px solid var(--accent);
    border-radius: 4px;
    color: var(--accent);
    font-family: var(--mono);
    font-size: 14px;
    letter-spacing: .18em;
    cursor: pointer;
    text-transform: uppercase;
    box-shadow: var(--glow);
    transition: background .2s, color .2s, box-shadow .2s;
    position: relative;
    overflow: hidden;
  }
  #run-btn:hover:not(:disabled) {
    background: var(--accent);
    color: var(--bg);
    box-shadow: 0 0 28px rgba(0,229,255,.6);
  }
  #run-btn:disabled { opacity: .4; cursor: not-allowed; }

  /* ── right column ───────────────────────────────────────────────────── */
  .right-col { display: flex; flex-direction: column; gap: 20px; }

  /* ── progress panel ────────────────────────────────────────────────── */
  #progress-panel { display: none; }

  .progress-steps {
    display: flex;
    gap: 0;
    margin-bottom: 20px;
    border: 1px solid var(--border);
    border-radius: 4px;
    overflow: hidden;
  }
  .step-item {
    flex: 1;
    padding: 10px 6px;
    text-align: center;
    font-size: 10px;
    letter-spacing: .08em;
    text-transform: uppercase;
    color: var(--dim);
    border-right: 1px solid var(--border);
    transition: background .3s, color .3s;
  }
  .step-item:last-child { border-right: none; }
  .step-item.active { background: rgba(0,229,255,.12); color: var(--accent); }
  .step-item.done   { background: rgba(0,229,255,.06); color: var(--text); }

  .prog-bar-outer {
    height: 4px;
    background: var(--border);
    border-radius: 2px;
    margin-bottom: 16px;
    overflow: hidden;
  }
  .prog-bar-inner {
    height: 100%;
    background: var(--accent);
    border-radius: 2px;
    box-shadow: var(--glow);
    width: 0%;
    transition: width .4s ease;
  }

  #log-box {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 12px 14px;
    height: 160px;
    overflow-y: auto;
    font-family: var(--mono);
    font-size: 11px;
    line-height: 1.9;
    color: var(--textlo);
  }
  #log-box .log-ok   { color: #40e090; }
  #log-box .log-warn { color: #ffb347; }
  #log-box .log-err  { color: #ff4f4f; }

  /* ── results ───────────────────────────────────────────────────────── */
  #results-panel { display: none; }

  .summary-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 24px;
  }
  .summary-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 14px;
    text-align: center;
  }
  .summary-card .sc-val {
    font-family: var(--mono);
    font-size: 26px;
    color: var(--accent);
    text-shadow: var(--glow);
    line-height: 1;
  }
  .summary-card .sc-label {
    font-size: 10px;
    letter-spacing: .12em;
    color: var(--dim);
    text-transform: uppercase;
    margin-top: 5px;
  }

  /* ── detection grid ────────────────────────────────────────────────── */
  #detections-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
    gap: 14px;
  }

  .det-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    overflow: hidden;
    cursor: pointer;
    transition: border-color .2s, transform .15s;
    position: relative;
  }
  .det-card:hover { border-color: var(--accent); transform: translateY(-2px); }
  .det-card.rank1 { border-color: var(--accent2); box-shadow: var(--glow2); }
  .det-card.rank1 .det-rank { background: var(--accent2); color: #fff; }

  .det-img-wrap {
    position: relative;
    aspect-ratio: 1;
    background: #000;
    overflow: hidden;
  }
  .det-img-wrap img {
    width: 100%; height: 100%;
    object-fit: cover;
    display: block;
    filter: brightness(.9) contrast(1.1);
    image-rendering: pixelated;
  }
  .det-bbox {
    position: absolute;
    border: 2px solid var(--accent2);
    box-shadow: 0 0 8px rgba(255,79,0,.6);
    pointer-events: none;
  }
  .rank1 .det-bbox { border-color: #ff0; box-shadow: 0 0 10px rgba(255,255,0,.7); }

  .det-rank {
    position: absolute;
    top: 6px; left: 6px;
    background: rgba(0,0,0,.75);
    border: 1px solid var(--border);
    border-radius: 3px;
    font-family: var(--mono);
    font-size: 10px;
    padding: 2px 6px;
    color: var(--dim);
    backdrop-filter: blur(4px);
  }

  .det-info {
    padding: 10px 12px;
  }
  .det-chip-name {
    font-family: var(--mono);
    font-size: 10px;
    color: var(--textlo);
    margin-bottom: 6px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .det-conf-row {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 6px;
  }
  .det-conf-label { font-size: 10px; color: var(--dim); letter-spacing: .08em; text-transform: uppercase; }
  .det-conf-val   { font-family: var(--mono); font-size: 14px; color: var(--accent); margin-left: auto; }
  .rank1 .det-conf-val { color: var(--accent2); }

  .det-bar-outer { height: 3px; background: var(--border); border-radius: 2px; overflow: hidden; }
  .det-bar-inner { height: 100%; background: var(--accent); border-radius: 2px; }
  .rank1 .det-bar-inner { background: var(--accent2); }

  .det-metrics {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 3px 8px;
    margin-top: 8px;
  }
  .det-metric {
    font-size: 9px;
    color: var(--textlo);
    display: flex;
    justify-content: space-between;
  }
  .det-metric span { color: var(--text); font-family: var(--mono); }

  /* ── modal ─────────────────────────────────────────────────────────── */
  #modal-overlay {
    display: none;
    position: fixed; inset: 0; z-index: 1000;
    background: rgba(4,6,13,.88);
    backdrop-filter: blur(6px);
    align-items: center;
    justify-content: center;
  }
  #modal-overlay.open { display: flex; }
  #modal-box {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 8px;
    max-width: 640px;
    width: 92%;
    padding: 28px;
    position: relative;
  }
  #modal-close {
    position: absolute;
    top: 14px; right: 16px;
    background: none; border: none;
    color: var(--dim); font-size: 20px; cursor: pointer;
  }
  #modal-close:hover { color: var(--accent2); }
  #modal-img { width: 100%; aspect-ratio: 1; object-fit: cover; border-radius: 4px; image-rendering: pixelated; margin-bottom: 18px; }
  #modal-title { font-family: var(--mono); font-size: 14px; color: var(--accent); margin-bottom: 16px; }
  .modal-metrics { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
  .mm-row { display: flex; flex-direction: column; gap: 4px; }
  .mm-label { font-size: 10px; letter-spacing: .12em; color: var(--dim); text-transform: uppercase; }
  .mm-bar-outer { height: 4px; background: var(--border); border-radius: 2px; }
  .mm-bar-inner { height: 100%; background: var(--accent); border-radius: 2px; }
  .mm-val { font-family: var(--mono); font-size: 12px; color: var(--text); }

  /* ── misc ──────────────────────────────────────────────────────────── */
  #export-btn {
    margin-top: 18px;
    padding: 10px 20px;
    background: transparent;
    border: 1px solid var(--border);
    border-radius: 4px;
    color: var(--dim);
    font-family: var(--mono);
    font-size: 12px;
    letter-spacing: .12em;
    cursor: pointer;
    transition: border-color .2s, color .2s;
  }
  #export-btn:hover { border-color: var(--accent); color: var(--accent); }

  .error-banner {
    background: rgba(255,79,79,.1);
    border: 1px solid #ff4f4f;
    border-radius: 4px;
    padding: 12px 16px;
    font-family: var(--mono);
    font-size: 12px;
    color: #ff8a8a;
    margin-top: 12px;
  }

  /* scan animation on logo */
  @keyframes scanY {
    0%,100% { transform: translateY(0); opacity: 1; }
    50%      { transform: translateY(28px); opacity: .4; }
  }
  .logo-cross { animation: scanY 3s ease-in-out infinite; }
</style>
</head>
<body>
<div class="shell">

  <!-- HEADER -->
  <header>
    <div class="logo-mark"><div class="logo-cross"></div></div>
    <div class="logo-text">
      <h1>XENARCH</h1>
      <p>Planetary Surface Technosignature Detection — Mk14</p>
    </div>
    <div class="header-badge" id="status-badge">
      BACKEND<br>
      <span id="badge-torch">LOADING…</span>
    </div>
  </header>

  <div class="main-grid">

    <!-- LEFT: Controls -->
    <aside>
      <div class="panel">
        <div class="panel-title">// Input Images</div>

        <div id="dropzone">
          <input type="file" id="file-input" multiple accept=".tif,.tiff,.png,.jpg,.jpeg,.npy" />
          <div class="dz-icon">⊕</div>
          <div class="dz-label">Drop planetary imagery here<br><span>or click to browse</span><br>TIF · PNG · JPG · NPY</div>
        </div>
        <div id="file-list"></div>

        <div class="panel-title" style="margin-top:24px">// Detection Parameters</div>
        <div class="param-grid">
          <div class="param-block">
            <label>Chip Size (px)</label>
            <select id="p-chip">
              <option value="128">128</option>
              <option value="256" selected>256</option>
              <option value="512">512</option>
            </select>
          </div>
          <div class="param-block">
            <label>Anomaly %ile</label>
            <input type="number" id="p-pct" value="92" min="50" max="99" step="1" />
          </div>
          <div class="param-block">
            <label>Epochs</label>
            <input type="number" id="p-epochs" value="15" min="1" max="50" />
          </div>
          <div class="param-block">
            <label>Latent Dim</label>
            <input type="number" id="p-latent" value="56" min="8" max="256" />
          </div>
          <div class="param-block">
            <label>Batch Size</label>
            <input type="number" id="p-batch" value="4" min="1" max="32" />
          </div>
          <div class="param-block">
            <label>Learn Rate</label>
            <input type="number" id="p-lr" value="0.0005" min="0.00001" max="0.01" step="0.00001" />
          </div>
        </div>

        <button id="run-btn" disabled>▶ RUN ANALYSIS</button>
        <div id="error-area"></div>
      </div>
    </aside>

    <!-- RIGHT: Progress + Results -->
    <div class="right-col">

      <!-- PROGRESS -->
      <div class="panel" id="progress-panel">
        <div class="panel-title">// Pipeline Status</div>
        <div class="progress-steps">
          <div class="step-item" id="s1">1 · Chip Extract</div>
          <div class="step-item" id="s2">2 · VAE Train</div>
          <div class="step-item" id="s3">3 · Normalise</div>
          <div class="step-item" id="s4">4 · Rank</div>
          <div class="step-item" id="s5">5 · Package</div>
        </div>
        <div class="prog-bar-outer"><div class="prog-bar-inner" id="prog-bar"></div></div>
        <div id="log-box"></div>
      </div>

      <!-- RESULTS -->
      <div class="panel" id="results-panel">
        <div class="panel-title">// Detection Results</div>
        <div class="summary-row" id="summary-row"></div>
        <div id="detections-grid"></div>
        <button id="export-btn">⬇ Export CSV</button>
      </div>

    </div><!-- /right-col -->
  </div><!-- /main-grid -->
</div><!-- /shell -->

<!-- MODAL -->
<div id="modal-overlay">
  <div id="modal-box">
    <button id="modal-close">✕</button>
    <div id="modal-title"></div>
    <img id="modal-img" src="" alt="chip" />
    <div class="modal-metrics" id="modal-metrics"></div>
  </div>
</div>

<script>
/* ── API base: auto-detect same host ───────────────────────────────────── */
const API = window.location.origin;

/* ── Status badge ──────────────────────────────────────────────────────── */
async function checkStatus() {
  try {
    const r = await fetch(`${API}/api/status`);
    const d = await r.json();
    const badge = document.getElementById('badge-torch');
    badge.textContent = d.torch ? 'VAE (PyTorch)' : 'NumPy scorer';
    badge.style.color = d.torch ? 'var(--accent)' : '#ffb347';
  } catch(e) {
    document.getElementById('badge-torch').textContent = 'OFFLINE';
  }
}
checkStatus();

/* ── File handling ─────────────────────────────────────────────────────── */
let selectedFiles = [];
const dz     = document.getElementById('dropzone');
const fi     = document.getElementById('file-input');
const fl     = document.getElementById('file-list');
const runBtn = document.getElementById('run-btn');

fi.addEventListener('change', () => addFiles(fi.files));
dz.addEventListener('dragover', e => { e.preventDefault(); dz.classList.add('drag'); });
dz.addEventListener('dragleave', () => dz.classList.remove('drag'));
dz.addEventListener('drop', e => { e.preventDefault(); dz.classList.remove('drag'); addFiles(e.dataTransfer.files); });

function addFiles(raw) {
  Array.from(raw).forEach(f => {
    if (!selectedFiles.find(x => x.name === f.name)) selectedFiles.push(f);
  });
  renderFileList();
}
function removeFile(name) {
  selectedFiles = selectedFiles.filter(f => f.name !== name);
  renderFileList();
}
function renderFileList() {
  fl.innerHTML = '';
  selectedFiles.forEach(f => {
    const tag = document.createElement('div');
    tag.className = 'file-tag';
    const kb = (f.size / 1024).toFixed(0);
    tag.innerHTML = `<span>${f.name} <span style="color:var(--textlo)">${kb}KB</span></span>
                     <button onclick="removeFile('${f.name}')">×</button>`;
    fl.appendChild(tag);
  });
  runBtn.disabled = selectedFiles.length === 0;
}

/* ── Run analysis ──────────────────────────────────────────────────────── */
let activeJobId = null;
let pollTimer   = null;
let csvData     = [];

runBtn.addEventListener('click', startAnalysis);

async function startAnalysis() {
  if (!selectedFiles.length) return;
  document.getElementById('error-area').innerHTML = '';

  const config = {
    chip_size:    +document.getElementById('p-chip').value,
    percentile:   +document.getElementById('p-pct').value,
    epochs:       +document.getElementById('p-epochs').value,
    latent_dim:   +document.getElementById('p-latent').value,
    batch_size:   +document.getElementById('p-batch').value,
    lr:           +document.getElementById('p-lr').value,
  };

  const fd = new FormData();
  selectedFiles.forEach(f => fd.append('files[]', f));
  fd.append('config', JSON.stringify(config));

  runBtn.disabled = true;
  showProgress(true);
  resetSteps();
  document.getElementById('results-panel').style.display = 'none';
  clearLog();

  try {
    const r = await fetch(`${API}/api/analyze`, { method:'POST', body: fd });
    const d = await r.json();
    if (d.error) throw new Error(d.error);
    activeJobId = d.job_id;
    pollTimer = setInterval(pollProgress, 1000);
  } catch(e) {
    showError(e.message);
    runBtn.disabled = false;
  }
}

async function pollProgress() {
  if (!activeJobId) return;
  try {
    const r = await fetch(`${API}/api/progress/${activeJobId}`);
    const d = await r.json();

    setProgress(d.pct, d.step);
    (d.logs || []).forEach(appendLog);

    if (d.error) {
      clearInterval(pollTimer);
      showError(d.error);
      runBtn.disabled = false;
      return;
    }

    if (d.done) {
      clearInterval(pollTimer);
      await loadResults();
      runBtn.disabled = false;
    }
  } catch(e) { /* network blip, retry */ }
}

async function loadResults() {
  const r = await fetch(`${API}/api/results/${activeJobId}`);
  const d = await r.json();
  if (d.error) { showError(d.error); return; }
  csvData = d.csv_rows || [];
  renderSummary(d.summary);
  renderDetections(d.detections);
  document.getElementById('results-panel').style.display = 'block';
}

/* ── UI helpers ────────────────────────────────────────────────────────── */
function showProgress(on) {
  document.getElementById('progress-panel').style.display = on ? 'block' : 'none';
}

function resetSteps() {
  for (let i = 1; i <= 5; i++) {
    const el = document.getElementById('s' + i);
    el.classList.remove('active', 'done');
  }
  document.getElementById('prog-bar').style.width = '0%';
}

function setProgress(pct, step) {
  document.getElementById('prog-bar').style.width = pct + '%';
  for (let i = 1; i <= 5; i++) {
    const el = document.getElementById('s' + i);
    el.classList.remove('active', 'done');
    if (i < step) el.classList.add('done');
    else if (i === step) el.classList.add('active');
  }
}

const logBox = document.getElementById('log-box');
const seenLogs = new Set();
function appendLog(entry) {
  const key = entry.t + entry.msg;
  if (seenLogs.has(key)) return;
  seenLogs.add(key);
  const cls = entry.level === 'success' ? 'log-ok' : entry.level === 'error' ? 'log-err' : entry.level === 'warning' ? 'log-warn' : '';
  logBox.innerHTML += `<div class="${cls}">[${entry.t}] ${entry.msg}</div>`;
  logBox.scrollTop = logBox.scrollHeight;
}
function clearLog() { logBox.innerHTML = ''; seenLogs.clear(); }

function showError(msg) {
  document.getElementById('error-area').innerHTML = `<div class="error-banner">ERROR: ${msg}</div>`;
}

/* ── Summary cards ─────────────────────────────────────────────────────── */
function renderSummary(s) {
  const row = document.getElementById('summary-row');
  row.innerHTML = [
    { val: s.total_chips,  label: 'Total Chips' },
    { val: s.n_anomalies,  label: 'Anomalies' },
    { val: s.n_high_conf,  label: 'High Conf' },
    { val: (s.top_conf*100).toFixed(2)+'%', label: 'Top Confidence' },
  ].map(c => `
    <div class="summary-card">
      <div class="sc-val">${c.val}</div>
      <div class="sc-label">${c.label}</div>
    </div>`).join('');
}

/* ── Detection grid ────────────────────────────────────────────────────── */
function renderDetections(dets) {
  const grid = document.getElementById('detections-grid');
  grid.innerHTML = '';
  dets.forEach((d, idx) => {
    const conf = (d.confidence * 100).toFixed(2);
    const card = document.createElement('div');
    card.className = 'det-card' + (idx === 0 ? ' rank1' : '');
    card.onclick = () => openModal(d);

    let bboxHtml = '';
    if (d.featureBbox) {
      const [y1n, x1n, y2n, x2n] = d.featureBbox;
      bboxHtml = `<div class="det-bbox" style="
        top:${(y1n*100).toFixed(1)}%;
        left:${(x1n*100).toFixed(1)}%;
        width:${((x2n-x1n)*100).toFixed(1)}%;
        height:${((y2n-y1n)*100).toFixed(1)}%;
      "></div>`;
    }

    const mkeys = ['mse','density','contextual','gradient','edge'];
    const mrows = mkeys.map(k =>
      `<div class="det-metric">${k} <span>${d.metrics[k]}</span></div>`
    ).join('');

    card.innerHTML = `
      <div class="det-img-wrap">
        <img src="${d.imgDataURI}" alt="chip" />
        ${bboxHtml}
        <div class="det-rank">#${d.rank}</div>
      </div>
      <div class="det-info">
        <div class="det-chip-name">${d.chipName}</div>
        <div class="det-conf-row">
          <div class="det-conf-label">CONF</div>
          <div class="det-conf-val">${conf}%</div>
        </div>
        <div class="det-bar-outer"><div class="det-bar-inner" style="width:${conf}%"></div></div>
        <div class="det-metrics">${mrows}</div>
      </div>`;
    grid.appendChild(card);
  });
}

/* ── Modal ─────────────────────────────────────────────────────────────── */
const overlay = document.getElementById('modal-overlay');
document.getElementById('modal-close').onclick = () => overlay.classList.remove('open');
overlay.addEventListener('click', e => { if (e.target === overlay) overlay.classList.remove('open'); });

function openModal(d) {
  document.getElementById('modal-title').textContent =
    `RANK #${d.rank}  ·  ${d.chipName}  ·  ${(d.confidence*100).toFixed(2)}% confidence`;
  document.getElementById('modal-img').src = d.imgDataURI;

  const mm = document.getElementById('modal-metrics');
  mm.innerHTML = Object.entries(d.metrics).map(([k, v]) => `
    <div class="mm-row">
      <div class="mm-label">${k}</div>
      <div class="mm-bar-outer"><div class="mm-bar-inner" style="width:${(v*100).toFixed(0)}%"></div></div>
      <div class="mm-val">${v}</div>
    </div>`).join('');

  overlay.classList.add('open');
}

/* ── CSV export ────────────────────────────────────────────────────────── */
document.getElementById('export-btn').addEventListener('click', () => {
  if (!csvData.length) return;
  const cols = ['chipName','confidence','score','source',
                'mse','density','contextual','gradient','edge'];
  const header = cols.join(',');
  const rows = csvData.map(r => [
    r.chipName, r.confidence, r.score, r.source,
    r.metrics.mse, r.metrics.density, r.metrics.contextual,
    r.metrics.gradient, r.metrics.edge
  ].join(','));
  const blob = new Blob([header+'\n'+rows.join('\n')], {type:'text/csv'});
  const a = Object.assign(document.createElement('a'), {
    href: URL.createObjectURL(blob), download: 'xenarch_results.csv'
  });
  a.click();
});
</script>
</body>
</html>"""

# ─────────────────────────────────────────────────────────────────────────────
# 1.  IMAGE LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_image_as_array(path: str) -> np.ndarray:
    if HAS_RASTERIO:
        try:
            with rasterio.open(path) as src:
                arr = src.read(1).astype(np.float32)
            lo, hi = np.percentile(arr, [1, 99])
            arr = np.clip(arr, lo, hi)
            arr = (arr - lo) / (hi - lo + 1e-8)
            return arr
        except Exception:
            pass
    img = Image.open(path).convert("L")
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr


# ─────────────────────────────────────────────────────────────────────────────
# 2.  CHIP EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_chips(image_path: str, output_dir: str,
                  chip_size: int = CHIP_SIZE,
                  max_chips: int = 300) -> List[Dict]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    arr = load_image_as_array(image_path)
    h, w = arr.shape
    stride = chip_size
    chips = []
    chip_id = 0
    source_stem = Path(image_path).stem

    for y in range(0, h - chip_size + 1, stride):
        for x in range(0, w - chip_size + 1, stride):
            if chip_id >= max_chips:
                break
            chip = arr[y:y+chip_size, x:x+chip_size]
            if chip.std() < 0.01:
                continue
            chip_filename = f"{source_stem}_chip_{chip_id:04d}.npy"
            chip_path = output_path / chip_filename
            np.save(str(chip_path), chip)
            chips.append({
                "chip_id":   chip_id,
                "chip_path": str(chip_path),
                "center_x":  x + chip_size // 2,
                "center_y":  y + chip_size // 2,
                "source":    Path(image_path).name,
            })
            chip_id += 1
        if chip_id >= max_chips:
            break

    logger.info(f"  Extracted {chip_id} chips from {Path(image_path).name}")
    return chips


# ─────────────────────────────────────────────────────────────────────────────
# 3.  NUMPY MULTI-METRIC SCORER
# ─────────────────────────────────────────────────────────────────────────────

class NumpyAnomalyScorer:
    def __init__(self, reference_chips: List[np.ndarray]):
        if reference_chips:
            stacked = np.stack([c.ravel() for c in reference_chips[:200]])
            self.ref_mean = stacked.mean(axis=0)
            self.ref_std  = stacked.std(axis=0) + 1e-8
        else:
            self.ref_mean = None
            self.ref_std  = None

    def mse_score(self, chip: np.ndarray) -> float:
        if self.ref_mean is None:
            return float(chip.std())
        return float(np.mean((chip.ravel() - self.ref_mean) ** 2))

    def density_score(self, chip: np.ndarray) -> float:
        if self.ref_mean is None:
            return float(np.abs(chip - chip.mean()).mean())
        diff = chip.ravel() - self.ref_mean
        return float(np.sqrt(np.sum((diff / self.ref_std) ** 2)) / chip.size)

    def contextual_score(self, chip: np.ndarray) -> Dict:
        mean_b = chip.mean()
        std_b  = chip.std() + 1e-8
        threshold   = mean_b + 2 * std_b
        bright_mask = chip > threshold
        brightness_a = 0.0
        if bright_mask.sum() >= 5:
            bright_a = np.mean(chip[bright_mask])
            brightness_a = min((bright_a - mean_b) / (std_b * 3), 1.0)
        try:
            local_std = generic_filter(chip, np.std, size=9)
            tex_mean  = local_std.mean()
            tex_std   = local_std.std() + 1e-8
            texture_a = float((np.abs(local_std - tex_mean) > 2 * tex_std).mean())
        except Exception:
            texture_a = 0.0
        labeled, n_regions = label(bright_mask)
        comp_a = 0.0
        best_bbox = None
        for rid in range(1, n_regions + 1):
            region = labeled == rid
            size   = region.sum()
            if size < 4:
                continue
            ys, xs = np.where(region)
            y1, y2 = int(ys.min()), int(ys.max())
            x1, x2 = int(xs.min()), int(xs.max())
            bbox_area   = (y2-y1+1) * (x2-x1+1)
            compactness = size / (bbox_area + 1e-8)
            if compactness > comp_a:
                comp_a = compactness
                best_bbox = [y1/chip.shape[0], x1/chip.shape[1],
                             y2/chip.shape[0], x2/chip.shape[1]]
        combined = float(0.35 * brightness_a + 0.35 * texture_a + 0.30 * comp_a)
        return {"score": combined, "bbox": best_bbox}

    def gradient_score(self, chip: np.ndarray) -> float:
        gx = np.gradient(chip, axis=0)
        gy = np.gradient(chip, axis=1)
        grad_mag = np.sqrt(gx**2 + gy**2)
        local_g  = gaussian_filter(grad_mag, sigma=4)
        return float(np.abs(grad_mag - local_g).mean())

    def edge_regularity_score(self, chip: np.ndarray) -> float:
        gx = np.gradient(chip, axis=0)
        gy = np.gradient(chip, axis=1)
        edge_str = np.sqrt(gx**2 + gy**2)
        thresh   = np.percentile(edge_str, 90)
        strong   = edge_str > thresh
        if strong.sum() < 10:
            return 0.0
        row_align = strong.sum(axis=1).max() / (strong.sum() + 1e-8)
        col_align = strong.sum(axis=0).max() / (strong.sum() + 1e-8)
        return float(max(row_align, col_align))

    def score(self, chip: np.ndarray) -> Dict:
        ctx_data = self.contextual_score(chip)
        return {
            "mse":         self.mse_score(chip),
            "density":     self.density_score(chip),
            "contextual":  ctx_data["score"],
            "gradient":    self.gradient_score(chip),
            "edge":        self.edge_regularity_score(chip),
            "feature_bbox": ctx_data["bbox"],
        }

    def normalize_scores(self, score_list: List[Dict]) -> List[Dict]:
        keys = ["mse", "density", "contextual", "gradient", "edge"]
        arrays = {k: np.array([s[k] for s in score_list]) for k in keys}
        norms  = {}
        for k, arr in arrays.items():
            lo, hi = arr.min(), arr.max()
            norms[k] = (arr - lo) / (hi - lo + 1e-8)
        result = []
        for i, s in enumerate(score_list):
            s_out = dict(s)
            for k in keys:
                s_out[f"{k}_norm"] = float(norms[k][i])
            s_out["combined"] = float(
                0.30 * norms["mse"][i]        +
                0.20 * norms["density"][i]    +
                0.30 * norms["contextual"][i] +
                0.15 * norms["gradient"][i]   +
                0.05 * norms["edge"][i]
            )
            result.append(s_out)
        return result


# ─────────────────────────────────────────────────────────────────────────────
# 4.  TORCH VAE
# ─────────────────────────────────────────────────────────────────────────────

if HAS_TORCH:
    class StableConvolutionalVAE(nn.Module):
        def __init__(self, latent_dim=56, input_size=256):
            super().__init__()
            self.latent_dim = latent_dim
            self.input_size = input_size
            final_size = input_size // 16
            self.encoder_conv = nn.Sequential(
                nn.Conv2d(1, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32, eps=1e-3), nn.ReLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64, eps=1e-3), nn.ReLU(),
                nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128, eps=1e-3), nn.ReLU(),
                nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256, eps=1e-3), nn.ReLU(),
            )
            self.fc_mu     = nn.Linear(256*final_size*final_size, latent_dim)
            self.fc_logvar = nn.Linear(256*final_size*final_size, latent_dim)
            nn.init.xavier_uniform_(self.fc_mu.weight, gain=0.01)
            nn.init.xavier_uniform_(self.fc_logvar.weight, gain=0.01)
            self.decoder_input = nn.Linear(latent_dim, 256*final_size*final_size)
            self.final_size = final_size
            self.decoder_conv = nn.Sequential(
                nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(128, eps=1e-3), nn.ReLU(),
                nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(64, eps=1e-3), nn.ReLU(),
                nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(32, eps=1e-3), nn.ReLU(),
                nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),
                nn.Sigmoid(),
            )

        def encode(self, x):
            h = torch.flatten(self.encoder_conv(x), 1)
            mu, logvar = self.fc_mu(h), self.fc_logvar(h)
            return mu, torch.clamp(logvar, -10, 10)

        def reparameterize(self, mu, logvar):
            return mu + torch.randn_like(torch.exp(0.5*logvar)) * torch.exp(0.5*logvar)

        def decode(self, z):
            h = self.decoder_input(z).view(-1, 256, self.final_size, self.final_size)
            return self.decoder_conv(h)

        def forward(self, x):
            mu, logvar = self.encode(x)
            return self.decode(self.reparameterize(mu, logvar)), mu, logvar

    class TorchDataset(torch.utils.data.Dataset):
        def __init__(self, chip_paths):
            self.paths = chip_paths
        def __len__(self): return len(self.paths)
        def __getitem__(self, idx):
            arr = np.load(self.paths[idx]).astype(np.float32)[np.newaxis]
            return torch.from_numpy(arr), self.paths[idx]

    def stable_vae_loss(recon, x, mu, logvar, beta=0.001, kl_weight=1.0):
        mse = F.mse_loss(recon, x, reduction="sum")
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return mse + beta * kl_weight * kld, mse, kld


# ─────────────────────────────────────────────────────────────────────────────
# 5.  CONFIDENCE
# ─────────────────────────────────────────────────────────────────────────────

def compute_confidence(scored_chips: List[Dict]) -> List[Dict]:
    scores    = np.array([c["combined"]                  for c in scored_chips])
    ctx_norms = np.array([c.get("contextual_norm", 0)   for c in scored_chips])
    mse_norms = np.array([c.get("mse_norm", 0)          for c in scored_chips])
    lo, hi    = scores.min(), scores.max()
    score_norm = (scores - lo) / (hi - lo + 1e-8)
    confidence = np.clip(0.50*score_norm + 0.30*ctx_norms + 0.20*mse_norms, 0, 1)
    out = []
    for i, c in enumerate(scored_chips):
        c2 = dict(c)
        c2["confidence"] = float(confidence[i])
        out.append(c2)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 6.  THUMBNAIL
# ─────────────────────────────────────────────────────────────────────────────

def chip_to_b64(chip_path: str, size: int = 256) -> str:
    try:
        arr = np.load(chip_path)
    except Exception:
        arr = np.zeros((CHIP_SIZE, CHIP_SIZE), dtype=np.float32)
    arr = (arr * 255).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(arr, mode="L").resize((size, size), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# ─────────────────────────────────────────────────────────────────────────────
# 7.  MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_analysis(job_id: str, image_paths: List[str], config: Dict, tmp_dir: str):
    def log(msg, level="info"):
        entry = {"t": time.strftime("%H:%M:%S.") + f"{int(time.time()*1000)%1000:03d}",
                 "msg": msg, "level": level}
        PROGRESS[job_id]["logs"].append(entry)
        getattr(logger, level)(msg)

    def set_step(n, pct):
        PROGRESS[job_id]["step"] = n
        PROGRESS[job_id]["pct"]  = pct

    try:
        PROGRESS[job_id] = {"step": 0, "pct": 0, "logs": [], "done": False, "error": None}

        chip_dir   = os.path.join(tmp_dir, "chips")
        chip_size  = config.get("chip_size", CHIP_SIZE)
        percentile = config.get("percentile", 92)
        epochs     = config.get("epochs", 15)
        latent_dim = config.get("latent_dim", 56)
        batch_size = config.get("batch_size", 4)
        lr         = config.get("lr", 0.0005)
        warmup     = config.get("warmup_epochs", 3)

        set_step(1, 5)
        log(f"Chip extractor: chip_size={chip_size}")
        all_chips = []
        for path in image_paths:
            log(f"Extracting chips from {Path(path).name}…")
            chips = extract_chips(path, chip_dir, chip_size=chip_size)
            all_chips.extend(chips)
            log(f"  → {len(chips)} chips extracted", "success" if chips else "warning")

        if not all_chips:
            raise ValueError("No chips extracted — check image dimensions (need ≥256×256).")

        log(f"Total chips: {len(all_chips)}", "success")
        set_step(1, 18)

        chip_paths = [c["chip_path"] for c in all_chips]
        chips_arr  = [np.load(p) for p in chip_paths]

        set_step(2, 20)
        model_used = "VAE (PyTorch)" if HAS_TORCH else "NumPy scorer"
        log(f"Fitting {model_used} on {len(all_chips)} chips…")

        if HAS_TORCH:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model  = StableConvolutionalVAE(latent_dim=latent_dim, input_size=chip_size).to(device)
            optim_ = torch.optim.Adam(model.parameters(), lr=lr)
            sched  = torch.optim.lr_scheduler.ReduceLROnPlateau(optim_, factor=0.5, patience=2)
            ds     = TorchDataset(chip_paths)
            loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)

            for epoch in range(epochs):
                model.train()
                kl_w = min(1.0, (epoch+1)/warmup)
                total_loss, total_mse, total_kld = 0, 0, 0
                for imgs, _ in loader:
                    imgs = imgs.to(device)
                    optim_.zero_grad()
                    recon, mu, logvar = model(imgs)
                    loss, mse, kld = stable_vae_loss(recon, imgs, mu, logvar, kl_weight=kl_w)
                    if torch.isnan(loss):
                        log(f"NaN at epoch {epoch+1}!", "error"); break
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optim_.step()
                    total_loss += loss.item(); total_mse += mse.item(); total_kld += kld.item()
                avg = total_loss / max(len(ds), 1)
                sched.step(avg)
                log(f"Epoch {epoch+1}/{epochs} [KL={kl_w:.2f}] loss={avg:.1f}")
                set_step(2, 20 + int((epoch+1)/epochs * 35))

            model.eval()
            log("Computing VAE reconstruction scores…")
            raw_scores = []
            with torch.no_grad():
                for imgs, _ in DataLoader(TorchDataset(chip_paths), batch_size=batch_size, shuffle=False):
                    imgs = imgs.to(device)
                    recon, mu, logvar = model(imgs)
                    mse   = torch.mean((imgs - recon)**2, dim=[1,2,3]).cpu().numpy()
                    dist  = torch.sqrt(torch.sum(mu**2, dim=1)).cpu().numpy()
                    dense = 1.0 - np.exp(-dist / latent_dim)
                    for j in range(len(mse)):
                        raw_scores.append({"mse": float(mse[j]), "density": float(dense[j]),
                                           "contextual": 0.0, "gradient": 0.0, "edge": 0.0})
            np_scorer = NumpyAnomalyScorer([])
            for i, arr in enumerate(chips_arr):
                ctx_data = np_scorer.contextual_score(arr)
                raw_scores[i]["contextual"]   = ctx_data["score"]
                raw_scores[i]["feature_bbox"] = ctx_data["bbox"]
                raw_scores[i]["gradient"]     = np_scorer.gradient_score(arr)
                raw_scores[i]["edge"]         = np_scorer.edge_regularity_score(arr)
        else:
            np_scorer  = NumpyAnomalyScorer(chips_arr)
            raw_scores = [np_scorer.score(arr) for arr in chips_arr]

        set_step(3, 58)
        log("Normalising scores…")
        scorer_tmp = NumpyAnomalyScorer([])
        scored = scorer_tmp.normalize_scores(raw_scores)
        set_step(3, 72)

        set_step(4, 73)
        scored = compute_confidence(scored)
        threshold = np.percentile([s["combined"] for s in scored], percentile)
        for i, s in enumerate(scored):
            s.update(all_chips[i])
            s["is_anomaly"] = bool(s["combined"] > threshold)
        scored.sort(key=lambda x: x["confidence"], reverse=True)
        n_anomaly = sum(1 for s in scored if s["is_anomaly"])
        n_high    = sum(1 for s in scored if s["confidence"] > 0.8)
        log(f"Anomalies: {n_anomaly}  |  High-conf >0.8: {n_high}", "success")
        set_step(4, 88)

        set_step(5, 90)
        log("Generating thumbnails…")
        top_n = min(12, len(scored))
        results_out = []
        for rank, s in enumerate(scored[:top_n], 1):
            results_out.append({
                "rank":        rank,
                "chipName":    Path(s["chip_path"]).stem,
                "confidence":  round(s["confidence"], 4),
                "score":       round(s["combined"], 4),
                "source":      s.get("source", ""),
                "imgDataURI":  chip_to_b64(s["chip_path"]),
                "featureBbox": s.get("feature_bbox"),
                "metrics": {
                    "mse":        round(s.get("mse_norm", 0), 3),
                    "density":    round(s.get("density_norm", 0), 3),
                    "contextual": round(s.get("contextual_norm", 0), 3),
                    "gradient":   round(s.get("gradient_norm", 0), 3),
                    "edge":       round(s.get("edge_norm", 0), 3),
                },
            })
        log(f"Rank 1: {results_out[0]['chipName']}  conf={results_out[0]['confidence']:.4f}", "success")
        set_step(5, 100)
        log("Analysis complete ✓", "success")

        RESULTS[job_id] = {
            "summary": {
                "total_chips": len(scored),
                "n_anomalies": n_anomaly,
                "n_high_conf": n_high,
                "top_conf":    round(results_out[0]["confidence"], 4) if results_out else 0,
                "model_used":  model_used,
            },
            "detections": results_out,
            "csv_rows": [
                {k: s[k] for k in ("chipName","confidence","score","source","metrics")}
                for s in results_out
            ],
        }
        PROGRESS[job_id]["done"] = True

    except Exception as exc:
        logger.error(traceback.format_exc())
        PROGRESS[job_id]["error"] = str(exc)
        PROGRESS[job_id]["done"]  = True


# ─────────────────────────────────────────────────────────────────────────────
# 8.  ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template_string(FRONTEND_HTML)

@app.route("/api/status")
def api_status():
    return jsonify({"status": "ok", "torch": HAS_TORCH, "rasterio": HAS_RASTERIO})

@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    if "files[]" not in request.files:
        return jsonify({"error": "No files uploaded"}), 400
    config = {}
    if "config" in request.form:
        try:
            config = json.loads(request.form["config"])
        except Exception:
            pass
    tmp_dir  = tempfile.mkdtemp(prefix="xenarch_")
    uploaded = []
    for f in request.files.getlist("files[]"):
        dest = os.path.join(tmp_dir, f.filename)
        f.save(dest)
        uploaded.append(dest)
    job_id = f"job_{int(time.time()*1000)}"
    threading.Thread(target=run_analysis, args=(job_id, uploaded, config, tmp_dir), daemon=True).start()
    return jsonify({"job_id": job_id})

@app.route("/api/progress/<job_id>")
def api_progress(job_id):
    p = PROGRESS.get(job_id)
    if p is None:
        return jsonify({"error": "unknown job"}), 404
    return jsonify({"step": p["step"], "pct": p["pct"],
                    "logs": p["logs"][-50:], "done": p["done"], "error": p["error"]})

@app.route("/api/results/<job_id>")
def api_results(job_id):
    r = RESULTS.get(job_id)
    if r is None:
        p = PROGRESS.get(job_id, {})
        if p.get("error"):
            return jsonify({"error": p["error"]}), 500
        return jsonify({"error": "results not ready"}), 202
    return jsonify(r)


# ─────────────────────────────────────────────────────────────────────────────
# 8.  FINETUNING  —  random search + successive halving over the VAE knobs
# ─────────────────────────────────────────────────────────────────────────────
#
# Same method as xenarch_hpo_sim.py, but on the *real* Mk17 VAE:
#   • sample N configs at random  (a grid would spend equal compute on the bad ones)
#   • train them all briefly, keep the best 1/ETA, train the survivors longer, repeat
#
# Training loss = MSE + β·KL + λ_grad·(gradient mismatch) + λ_con·(local-contrast mismatch),
# so the VAE learns to rebuild edges and texture instead of a blurry average.
#
# Objective = detection AUROC on held-out natural terrain: every val chip is scored clean
# and with a synthetic artificial object pasted in (rectangle, ring or straight line).
# Each chip gets four metrics — reconstruction MSE, gradient residual, contrast residual,
# edge regularity — and the combined score (mean of their z-scores) should rank the
# injected chips above the clean ones. Held-out MSE is kept as a tiebreak.
#
# Synthetic objects are a stand-in: swap in labelled lander chips when they exist.

_FT_SEED = 0
_FT_METRICS = ("mse", "gradient", "contrast", "edge")


def _ft_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _ft_prepare_chips(image_dir, chip_size, tmp_dir, max_total, val_frac, per_image=6):
    exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
    imgs = sorted(p for p in Path(image_dir).rglob("*") if p.suffix.lower() in exts)
    if not imgs:
        raise SystemExit(f"finetune: no images found in '{image_dir}'")
    rng = np.random.default_rng(_FT_SEED)
    rng.shuffle(imgs)                            # sample across all sub-folders / missions
    chip_dir = os.path.join(tmp_dir, "ft_chips")
    paths = []
    for im in imgs:
        if len(paths) >= max_total:
            break
        for c in extract_chips(str(im), chip_dir, chip_size=chip_size, max_chips=per_image):
            paths.append(c["chip_path"])
    if len(paths) < 8:
        raise SystemExit(f"finetune: only {len(paths)} chips — need images ≥ {chip_size}px")
    rng.shuffle(paths)
    paths = paths[:max_total]
    n_val = max(4, int(len(paths) * val_frac))
    return paths[n_val:], paths[:n_val]          # train, val


def _ft_sample_cfg(rng):
    return dict(
        latent_dim    = int(rng.choice([32, 48, 56, 64, 96])),
        lr            = float(10 ** rng.uniform(-4.0, -2.7)),     # 1e-4 .. ~2e-3
        batch_size    = int(rng.choice([2, 4, 8])),
        warmup_epochs = int(rng.choice([1, 2, 3, 5])),
        beta          = float(10 ** rng.uniform(-4, -2)),         # KL weight 1e-4 .. 1e-2
        lam_grad      = float(rng.choice([0.0, 0.3, 1.0, 3.0])),  # 0 = plain MSE baseline
        lam_con       = float(rng.choice([0.0, 0.3, 1.0, 3.0])),
    )


def _ft_diffs(t):
    return t[..., :, 1:] - t[..., :, :-1], t[..., 1:, :] - t[..., :-1, :]


def _ft_grad_mag(t):
    dx, dy = _ft_diffs(t)
    return torch.sqrt(F.pad(dx, (0, 1)) ** 2 + F.pad(dy, (0, 0, 0, 1)) ** 2 + 1e-12)


def _ft_local_std(t, k=7):
    mu = F.avg_pool2d(t, k, stride=1, padding=k // 2, count_include_pad=False)
    sq = F.avg_pool2d(t * t, k, stride=1, padding=k // 2, count_include_pad=False)
    return torch.sqrt(torch.clamp(sq - mu * mu, min=1e-6))


def _ft_loss(recon, x, mu, logvar, cfg, kl_w):
    loss, mse, kld = stable_vae_loss(recon, x, mu, logvar, beta=cfg["beta"], kl_weight=kl_w)
    if cfg["lam_grad"]:
        (rx, ry), (xx, xy) = _ft_diffs(recon), _ft_diffs(x)
        loss = loss + cfg["lam_grad"] * (torch.sum((rx - xx) ** 2) + torch.sum((ry - xy) ** 2))
    if cfg["lam_con"]:
        loss = loss + cfg["lam_con"] * torch.sum((_ft_local_std(recon) - _ft_local_std(x)) ** 2)
    return loss


def _ft_inject(chip, rng):
    """Paste one sharp-edged artificial object (rectangle / ring / line) into a chip."""
    h, w = chip.shape
    out = chip.copy()
    size = int(rng.integers(14, 40))
    cy, cx = int(rng.integers(size, h - size)), int(rng.integers(size, w - size))
    delta = float(rng.uniform(0.15, 0.35) * rng.choice([-1, 1]))
    yy, xx = np.ogrid[:h, :w]
    kind = int(rng.integers(3))
    if kind == 0:                                # rectangle (lander deck, module)
        mask = (np.abs(yy - cy) <= size // 2) & (np.abs(xx - cx) <= size // 3)
    elif kind == 1:                              # ring (circular structure)
        mask = np.abs(np.hypot(yy - cy, xx - cx) - size / 2) <= 1.5
    else:                                        # straight line (track, cable)
        th = rng.uniform(0, np.pi)
        along = (xx - cx) * np.cos(th) + (yy - cy) * np.sin(th)
        across = -(xx - cx) * np.sin(th) + (yy - cy) * np.cos(th)
        mask = (np.abs(across) <= 1.0) & (np.abs(along) <= size)
    out[mask] = np.clip(out[mask] + delta, 0, 1)
    return out


def _ft_build_eval_set(val_paths, n_inject=3):
    """Clean val chips (label 0) + n_inject synthetic-object copies of each (label 1)."""
    rng = np.random.default_rng(_FT_SEED + 1)
    clean = [np.load(p).astype(np.float32) for p in val_paths]
    chips = clean + [_ft_inject(c, rng) for c in clean for _ in range(n_inject)]
    labels = np.array([0] * len(clean) + [1] * (len(chips) - len(clean)))
    scorer = NumpyAnomalyScorer([])
    edge = np.array([scorer.edge_regularity_score(c) for c in chips])  # image-only metric
    return np.stack(chips)[:, None], labels, edge


def _ft_train(cfg, epochs, train_paths, chip_size, device):
    """Fresh model trained `epochs` epochs; None if the loss went NaN."""
    torch.manual_seed(_FT_SEED)
    model = StableConvolutionalVAE(latent_dim=cfg["latent_dim"], input_size=chip_size).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    loader = DataLoader(TorchDataset(train_paths), batch_size=cfg["batch_size"],
                        shuffle=True, num_workers=0)
    for epoch in range(epochs):
        model.train()
        kl_w = min(1.0, (epoch + 1) / cfg["warmup_epochs"])
        for imgs, _ in loader:
            imgs = imgs.to(device)
            opt.zero_grad()
            recon, mu, logvar = model(imgs)
            loss = _ft_loss(recon, imgs, mu, logvar, cfg, kl_w)
            if torch.isnan(loss):
                return None
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    return model


def _ft_evaluate(model, eval_set, device, bs=8):
    """Held-out MSE on the clean chips + detection AUROC per metric and combined."""
    from sklearn.metrics import roc_auc_score
    fail = {"val_mse": float("inf"), "auc": 0.0, **{f"auc_{m}": 0.0 for m in _FT_METRICS}}
    if model is None:
        return fail
    chips, labels, edge = eval_set
    model.eval()
    per = {m: [] for m in ("mse", "gradient", "contrast")}
    sq_err = []
    with torch.no_grad():
        for i in range(0, len(chips), bs):
            x = torch.from_numpy(chips[i:i + bs]).to(device)
            mu, _ = model.encode(x)
            r = model.decode(mu)                           # deterministic reconstruction
            maps = {"mse":      (x - r) ** 2,
                    "gradient": (_ft_grad_mag(x) - _ft_grad_mag(r)).abs(),
                    "contrast": (_ft_local_std(x) - _ft_local_std(r)).abs()}
            for m, mp in maps.items():
                # 99th percentile, not the mean: a 20-px object barely moves a 256² mean
                per[m].append(torch.quantile(mp.flatten(1), 0.99, dim=1).cpu().numpy())
            sq_err.append(((x - r) ** 2).mean(dim=(1, 2, 3)).cpu().numpy())
    per = {m: np.concatenate(v) for m, v in per.items()}
    per["edge"] = edge
    val_mse = float(np.concatenate(sq_err)[labels == 0].mean())
    if not np.isfinite(val_mse):
        return fail
    clean = labels == 0
    z = [(per[m] - per[m][clean].mean()) / (per[m][clean].std() + 1e-8) for m in _FT_METRICS]
    out = {"val_mse": val_mse, "auc": float(roc_auc_score(labels, np.mean(z, axis=0)))}
    for m in _FT_METRICS:
        out[f"auc_{m}"] = float(roc_auc_score(labels, per[m]))
    return out


def _ft_rank_key(res):
    return (-res["auc"], res["val_mse"])          # best detection first, then best recon


def _ft_write_dashboard(record, path=Path("dashboard") / "finetune.json"):
    """Latest tuning run for the Mission Control page (dashboard/index.html, section 04)."""
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(record, indent=1, default=float))
    print(f"  wrote {path}")


def run_finetuning(image_dir="training data", chip_size=256, n_configs=9, eta=3,
                   rungs=(1, 3, 8), confirm=10, max_total=150, val_frac=0.15,
                   out_csv="results/mk17_finetune.csv",
                   out_model="data/models/mk17_finetuned.pt", live=None):
    if not HAS_TORCH:
        raise SystemExit("finetune: PyTorch is required")
    import csv
    device = _ft_device()
    t0 = time.time()
    Path("results").mkdir(exist_ok=True)
    knobs = ("latent_dim", "lr", "batch_size", "warmup_epochs", "beta", "lam_grad", "lam_con")

    def fmt(c):
        return (f"latent={c['latent_dim']:>2} lr={c['lr']:.2e} bs={c['batch_size']} "
                f"wu={c['warmup_epochs']} beta={c['beta']:.1e} "
                f"λg={c['lam_grad']:g} λc={c['lam_con']:g}")

    def fmt_res(r):
        return (f"AUROC {r['auc']:.3f} (mse {r['auc_mse']:.2f} grad {r['auc_gradient']:.2f} "
                f"con {r['auc_contrast']:.2f} edge {r['auc_edge']:.2f}) | val_mse {r['val_mse']:.5f}")

    with tempfile.TemporaryDirectory() as tmp:
        train_p, val_p = _ft_prepare_chips(image_dir, chip_size, tmp, max_total, val_frac)
        eval_set = _ft_build_eval_set(val_p)
        if live:
            live.samples([np.load(q) for q in train_p[:12]])
            live.log(f"{len(train_p)} train / {len(val_p)} val chips · {n_configs} configs · rungs {list(rungs)}")
        logger.info(f"finetune: {len(train_p)} train / {len(val_p)} val chips "
                    f"(+{int(eval_set[1].sum())} with synthetic objects) | device={device}")
        logger.info(f"random search + successive halving | {n_configs} configs, rungs {list(rungs)}")

        rng = np.random.default_rng(_FT_SEED)
        configs = [_ft_sample_cfg(rng) for _ in range(n_configs)]
        alive, history = list(range(n_configs)), []

        for r, ep in enumerate(rungs):
            scored = []
            if live:
                live.step(f"Rung {r} · {ep} epochs")
            for k, idx in enumerate(alive):
                if live:
                    live.progress(k, len(alive), "configs")
                res = _ft_evaluate(_ft_train(configs[idx], ep, train_p, chip_size, device),
                                   eval_set, device)
                scored.append((_ft_rank_key(res), idx, res))
                history.append({"rung": r, "epochs": ep, **res, **configs[idx]})
                logger.info(f"  r{r} cfg{idx:2d} [{ep}ep] {fmt(configs[idx])} -> {fmt_res(res)}")
                if live:
                    live.metric(rung=r, cfg=idx, epochs=ep, auroc=res["auc"], val_mse=res["val_mse"])
            scored.sort(key=lambda s: s[0])
            keep = max(1, len(alive) // eta)
            alive = [i for _, i, _ in scored[:keep]]
            logger.info(f"rung {r}: {len(scored)} configs @ {ep}ep -> keep {keep} "
                        f"(best AUROC {scored[0][2]['auc']:.3f})")

        best = configs[alive[0]]
        if live:
            live.step(f"Confirming best config · {confirm} epochs")
        best_model = _ft_train(best, confirm, train_p, chip_size, device)
        best_res = _ft_evaluate(best_model, eval_set, device)
        if best_model is not None:
            Path(out_model).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"config": best, "result": best_res, "chip_size": chip_size,
                        "state_dict": best_model.state_dict()}, out_model)
        dt = time.time() - t0

        print("\n" + "=" * 70)
        print("BEST CONFIG   (drop into config={...} of run_analysis / the web form)")
        print("=" * 70)
        for k in knobs:
            print(f"  {k:<14}: {best[k]}")
        print(f"  @{confirm} epochs    : {fmt_res(best_res)}")
        if best_model is not None:
            print(f"  weights       : {out_model}")
        spent = sum(len([h for h in history if h['rung'] == r]) * ep
                    for r, ep in enumerate(rungs))
        print(f"\n  epoch-units spent : {spent}   (wall {dt/60:.1f} min)")
        print(f"  a flat grid at {rungs[-1]} ep/config would test only "
              f"~{spent // rungs[-1]} configs for the same cost")

        deepest = {}
        for h in history:
            k = tuple(h[n] for n in knobs)
            if k not in deepest or h['rung'] >= deepest[k]['rung']:
                deepest[k] = h
        board = sorted(deepest.values(), key=_ft_rank_key)
        print("\n  rank  AUROC  mse   grad  con   edge  val_mse   latent lr       bs wu "
              "beta     λg  λc  rung")
        for i, h in enumerate(board, 1):
            print(f"  {i:>3}   {h['auc']:.3f}  {h['auc_mse']:.2f}  {h['auc_gradient']:.2f}  "
                  f"{h['auc_contrast']:.2f}  {h['auc_edge']:.2f}  {h['val_mse']:.5f}  "
                  f"{h['latent_dim']:>5}  {h['lr']:.1e}  {h['batch_size']:>2} "
                  f"{h['warmup_epochs']:>2} {h['beta']:.1e}  {h['lam_grad']:<3g} "
                  f"{h['lam_con']:<3g} {h['rung']}")

        fields = ["rung", "epochs", "auc", *(f"auc_{m}" for m in _FT_METRICS), "val_mse", *knobs]
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(history)
        print(f"\n  wrote {out_csv}")

        _ft_write_dashboard({
            "id": time.strftime("%Y-%m-%d %H:%M"), "script": Path(__file__).name,
            "data": str(image_dir), "device": device, "minutes": round(dt / 60, 1),
            "train_chips": len(train_p), "val_chips": len(val_p),
            "injected": int(eval_set[1].sum()), "rungs": list(rungs), "confirm": confirm,
            "configs": configs, "best": {"cfg": alive[0], "config": best, "result": best_res},
            "history": [{"cfg": configs.index({k: h[k] for k in knobs}), **h} for h in history],
        })

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(7, 4.5))
            for r in range(len(rungs)):
                pts = [h for h in history if h["rung"] == r]
                ax.scatter([p["lr"] for p in pts], [p["auc"] for p in pts],
                           s=[16 + p["latent_dim"] for p in pts], alpha=.7,
                           label=f"rung {r} ({rungs[r]} ep)")
            ax.scatter([best["lr"]], [best_res["auc"]], marker="*", s=340, c="#c0392b",
                       zorder=5, label="best")
            ax.set_xscale("log")
            ax.set_xlabel("learning rate")
            ax.set_ylabel("detection AUROC (synthetic objects)")
            ax.set_title("Mk17 finetuning — successive halving (point size = latent dim)")
            ax.legend(fontsize=8)
            ax.grid(alpha=.25, which="both")
            fig.tight_layout()
            fig.savefig("results/mk17_finetune.png", dpi=140)
            print("  wrote results/mk17_finetune.png")
        except Exception as e:
            print(f"  (plot skipped: {e})")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def _cli_arg(flag, default=None, cast=str):
    import sys
    return cast(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default


def _tune_main(live=None):
    run_finetuning(
        image_dir = _cli_arg("--data", "training data"),
        chip_size = _cli_arg("--chip-size", 256, int),
        n_configs = _cli_arg("--configs", 9, int),
        rungs     = tuple(int(x) for x in _cli_arg("--rungs", "1,3,8").split(",")),
        confirm   = _cli_arg("--confirm", 10, int),
        max_total = _cli_arg("--max-chips", 150, int),
        live      = live,
    )


if __name__ == "__main__":
    import sys

    if "--tune" in sys.argv or "--finetune" in sys.argv:
        from xenarch_live import LiveRun
        live = LiveRun(__file__, _cli_arg("--data", "training data"),
                       [f"Rung {r}" for r in range(len(_cli_arg("--rungs", "1,3,8").split(",")))] + ["Confirming best config"])
        try:
            _tune_main(live)
            live.done()
        except BaseException as e:
            live.fail(e)
            raise
        sys.exit(0)

    port = int(os.environ.get("PORT", 5000))
    host = os.environ.get("HOST", "0.0.0.0")
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    logger.info("=" * 60)
    logger.info("XENARCH Mk14 — Production Web Edition + finetuning")
    logger.info(f"  PyTorch  : {'enabled' if HAS_TORCH else 'DISABLED (numpy fallback)'}")
    logger.info(f"  Rasterio : {'enabled' if HAS_RASTERIO else 'disabled (Pillow fallback)'}")
    logger.info(f"  CORS     : {CORS_ORIGINS}")
    logger.info(f"  Serving  : http://{host}:{port}")
    logger.info("  Tune     : python xenarch_mk17_with_finetuning.py --tune")
    logger.info("=" * 60)
    app.run(host=host, port=port, debug=debug, threaded=True)
 