// Tides demo: precomputed model predictions + an in-browser angle quantum kernel.
// Data: data/tides.json, written by scripts/build_tide_demo.py.
"use strict";

const css = (name) => getComputedStyle(document.querySelector(".viz-root")).getPropertyValue(name).trim();
const EPOCH_S = Date.parse("2000-01-01T00:00:00Z") / 1000;
const WEEK_H = 7 * 24;
const fmtR2 = (v) => v.toFixed(3);

// Deterministic RNG so a "sample" is reproducible.
function mulberry32(seed) {
  return function () {
    seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// --- quantum kernel ------------------------------------------------------------------------
// Each hour -> phases phi_j = speed_j * hours (mod 2 pi). The angle-encoding fidelity kernel is
// prod_j cos^2((a_j - b_j)/2) = prod_j (1 + cos a_j cos b_j + sin a_j sin b_j) / 2, so we keep
// cos and sin of each phase and avoid trig in the inner loops.
function phaseFeatures(hours, speedsRad) {
  const n = hours.length, d = speedsRad.length;
  const c = new Float64Array(n * d), s = new Float64Array(n * d);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < d; j++) {
      const p = (hours[i] * speedsRad[j]) % (2 * Math.PI);
      c[i * d + j] = Math.cos(p);
      s[i * d + j] = Math.sin(p);
    }
  }
  return { c, s, n, d };
}

function kernel(A, i, B, k) {
  const d = A.d;
  let v = 1;
  for (let j = 0; j < d; j++) {
    v *= 0.5 * (1 + A.c[i * d + j] * B.c[k * d + j] + A.s[i * d + j] * B.s[k * d + j]);
  }
  return v;
}

// Kernel ridge regression: (K + alpha I) w = y - mean(y), via Cholesky.
function fitKRR(F, idx, y, alpha) {
  const n = idx.length;
  const ym = idx.reduce((a, i) => a + y[i], 0) / n;
  const L = new Float64Array(n * n);
  for (let a = 0; a < n; a++) {
    for (let b = 0; b <= a; b++) {
      L[a * n + b] = kernel(F, idx[a], F, idx[b]) + (a === b ? alpha : 0);
    }
  }
  for (let j = 0; j < n; j++) {               // in-place Cholesky (lower triangle)
    let sum = L[j * n + j];
    for (let k = 0; k < j; k++) sum -= L[j * n + k] * L[j * n + k];
    const djj = Math.sqrt(Math.max(sum, 1e-300));
    L[j * n + j] = djj;
    for (let i = j + 1; i < n; i++) {
      let s2 = L[i * n + j];
      for (let k = 0; k < j; k++) s2 -= L[i * n + k] * L[j * n + k];
      L[i * n + j] = s2 / djj;
    }
  }
  const z = new Float64Array(n);              // solve L z = y - ym, then L^T w = z
  for (let i = 0; i < n; i++) {
    let s3 = y[idx[i]] - ym;
    for (let k = 0; k < i; k++) s3 -= L[i * n + k] * z[k];
    z[i] = s3 / L[i * n + i];
  }
  const w = new Float64Array(n);
  for (let i = n - 1; i >= 0; i--) {
    let s4 = z[i];
    for (let k = i + 1; k < n; k++) s4 -= L[k * n + i] * w[k];
    w[i] = s4 / L[i * n + i];
  }
  return { idx, w, ym };
}

function predictKRR(model, Ftrain, Ftest) {
  const out = new Float64Array(Ftest.n);
  for (let t = 0; t < Ftest.n; t++) {
    let v = model.ym;
    for (let a = 0; a < model.idx.length; a++) v += model.w[a] * kernel(Ftest, t, Ftrain, model.idx[a]);
    out[t] = v;
  }
  return out;
}

function r2(y, p) {
  let m = 0;
  for (const v of y) m += v;
  m /= y.length;
  let ss = 0, sr = 0;
  for (let i = 0; i < y.length; i++) { ss += (y[i] - m) ** 2; sr += (y[i] - p[i]) ** 2; }
  return 1 - sr / ss;
}

// --- page ----------------------------------------------------------------------------------
const SERIES = [ // key, label, css color var, dashed, initially shown
  ["qkernel", "Quantum kernel", "--qkernel", false, true],
  ["vqc", "Trainable quantum circuit", "--vqc", false, false, "Quantum circuit"],
  ["gp", "Gaussian process", "--gp", false, false],
  ["utide", "UTide harmonic analysis", "--utide", true, true, "UTide"],
  ["noaa", "NOAA official prediction", "--noaa", false, false, "NOAA forecast"],
];

function tile(label, value, sub, color, dashed) {
  const sw = color ? `<span class="swatch${dashed ? " dash" : ""}" style="${dashed ? "border-color" : "background"}:${color}"></span>` : "";
  return `<div class="tile"><div class="label">${sw}${label}</div><div class="value">${value}</div>` +
    (sub ? `<div class="sub">${sub}</div>` : "") + `</div>`;
}

function axes() {
  const ink = css("--text-secondary"), grid = css("--grid");
  return [
    { stroke: ink, grid: { stroke: grid, width: 1 }, ticks: { stroke: grid } },
    { stroke: ink, grid: { stroke: grid, width: 1 }, ticks: { stroke: grid }, label: "water level (metres)",
      labelSize: 22, size: 52, values: (u, v) => v.map((x) => x.toFixed(1)) },
  ];
}

function makeChart(el, series, data) {
  const width = () => Math.max(300, el.clientWidth);
  const u = new uPlot({
    width: width(), height: 320,
    scales: { x: { time: true } },
    axes: axes(),
    cursor: { points: { size: 6 } },
    legend: { live: true },
    series: [{ label: "Time", value: "{MMM} {D}, {h}{aa}" }, ...series],
  }, data, el);
  new ResizeObserver(() => u.setSize({ width: width(), height: 320 })).observe(el);
  return u;
}

async function main() {
  const D = await (await fetch("data/tides.json")).json();
  const speeds = D.speeds_deg_per_hour.map((s) => (s * Math.PI) / 180);
  const T = D.test;
  const xs = T.hours.map((h) => EPOCH_S + h * 3600);
  const obs = Float64Array.from(T.observed);

  document.getElementById("repolink").href = D.repo || "#";

  // Section 1: tiles + chart of every model.
  document.getElementById("tiles").innerHTML = SERIES.map(([k, lab, c, dash, , short]) =>
    tile(short || lab, fmtR2(T.r2[k]), "score for 2023", css(c), dash)).join("");
  const s1 = [{ label: "Actual water level", stroke: css("--observed"), width: 1.5 }];
  const d1 = [xs, obs];
  for (const [k, lab, c, dash, show] of SERIES) {
    s1.push({ label: lab, stroke: css(c), width: 1.5, dash: dash ? [5, 4] : undefined, show });
    d1.push(T.predictions[k]);
  }
  const c1 = makeChart(document.getElementById("chart1"), s1, d1);

  // Section 2: live quantum kernel.
  const P = D.train_pool;
  const Fpool = phaseFeatures(P.hours, speeds);
  const Ftest = phaseFeatures(T.hours, speeds);
  const ypool = Float64Array.from(P.level);
  const alphaEl = document.getElementById("alpha");
  alphaEl.value = Math.log10(D.params.qkernel.alpha || 0.1).toFixed(2);
  let sampleSeed = 1;
  let live = new Float64Array(T.hours.length).fill(NaN);
  const c2 = makeChart(document.getElementById("chart2"), [
    { label: "Actual water level", stroke: css("--observed"), width: 1.5 },
    { label: "Your quantum model", stroke: css("--qkernel"), width: 1.8 },
  ], [xs, obs, live]);

  function train() {
    const n = +document.getElementById("ntrain").value;
    const alpha = 10 ** +alphaEl.value;
    document.getElementById("alphalabel").textContent = alpha < 0.01 ? alpha.toExponential(1) : alpha.toPrecision(2);
    const rng = mulberry32(sampleSeed * 7919 + n);
    const all = Array.from({ length: Fpool.n }, (_, i) => i);
    for (let i = all.length - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); [all[i], all[j]] = [all[j], all[i]]; }
    const idx = all.slice(0, n);
    document.getElementById("livetiles").innerHTML = tile("Training…", "–", `learning from ${n} hours`);
    setTimeout(() => {
      const t0 = performance.now();
      const model = fitKRR(Fpool, idx, ypool, alpha);
      live = predictKRR(model, Fpool, Ftest);
      const ms = performance.now() - t0;
      const score = r2(obs, live);
      document.getElementById("livetiles").innerHTML =
        tile("Your quantum model", fmtR2(score), "score for 2023", css("--qkernel")) +
        tile("Our tuned version", fmtR2(T.r2.qkernel), "1000 hours, best smoothing", css("--qkernel")) +
        tile("Hours learned from", n, `random pick #${sampleSeed}`) +
        tile("Time to train and predict", `${(ms / 1000).toFixed(2)} s`, `${T.hours.length.toLocaleString()} hours of 2023`);
      c2.setData([xs, obs, live], false);
      setWeek(+weekEl.value);
    }, 20);
  }
  document.getElementById("ntrain").addEventListener("change", train);
  alphaEl.addEventListener("change", train);
  alphaEl.addEventListener("input", () => {
    const a = 10 ** +alphaEl.value;
    document.getElementById("alphalabel").textContent = a < 0.01 ? a.toExponential(1) : a.toPrecision(2);
  });
  document.getElementById("resample").addEventListener("click", () => { sampleSeed++; train(); });

  // Shared week selector.
  const weekEl = document.getElementById("week");
  const t0h = T.hours[0];
  function setWeek(w) {
    const lo = EPOCH_S + (t0h + w * WEEK_H) * 3600, hi = lo + WEEK_H * 3600;
    for (const u of [c1, c2]) u.setScale("x", { min: lo, max: hi });
    document.getElementById("weeklabel").textContent =
      new Date(lo * 1000).toLocaleDateString("en-US", { month: "long", day: "numeric", year: "numeric", timeZone: "UTC" });
  }
  weekEl.addEventListener("input", () => setWeek(+weekEl.value));
  document.getElementById("prev").addEventListener("click", () => { weekEl.value = Math.max(0, +weekEl.value - 1); setWeek(+weekEl.value); });
  document.getElementById("next").addEventListener("click", () => { weekEl.value = Math.min(51, +weekEl.value + 1); setWeek(+weekEl.value); });
  setWeek(+weekEl.value);
  train();

  // Section 3: ceilings vs models (benchmark, n = 1000).
  const C = D.benchmark.ceilings, B = D.benchmark.curves;
  const rows = [
    ["Ceiling: every note the circuit can play", C["full, |w_j|<=1"], null, true],
    ["Ceiling: each rhythm on its own, no combinations", C["additive, |w|<=1"], null, true],
    "sep",
    ["Gaussian process", B.gp["1000"], "--gp"],
    ["UTide (oceanographers\u2019 method)", B.utide["1000"], "--utide"],
    ["Quantum kernel", B.qkernel["1000"], "--qkernel"],
    ["Trainable quantum circuit", B.vqc["1000"], "--vqc"],
  ];
  const lo = 0.80, hi = 0.95, pct = (v) => (100 * (v - lo)) / (hi - lo);
  let html = "";
  for (const r of rows) {
    if (r === "sep") { html += `<div class="sep"></div>`; continue; }
    const [name, v, c, ceiling] = r;
    const style = ceiling ? "" : `background:${css(c)}`;
    html += `<div class="name">${name}</div><div class="track"><div class="bar${ceiling ? " ceiling" : ""}" ` +
      `style="width:${pct(v).toFixed(1)}%;${style}" title="${name}: score ${fmtR2(v)}"></div></div><div class="val">${fmtR2(v)}</div>`;
  }
  html += `<div></div><div class="axis"><span>${lo.toFixed(2)}</span><span>score for 2023</span><span>${hi.toFixed(2)}</span></div><div></div>`;
  document.getElementById("bars").innerHTML = html;
}

main().catch((e) => {
  document.querySelector("main").insertAdjacentHTML("afterbegin",
    `<p class="note">Could not load data/tides.json (${e.message}). Serve this folder over HTTP, e.g. <code>python -m http.server</code>.</p>`);
});
