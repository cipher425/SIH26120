/* ThermoLift Twin dashboard - vanilla JS + ECharts */
const $ = (s) => document.querySelector(s);
const C = { accent: "#19c3b1", amber: "#f5a524", red: "#ef5350", green: "#43c46b", blue: "#4ea1ff",
  purple: "#a47cf3", muted: "#8b9bb0", line: "#243244", text: "#e6edf3", grid: "#1a2533", panel: "#172230", bg: "#0b1016" };
const state = { well: null, page: "overview", mode: "manual", playing: false, timer: null, busy: false,
  modeByDay: {}, cache: { css: {}, well: {} }, lastCard: null, meta: null };
const charts = {};
const CLS = { normal: ["Normal", "good"], fluid_pound: ["Fluid pound", "warn"], gas_interference: ["Gas interference", "warn"],
  rod_float: ["Rod floating", "bad"], pump_unseated: ["Pump unseated", "bad"], tv_leak: ["Travelling-valve leak", "warn"],
  sv_leak: ["Standing-valve leak", "warn"] };
const TITLES = {
  overview: ["Field Overview", "Reservoir → wellbore → surface, one integrated view"],
  live: ["Live Well Twin", "Closed-loop demo: the twin watches the well and drives the VFD"],
  css: ["CSS Optimizer", "Steam volume, pressure, soak & cut-off chosen with the pump in the loop"],
  srp: ["SRP Optimizer", "Stroke speed, SPM & VFD profile from reservoir temperature and viscosity"],
  steam: ["Steam Scheduler", "Allocate limited steam generators to the highest-value wells"],
  rel: ["Reliability & Alerts", "Predict rod failures, detect sensor & equipment anomalies"],
  models: ["AI Models", "Held-out performance of every model in the twin"],
  arch: ["Architecture", "How the digital twin is built"] };

// ------------------------------------------------------------ helpers
const fmt = (n, d = 0) => (n == null || Number.isNaN(n)) ? "—" :
  Number(n).toLocaleString("en-IN", { maximumFractionDigits: d, minimumFractionDigits: d });
const lakh = (n) => n == null ? "—" : `₹${fmt(n / 1e5, 2)} L`;
const pct = (a, b) => (b ? (a / b - 1) * 100 : 0);
async function get(url, opts) {
  const r = await fetch(url, opts);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}
function toast(msg, ms = 2500) {
  const t = $("#toast"); t.textContent = msg; t.style.display = "block";
  clearTimeout(t._h); t._h = setTimeout(() => (t.style.display = "none"), ms);
}
function ch(id) {
  if (!charts[id]) charts[id] = echarts.init(document.getElementById(id), null, { renderer: "canvas" });
  return charts[id];
}
const ax = (o = {}) => ({ axisLine: { lineStyle: { color: C.line } }, axisTick: { show: false },
  axisLabel: { color: C.muted, fontSize: 11 }, splitLine: { lineStyle: { color: C.grid } },
  nameTextStyle: { color: C.muted, fontSize: 11 }, ...o });
const base = (o = {}) => {
  const opt = { backgroundColor: "transparent", animationDuration: 300,
    textStyle: { color: C.text, fontFamily: "Segoe UI, Inter, sans-serif" },
    grid: { left: 12, right: 16, top: 36, bottom: 12, containLabel: true },
    tooltip: { trigger: "axis", backgroundColor: C.panel, borderColor: C.line, textStyle: { color: C.text, fontSize: 12 } },
    legend: { textStyle: { color: C.muted, fontSize: 11 }, top: 0, itemWidth: 14, itemHeight: 8 }, ...o };
  // keep legend swatches in sync with custom line colours
  (opt.series || []).forEach((s) => {
    if (s.lineStyle && s.lineStyle.color && !s.itemStyle) s.itemStyle = { color: s.lineStyle.color };
  });
  return opt;
};
function tColor(T) {
  const stops = [[30, [78, 161, 255]], [90, [25, 195, 177]], [150, [245, 165, 36]], [260, [239, 83, 80]]];
  T = Math.max(30, Math.min(260, T));
  for (let i = 0; i < stops.length - 1; i++) {
    const [t0, c0] = stops[i], [t1, c1] = stops[i + 1];
    if (T <= t1) { const f = (T - t0) / (t1 - t0); return `rgb(${c0.map((c, k) => Math.round(c + f * (c1[k] - c))).join(",")})`; }
  }
  return "rgb(239,83,80)";
}
function resizeAll() { setTimeout(() => Object.values(charts).forEach((c) => c.resize()), 60); }
window.addEventListener("resize", resizeAll);

// ------------------------------------------------------------ navigation
function showPage(p) {
  state.page = p;
  document.querySelectorAll(".page").forEach((s) => s.classList.toggle("active", s.id === `page-${p}`));
  document.querySelectorAll("#nav button").forEach((b) => b.classList.toggle("active", b.dataset.page === p));
  $("#pageTitle").textContent = TITLES[p][0];
  $("#pageSub").textContent = TITLES[p][1];
  if (p !== "live") stopPlay();
  const r = RENDER[p];
  if (r) r().catch((e) => { console.error(e); toast("Error: " + e.message.slice(0, 120), 5000); });
  resizeAll();
}
document.querySelectorAll("#nav button").forEach((b) => b.addEventListener("click", () => showPage(b.dataset.page)));

// ============================================================ OVERVIEW
async function renderOverview() {
  const d = state.cache.overview || (state.cache.overview = await get("/api/overview"));
  const k = d.kpi;
  $("#asof").textContent = "Data as of " + k.as_of;
  const tiles = [["Field oil", fmt(k.field_oil_bopd), "bopd", ""], ["cSOR · 12 m", fmt(k.csor_12m, 2), "t/m³", "warn"],
    ["Lift energy", fmt(k.kwh_per_bbl_30d, 2), "kWh/bbl", "warn"], ["Failures · 12 m", k.failures_12m, "rod/pump", "bad"],
    ["Fluid pound", fmt(k.fluid_pound_pct_90d, 0), "% days", "warn"], ["Rod floating", fmt(k.rod_float_pct_90d, 1), "% days", "warn"],
    ["Producing", k.wells_producing, `/ ${k.wells} wells`, "good"]];
  $("#kpis").innerHTML = tiles.map(([l, v, u, c]) => `<div class="kpi ${c}"><div class="l">${l}</div><div class="v">${v}<span class="u">${u}</span></div></div>`).join("");

  const tv = d.ablation.truth_validation;
  const A = tv.A_historical;
  const imp = (s) => [pct(s.oil_per_day, A.oil_per_day), pct(s.value_rate, A.value_rate), -pct(s.csor, A.csor),
    -pct(s.kwh_per_bbl, A.kwh_per_bbl), -pct(s.fail_per_year, A.fail_per_year)];
  const cats = ["Oil rate", "Net value", "cSOR reduction", "kWh/bbl reduction", "Failure reduction"];
  const ser = [["B · AI SRP only", tv.B_ai_srp_only, C.blue], ["C · AI CSS only", tv.C_ai_css_only, C.purple], ["D · Joint AI (ours)", tv.D_joint_ai, C.accent]];
  ch("chImpact").setOption(base({
    tooltip: { trigger: "axis", valueFormatter: (v) => fmt(v, 1) + " %", backgroundColor: C.panel, borderColor: C.line, textStyle: { color: C.text } },
    xAxis: ax({ type: "category", data: cats }), yAxis: ax({ type: "value", name: "% better", nameTextStyle: { color: C.muted, align: "left" } }),
    series: ser.map(([n, s, c]) => ({ name: n, type: "bar", data: imp(s).map((v) => +v.toFixed(1)), itemStyle: { color: c, borderRadius: [4, 4, 0, 0] },
      barGap: "15%", label: { show: n.startsWith("D"), position: "top", color: C.text, fontSize: 11, formatter: (p) => `${p.value > 0 ? "+" : ""}${p.value}%` } }))
  }), true);
  const D = tv.D_joint_ai;
  const rows = [["Oil rate", `${fmt(A.oil_per_day)} bopd`, `${fmt(D.oil_per_day)} bopd`, pct(D.oil_per_day, A.oil_per_day), true],
    ["cSOR", fmt(A.csor, 2), fmt(D.csor, 2), pct(D.csor, A.csor), false],
    ["Energy", `${fmt(A.kwh_per_bbl, 2)} kWh/bbl`, `${fmt(D.kwh_per_bbl, 2)} kWh/bbl`, pct(D.kwh_per_bbl, A.kwh_per_bbl), false],
    ["Rod/pump failures", `${fmt(A.fail_per_year, 1)} /yr`, `${fmt(D.fail_per_year, 1)} /yr`, pct(D.fail_per_year, A.fail_per_year), false],
    ["Net value", lakh(A.value_rate) + "/d", lakh(D.value_rate) + "/d", pct(D.value_rate, A.value_rate), true]];
  $("#impactTable").innerHTML = `<table><thead><tr><th>Metric</th><th class="num">Today</th><th class="num">Joint AI</th><th class="num">Δ</th></tr></thead><tbody>${
    rows.map(([m, a, b, dl, upGood]) => `<tr><td>${m}</td><td class="num">${a}</td><td class="num">${b}</td><td class="num ${(dl > 0) === upGood ? "up" : "dn"}">${dl > 0 ? "+" : ""}${fmt(dl, 1)}%</td></tr>`).join("")
  }</tbody></table><p class="hint" style="margin-top:12px">Twin-internal estimate: value ${
    fmt(pct(d.ablation.D_joint_ai.value_rate, d.ablation.A_historical.value_rate), 0)}% · ground-truth check: ${fmt(pct(D.value_rate, A.value_rate), 0)}%</p>`;

  const m = d.monthly;
  ch("chMonthly").setOption(base({
    grid: [{ left: 12, right: 16, top: 36, height: "52%", containLabel: true }, { left: 12, right: 16, top: "72%", bottom: 8, containLabel: true }],
    xAxis: [ax({ type: "category", data: m.map((r) => r.month), axisLabel: { show: false } }), ax({ type: "category", data: m.map((r) => r.month), gridIndex: 1 })],
    yAxis: [ax({ type: "value", name: "oil bopd" }), ax({ type: "value", name: "steam kt", splitLine: { show: false } }),
      ax({ type: "value", name: "failures", gridIndex: 1, minInterval: 1 })],
    tooltip: { trigger: "axis", backgroundColor: C.panel, borderColor: C.line, textStyle: { color: C.text } },
    axisPointer: { link: [{ xAxisIndex: "all" }] },
    series: [{ name: "Steam (kt/month)", type: "bar", yAxisIndex: 1, data: m.map((r) => +(r.steam / 1000).toFixed(2)), itemStyle: { color: "rgba(245,165,36,.35)" } },
      { name: "Oil (bopd)", type: "line", smooth: true, symbol: "none", data: m.map((r) => +r.bopd.toFixed(1)), lineStyle: { width: 2.5, color: C.accent }, areaStyle: { color: "rgba(25,195,177,.08)" } },
      { name: "Failures", type: "bar", xAxisIndex: 1, yAxisIndex: 2, data: m.map((r) => r.failures), itemStyle: { color: C.red } }]
  }), true);

  const phaseCol = { produce: C.green, inject: C.amber, soak: C.amber, down: C.red, idle: C.muted };
  ch("chMap").setOption(base({
    tooltip: { trigger: "item", backgroundColor: C.panel, borderColor: C.line, textStyle: { color: C.text },
      formatter: (p) => { const w = p.data.w; return `<b>${w.well_id}</b><br>${w.phase} · ${fmt(w.oil_bopd, 1)} bopd<br>risk ${fmt(w.risk_14d * 100, 1)}% · RFI ${fmt(w.RFI, 2)}`; } },
    xAxis: ax({ type: "value", scale: true, name: "lon", axisLabel: { show: false } }), yAxis: ax({ type: "value", scale: true, name: "lat", axisLabel: { show: false } }),
    series: [{ type: "scatter", data: d.wells.map((w) => ({ value: [w.lon, w.lat], w, symbolSize: 10 + w.oil_bopd / 2.5,
      itemStyle: { color: phaseCol[w.phase] || C.muted, borderColor: w.risk_14d > 0.15 ? C.red : "rgba(255,255,255,.3)", borderWidth: w.risk_14d > 0.15 ? 3 : 1, opacity: .9 } })),
      label: { show: true, formatter: (p) => p.data.w.well_id.slice(4), position: "right", color: C.muted, fontSize: 10 } }]
  }), true);
  ch("chMap").off("click");
  ch("chMap").on("click", (p) => openLive(p.data.w.well_id));

  $("#wellTable").innerHTML = `<table><thead><tr><th>Well</th><th>Phase</th><th class="num">Oil bopd</th><th class="num">SPM</th><th class="num">Cycle</th><th class="num">Avg cSOR</th><th class="num">RFI</th><th class="num">Risk 14 d</th><th class="num">Failures</th></tr></thead><tbody>${
    d.wells.map((w) => `<tr class="click" data-w="${w.well_id}"><td><b>${w.well_id}</b></td><td><span class="tag ${w.phase}">${w.phase}</span></td><td class="num">${fmt(w.oil_bopd, 1)}</td><td class="num">${fmt(w.spm, 2)}</td><td class="num">${w.cycle}</td><td class="num">${fmt(w.csor_avg, 2)}</td><td class="num">${fmt(w.RFI, 2)}</td><td class="num"><span class="tag ${w.risk_14d > 0.15 ? "hi" : w.risk_14d > 0.05 ? "md" : "lo"}">${fmt(w.risk_14d * 100, 1)}%</span></td><td class="num">${w.failures_total}</td></tr>`).join("")
  }</tbody></table>`;
  document.querySelectorAll("#wellTable tr.click").forEach((tr) => tr.addEventListener("click", () => openLive(tr.dataset.w)));
}
function openLive(w) { setWell(w); showPage("live"); }

// ============================================================ LIVE
async function liveReset() {
  stopPlay();
  state.modeByDay = {};
  state.lastCard = null;
  await get(`/api/live/${state.well}/reset?plan=${$("#planSel").value}`, { method: "POST" });
  await liveTick(1);
}
async function liveTick(n) {
  if (state.busy) return;
  state.busy = true;
  try {
    const o = await get(`/api/live/${state.well}/tick?n=${n}&mode=${state.mode}`);
    renderLive(o);
  } finally { state.busy = false; }
}
function startPlay() {
  state.playing = true; $("#playBtn").textContent = "❚❚ Pause";
  state.timer = setInterval(() => liveTick(+$("#speedSel").value), 800);
}
function stopPlay() {
  state.playing = false; $("#playBtn").textContent = "▶ Play";
  clearInterval(state.timer);
}
$("#playBtn").addEventListener("click", () => (state.playing ? stopPlay() : startPlay()));
$("#stepBtn").addEventListener("click", () => liveTick(1));
$("#ffBtn").addEventListener("click", () => liveTick(30));
$("#resetBtn").addEventListener("click", () => liveReset());
$("#planSel").addEventListener("change", () => liveReset());
document.querySelectorAll("#modeSwitch button").forEach((b) => b.addEventListener("click", () => {
  state.mode = b.dataset.mode;
  document.querySelectorAll("#modeSwitch button").forEach((x) => x.classList.toggle("on", x === b));
  toast(state.mode === "ai" ? "AI closed-loop control ON — twin now sets SPM & VFD profile daily" : "Manual control — operator practice (weekly, reactive)");
}));

function renderLive(o) {
  if (o.done) { stopPlay(); toast("Cycle finished — press ↺ New cycle"); return; }
  const r = o.record || {};
  (o.history || []).forEach((h) => { if (!(h.day_in_cycle in state.modeByDay)) state.modeByDay[h.day_in_cycle] = state.mode; });
  $("#dayNo").textContent = r.day_in_cycle ?? 0;
  const ph = $("#phaseTag"); ph.className = `tag ${r.phase}`; ph.textContent = r.phase;
  if (r.failure) toast(`⚠ ${r.failure === "rod_part" ? "Rod parted" : "Pump unseated"} on day ${r.day_in_cycle} — workover`, 4000);

  // dyno card
  const producing = r.phase === "produce" && o.card;
  if (producing) state.lastCard = o;
  const cd = state.lastCard;
  if (cd && r.phase !== "inject" && r.phase !== "soak") {
    const sev = (CLS[cd.diagnosis] || ["", "good"])[1];
    const col = sev === "bad" ? C.red : sev === "warn" ? C.amber : C.accent;
    const close = (xs, ys) => xs.map((x, i) => [x, ys[i]]).concat([[xs[0], ys[0]]]);
    ch("chCard").setOption(base({
      tooltip: { show: false },
      xAxis: ax({ type: "value", name: "Polished-rod position (m)", nameLocation: "middle", nameGap: 24, min: 0 }),
      yAxis: ax({ type: "value", name: "Load (kN)" }),
      series: [{ name: "Surface card", type: "line", data: close(cd.card.x, cd.card.y), symbol: "none", lineStyle: { width: 2.5, color: col }, areaStyle: { color: col, opacity: 0.08 },
        markLine: { silent: true, symbol: "none", data: [{ yAxis: 0 }], lineStyle: { color: C.red, type: "dotted" }, label: { formatter: "zero load", color: C.red, position: "insideEndTop" } } },
        { name: "Downhole card", type: "line", data: close(cd.card.px, cd.card.py), symbol: "none", lineStyle: { width: 1.5, type: "dashed", color: C.muted } }]
    }), true);
    const [nm, sv] = CLS[cd.diagnosis] || [cd.diagnosis, "good"];
    const dm = $("#diagMain"); dm.textContent = nm; dm.className = `diag-main ${sv}`;
    $("#diagProbs").innerHTML = Object.entries(cd.diag_probs).sort((a, b) => b[1] - a[1]).slice(0, 5)
      .map(([k, v]) => `<div class="probrow"><span>${CLS[k][0]}</span><div class="probbar"><i style="width:${v * 100}%;background:${CLS[k][1] === "bad" ? C.red : CLS[k][1] === "warn" ? C.amber : C.green}"></i></div><span>${fmt(v * 100, 1)}%</span></div>`).join("");
  } else {
    ch("chCard").setOption(base({ xAxis: ax({ type: "value", show: false }), yAxis: ax({ type: "value", show: false }), series: [],
      graphic: { type: "text", left: "center", top: "middle", style: { text: r.phase === "inject" ? "♨ Steam injection in progress — pump idle" : r.phase === "soak" ? "Soaking — heat spreading into the sand" : "Waiting for production", fill: C.muted, fontSize: 15 } } }), true);
    $("#diagMain").textContent = "—"; $("#diagMain").className = "diag-main"; $("#diagProbs").innerHTML = "";
  }

  // rod-float alert
  let al = "";
  if (r.phase === "produce") {
    if (r.RFI > 1) al = `<div class="alert bad">⛔<div><b>Rod floating now</b> — RFI ${fmt(r.RFI, 2)}. Rods fall slower than the polished rod: carrier-bar separation and impact loading.</div></div>`;
    else if (o.float_in_days != null) al = `<div class="alert warn">⏱<div><b>Rod float predicted in ${o.float_in_days} days</b> at current settings as the heated zone cools (twin look-ahead).</div></div>`;
    else al = `<div class="alert good">✓<div>No rod floating expected in the next 60 days at current settings.</div></div>`;
  }
  $("#floatAlert").innerHTML = al;

  drawSchematic(r, o.profile);

  const T = r.T_res, mu = r.mu_tubing_cp, rfi = r.RFI, fill = r.fillage, risk = o.risk_14d;
  gauge("gT", "Reservoir T (°C)", T, 40, 300, "°", [[0.35, C.blue], [0.65, C.amber], [1, C.red]]);
  gauge("gMu", "Tubing viscosity (cP)", mu, 0, 5000, "", [[0.3, C.green], [0.6, C.amber], [1, C.red]]);
  gauge("gRfi", "Rod-Float-Index", rfi, 0, 2, "", [[0.45, C.green], [0.5, C.amber], [1, C.red]], 2);
  gauge("gFill", "Pump fillage (%)", fill != null ? fill * 100 : null, 0, 100, "%", [[0.6, C.red], [0.8, C.amber], [1, C.green]]);
  gauge("gRisk", "Failure risk 14 d (%)", risk != null ? risk * 100 : null, 0, 40, "%", [[0.15, C.green], [0.4, C.amber], [1, C.red]], 1);

  // timeline
  const h = (o.history || []).filter((x) => x.phase !== "inject" && x.phase !== "soak");
  const aiRanges = [];
  let st = null;
  h.forEach((x, i) => {
    const isAi = state.modeByDay[x.day_in_cycle] === "ai";
    if (isAi && st == null) st = x.day_in_cycle;
    if ((!isAi || i === h.length - 1) && st != null) { aiRanges.push([{ xAxis: st }, { xAxis: x.day_in_cycle }]); st = null; }
  });
  const area = { silent: true, itemStyle: { color: "rgba(25,195,177,.08)" }, data: aiRanges };
  const fails = h.filter((x) => x.failure).map((x) => ({ coord: [x.day_in_cycle, 0], value: x.failure === "rod_part" ? "rod" : "unseat" }));
  ch("chTimeline").setOption(base({
    grid: [{ left: 12, right: 16, top: 30, height: "44%", containLabel: true }, { left: 12, right: 16, top: "64%", bottom: 8, containLabel: true }],
    axisPointer: { link: [{ xAxisIndex: "all" }] },
    xAxis: [ax({ type: "value", min: "dataMin", max: "dataMax", axisLabel: { show: false } }), ax({ type: "value", min: "dataMin", max: "dataMax", gridIndex: 1, name: "cycle day" })],
    yAxis: [ax({ type: "value", name: "oil bbl/d" }), ax({ type: "value", name: "T °C", splitLine: { show: false } }),
      ax({ type: "value", name: "SPM", gridIndex: 1 }), ax({ type: "value", name: "RFI", gridIndex: 1, splitLine: { show: false }, max: (v) => Math.max(1.4, v.max) })],
    series: [
      { name: "Oil", type: "line", data: h.map((x) => [x.day_in_cycle, x.oil_bbl]), symbol: "none", lineStyle: { color: C.green, width: 2 }, markArea: area,
        markPoint: { data: fails, symbol: "pin", symbolSize: 30, itemStyle: { color: C.red }, label: { fontSize: 8 } } },
      { name: "Reservoir T", type: "line", yAxisIndex: 1, data: h.map((x) => [x.day_in_cycle, x.T_res]), symbol: "none", lineStyle: { color: C.amber, width: 1.5 } },
      { name: "SPM", type: "line", step: "end", xAxisIndex: 1, yAxisIndex: 2, data: h.map((x) => [x.day_in_cycle, x.spm]), symbol: "none", lineStyle: { color: C.blue, width: 2 }, markArea: area },
      { name: "RFI", type: "line", xAxisIndex: 1, yAxisIndex: 3, data: h.map((x) => [x.day_in_cycle, x.RFI]), symbol: "none", lineStyle: { color: C.red, width: 1.5 },
        markLine: { silent: true, symbol: "none", data: [{ yAxis: 1 }], lineStyle: { color: C.red, type: "dashed" }, label: { formatter: "float", color: C.red } } }]
  }), true);

  // recommendation
  const rc = o.recommendation;
  if (rc && r.phase === "produce") {
    $("#recoBig").innerHTML = `<div>SPM<b>${fmt(rc.spm, 2)}</b></div><div>Down-stroke<b>${fmt(rc.down_frac * 100, 0)}%</b></div><div>Oil<b>${fmt(rc.oil_bbl, 1)}</b></div><div>RFI<b>${fmt(rc.RFI, 2)}</b></div><div>Energy<b>${fmt(rc.kwh, 0)} kWh</b></div>`;
    $("#recoReasons").innerHTML = (o.reasons || []).map((x) => `<li>${x}</li>`).join("") +
      (state.mode === "manual" ? `<li style="color:var(--muted)">Manual mode: operator ignores this until the weekly review. Switch to <b>AI closed-loop</b> to apply automatically.</li>` : `<li style="color:var(--accent)">Applied automatically via VFD set-points.</li>`);
  } else if (r.phase !== "produce") {
    $("#recoBig").innerHTML = ""; $("#recoReasons").innerHTML = `<li style="color:var(--muted)">Pump idle during ${r.phase}.</li>`;
  }
  // scoreboard
  const cm = o.cum || {};
  const row = (k) => cm[k] || { oil: 0, kwh: 0, days: 0, fail: 0, float: 0 };
  const M = row("manual"), A = row("ai");
  const cell = (s, f, d = 1) => (s.days ? fmt(f(s), d) : "—");
  $("#scoreboard").innerHTML = `<div class="h">Metric</div><div class="h mn">Manual</div><div class="h ai">AI</div>
    <div>Producing days</div><div class="mn">${M.days}</div><div class="ai">${A.days}</div>
    <div>Oil (bopd)</div><div class="mn">${cell(M, (s) => s.oil / s.days)}</div><div class="ai">${cell(A, (s) => s.oil / s.days)}</div>
    <div>Energy (kWh/bbl)</div><div class="mn">${cell(M, (s) => s.kwh / Math.max(s.oil, 1), 2)}</div><div class="ai">${cell(A, (s) => s.kwh / Math.max(s.oil, 1), 2)}</div>
    <div>Rod-float days</div><div class="mn">${M.float}</div><div class="ai">${A.float}</div>
    <div>Failures</div><div class="mn">${M.fail}</div><div class="ai">${A.fail}</div>`;
}

function gauge(id, name, val, min, max, unit, bands, dec = 0) {
  ch(id).setOption({ series: [{ type: "gauge", min, max, radius: "92%", center: ["50%", "58%"], startAngle: 210, endAngle: -30,
    axisLine: { lineStyle: { width: 9, color: bands } }, pointer: { width: 4, length: "58%", itemStyle: { color: C.text } },
    axisTick: { show: false }, splitLine: { show: false }, axisLabel: { show: false }, anchor: { show: true, size: 8, itemStyle: { color: C.text } },
    title: { offsetCenter: [0, "72%"], color: C.muted, fontSize: 11 },
    detail: { valueAnimation: true, offsetCenter: [0, "35%"], fontSize: 17, fontWeight: 700, color: C.text, formatter: (v) => (val == null ? "—" : fmt(val > max ? val : v, dec) + unit) },
    data: [{ value: val == null ? min : Math.min(max, val), name }] }] }, true);
}

function drawSchematic(r, prof) {
  const W = 200, top = 70, bot = 270;
  let stops = "";
  if (prof && prof.T) {
    const n = prof.T.length;
    stops = prof.T.map((T, i) => `<stop offset="${(i / (n - 1)) * 100}%" stop-color="${tColor(T)}"/>`).join("");
  } else {
    const c = r.phase === "inject" ? tColor(260) : tColor(r.T_res || 47);
    stops = `<stop offset="0%" stop-color="${c}"/><stop offset="100%" stop-color="${c}"/>`;
  }
  const Tr = r.T_res || 47;
  const rx = 18 + Math.max(0, (Tr - 47) / 220) * 70;
  const beamTilt = r.phase === "produce" ? (Date.now() / 300) % 2 < 1 ? -8 : 8 : 0;
  $("#schem").innerHTML = `<svg viewBox="0 0 ${W} 340" width="100%" height="100%">
    <defs><linearGradient id="tg" x1="0" y1="0" x2="0" y2="1">${stops}</linearGradient>
      <radialGradient id="hz"><stop offset="0%" stop-color="${tColor(Tr)}" stop-opacity=".95"/><stop offset="100%" stop-color="${tColor(Tr)}" stop-opacity="0"/></radialGradient></defs>
    <rect x="20" y="58" width="160" height="3" fill="${C.line}"/>
    <g transform="rotate(${beamTilt} 70 22)"><rect x="20" y="19" width="100" height="6" rx="2" fill="${C.muted}"/><path d="M120 14 q14 8 0 22 z" fill="${C.muted}"/></g>
    <polygon points="62,58 70,24 78,58" fill="none" stroke="${C.muted}" stroke-width="2"/>
    <line x1="126" y1="36" x2="100" y2="${top}" stroke="${C.muted}" stroke-width="1.2"/>
    <rect x="84" y="${top}" width="32" height="${bot - top + 18}" fill="none" stroke="${C.line}" stroke-width="2"/>
    <rect x="93" y="${top}" width="14" height="${bot - top}" fill="url(#tg)"/>
    <rect x="89" y="${bot}" width="22" height="16" rx="2" fill="${C.panel}" stroke="${C.muted}"/>
    <text x="100" y="${bot + 11}" fill="${C.muted}" font-size="8" text-anchor="middle">PUMP</text>
    <rect x="0" y="296" width="${W}" height="40" fill="rgba(160,120,70,.18)"/>
    <ellipse cx="100" cy="316" rx="${rx}" ry="16" fill="url(#hz)"/>
    <text x="4" y="332" fill="${C.muted}" font-size="9">Jodhpur Sst.</text>
    <text x="124" y="${top + 8}" fill="${C.text}" font-size="11">${r.T_wellhead != null ? fmt(r.T_wellhead, 0) + "°C" : ""}</text>
    <text x="124" y="${top + 20}" fill="${C.muted}" font-size="8">wellhead</text>
    <text x="124" y="${(top + bot) / 2}" fill="${C.text}" font-size="11">${r.mu_tubing_cp != null ? fmt(r.mu_tubing_cp, 0) + " cP" : ""}</text>
    <text x="124" y="${(top + bot) / 2 + 12}" fill="${C.muted}" font-size="8">${r.mu_tubing_cp != null ? "tubing μ" : ""}</text>
    <text x="124" y="${bot + 6}" fill="${C.text}" font-size="11">${r.T_pump != null ? fmt(r.T_pump, 0) + "°C" : ""}</text>
    <text x="196" y="318" fill="${C.text}" font-size="12" text-anchor="end" font-weight="700">${fmt(Tr, 0)}°C</text>
    <text x="196" y="330" fill="${C.muted}" font-size="8" text-anchor="end">heated zone</text>
    <text x="8" y="${top + 8}" fill="${C.muted}" font-size="8">0 m</text>
    <text x="8" y="${bot}" fill="${C.muted}" font-size="8">~${fmt((state.meta && state.meta.pump_depth) || 980)} m</text>
  </svg>`;
}

// ============================================================ CSS
const SC = [["A_historical", "A · Historical", "today's CSS + manual SRP"], ["B_ai_srp_only", "B · AI SRP only", "today's CSS + AI pump control"],
  ["C_ai_css_only", "C · AI CSS only", "AI steam plan + manual SRP"], ["D_joint_ai", "D · Joint AI", "AI steam plan + AI pump control"]];
async function renderCss() {
  const w = state.well;
  const [sc, wd] = await Promise.all([state.cache.css[w] ? state.cache.css[w] : get(`/api/css/${w}`), wellInfo(w)]);
  state.cache.css[w] = sc;
  const A = sc.scenarios.A_historical.summary;
  const vr = (s) => s.value_per_day;
  const bestK = SC.map((x) => x[0]).reduce((a, b) => (vr(sc.scenarios[b].summary) > vr(sc.scenarios[a].summary) ? b : a));
  $("#scenCards").innerHTML = SC.map(([k, t, sub]) => {
    const s = sc.scenarios[k].summary, c = sc.scenarios[k].css;
    const dv = pct(vr(s), vr(A));
    return `<div class="card ${k === bestK ? "best" : ""}"><h4>${t}</h4><div class="hint" style="margin:0">${sub}</div>
      <div class="v">${lakh(vr(s))}<span class="u">/day</span></div><div class="d ${dv >= 0 ? "up" : "dn"}">${k === "A_historical" ? "baseline" : (dv >= 0 ? "+" : "") + fmt(dv, 1) + "% vs A"}</div>
      <div class="kv"><span>Oil / day</span><b>${fmt(s.cum_oil_bbl / s.cycle_days, 1)} bopd</b><span>Oil / cycle</span><b>${fmt(s.cum_oil_bbl)} bbl</b>
      <span>Cycle length</span><b>${s.cycle_days} d</b><span>cSOR</span><b>${fmt(s.csor, 2)}</b><span>Energy</span><b>${fmt(s.kwh_per_bbl, 2)} kWh/bbl</b>
      <span>Exp. failures</span><b>${fmt(s.exp_failures, 2)}</b><span>Steam</span><b>${fmt(c.steam_t)} t · x=${fmt(c.quality, 2)}</b>
      <span>Inj. pressure</span><b>${fmt(c.inj_pressure_mpa, 2)} MPa</b><span>Soak · cut-off</span><b>${c.soak_days} d · ${fmt(c.cutoff_bopd, 1)} bopd</b></div></div>`;
  }).join("");

  const a = sc.scenarios.A_historical, d = sc.scenarios.D_joint_ai;
  ch("chCssCurve").setOption(base({
    xAxis: ax({ type: "value", name: "cycle day", nameLocation: "middle", nameGap: 22 }),
    yAxis: [ax({ type: "value", name: "oil bbl/d" }), ax({ type: "value", name: "margin ₹k/d", splitLine: { show: false } })],
    series: [
      { name: "Oil · historical", type: "line", data: a.curve.map((x) => [x.d, x.oil]), symbol: "none", lineStyle: { color: C.amber, width: 2 } },
      { name: "Oil · joint AI", type: "line", data: d.curve.map((x) => [x.d, x.oil]), symbol: "none", lineStyle: { color: C.accent, width: 2.5 },
        markLine: { symbol: "none", data: [{ xAxis: d.summary.cycle_days, label: { formatter: "AI cut-off → re-steam", color: C.accent } }], lineStyle: { color: C.accent, type: "dashed" } } },
      { name: "Daily margin · AI", type: "line", yAxisIndex: 1, data: d.curve.map((x) => [x.d, x.margin / 1000]), symbol: "none", lineStyle: { color: C.purple, width: 1.2, type: "dotted" },
        markLine: { symbol: "none", data: [{ yAxis: d.summary.value_per_day / 1000, label: { formatter: "cycle avg value rate", color: C.purple, position: "insideEndTop" } }], lineStyle: { color: C.purple } } }]
  }), true);

  const tr = sc.trials;
  const bi = tr.reduce((b, x, i) => (x.value_per_day > tr[b].value_per_day ? i : b), 0);
  ch("chTrials").setOption(base({
    tooltip: { trigger: "item", backgroundColor: C.panel, borderColor: C.line, textStyle: { color: C.text },
      formatter: (p) => { const x = tr[p.dataIndex] || tr[bi]; return `steam ${fmt(x.steam_t)} t · P ${fmt(x.inj_pressure_mpa, 1)} MPa<br>soak ${x.soak_days} d · cut-off ${fmt(x.cutoff_bopd, 1)}<br>value ${lakh(x.value_per_day)}/d`; } },
    xAxis: ax({ type: "value", name: "cSOR", nameLocation: "middle", nameGap: 22, scale: true }), yAxis: ax({ type: "value", name: "oil / day", scale: true }),
    visualMap: { min: Math.min(...tr.map((x) => x.value_per_day)), max: Math.max(...tr.map((x) => x.value_per_day)), dimension: 2, orient: "vertical", right: 0, top: "middle", itemHeight: 120,
      inRange: { color: ["#3b4a60", C.blue, C.accent] }, textStyle: { color: C.muted }, formatter: (v) => fmt(v / 1e5, 1) + "L", seriesIndex: 0 },
    grid: { left: 12, right: 70, top: 36, bottom: 12, containLabel: true },
    series: [{ type: "scatter", data: tr.map((x) => [x.csor, x.oil / x.cycle_days, x.value_per_day]), symbolSize: 9 },
      { type: "scatter", data: [[tr[bi].csor, tr[bi].oil / tr[bi].cycle_days, tr[bi].value_per_day]], symbol: "path://M50 0 L61 35 L98 35 L68 57 L79 91 L50 70 L21 91 L32 57 L2 35 L39 35 Z", symbolSize: 22, itemStyle: { color: C.amber } }]
  }), true);

  buildWhatIf(wd.historical_css);
  if (state.wiWell !== w) { state.wiWell = w; $("#wiBtn").click(); }
  const cy = wd.cycles;
  const nt = wd.calibration.n_train_cycles;
  ch("chCycles").setOption(base({
    xAxis: ax({ type: "category", data: cy.map((c) => "C" + c.cycle) }),
    yAxis: [ax({ type: "value", name: "cycle oil bbl" }), ax({ type: "value", name: "cSOR", splitLine: { show: false } })],
    series: [{ name: "Actual oil", type: "bar", data: cy.map((c) => c.cum_oil_bbl), itemStyle: { color: "rgba(78,161,255,.55)", borderRadius: [3, 3, 0, 0] },
      markArea: { silent: true, itemStyle: { color: "rgba(245,165,36,.07)" }, label: { color: C.amber, formatter: "out-of-sample" }, data: cy.length > nt ? [[{ xAxis: "C" + cy[nt].cycle }, { xAxis: "C" + cy[cy.length - 1].cycle }]] : [] } },
      { name: "Twin (history-matched)", type: "line", data: cy.map((c) => c.twin_oil), symbol: "circle", symbolSize: 5, lineStyle: { color: C.accent, width: 2 } },
      { name: "cSOR", type: "line", yAxisIndex: 1, data: cy.map((c) => c.csor), symbol: "none", lineStyle: { color: C.amber, type: "dotted" } }]
  }), true);
}
async function wellInfo(w) { return state.cache.well[w] || (state.cache.well[w] = await get(`/api/well/${w}`)); }

const WI = [["steam_t", "Steam volume (t)", 800, 3500, 50, 0], ["quality", "Steam quality", 0.6, 0.85, 0.01, 2],
  ["inj_pressure_mpa", "Inj. pressure (MPa)", 4, 8.5, 0.1, 1], ["soak_days", "Soak (days)", 1, 15, 1, 0], ["cutoff_bopd", "Cut-off (bopd)", 5, 25, 0.5, 1]];
function buildWhatIf(css) {
  $("#wiSliders").innerHTML = WI.map(([k, l, mn, mx, st, d]) => `<div class="slider"><span>${l}</span><input type="range" id="wi_${k}" min="${mn}" max="${mx}" step="${st}" value="${css[k]}"><output id="wo_${k}">${fmt(css[k], d)}</output></div>`).join("");
  WI.forEach(([k, , , , , d]) => $(`#wi_${k}`).addEventListener("input", (e) => ($(`#wo_${k}`).textContent = fmt(+e.target.value, d))));
}
$("#wiBtn").addEventListener("click", async () => {
  const body = { srp_mode: $("#wiMode").value };
  WI.forEach(([k]) => (body[k] = +$(`#wi_${k}`).value));
  $("#wiBtn").disabled = true;
  try {
    const r = await get(`/api/whatif/${state.well}`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    const s = r.summary;
    $("#wiOut").innerHTML = `<span>Cycle oil</span><b>${fmt(s.cum_oil_bbl)} bbl</b><span>Cycle length</span><b>${s.cycle_days} d</b><span>Oil / day</span><b>${fmt(s.cum_oil_bbl / s.cycle_days, 1)} bopd</b>
      <span>cSOR</span><b>${fmt(s.csor, 2)}</b><span>Energy</span><b>${fmt(s.kwh_per_bbl, 2)} kWh/bbl</b><span>Rod-float days</span><b>${s.float_days}</b>
      <span>Exp. failures</span><b>${fmt(s.exp_failures, 2)}</b><span>Heated radius</span><b>${fmt(s.heated_radius_m, 1)} m</b><span>M1 correction</span><b>×${fmt(s.m1_correction, 2)}</b>
      <span>Net value</span><b>${lakh(s.value_per_day)}/day</b>`;
    const c = r.curve;
    ch("chWhatIf").setOption(base({
      xAxis: ax({ type: "value", name: "cycle day", nameLocation: "middle", nameGap: 22 }),
      yAxis: [ax({ type: "value", name: "oil bbl/d · SPM×5" }), ax({ type: "value", name: "T °C", splitLine: { show: false } })],
      series: [{ name: "Oil", type: "line", data: c.map((x) => [x.d, x.oil]), symbol: "none", lineStyle: { color: C.green, width: 2 } },
        { name: "SPM ×5", type: "line", step: "end", data: c.map((x) => [x.d, x.spm * 5]), symbol: "none", lineStyle: { color: C.blue, width: 1.2 } },
        { name: "Reservoir T", type: "line", yAxisIndex: 1, data: c.map((x) => [x.d, x.T]), symbol: "none", lineStyle: { color: C.amber, width: 1.5 } }]
    }), true);
  } finally { $("#wiBtn").disabled = false; }
});
$("#optBtn").addEventListener("click", async () => {
  $("#optBtn").disabled = true; $("#optBtn").textContent = "Optimising…";
  try {
    state.cache.css[state.well] = await get(`/api/css/${state.well}/optimize?trials=${$("#trials").value}`, { method: "POST" });
    await renderCss(); toast("Optimisation complete");
  } finally { $("#optBtn").disabled = false; $("#optBtn").textContent = "Re-optimise"; }
});

// ============================================================ SRP
async function renderSrp() {
  const day = $("#srpDay").value, stroke = $("#srpStroke").value;
  const d = await get(`/api/srp/${state.well}?prod_day=${day}${stroke ? "&stroke=" + stroke : ""}`);
  const o = d.obs;
  const tiles = [["Reservoir T", fmt(o.T_avg, 0), "°C"], ["Tubing viscosity", fmt(o.mu_tubing_cp, 0), "cP"], ["Inflow", fmt(o.q_in, 0), "bbl/d"],
    ["Water cut", fmt(o.wc * 100, 0), "%"], ["Current SPM", fmt(d.current.spm, 2), ""], ["Current RFI", fmt(d.current.RFI, 2), ""]];
  $("#srpKpis").innerHTML = tiles.map(([l, v, u], i) => `<div class="kpi ${i === 5 && d.current.RFI > 1 ? "bad" : ""}"><div class="l">${l}</div><div class="v">${v}<span class="u">${u}</span></div></div>`).join("");
  const hm = d.heatmap;
  const xs = hm.spm.map(String), ys = hm.down_frac.map((v) => Math.round(v * 100) + "%");
  const val = [], rfi = [], flt = [];
  hm.value.forEach((row, j) => row.forEach((v, i) => { val.push([i, j, v]); rfi.push([i, j, hm.RFI[j][i]]); if (hm.RFI[j][i] > 1) flt.push([i, j]); }));
  const bi = hm.spm.findIndex((s) => Math.abs(s - d.best.spm) < 1e-6), bj = hm.down_frac.findIndex((s) => Math.abs(s - d.best.down_frac) < 1e-6);
  const heatBase = (vm, data, name) => base({
    tooltip: { trigger: "item", backgroundColor: C.panel, borderColor: C.line, textStyle: { color: C.text },
      formatter: (p) => p.seriesType === "heatmap" ? `SPM ${xs[p.value[0]]} · down ${ys[p.value[1]]}<br>${name}: ${fmt(p.value[2], 2)}<br>oil ${fmt(hm.oil[p.value[1]][p.value[0]], 1)} bopd · fill ${fmt(hm.fillage[p.value[1]][p.value[0]] * 100, 0)}%` : "" },
    grid: { left: 12, right: 16, top: 16, bottom: 58, containLabel: true },
    xAxis: ax({ type: "category", data: xs, name: "SPM", nameLocation: "middle", nameGap: 24, axisLabel: { color: C.muted, interval: 3 } }),
    yAxis: ax({ type: "category", data: ys, name: "down-stroke" }), visualMap: vm,
    series: [{ type: "heatmap", data, itemStyle: { borderColor: C.bg, borderWidth: 0.5 } },
      { type: "scatter", data: flt, symbol: "path://M0 0 L10 10 M10 0 L0 10", symbolSize: 7, itemStyle: { color: "rgba(255,255,255,.55)", borderColor: "rgba(255,255,255,.6)", borderWidth: 1.5 } },
      { type: "scatter", data: bi >= 0 && bj >= 0 ? [[bi, bj]] : [], symbolSize: 16, itemStyle: { color: "transparent", borderColor: "#fff", borderWidth: 3 } }]
  });
  const vv = hm.value.flat();
  const vmax = Math.max(...vv);
  ch("chHeat").setOption(heatBase({ min: Math.max(Math.min(...vv), vmax - Math.max(40, Math.abs(vmax) * 0.6)), max: vmax, orient: "horizontal", left: "center", bottom: 0, itemHeight: 200, calculable: true,
    inRange: { color: ["#2a1d3a", "#3b4a60", C.blue, C.accent, "#b8ffef"] }, textStyle: { color: C.muted }, seriesIndex: 0 }, val, "₹k/day"), true);
  ch("chRfi").setOption(heatBase({ type: "piecewise", pieces: [{ lt: 0.7, color: "#1f5c47", label: "< 0.7 safe" }, { gte: 0.7, lt: 0.9, color: "#7a6a1c", label: "0.7–0.9" },
      { gte: 0.9, lt: 1, color: "#a45a17", label: "0.9–1.0 margin" }, { gte: 1, color: "#8f2323", label: "≥ 1 floats" }],
    orient: "horizontal", left: "center", bottom: 0, textStyle: { color: C.muted }, seriesIndex: 0 }, rfi, "RFI"), true);

  const p = d.plan;
  ch("chPlan").setOption(base({
    grid: [{ left: 12, right: 16, top: 30, height: "40%", containLabel: true }, { left: 12, right: 16, top: "62%", bottom: 8, containLabel: true }],
    axisPointer: { link: [{ xAxisIndex: "all" }] },
    xAxis: [ax({ type: "value", min: "dataMin", max: "dataMax", axisLabel: { show: false } }), ax({ type: "value", min: "dataMin", max: "dataMax", gridIndex: 1, name: "prod. day" })],
    yAxis: [ax({ type: "value", name: "SPM" }), ax({ type: "value", name: "down %", min: 45, max: 70, splitLine: { show: false } }),
      ax({ type: "value", name: "T °C", gridIndex: 1 }), ax({ type: "value", name: "RFI", gridIndex: 1, max: 1.2, splitLine: { show: false } })],
    series: [{ name: "SPM", type: "line", step: "end", data: p.map((x) => [x.prod_day, x.spm]), symbol: "none", lineStyle: { color: C.blue, width: 2 } },
      { name: "Down-stroke %", type: "line", step: "end", yAxisIndex: 1, data: p.map((x) => [x.prod_day, x.down_frac * 100]), symbol: "none", lineStyle: { color: C.purple, width: 2 } },
      { name: "Reservoir T", type: "line", xAxisIndex: 1, yAxisIndex: 2, data: p.map((x) => [x.prod_day, x.T]), symbol: "none", lineStyle: { color: C.amber } },
      { name: "RFI", type: "line", xAxisIndex: 1, yAxisIndex: 3, data: p.map((x) => [x.prod_day, x.rfi]), symbol: "none", lineStyle: { color: C.red },
        markLine: { silent: true, symbol: "none", data: [{ yAxis: 1 }], lineStyle: { color: C.red, type: "dashed" } } }]
  }), true);
  const b = d.best, ba = d.best_any_stroke;
  $("#srpReco").innerHTML = `<div>SPM<b>${fmt(b.spm, 2)}</b></div><div>Down-stroke<b>${fmt(b.down_frac * 100, 0)}%</b></div><div>Oil<b>${fmt(b.oil_bbl, 1)}</b></div><div>kWh/bbl<b>${fmt(b.kwh / Math.max(b.oil_bbl, 0.1), 2)}</b></div><div>RFI<b>${fmt(b.RFI, 2)}</b></div>`;
  $("#srpReasons").innerHTML = d.reasons.map((x) => `<li>${x}</li>`).join("") +
    (Math.abs(ba.stroke_m - b.stroke_m) > 0.01 && ba.value_inr - b.value_inr > 1000 ?`<li>At next rod job consider a ${fmt(ba.stroke_m, 2)} m stroke: ${lakh(ba.value_inr - b.value_inr)}/day more value (longer, slower strokes suit viscous crude).</li>` : "");
  const pr = d.profile;
  ch("chProfile").setOption(base({ grid: { left: 12, right: 16, top: 26, bottom: 8, containLabel: true },
    xAxis: ax({ type: "value", name: "°C", scale: true }), yAxis: ax({ type: "value", inverse: true, name: "depth m" }),
    series: [{ name: "Fluid in tubing", type: "line", data: pr.z.map((z, i) => [pr.T[i], z]), symbol: "none", lineStyle: { color: C.amber, width: 2 } },
      { name: "Earth (geotherm)", type: "line", data: pr.z.map((z) => [30 + 0.017 * z, z]), symbol: "none", lineStyle: { color: C.muted, type: "dashed" } }] }), true);
}
$("#srpDay").addEventListener("input", (e) => ($("#srpDayOut").textContent = e.target.value));
$("#srpDay").addEventListener("change", () => renderSrp());
$("#srpBtn").addEventListener("click", () => renderSrp());
$("#srpStroke").addEventListener("change", () => renderSrp());

// ============================================================ STEAM
async function renderSteam() {
  const g = $("#genSel").value;
  const d = await get(`/api/steam_schedule?generators=${g}`);
  const rows = d.rows;
  const actCol = { "re-steam": C.accent, "continue producing": C.muted, "steaming now": C.amber };
  const sorted = [...rows].reverse();
  ch("chGain").setOption(base({
    tooltip: { trigger: "axis", axisPointer: { type: "shadow" }, backgroundColor: C.panel, borderColor: C.line, textStyle: { color: C.text }, valueFormatter: (v) => "₹" + fmt(v * 1000) + "/day" },
    xAxis: ax({ type: "value", name: "₹k/day" }), yAxis: ax({ type: "category", data: sorted.map((r) => r.well_id) }),
    series: [{ type: "bar", data: sorted.map((r) => ({ value: +(r.gain_inr_d / 1000).toFixed(1), itemStyle: { color: actCol[r.action], borderRadius: 3 } })) }]
  }), true);
  const jobs = rows.filter((r) => r.action === "re-steam");
  const gens = Array.from({ length: +g }, (_, i) => `SG-${i + 1}`);
  const t0 = jobs.length ? Math.min(...jobs.map((j) => +new Date(j.start))) : Date.now();
  ch("chGantt").setOption(base({
    tooltip: { trigger: "item", backgroundColor: C.panel, borderColor: C.line, textStyle: { color: C.text }, formatter: (p) => `${p.data.name}<br>${p.data.s} → ${p.data.e}` },
    grid: { left: 12, right: 20, top: 20, bottom: 12, containLabel: true },
    xAxis: ax({ type: "time", min: t0, axisLabel: { color: C.muted, formatter: "{dd} {MMM}" } }), yAxis: ax({ type: "category", data: gens }),
    series: [{ type: "custom", renderItem: (params, api) => {
      const cat = api.value(0), s = api.coord([api.value(1), cat]), e = api.coord([api.value(2), cat]);
      const h = api.size([0, 1])[1] * 0.55;
      return { type: "group", children: [
        { type: "rect", shape: { x: s[0], y: s[1] - h / 2, width: Math.max(2, e[0] - s[0] - 2), height: h, r: 4 }, style: { fill: C.accent, opacity: 0.85 } },
        { type: "text", style: { x: s[0] + 6, y: s[1], text: api.value(3), fill: "#04211d", fontSize: 11, fontWeight: 700, verticalAlign: "middle" } }] };
    }, encode: { x: [1, 2], y: 0 },
      data: jobs.map((j) => ({ name: j.well_id, s: j.start, e: j.end, value: [gens.indexOf(j.generator), +new Date(j.start), +new Date(j.end) + 864e5, j.well_id] })) }]
  }), true);
  $("#steamTable").innerHTML = `<table><thead><tr><th>Priority</th><th>Well</th><th>Phase</th><th class="num">Oil now</th><th class="num">Margin now ₹/d</th><th class="num">Next-cycle value ₹/d</th><th class="num">Gain ₹/d</th><th class="num">Steam t</th><th>Action</th><th>Generator</th><th>Window</th></tr></thead><tbody>${
    rows.map((r, i) => `<tr><td>${i + 1}</td><td><b>${r.well_id}</b></td><td><span class="tag ${r.phase}">${r.phase}</span></td><td class="num">${fmt(r.current_oil_bopd, 1)}</td><td class="num">${fmt(r.current_margin_inr_d)}</td><td class="num">${fmt(r.next_cycle_value_inr_d)}</td><td class="num ${r.gain_inr_d > 0 ? "up" : "dn"}">${fmt(r.gain_inr_d)}</td><td class="num">${fmt(r.steam_t)}</td><td>${r.action}</td><td>${r.generator || ""}</td><td>${r.start ? r.start + " → " + r.end : ""}</td></tr>`).join("")
  }</tbody></table>`;
}
$("#genSel").addEventListener("change", () => renderSteam());

// ============================================================ RELIABILITY
async function renderRel() {
  const [d, an] = await Promise.all([get("/api/reliability"), get("/api/anomalies")]);
  const rk = [...d.risk].reverse();
  ch("chRisk").setOption(base({
    tooltip: { trigger: "axis", axisPointer: { type: "shadow" }, backgroundColor: C.panel, borderColor: C.line, textStyle: { color: C.text },
      formatter: (p) => { const r = rk[p[0].dataIndex]; return `<b>${r.well_id}</b><br>risk ${fmt(r.risk_14d * 100, 1)}%<br>RFI ${r.RFI} · stress ${r.stress}<br>float days (30d) ${r.float_30d}<br>μ ${fmt(r.mu_tubing_cp)} cP · SPM ${r.spm}`; } },
    xAxis: ax({ type: "value", name: "% in 14 days" }), yAxis: ax({ type: "category", data: rk.map((r) => r.well_id) }),
    series: [{ type: "bar", data: rk.map((r) => ({ value: +(r.risk_14d * 100).toFixed(2), itemStyle: { color: r.risk_14d > 0.15 ? C.red : r.risk_14d > 0.05 ? C.amber : C.green, borderRadius: 3 } })) }]
  }), true);
  const imp = Object.entries(d.importance).sort((a, b) => a[1] - b[1]);
  ch("chImp").setOption(base({ tooltip: { trigger: "axis", axisPointer: { type: "shadow" }, backgroundColor: C.panel, borderColor: C.line, textStyle: { color: C.text } },
    xAxis: ax({ type: "value" }), yAxis: ax({ type: "category", data: imp.map((x) => x[0]) }),
    series: [{ type: "bar", data: imp.map((x) => +x[1].toFixed(4)), itemStyle: { color: C.purple, borderRadius: 3 } }] }), true);
  ch("chCause").setOption(base({ tooltip: { trigger: "item", backgroundColor: C.panel, borderColor: C.line, textStyle: { color: C.text } },
    legend: { bottom: 0, top: "auto", textStyle: { color: C.muted } },
    series: [{ type: "pie", radius: ["38%", "62%"], center: ["50%", "42%"], itemStyle: { borderColor: C.bg, borderWidth: 2 }, label: { color: C.text, formatter: "{d}%" },
      data: Object.entries(d.by_cause).map(([k, v], i) => ({ name: k, value: v, itemStyle: { color: [C.red, C.amber, C.blue, C.purple][i % 4] } })) }] }), true);
  const tn = { load_cell_drift: "Load-cell drift", tubing_leak: "Tubing leak", motor_fault: "Motor / drive fault", sensor_freeze: "Sensor freeze" };
  $("#anomTable").innerHTML = `<table><thead><tr><th>Date</th><th>Well</th><th>Detected fault</th><th class="num">Score</th><th class="num">kWh ratio</th><th class="num">Load ratio</th><th class="num">Liquid ratio</th></tr></thead><tbody>${
    an.rows.map((r) => `<tr><td>${r.date}</td><td><b>${r.well_id}</b></td><td><span class="tag md">${tn[r.type] || r.type}</span></td><td class="num">${fmt(r.score, 2)}</td><td class="num">${fmt(r.kwh_ratio, 2)}</td><td class="num">${fmt(r.pprl_ratio, 2)}</td><td class="num">${fmt(r.liquid_ratio, 2)}</td></tr>`).join("")}</tbody></table>`;
  $("#failTable").innerHTML = `<table><thead><tr><th>Date</th><th>Well</th><th>Cycle</th><th>Failure</th><th>Probable cause</th><th class="num">SPM</th><th class="num">Downtime d</th></tr></thead><tbody>${
    d.failures.map((r) => `<tr><td>${r.date}</td><td><b>${r.well_id}</b></td><td>${r.cycle}</td><td><span class="tag hi">${r.failure === "rod_part" ? "rod parted" : "pump unseated"}</span></td><td>${r.probable_cause}</td><td class="num">${fmt(r.spm, 2)}</td><td class="num">${r.downtime_days}</td></tr>`).join("")}</tbody></table>`;
}

// ============================================================ MODELS
async function renderModels() {
  const [m, fc] = await Promise.all([state.cache.models || get("/api/models"), get(`/api/well/${state.well}/forecast`)]);
  state.cache.models = m;
  const t = [["M1 hybrid reservoir", fmt(m.M1.mape_hybrid, 1) + "%", "cycle-oil MAPE (physics only " + fmt(m.M1.mape_calibrated_physics, 0) + "%)"],
    ["M2 hybrid LSTM", fmt(m.M2.mape_lstm, 1) + "%", "14-day oil MAPE (decline curve " + fmt(m.M2.mape_decline_curve, 1) + "%)"],
    ["M3 card CNN", fmt(m.M3.accuracy * 100, 1) + "%", "accuracy · 7 pump conditions"],
    ["M4 reliability", fmt(m.M4.roc_auc, 2), "ROC-AUC · oracle ceiling " + fmt(m.M4.oracle_roc_auc, 2)],
    ["M5 anomaly", fmt(m.M5.f1, 2), "F1 · fault-type accuracy " + fmt(m.M5.type_accuracy_on_detected * 100, 0) + "%"]];
  $("#modelKpis").innerHTML = t.map(([l, v, s]) => `<div class="kpi good"><div class="l">${l}</div><div class="v">${v}</div><div class="metric-sub">${s}</div></div>`).join("");
  ch("chM1").setOption(base({ tooltip: { trigger: "axis", backgroundColor: C.panel, borderColor: C.line, textStyle: { color: C.text }, valueFormatter: (v) => fmt(v, 1) + "%" },
    xAxis: ax({ type: "category", data: ["Physics (nominal)", "Physics (history-matched)", "Physics + ML (M1)"] }), yAxis: ax({ type: "value", name: "MAPE %" }),
    series: [{ type: "bar", barWidth: "45%", data: [m.M1.mape_nominal_physics, m.M1.mape_calibrated_physics, m.M1.mape_hybrid].map((v, i) => ({ value: +v.toFixed(1), itemStyle: { color: [C.muted, C.blue, C.accent][i], borderRadius: [4, 4, 0, 0] } })),
      label: { show: true, position: "top", color: C.text, formatter: "{c}%" } }] }), true);
  const s = m.M1.scatter;
  const mx = Math.max(...s.map((x) => Math.max(x.actual, x.physics, x.hybrid)));
  ch("chM1s").setOption(base({ tooltip: { trigger: "item", backgroundColor: C.panel, borderColor: C.line, textStyle: { color: C.text } },
    xAxis: ax({ type: "value", name: "actual bbl", nameLocation: "middle", nameGap: 22, max: Math.ceil(mx) }), yAxis: ax({ type: "value", name: "predicted bbl", max: Math.ceil(mx) }),
    series: [{ name: "Physics only", type: "scatter", data: s.map((x) => [x.actual, x.physics]), symbolSize: 6, itemStyle: { color: C.blue, opacity: 0.6 } },
      { name: "Hybrid (M1)", type: "scatter", data: s.map((x) => [x.actual, x.hybrid]), symbolSize: 7, itemStyle: { color: C.accent } },
      { name: "Perfect", type: "line", data: [[0, 0], [mx, mx]], symbol: "none", lineStyle: { color: C.muted, type: "dashed" } }] }), true);
  const hz = m.M2.mae_by_horizon_lstm.map((_, i) => "d+" + (i + 1));
  ch("chM2").setOption(base({ xAxis: ax({ type: "category", data: hz }), yAxis: ax({ type: "value", name: "MAE bbl/d" }),
    series: [{ name: "Decline curve", type: "line", data: m.M2.mae_by_horizon_decline, lineStyle: { color: C.amber }, itemStyle: { color: C.amber } },
      { name: "Hybrid LSTM", type: "line", data: m.M2.mae_by_horizon_lstm, lineStyle: { color: C.accent, width: 2.5 }, itemStyle: { color: C.accent } }] }), true);
  const hist = fc.history.slice(-60);
  const fx = fc.forecast.map((v, i) => [fc.start_day + i, v]);
  ch("chM2f").setOption(base({ xAxis: ax({ type: "value", name: "prod. day", nameLocation: "middle", nameGap: 22, min: "dataMin" }), yAxis: ax({ type: "value", name: "oil bbl/d", scale: true }),
    series: [{ name: "Observed", type: "line", data: hist.map((x) => [x.d, x.oil]), symbol: "none", lineStyle: { color: C.muted } },
      { name: "Decline curve", type: "line", data: fc.decline.map((v, i) => [fc.start_day + i, v]), symbol: "none", lineStyle: { color: C.amber, type: "dashed" } },
      { name: "Hybrid LSTM", type: "line", data: fx, symbol: "circle", symbolSize: 4, lineStyle: { color: C.accent, width: 2.5 } }] }), true);
  const cm = m.M3.confusion_matrix, cls = m.M3.classes.map((c) => CLS[c][0]);
  const data = [];
  cm.forEach((row, i) => row.forEach((v, j) => data.push([j, i, v])));
  ch("chM3").setOption(base({ tooltip: { trigger: "item", backgroundColor: C.panel, borderColor: C.line, textStyle: { color: C.text }, formatter: (p) => `true ${cls[p.value[1]]}<br>pred ${cls[p.value[0]]}: ${p.value[2]}` },
    grid: { left: 12, right: 16, top: 16, bottom: 12, containLabel: true },
    xAxis: ax({ type: "category", data: cls, name: "predicted", axisLabel: { color: C.muted, rotate: 30, fontSize: 10 } }), yAxis: ax({ type: "category", data: cls, inverse: true, axisLabel: { color: C.muted, fontSize: 10 } }),
    visualMap: { show: false, min: 0, max: Math.max(...data.map((x) => x[2])), inRange: { color: ["#121a23", "#1d4a52", C.accent] } },
    series: [{ type: "heatmap", data, label: { show: true, color: C.text, fontSize: 11 }, itemStyle: { borderColor: C.bg, borderWidth: 1 } }] }), true);
  const mm = [["M4 ROC-AUC", m.M4.roc_auc, C.purple], ["M4 oracle ceiling", m.M4.oracle_roc_auc, C.muted], ["M5 precision", m.M5.precision, C.accent],
    ["M5 recall", m.M5.recall, C.accent], ["M5 F1", m.M5.f1, C.accent], ["M5 type accuracy", m.M5.type_accuracy_on_detected, C.blue]];
  ch("chM45").setOption(base({ tooltip: { trigger: "axis", backgroundColor: C.panel, borderColor: C.line, textStyle: { color: C.text } },
    xAxis: ax({ type: "category", data: mm.map((x) => x[0]), axisLabel: { color: C.muted, rotate: 20, fontSize: 10 } }), yAxis: ax({ type: "value", max: 1 }),
    series: [{ type: "bar", barWidth: "50%", data: mm.map((x) => ({ value: +x[1].toFixed(3), itemStyle: { color: x[2], borderRadius: [4, 4, 0, 0] } })), label: { show: true, position: "top", color: C.text } }],
    graphic: { type: "text", right: 14, top: 6, style: { text: `M4 top-10% lift ×${fmt(m.M4.lift_top10pct, 1)}`, fill: C.muted, fontSize: 11 } } }), true);
}

// ============================================================ ARCHITECTURE
function renderArch() {
  const box = (x, y, w, h, title, lines, col) => `<g><rect x="${x}" y="${y}" width="${w}" height="${h}" rx="10" fill="${C.panel}" stroke="${col}" stroke-width="1.5"/>
    <text x="${x + 12}" y="${y + 22}" fill="${col}" font-size="13" font-weight="700">${title}</text>
    ${lines.map((l, i) => `<text x="${x + 12}" y="${y + 42 + i * 16}" fill="${C.text}" font-size="11.5">${l}</text>`).join("")}</g>`;
  const arrow = (x1, y1, x2, y2, col = C.muted) => `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${col}" stroke-width="1.6" marker-end="url(#ah)"/>`;
  $("#archCard").innerHTML = `<h3>ThermoLift Twin — well-to-surface digital twin</h3><p class="hint">Physics gives structure, ML learns what physics misses, optimisers act, the field feeds back.</p>
  <svg viewBox="0 0 1200 560"><defs><marker id="ah" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8 z" fill="${C.muted}"/></marker></defs>
  ${box(10, 30, 210, 190, "FIELD DATA", ["SCADA / VFD (SPM, kWh)", "Load cell → dyno cards", "CSS logs (steam, P, soak)", "Production tests", "PVT & completion data", "Failure / workover logs"], C.blue)}
  ${box(260, 30, 200, 190, "DATA LAYER", ["Time-series historian", "Cleaning & alignment", "Physics features:", "  tubing μ, RFI, Goodman", "  residual z-scores", "FastAPI services"], C.muted)}
  ${box(500, 30, 330, 190, "PHYSICS TWIN (history-matched)", ["Reservoir: Marx-Langenheim heating,", "  Boberg-Lantz cooling & inflow", "Wellbore: Ramey heat loss → μ(T) profile", "Rod pump: viscous drag, rod float,", "  Goodman stress, pump fill, energy", "Calibrated per well (k, heat efficiency)"], C.amber)}
  ${box(870, 30, 320, 190, "AI LAYER", ["M1 hybrid reservoir (physics × GBM)", "M2 hybrid LSTM production forecast", "M3 1-D CNN dyno-card diagnosis", "M4 failure-risk model (14 days)", "M5 anomaly detection + fault type", ""], C.purple)}
  ${box(500, 290, 330, 150, "OPTIMISERS", ["CSS: Bayesian opt. (Optuna TPE) with", "  pump controller inside every trial", "SRP: model-predictive SPM + VFD profile", "Field steam scheduler (generators)"], C.accent)}
  ${box(870, 290, 320, 150, "DECISIONS", ["VFD set-points (SPM, down-stroke)", "Next-cycle steam plan & cut-off", "Rod-float & failure early warnings", "Explained in plain language"], C.green)}
  ${box(10, 290, 450, 150, "OPERATOR DASHBOARD", ["Live twin · CSS & SRP optimisers · steam schedule", "Reliability & alerts · model performance", "Advisory mode first → closed loop after trust"], C.text)}
  ${arrow(220, 125, 258, 125)}${arrow(460, 125, 498, 125)}${arrow(830, 125, 868, 125)}
  ${arrow(665, 220, 665, 288)}${arrow(1030, 220, 900, 288)}${arrow(830, 365, 868, 365)}${arrow(498, 365, 462, 365)}
  <path d="M1030 440 L1030 520 L2 520 L2 125 L8 125" fill="none" stroke="${C.green}" stroke-width="1.6" stroke-dasharray="6 4" marker-end="url(#ah)"/>
  <text x="560" y="540" fill="${C.green}" font-size="12">closed loop: new data re-calibrates the twin & retrains models</text>
  </svg>`;
}

// ============================================================ boot
const RENDER = { overview: renderOverview, live: async () => { if (!state.liveInit || state.liveInit !== state.well) { state.liveInit = state.well; await liveReset(); } },
  css: renderCss, srp: renderSrp, steam: renderSteam, rel: renderRel, models: renderModels, arch: async () => renderArch() };
function setWell(w) {
  state.well = w; $("#wellSel").value = w;
  wellInfo(w).then((d) => { state.meta = { ...(state.meta || {}), pump_depth: d.props.pump_depth_m }; }).catch(() => {});
}
$("#wellSel").addEventListener("change", (e) => { setWell(e.target.value); showPage(state.page); });
(async function boot() {
  const meta = await get("/api/meta");
  $("#asof").textContent = "Data as of " + meta.as_of;
  $("#wellSel").innerHTML = meta.wells.map((w) => `<option>${w}</option>`).join("");
  // deep links: #page  or  #live/ff=90  or  #live/ff=90/ai
  const [pg, ...opts] = location.hash.replace("#", "").split("/");
  setWell(meta.demo_well);
  if (pg && TITLES[pg]) {
    if (opts.includes("ai")) document.querySelector('#modeSwitch button[data-mode="ai"]').click();
    showPage(pg);
    const ff = opts.find((o) => o.startsWith("ff="));
    if (pg === "live" && ff) {
      await new Promise((r) => setTimeout(r, 400));
      let n = +ff.slice(3);
      while (n > 0) { const k = Math.min(n, 30); state.busy = false; await liveTick(k); n -= k; }
    }
  } else showPage("overview");
})();
