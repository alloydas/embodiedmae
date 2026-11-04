// Interactive partial-cloud viewer (1 vs 3 vs 10 cameras / camera 00 vs cameras 00-02 vs complete cloud).
// Embedded inline by figures/make_sorghum_ledger.py and figures/update_maize_d2_page_nova.py.
// Data: D.pcint from figures/make_pcview_gallery.py (reports/vis_pcview_<species>/interactive.json):
//   {species, ref, cond_order, cond_names, epoch, tau, plants:[{plant, ref(b64 int16 xyz*1e4), cat(b64 u8),
//     cond:{<c>:{input, pred (b64 int16), miss (b64 u8, distance*1000), cov, chamfer, missed_frac}}}]}
// Needs three.js r128 (global THREE) and an element #pcint.
(function () {
  const host = document.getElementById('pcint');
  if (!host || typeof D === 'undefined' || !D.pcint) return;
  const P = D.pcint;
  const h = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  if (typeof THREE === 'undefined') { host.innerHTML = '<p class="small muted">3D viewer needs WebGL (three.js did not load).</p>'; return; }
  const b64 = s => { const bin = atob(s), u = new Uint8Array(bin.length); for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i); return u; };
  const xyz = s => { const u = b64(s), a = new Int16Array(u.buffer, u.byteOffset, u.byteLength >> 1), f = new Float32Array(a.length); for (let i = 0; i < a.length; i++) f[i] = a[i] / 1e4; return f; };
  const cache = {};
  const plantData = i => cache[i] || (cache[i] = (() => {
    const p = P.plants[i], o = { ref: xyz(p.ref), cat: b64(p.cat), cond: {} };
    P.cond_order.forEach(c => { const q = p.cond[c]; o.cond[c] = { input: xyz(q.input), pred: xyz(q.pred), miss: b64(q.miss) }; });
    return o; })());

  const sp = P.species, isM = sp === 'maize';
  const short = { '1view': isM ? '1 camera' : 'camera 00', '3view': isM ? '3 cameras' : 'cameras 00-02', 'full': isM ? '10 cameras' : 'complete cloud' };
  const catLab = isM ? ['in camera 00’s cloud', 'added by cameras 01-02', 'only cameras 03-09'] : ['seen by camera 00', 'only camera 01 or 02', 'hidden from all three'];
  const CATC = [[0.20, 0.55, 0.30], [0.80, 0.58, 0.10], [0.82, 0.25, 0.20]];
  const IN_C = [0.20, 0.55, 0.30], PR_C = [0.78, 0.55, 0.08];
  // miss ramp: near = light yellow, far (>= 0.08) = dark purple
  const RAMP = [[0.99, 0.95, 0.62], [0.98, 0.65, 0.25], [0.85, 0.30, 0.35], [0.45, 0.12, 0.45], [0.12, 0.05, 0.20]];
  const ramp = t => { t = Math.max(0, Math.min(1, t)) * (RAMP.length - 1); const i = Math.min(RAMP.length - 2, Math.floor(t)), f = t - i, a = RAMP[i], b = RAMP[i + 1]; return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f]; };

  host.innerHTML = `
    <div class="pcint-ctl">
      <div class="toggle" role="group" aria-label="Plant" id="pcint_plant"></div>
      <div class="toggle" role="group" aria-label="Input cameras" id="pcint_cond"></div>
      <div class="toggle" role="group" aria-label="Reference colouring" id="pcint_mode">
        <button data-m="miss" aria-pressed="true">reference: miss map</button><button data-m="cov" aria-pressed="false">reference: coverage</button><button data-m="plain" aria-pressed="false">reference: plain</button></div>
      <label class="small" style="margin-left:6px"><input type="checkbox" id="pcint_overlay"> overlay the reference (grey) on the input and reconstruction</label>
    </div>
    <div class="pcint-grid">
      <figure><canvas id="pcint_c0"></canvas><figcaption id="pcint_t0"></figcaption></figure>
      <figure><canvas id="pcint_c1"></canvas><figcaption id="pcint_t1"></figcaption></figure>
      <figure><canvas id="pcint_c2"></canvas><figcaption id="pcint_t2"></figcaption></figure>
    </div>
    <div class="small muted" id="pcint_legend" style="margin-top:6px"></div>
    <p class="small muted">Drag to rotate (all three panels move together), scroll to zoom, double-click to reset. Plants are sorted by single-camera coverage.</p>`;

  const st = { plant: 0, cond: '1view', mode: 'miss', overlay: false, theta: -0.6, phi: 0.25, dist: 2.6 };
  const rigs = [0, 1, 2].map(k => {
    const cv = document.getElementById('pcint_c' + k);
    const r = new THREE.WebGLRenderer({ canvas: cv, antialias: true, alpha: true });
    r.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    const s = new THREE.Scene(), cam = new THREE.PerspectiveCamera(40, 1, 0.01, 100);
    const g = new THREE.BufferGeometry(), m = new THREE.PointsMaterial({ size: 0.012, vertexColors: true, sizeAttenuation: true });
    s.add(new THREE.Points(g, m));
    return { cv, r, s, cam, g };
  });
  let center = [0, 0, 0], scale = 1;
  function setGeom(rg, parts) {
    let n = 0; parts.forEach(p => n += p.xyz.length);
    const pos = new Float32Array(n), col = new Float32Array(n); let o = 0;
    parts.forEach(p => { pos.set(p.xyz, o); for (let i = 0; i < p.xyz.length / 3; i++) { const c = p.color(i); col[o + 3 * i] = c[0]; col[o + 3 * i + 1] = c[1]; col[o + 3 * i + 2] = c[2]; } o += p.xyz.length; });
    rg.g.setAttribute('position', new THREE.BufferAttribute(pos, 3)); rg.g.setAttribute('color', new THREE.BufferAttribute(col, 3));
    rg.g.computeBoundingSphere();
  }
  function render() {
    const cx = center[0] + st.dist * scale * Math.cos(st.phi) * Math.sin(st.theta), cy = center[1] + st.dist * scale * Math.sin(st.phi), cz = center[2] + st.dist * scale * Math.cos(st.phi) * Math.cos(st.theta);
    rigs.forEach(rg => {
      const w = rg.cv.clientWidth || 300, hh = rg.cv.clientHeight || 300;
      rg.r.setSize(w, hh, false); rg.cam.aspect = w / hh; rg.cam.updateProjectionMatrix();
      rg.cam.position.set(cx, cy, cz); rg.cam.lookAt(center[0], center[1], center[2]); rg.r.render(rg.s, rg.cam);
    });
  }
  function update() {
    const d = plantData(st.plant), p = P.plants[st.plant], q = p.cond[st.cond], dc = d.cond[st.cond];
    const R = d.ref; let mn = [1e9, 1e9, 1e9], mx = [-1e9, -1e9, -1e9];
    for (let i = 0; i < R.length; i += 3) for (let k = 0; k < 3; k++) { mn[k] = Math.min(mn[k], R[i + k]); mx[k] = Math.max(mx[k], R[i + k]); }
    center = [0, 1, 2].map(k => (mn[k] + mx[k]) / 2); scale = Math.max(mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2]) / 2 || 1;
    const grey = () => [0.62, 0.64, 0.62];
    const ov = st.overlay ? [{ xyz: R, color: grey }] : [];
    setGeom(rigs[0], ov.concat([{ xyz: dc.input, color: () => IN_C }]));
    setGeom(rigs[1], ov.concat([{ xyz: dc.pred, color: () => PR_C }]));
    const refColor = st.mode === 'cov' ? (i => CATC[d.cat[i]]) : st.mode === 'miss' ? (i => ramp(dc.miss[i] / 80)) : (() => [0.12, 0.44, 0.55]);
    setGeom(rigs[2], [{ xyz: R, color: refColor }]);
    rigs.forEach(rg => rg.s.children[0].material.size = 0.012 * scale);
    document.getElementById('pcint_t0').innerHTML = `<b>Input:</b> ${h(short[st.cond])} — ${(q.cov * 100).toFixed(0)} % of the ${h(P.ref)}`;
    document.getElementById('pcint_t1').innerHTML = `<b>E1 reconstruction</b> (epoch ${P.epoch}, mask ratio 0.80) — Chamfer to the ${h(P.ref)} ${(q.chamfer * 1e3).toFixed(2)}×10⁻³`;
    document.getElementById('pcint_t2').innerHTML = `<b>Reference:</b> ${h(P.ref)}` + (st.mode === 'miss' ? ` — ${(q.missed_frac * 100).toFixed(0)} % farther than ${P.tau} from the reconstruction` : '');
    const sw = c => `<i style="display:inline-block;width:10px;height:10px;border-radius:2px;vertical-align:middle;margin:0 4px 0 10px;background:rgb(${c.map(v => Math.round(v * 255)).join(',')})"></i>`;
    document.getElementById('pcint_legend').innerHTML = st.mode === 'cov'
      ? 'Reference coverage:' + catLab.map((l, i) => sw(CATC[i]) + h(l)).join('') + ` <span>(nearest input point within ${P.tau})</span>`
      : st.mode === 'miss' ? 'Reference miss map, distance to the nearest reconstructed point:' + [0, 0.02, 0.04, 0.06, 0.08].map(v => sw(ramp(v / 0.08)) + v.toFixed(2)).join('') + ' + (normalised units; dark = not rebuilt)'
      : '';
    ['pcint_plant', 'pcint_cond', 'pcint_mode'].forEach(id => document.querySelectorAll('#' + id + ' button').forEach(b => {
      const on = id === 'pcint_plant' ? +b.dataset.i === st.plant : id === 'pcint_cond' ? b.dataset.c === st.cond : b.dataset.m === st.mode;
      b.setAttribute('aria-pressed', on ? 'true' : 'false'); }));
    render();
  }
  document.getElementById('pcint_plant').innerHTML = P.plants.map((p, i) => `<button data-i="${i}">${h(p.plant.replace('plant_', ''))} · ${(p.cond['1view'].cov * 100).toFixed(0)} %</button>`).join('');
  document.getElementById('pcint_cond').innerHTML = P.cond_order.map(c => `<button data-c="${c}">${h(short[c])}</button>`).join('');
  document.querySelectorAll('#pcint_plant button').forEach(b => b.addEventListener('click', () => { st.plant = +b.dataset.i; update(); }));
  document.querySelectorAll('#pcint_cond button').forEach(b => b.addEventListener('click', () => { st.cond = b.dataset.c; update(); }));
  document.querySelectorAll('#pcint_mode button').forEach(b => b.addEventListener('click', () => { st.mode = b.dataset.m; update(); }));
  document.getElementById('pcint_overlay').addEventListener('change', e => { st.overlay = e.target.checked; update(); });
  rigs.forEach(rg => {
    let drag = null;
    rg.cv.addEventListener('pointerdown', e => { drag = [e.clientX, e.clientY]; rg.cv.setPointerCapture(e.pointerId); });
    rg.cv.addEventListener('pointermove', e => { if (!drag) return; st.theta -= (e.clientX - drag[0]) * 0.01; st.phi = Math.max(-1.45, Math.min(1.45, st.phi + (e.clientY - drag[1]) * 0.01)); drag = [e.clientX, e.clientY]; render(); });
    rg.cv.addEventListener('pointerup', () => drag = null); rg.cv.addEventListener('pointercancel', () => drag = null);
    rg.cv.addEventListener('wheel', e => { e.preventDefault(); st.dist = Math.max(0.8, Math.min(8, st.dist * (e.deltaY > 0 ? 1.1 : 0.9))); render(); }, { passive: false });
    rg.cv.addEventListener('dblclick', () => { st.theta = -0.6; st.phi = 0.25; st.dist = 2.6; render(); });
  });
  window.addEventListener('resize', render);
  update();
})();
