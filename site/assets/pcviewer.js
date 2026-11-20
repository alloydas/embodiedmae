/* CropCode point-cloud viewer: WebGL points, one shared orbit camera per group of panels, no dependencies.
 *
 *   const g = PCV.group();                       // panels in a group rotate and zoom together
 *   const p = PCV.panel(canvas, g);              // a canvas becomes a panel
 *   p.set([{pos: Float32Array, col: Float32Array, size: 1}])   // layers drawn in order
 *   g.frame(bounds)                               // fit the camera to [[minx,miny,minz],[maxx,maxy,maxz]]
 *   const item = await PCV.load(url)             // manifest + binary: {meta, clouds: {name: {pos, err, mark, n}}}
 *   PCV.missed(truthPos, predPos, tau)            // true points with no prediction within tau: {pos, n}
 *   PCV.procedural(recipe, species)               // a 3D plant built from a decoded recipe (metres, y up), as points
 *   PCV.proceduralMesh(recipe, species, flat)     // the same plant as shaded, textured leaf surfaces (as the paper's
 *                                                 // figures): {pos, nrm, col}; flat = '#rrggbb' for a one-colour ghost
 *   p.set([{tri: true, pos, nrm, col, alpha}])     // a mesh layer; alpha < 1 draws after the opaque ones, without depth writes
 * Coordinates: camera frame of the capture (x right, y up, z toward the camera). Mouse / one finger: orbit;
 * wheel / pinch: zoom; double click: reset.
 */
(function () {
  'use strict';
  const VS = `attribute vec3 p; attribute vec3 c; uniform mat4 m; uniform float s; varying vec3 v;
    void main(){ gl_Position = m * vec4(p, 1.0); gl_PointSize = s; v = c; }`;
  const FS = `precision mediump float; varying vec3 v;
    void main(){ vec2 d = gl_PointCoord - 0.5; if (dot(d, d) > 0.25) discard; gl_FragColor = vec4(v, 1.0); }`;
  // meshes: two-sided Lambert shading (0.55 ambient + 0.45 diffuse, as figures/_proc3d.py), per-vertex colour, one alpha
  const VSM = `attribute vec3 p; attribute vec3 nr; attribute vec3 c; uniform mat4 m; uniform vec3 L; varying vec3 v;
    void main(){ gl_Position = m * vec4(p, 1.0); v = c * (0.55 + 0.45 * abs(dot(normalize(nr), L))); }`;
  const FSM = `precision mediump float; varying vec3 v; uniform float a; void main(){ gl_FragColor = vec4(v, a); }`;

  // ---------------------------------------------------------------- tiny matrix helpers (column-major)
  function persp(fovy, aspect, n, f) {
    const t = 1 / Math.tan(fovy / 2), nf = 1 / (n - f);
    return [t / aspect, 0, 0, 0, 0, t, 0, 0, 0, 0, (f + n) * nf, -1, 0, 0, 2 * f * n * nf, 0];
  }
  function lookAt(e, c, u) {
    let z = [e[0] - c[0], e[1] - c[1], e[2] - c[2]]; let l = Math.hypot(...z); z = z.map(x => x / l);
    let x = [u[1] * z[2] - u[2] * z[1], u[2] * z[0] - u[0] * z[2], u[0] * z[1] - u[1] * z[0]]; l = Math.hypot(...x); x = x.map(q => q / l);
    const y = [z[1] * x[2] - z[2] * x[1], z[2] * x[0] - z[0] * x[2], z[0] * x[1] - z[1] * x[0]];
    return [x[0], y[0], z[0], 0, x[1], y[1], z[1], 0, x[2], y[2], z[2], 0,
      -(x[0] * e[0] + x[1] * e[1] + x[2] * e[2]), -(y[0] * e[0] + y[1] * e[1] + y[2] * e[2]), -(z[0] * e[0] + z[1] * e[1] + z[2] * e[2]), 1];
  }
  function mul(a, b) {
    const o = new Array(16).fill(0);
    for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) for (let k = 0; k < 4; k++) o[j * 4 + i] += a[k * 4 + i] * b[j * 4 + k];
    return o;
  }

  // ---------------------------------------------------------------- shared camera
  function group() {
    const g = { panels: [], theta: 0, phi: 0.12, dist: 3, target: [0, 0, 0], fov: 0.6, home: null, size: 1 };
    g.frame = function (b, opts) {
      const c = [0, 1, 2].map(i => (b[0][i] + b[1][i]) / 2);
      const r = Math.max(...[0, 1, 2].map(i => b[1][i] - b[0][i])) / 2 * 1.08;
      g.target = c; g.dist = r / Math.tan(g.fov / 2) + r;
      g.theta = (opts && opts.theta) || 0; g.phi = (opts && opts.phi != null) ? opts.phi : 0.12;
      g.home = { theta: g.theta, phi: g.phi, dist: g.dist, target: c.slice() };
      g.draw();
    };
    g.reset = function () { if (g.home) { Object.assign(g, JSON.parse(JSON.stringify(g.home))); g.draw(); } };
    g.draw = function () { g.panels.forEach(p => p.draw()); };
    g.eye = function () {
      const cp = Math.cos(g.phi);
      return [g.target[0] + g.dist * cp * Math.sin(g.theta), g.target[1] + g.dist * Math.sin(g.phi), g.target[2] + g.dist * cp * Math.cos(g.theta)];
    };
    return g;
  }

  // ---------------------------------------------------------------- one shared WebGL context
  // Browsers keep only ~16 live WebGL contexts per page and silently drop the oldest, which blanked panels on a page
  // with 25 of them. Every panel therefore renders through ONE hidden WebGL canvas and copies the frame into its own 2D canvas.
  let SH = null;
  function shared() {
    if (SH) return SH;
    const cv = document.createElement('canvas');
    const gl = cv.getContext('webgl', { antialias: true, premultipliedAlpha: false, preserveDrawingBuffer: true });
    if (!gl) return (SH = { gl: null });
    const sh = (t, s) => { const o = gl.createShader(t); gl.shaderSource(o, s); gl.compileShader(o); return o; };
    const prog = gl.createProgram(); gl.attachShader(prog, sh(gl.VERTEX_SHADER, VS)); gl.attachShader(prog, sh(gl.FRAGMENT_SHADER, FS)); gl.linkProgram(prog);
    const loc = { p: gl.getAttribLocation(prog, 'p'), c: gl.getAttribLocation(prog, 'c'), m: gl.getUniformLocation(prog, 'm'), s: gl.getUniformLocation(prog, 's') };
    const progM = gl.createProgram(); gl.attachShader(progM, sh(gl.VERTEX_SHADER, VSM)); gl.attachShader(progM, sh(gl.FRAGMENT_SHADER, FSM)); gl.linkProgram(progM);
    const locM = { p: gl.getAttribLocation(progM, 'p'), n: gl.getAttribLocation(progM, 'nr'), c: gl.getAttribLocation(progM, 'c'),
                   m: gl.getUniformLocation(progM, 'm'), L: gl.getUniformLocation(progM, 'L'), a: gl.getUniformLocation(progM, 'a') };
    return (SH = { cv, gl, prog, loc, progM, locM });
  }

  // ---------------------------------------------------------------- a panel
  function panel(canvas, g) {
    const S = shared(), gl = S.gl, ctx = canvas.getContext('2d');
    const p = { canvas, layers: [], bg: [1, 1, 1] };
    if (!gl || !ctx) { p.set = () => {}; p.draw = () => {}; canvas.replaceWith(Object.assign(document.createElement('p'), { textContent: 'WebGL is not available in this browser.' })); return p; }
    const { prog, loc, progM, locM } = S;
    const buf = a => { const b = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, b); gl.bufferData(gl.ARRAY_BUFFER, a, gl.STATIC_DRAW); return b; };
    p.set = function (layers) {
      p.layers.forEach(l => { gl.deleteBuffer(l.bp); gl.deleteBuffer(l.bc); if (l.bn) gl.deleteBuffer(l.bn); });
      p.layers = layers.filter(l => l && l.pos && l.pos.length).map(l => l.tri
        ? { tri: true, bp: buf(l.pos), bn: buf(l.nrm), bc: buf(l.col), n: l.pos.length / 3, alpha: l.alpha == null ? 1 : l.alpha }
        : { bp: buf(l.pos), bc: buf(l.col), n: l.pos.length / 3, size: l.size || 1 });
      p.draw();
    };
    p.draw = function () {
      const dpr = window.devicePixelRatio || 1, w = Math.max(1, Math.round(canvas.clientWidth * dpr)), h = Math.max(1, Math.round(canvas.clientHeight * dpr));
      if (!canvas.clientWidth) return;                                   // hidden panel: nothing to draw
      if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
      if (S.cv.width !== w || S.cv.height !== h) { S.cv.width = w; S.cv.height = h; }
      gl.viewport(0, 0, w, h); gl.clearColor(p.bg[0], p.bg[1], p.bg[2], 1); gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
      gl.enable(gl.DEPTH_TEST); gl.depthMask(true); gl.disable(gl.BLEND);
      const m = new Float32Array(mul(persp(g.fov, w / h, g.dist / 50, g.dist * 4), lookAt(g.eye(), g.target, [0, 1, 0])));
      const base = Math.max(1.2, Math.min(w, h) / 260) * g.size;
      const pts = p.layers.filter(l => !l.tri), solid = p.layers.filter(l => l.tri && l.alpha >= 1), ghost = p.layers.filter(l => l.tri && l.alpha < 1);
      if (pts.length) {
        gl.useProgram(prog); gl.uniformMatrix4fv(loc.m, false, m);
        pts.forEach(l => {
          gl.uniform1f(loc.s, base * l.size);
          gl.bindBuffer(gl.ARRAY_BUFFER, l.bp); gl.enableVertexAttribArray(loc.p); gl.vertexAttribPointer(loc.p, 3, gl.FLOAT, false, 0, 0);
          gl.bindBuffer(gl.ARRAY_BUFFER, l.bc); gl.enableVertexAttribArray(loc.c); gl.vertexAttribPointer(loc.c, 3, gl.FLOAT, false, 0, 0);
          gl.drawArrays(gl.POINTS, 0, l.n);
        });
      }
      if (solid.length || ghost.length) {
        gl.useProgram(progM); gl.uniformMatrix4fv(locM.m, false, m);
        const Lw = [0.35, 0.8, 0.5], Ln = Math.hypot(...Lw); gl.uniform3f(locM.L, Lw[0] / Ln, Lw[1] / Ln, Lw[2] / Ln);
        const mesh = l => {
          gl.uniform1f(locM.a, l.alpha);
          gl.bindBuffer(gl.ARRAY_BUFFER, l.bp); gl.enableVertexAttribArray(locM.p); gl.vertexAttribPointer(locM.p, 3, gl.FLOAT, false, 0, 0);
          gl.bindBuffer(gl.ARRAY_BUFFER, l.bn); gl.enableVertexAttribArray(locM.n); gl.vertexAttribPointer(locM.n, 3, gl.FLOAT, false, 0, 0);
          gl.bindBuffer(gl.ARRAY_BUFFER, l.bc); gl.enableVertexAttribArray(locM.c); gl.vertexAttribPointer(locM.c, 3, gl.FLOAT, false, 0, 0);
          gl.drawArrays(gl.TRIANGLES, 0, l.n);
        };
        solid.forEach(mesh);
        if (ghost.length) { gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA); gl.depthMask(false); ghost.forEach(mesh); gl.depthMask(true); gl.disable(gl.BLEND); }
        gl.disableVertexAttribArray(locM.n);                              // the point program has no normal attribute
      }
      ctx.drawImage(S.cv, 0, 0);
    };
    // pointer: orbit with one pointer, pinch-zoom with two, wheel zoom, double-click reset
    const pts = new Map(); let pinch = null;
    canvas.style.touchAction = 'none';
    canvas.addEventListener('pointerdown', e => { canvas.setPointerCapture(e.pointerId); pts.set(e.pointerId, [e.clientX, e.clientY]); pinch = null; });
    canvas.addEventListener('pointermove', e => {
      if (!pts.has(e.pointerId)) return;
      const prev = pts.get(e.pointerId); pts.set(e.pointerId, [e.clientX, e.clientY]);
      if (pts.size === 1) {
        g.theta -= (e.clientX - prev[0]) * 0.01;
        g.phi = Math.max(-1.45, Math.min(1.45, g.phi + (e.clientY - prev[1]) * 0.01));
      } else if (pts.size === 2) {
        const [a, b] = [...pts.values()]; const d = Math.hypot(a[0] - b[0], a[1] - b[1]);
        if (pinch) g.dist = Math.max(0.2, Math.min(40, g.dist * pinch / d));
        pinch = d;
      }
      g.draw();
    });
    const up = e => { pts.delete(e.pointerId); pinch = null; };
    canvas.addEventListener('pointerup', up); canvas.addEventListener('pointercancel', up);
    canvas.addEventListener('wheel', e => { e.preventDefault(); g.dist = Math.max(0.2, Math.min(40, g.dist * Math.exp(e.deltaY * 0.0012))); g.draw(); }, { passive: false });
    canvas.addEventListener('dblclick', () => g.reset());
    g.panels.push(p);
    return p;
  }

  // ---------------------------------------------------------------- data
  async function load(url) {
    const meta = await (await fetch(url)).json();
    const base = url.slice(0, url.lastIndexOf('/') + 1);
    let buf;
    if (meta.b64) { const bs = atob(meta.b64); const u = new Uint8Array(bs.length); for (let i = 0; i < bs.length; i++) u[i] = bs.charCodeAt(i); buf = u.buffer; delete meta.b64; }
    else buf = await (await fetch(base + meta.bin)).arrayBuffer();
    const clouds = {};
    for (const [name, c] of Object.entries(meta.clouds)) {
      const q = new Int16Array(buf, c.off, c.n * 3), pos = new Float32Array(c.n * 3);
      for (let i = 0; i < pos.length; i++) pos[i] = q[i] / meta.q;
      let o = c.off + c.n * 6, err = null, mark = null;
      if (c.err) { err = new Float32Array(c.n); const e = new Uint8Array(buf, o, c.n); for (let i = 0; i < c.n; i++) err[i] = e[i] / 255 * meta.err_max_mm; o += c.n; }
      if (c.mark) { mark = new Uint8Array(buf, o, c.n).slice(); }
      clouds[name] = { pos, err, mark, n: c.n };
    }
    return { meta, base, clouds };
  }
  function bounds(...arrs) {
    const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    arrs.forEach(a => { for (let i = 0; i < a.length; i += 3) for (let k = 0; k < 3; k++) { lo[k] = Math.min(lo[k], a[i + k]); hi[k] = Math.max(hi[k], a[i + k]); } });
    return [lo, hi];
  }

  // ---------------------------------------------------------------- colours
  const hex = h => [1, 3, 5].map(i => parseInt(h.slice(i, i + 2), 16) / 255);
  const ramp = stops => { const s = stops.map(hex); return t => { t = Math.max(0, Math.min(1, t)) * (s.length - 1); const i = Math.min(s.length - 2, Math.floor(t)), f = t - i; return [0, 1, 2].map(k => s[i][k] * (1 - f) + s[i + 1][k] * f); }; };
  const LEAF = ramp(['#1f4d22', '#2f7d32', '#5aa83e', '#a8d26a']);
  const VIRIDIS = ramp(['#440154', '#414487', '#2a788e', '#22a884', '#7ad151', '#fde725']);   // colour-blind safe, sequential
  function plantColors(pos) {
    const n = pos.length / 3, c = new Float32Array(pos.length); let lo = Infinity, hi = -Infinity;
    for (let i = 0; i < n; i++) { lo = Math.min(lo, pos[3 * i + 1]); hi = Math.max(hi, pos[3 * i + 1]); }
    for (let i = 0; i < n; i++) { const v = LEAF(0.12 + 0.88 * (pos[3 * i + 1] - lo) / Math.max(1e-9, hi - lo)); c.set(v, 3 * i); }
    return c;
  }
  function errorColors(err, max) { const c = new Float32Array(err.length * 3); for (let i = 0; i < err.length; i++) c.set(VIRIDIS(err[i] / max), 3 * i); return c; }
  function flat(n, h) { const c = new Float32Array(n * 3), v = hex(h); for (let i = 0; i < n; i++) c.set(v, 3 * i); return c; }
  function markColors(pos, mark, other) {
    const c = plantColors(pos), o = hex(other || '#c9c2b0');
    for (let i = 0; i < mark.length; i++) if (!mark[i]) c.set(o, 3 * i);
    return c;
  }

  // ---------------------------------------------------------------- the procedural model: a plant built from its recipe
  // rec = {h, n, f: [leaf field names], L: [[values per leaf]]} (the decoded recipe, generator units: metres, degrees).
  // The geometry of the paper's stem-and-leaves schematic (figures/_fig12_kit.glyph), in 3D: each leaf a ribbon along a
  // quadratic Bezier in its own vertical plane, turned about the stem by its azimuth (maize: 180 deg x leaf index +
  // jitter; sorghum: the roll angle). Returns points in metres, y up, stem base at the origin.
  function procedural(rec, species) {
    const out = [], col = name => rec.f.indexOf(name), rad = d => d * Math.PI / 180;
    const H = rec.h, sr = species === 'maize' ? 0.011 : 0.009;
    for (let y = 0; y <= H; y += 0.003) for (let k = 0; k < 8; k++) { const a = k * Math.PI / 4; out.push(sr * Math.cos(a), y, sr * Math.sin(a)); }
    let y = 0;
    rec.L.forEach((l, i) => {
      let y0, len, w, az, phi0, phi1;
      if (species === 'maize') {
        y += l[col('distance')]; y0 = y; len = l[col('leafLength')]; w = l[col('leafWidth')];
        az = rad(180 * i + l[col('azJitterDeg')]); phi0 = rad(35 + l[col('leafAngle')]); phi1 = phi0 - rad(l[col('droopiness')]);
      } else {
        y0 = l[col('starting_point')] * H; len = l[col('length')]; w = l[col('width')];
        az = rad(l[col('roll_angle')]); phi0 = rad(l[col('branching_angle')]); phi1 = phi0 + rad(75);
      }
      if (!(len > 0)) return;
      const d = [Math.cos(az), Math.sin(az)], e = [-Math.sin(az), Math.cos(az)];
      const p1 = [0.5 * len * Math.sin(phi0), y0 + 0.5 * len * Math.cos(phi0)];
      const p2 = [p1[0] + 0.5 * len * Math.sin(phi1), p1[1] + 0.5 * len * Math.cos(phi1)];
      const steps = Math.max(24, Math.round(len / 0.004));
      for (let s = 0; s <= steps; s++) {
        const t = s / steps, a = (1 - t) * (1 - t), b = 2 * (1 - t) * t, c = t * t;
        const hz = b * p1[0] + c * p2[0], vy = a * y0 + b * p1[1] + c * p2[1];
        const hw = 0.5 * w * Math.pow(Math.sin(Math.PI * Math.min(1, 0.15 + 0.85 * t)), 0.6);
        for (let k = -3; k <= 3; k++) { const o = hw * k / 3; out.push(hz * d[0] + o * e[0], vy, hz * d[1] + o * e[1]); }
      }
    });
    return new Float32Array(out);
  }

  // the same plant as shaded, textured surfaces (figures/_proc3d.py mesh() + texture(), the paper's 3D plants): leaf
  // ribbons 8 strips across, V-folded about the midrib; leaves dark green at the base to lighter at the tip, a pale
  // midrib, faint veins, darker margins, a yellowing tip; the stem a green-brown sheath with darker nodes.
  // flat: one colour instead (the grey ghost of the true plant). Returns non-indexed triangles {pos, nrm, col}.
  function proceduralMesh(rec, species, flat) {
    const P = [], N = [], C = [], col = name => rec.f.indexOf(name), rad = d => d * Math.PI / 180, F = flat ? hex(flat) : null;
    const H = rec.h, sr = species === 'maize' ? 0.011 : 0.009;
    const mix = (a, b, t) => [0, 1, 2].map(k => a[k] * (1 - t) + b[k] * t);
    function tex(kind, t, o) {
      if (F) return F;
      if (kind === 'stem') { const f = (t * 12) % 1 < 0.12 ? 0.85 : 1; return [0.34 * f, 0.46 * f, 0.20 * f]; }
      let c = mix([0.16, 0.40, 0.13], [0.40, 0.66, 0.24], t); const a = Math.abs(o);
      if (a < 0.14) c = mix(c, [0.80, 0.88, 0.58], 0.55);
      c = c.map(v => v * (1 + 0.05 * Math.cos(a * Math.PI * 9)));
      if (a > 0.82) c = c.map(v => v * 0.86);
      if (t > 0.9) c = mix(c, [0.70, 0.66, 0.30], 0.25);
      return c.map(v => Math.max(0, Math.min(1, v)));
    }
    function quad(a, b, c, d, rgb) {
      const u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]], v = [d[0] - a[0], d[1] - a[1], d[2] - a[2]];
      let n = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]]; const l = Math.hypot(...n) || 1; n = n.map(x => x / l);
      for (const q of [a, b, c, a, c, d]) { P.push(q[0], q[1], q[2]); N.push(n[0], n[1], n[2]); C.push(rgb[0], rgb[1], rgb[2]); }
    }
    const ny = Math.max(2, Math.floor(H / 0.05) + 1);
    for (let k = 0; k < 8; k++) {
      const a0 = k * Math.PI / 4, a1 = (k + 1) * Math.PI / 4;
      for (let j = 0; j < ny - 1; j++) {
        const y0 = H * j / (ny - 1), y1 = H * (j + 1) / (ny - 1), rgb = tex('stem', (y0 + y1) / 2 / Math.max(H, 1e-6), 0);
        quad([sr * Math.cos(a0), y0, sr * Math.sin(a0)], [sr * Math.cos(a1), y0, sr * Math.sin(a1)], [sr * Math.cos(a1), y1, sr * Math.sin(a1)], [sr * Math.cos(a0), y1, sr * Math.sin(a0)], rgb);
      }
    }
    let y = 0; const across = 8, fold = 0.18;
    rec.L.forEach((l, i) => {
      let y0, len, w, az, phi0, phi1;
      if (species === 'maize') {
        y += l[col('distance')]; y0 = y; len = l[col('leafLength')]; w = l[col('leafWidth')];
        az = rad(180 * i + l[col('azJitterDeg')]); phi0 = rad(35 + l[col('leafAngle')]); phi1 = phi0 - rad(l[col('droopiness')]);
      } else {
        y0 = l[col('starting_point')] * H; len = l[col('length')]; w = l[col('width')];
        az = rad(l[col('roll_angle')]); phi0 = rad(l[col('branching_angle')]); phi1 = phi0 + rad(75);
      }
      if (!(len > 0)) return;
      const d = [Math.cos(az), Math.sin(az)], e = [-Math.sin(az), Math.cos(az)];
      const p1 = [0.5 * len * Math.sin(phi0), y0 + 0.5 * len * Math.cos(phi0)];
      const p2 = [p1[0] + 0.5 * len * Math.sin(phi1), p1[1] + 0.5 * len * Math.cos(phi1)];
      const steps = Math.max(12, Math.floor(len / 0.012) + 1), G = [];
      for (let s = 0; s < steps; s++) {
        const t = s / (steps - 1), a = (1 - t) * (1 - t), b = 2 * (1 - t) * t, c = t * t;
        const hz = b * p1[0] + c * p2[0], vy = a * y0 + b * p1[1] + c * p2[1];
        const hw = 0.5 * w * Math.pow(Math.sin(Math.PI * Math.min(1, 0.15 + 0.85 * t)), 0.6), row = [];
        for (let k = 0; k <= across; k++) { const o = -1 + 2 * k / across, off = hw * o; row.push([hz * d[0] + off * e[0], vy + fold * hw * Math.abs(o), hz * d[1] + off * e[1]]); }
        G.push(row);
      }
      for (let s = 0; s < steps - 1; s++) for (let k = 0; k < across; k++) {
        const t = (s + 0.5) / (steps - 1), o = -1 + 2 * (k + 0.5) / across;
        quad(G[s][k], G[s][k + 1], G[s + 1][k + 1], G[s + 1][k], tex('leaf', t, o));
      }
    });
    return { tri: true, pos: new Float32Array(P), nrm: new Float32Array(N), col: new Float32Array(C) };
  }

  // ---------------------------------------------------------------- missed points
  // the true points with no predicted point within tau (same units as the clouds): a uniform grid of cell tau over the
  // prediction, each true point checks its 27 neighbouring cells. Returns the missed points' positions and their count.
  function missed(T, P, tau) {
    const inv = 1 / tau, grid = new Map(), key = (a, b, c) => a + ',' + b + ',' + c, t2 = tau * tau;
    for (let i = 0; i < P.length; i += 3) {
      const k = key(Math.floor(P[i] * inv), Math.floor(P[i + 1] * inv), Math.floor(P[i + 2] * inv));
      let a = grid.get(k); if (!a) grid.set(k, a = []); a.push(i);
    }
    const out = [];
    for (let i = 0; i < T.length; i += 3) {
      const x = T[i], y = T[i + 1], z = T[i + 2], cx = Math.floor(x * inv), cy = Math.floor(y * inv), cz = Math.floor(z * inv);
      let hit = false;
      for (let dx = -1; dx <= 1 && !hit; dx++) for (let dy = -1; dy <= 1 && !hit; dy++) for (let dz = -1; dz <= 1 && !hit; dz++) {
        const a = grid.get(key(cx + dx, cy + dy, cz + dz)); if (!a) continue;
        for (const j of a) { const ex = P[j] - x, ey = P[j + 1] - y, ez = P[j + 2] - z; if (ex * ex + ey * ey + ez * ez <= t2) { hit = true; break; } }
      }
      if (!hit) out.push(x, y, z);
    }
    return { pos: new Float32Array(out), n: out.length / 3 };
  }
  window.PCV = { group, panel, load, bounds, plantColors, errorColors, flat, markColors, missed, procedural, proceduralMesh, VIRIDIS, hex };
})();
