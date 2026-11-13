/* CropCode point-cloud viewer: WebGL points, one shared orbit camera per group of panels, no dependencies.
 *
 *   const g = PCV.group();                       // panels in a group rotate and zoom together
 *   const p = PCV.panel(canvas, g);              // a canvas becomes a panel
 *   p.set([{pos: Float32Array, col: Float32Array, size: 1}])   // layers drawn in order
 *   g.frame(bounds)                               // fit the camera to [[minx,miny,minz],[maxx,maxy,maxz]]
 *   const item = await PCV.load(url)             // manifest + binary: {meta, clouds: {name: {pos, err, mark, n}}}
 * Coordinates: camera frame of the capture (x right, y up, z toward the camera). Mouse / one finger: orbit;
 * wheel / pinch: zoom; double click: reset.
 */
(function () {
  'use strict';
  const VS = `attribute vec3 p; attribute vec3 c; uniform mat4 m; uniform float s; varying vec3 v;
    void main(){ gl_Position = m * vec4(p, 1.0); gl_PointSize = s; v = c; }`;
  const FS = `precision mediump float; varying vec3 v;
    void main(){ vec2 d = gl_PointCoord - 0.5; if (dot(d, d) > 0.25) discard; gl_FragColor = vec4(v, 1.0); }`;

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
    return (SH = { cv, gl, prog, loc });
  }

  // ---------------------------------------------------------------- a panel
  function panel(canvas, g) {
    const S = shared(), gl = S.gl, ctx = canvas.getContext('2d');
    const p = { canvas, layers: [], bg: [1, 1, 1] };
    if (!gl || !ctx) { p.set = () => {}; p.draw = () => {}; canvas.replaceWith(Object.assign(document.createElement('p'), { textContent: 'WebGL is not available in this browser.' })); return p; }
    const { prog, loc } = S;
    p.set = function (layers) {
      p.layers.forEach(l => { gl.deleteBuffer(l.bp); gl.deleteBuffer(l.bc); });
      p.layers = layers.map(l => {
        const bp = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, bp); gl.bufferData(gl.ARRAY_BUFFER, l.pos, gl.STATIC_DRAW);
        const bc = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, bc); gl.bufferData(gl.ARRAY_BUFFER, l.col, gl.STATIC_DRAW);
        return { bp, bc, n: l.pos.length / 3, size: l.size || 1 };
      });
      p.draw();
    };
    p.draw = function () {
      const dpr = window.devicePixelRatio || 1, w = Math.max(1, Math.round(canvas.clientWidth * dpr)), h = Math.max(1, Math.round(canvas.clientHeight * dpr));
      if (!canvas.clientWidth) return;                                   // hidden panel: nothing to draw
      if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
      if (S.cv.width !== w || S.cv.height !== h) { S.cv.width = w; S.cv.height = h; }
      gl.viewport(0, 0, w, h); gl.clearColor(p.bg[0], p.bg[1], p.bg[2], 1); gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
      gl.enable(gl.DEPTH_TEST); gl.useProgram(prog);
      const m = mul(persp(g.fov, w / h, g.dist / 50, g.dist * 4), lookAt(g.eye(), g.target, [0, 1, 0]));
      gl.uniformMatrix4fv(loc.m, false, new Float32Array(m));
      const base = Math.max(1.2, Math.min(w, h) / 260) * g.size;
      p.layers.forEach(l => {
        gl.uniform1f(loc.s, base * l.size);
        gl.bindBuffer(gl.ARRAY_BUFFER, l.bp); gl.enableVertexAttribArray(loc.p); gl.vertexAttribPointer(loc.p, 3, gl.FLOAT, false, 0, 0);
        gl.bindBuffer(gl.ARRAY_BUFFER, l.bc); gl.enableVertexAttribArray(loc.c); gl.vertexAttribPointer(loc.c, 3, gl.FLOAT, false, 0, 0);
        gl.drawArrays(gl.POINTS, 0, l.n);
      });
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
  window.PCV = { group, panel, load, bounds, plantColors, errorColors, flat, markColors, VIRIDIS, hex };
})();
