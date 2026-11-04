// Smoke test for the sorghum ledger page built by figures/make_sorghum_ledger.py.
// Runs the page's inline script against a permissive DOM stub (no browser), reports
// script errors and how many charts drew, and prints chosen tables as text.
// usage: node figures/ledger_smoke.js <ledger.html> [tableId ...]
const fs = require('fs');
const [file, ...ids] = process.argv.slice(2);
const html = fs.readFileSync(file, 'utf8');
const dataJson = (html.match(/<script id="data" type="application\/json">([\s\S]*?)<\/script>/) || [])[1];
const scripts = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(m => m[1]);
const els = {};
function mkEl(id) {
  const fn = function () { return el; };
  const store = { id, innerHTML: '', textContent: '', style: {}, dataset: {}, children: [], attrs: {},
    classList: { add() {}, remove() {}, toggle() {}, contains() { return false; } } };
  const el = new Proxy(fn, {
    get(t, k) {
      if (k in store) return store[k];
      if (k === 'setAttribute') return (a, v) => { store.attrs[a] = v; };
      if (k === 'getAttribute') return a => store.attrs[a] ?? null;
      if (k === 'querySelectorAll') return () => [];
      if (k === 'querySelector') return () => mkEl('_q');
      if (k === 'getBoundingClientRect') return () => ({ width: 640, height: 300, left: 0, top: 0 });
      if (k === 'clientWidth' || k === 'offsetWidth') return 640;
      if (k === Symbol.iterator) return function* () {};
      if (k === 'length') return 0;
      return function () { return mkEl('_f'); };
    },
    set(t, k, v) { store[k] = v; return true; },
    apply() { return el; },
  });
  return el;
}
const document = new Proxy({}, { get(t, k) {
  if (k === 'getElementById') return id => (id === 'data' ? { textContent: dataJson } : (els[id] ||= mkEl(id)));
  if (k === 'querySelectorAll') return () => [];
  if (k === 'querySelector') return () => mkEl('_q');
  if (k === 'createElement' || k === 'createElementNS') return () => mkEl('_c');
  if (k === 'documentElement' || k === 'body') return mkEl('_root');
  if (k === 'addEventListener') return () => {};
  return mkEl('_d');
} });
const window = { addEventListener() {}, matchMedia: () => ({ matches: false, addEventListener() {} }), innerWidth: 1200, devicePixelRatio: 1, location: { hash: '' } };
const ctx = { document, window, localStorage: { getItem: () => null, setItem() {} }, ResizeObserver: class { observe() {} }, IntersectionObserver: class { observe() {} }, requestAnimationFrame: () => 0, getComputedStyle: () => ({ getPropertyValue: () => '' }), console };
// minimal three.js stub so 3D viewers run their data path (CONCAT=1 runs all inline scripts as one, sharing globals)
class V3 { constructor() { this.x = 0; this.y = 0; this.z = 0; } set(x, y, z) { this.x = x; this.y = y; this.z = z; } }
ctx.THREE = { WebGLRenderer: class { setPixelRatio() {} setSize() {} render() { ctx.__renders = (ctx.__renders || 0) + 1; } },
  Scene: class { constructor() { this.children = []; } add(o) { this.children.push(o); } },
  PerspectiveCamera: class { constructor() { this.position = new V3(); } updateProjectionMatrix() {} lookAt() {} },
  BufferGeometry: class { constructor() { this.attr = {}; } setAttribute(k, v) { this.attr[k] = v; } computeBoundingSphere() {} },
  BufferAttribute: class { constructor(a, n) { this.array = a; this.itemSize = n; } },
  PointsMaterial: class { constructor(o) { Object.assign(this, o); } }, Points: class { constructor(g, m) { this.geometry = g; this.material = m; } },
  Color: class { constructor() {} setHSL() { return this; } } };
ctx.atob = s => Buffer.from(s, 'base64').toString('binary');
let ok = true;
const runs = process.env.CONCAT ? [scripts.join('\n;\n')] : scripts;
for (const s of runs) {
  try { new Function(...Object.keys(ctx), s)(...Object.values(ctx)); }
  catch (e) { ok = false; console.log('SCRIPT ERROR:', e.message, (e.stack || '').split('\n')[1]); }
}
if (ctx.__renders) console.log('3D renders:', ctx.__renders);
const charts = Object.keys(els).filter(k => k.startsWith('ch_'));
const drawn = charts.filter(k => /<svg/.test(String(els[k].innerHTML)));
console.log(ok ? 'js ok;' : 'js FAILED;', 'charts drawn:', drawn.length, '/', charts.length, charts.filter(k => !drawn.includes(k)).join(' '));
const strip = h => String(h).replace(/<tr[^>]*>/g, '\n').replace(/<[^>]+>/g, ' ').replace(/[ \t]+/g, ' ');
for (const id of ids) console.log('== ' + id + strip(els[id] ? els[id].innerHTML : '(missing)'));
