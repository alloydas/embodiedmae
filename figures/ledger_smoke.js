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
let ok = true;
for (const s of scripts) {
  try { new Function(...Object.keys(ctx), s)(...Object.values(ctx)); }
  catch (e) { ok = false; console.log('SCRIPT ERROR:', e.message); }
}
const charts = Object.keys(els).filter(k => k.startsWith('ch_'));
const drawn = charts.filter(k => /<svg/.test(String(els[k].innerHTML)));
console.log(ok ? 'js ok;' : 'js FAILED;', 'charts drawn:', drawn.length, '/', charts.length, charts.filter(k => !drawn.includes(k)).join(' '));
const strip = h => String(h).replace(/<tr[^>]*>/g, '\n').replace(/<[^>]+>/g, ' ').replace(/[ \t]+/g, ' ');
for (const id of ids) console.log('== ' + id + strip(els[id] ? els[id].innerHTML : '(missing)'));
