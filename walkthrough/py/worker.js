// Runs the release's Python in the browser: Pyodide (CPython compiled to WebAssembly) with
// NumPy, OpenCV and Pillow, inside a worker so the page never freezes while it loads or runs.
const PYODIDE = 'https://cdn.jsdelivr.net/pyodide/v0.28.3/full/';
const ASSETS = ['canny_boat_input.webp', 'canny_boat_mask.webp', 'canny_octopus_input.webp', 'canny_octopus_mask.webp',
  'raw_octopus_pink.webp', 'exemplar_pink.webp'];
let py;

const get = async (url) => { const r = await fetch(url); if (!r.ok) throw new Error(`${url} answered ${r.status}`); return r; };

async function boot() {
  postMessage({ type: 'status', text: 'Downloading Python (5 MB)…' });
  importScripts(PYODIDE + 'pyodide.js');
  py = await loadPyodide({ indexURL: PYODIDE });
  postMessage({ type: 'status', text: 'Downloading NumPy, OpenCV and Pillow (16 MB)…' });
  const here = (p) => new URL(p, self.location).href;
  const [, helpers, ...images] = await Promise.all([
    py.loadPackage(['numpy', 'opencv-python', 'pillow'], { messageCallback: () => {} }),
    get(here('page_helpers.py')).then((r) => r.text()),
    ...ASSETS.map((a) => get(here('../assets/' + a)).then((r) => r.arrayBuffer()))]);
  py.FS.mkdir('/assets');
  ASSETS.forEach((a, i) => py.FS.writeFile('/assets/' + a, new Uint8Array(images[i])));
  py.FS.writeFile('/home/pyodide/page_helpers.py', helpers);
  postMessage({ type: 'status', text: 'Starting Python…' });
  await py.runPythonAsync('import page_helpers, sys, numpy, cv2, PIL');
  const v = (m) => py.runPython(m);
  postMessage({ type: 'ready',
    versions: `Python ${v('sys.version.split()[0]')} · NumPy ${v('numpy.__version__')} · OpenCV ${v('cv2.__version__')} · Pillow ${v('PIL.__version__')}` });
}

self.onmessage = async ({ data: m }) => {
  if (m.type === 'boot') {
    try { await boot(); } catch (e) { postMessage({ type: 'failed', text: String(e && e.message || e) }); }
  } else if (m.type === 'run') {
    let res;
    try {
      const r = py.pyimport('page_helpers').run_cell(m.fn, m.first, m.call);
      res = r.toJs({ dict_converter: Object.fromEntries }); r.destroy();
    } catch (e) { res = { image: null, stdout: '', error: String(e && e.message || e), steps: [], ms: 0 }; }
    const transfer = res.image ? [res.image[2].buffer] : [];
    postMessage({ type: 'result', id: m.id, ...res }, transfer);
  }
};
