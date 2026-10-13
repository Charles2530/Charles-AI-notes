/* Run after mkdocs build: PLAYWRIGHT_MODULE=/path/to/playwright node --test tests/browser/paper_library_test.cjs */
const { test, before, after } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');
const http = require('node:http');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const root = path.resolve(__dirname, '../..');
let browser, server, base;

// A real, deterministic PDF with a searchable text layer; no downloaded fixtures.
function pdfFixture(count = 3, withText = true) {
  const objects = ['<< /Type /Catalog /Pages 2 0 R >>', '', '<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>'];
  const pages = [];
  for (let n = 1; n <= count; n++) {
    const id = objects.length + 1;
    pages.push(`${id} 0 R`);
    const content = withText ? `BT /F1 20 Tf 40 720 Td (Research page ${n} searchable evidence) Tj ET` : '';
    objects.push(`<< /Type /Page /Parent 2 0 R /MediaBox [0 0 600 800] /Resources << /Font << /F1 3 0 R >> >> /Contents ${id + 1} 0 R >>`);
    objects.push(`<< /Length ${content.length} >>\nstream\n${content}\nendstream`);
  }
  objects[1] = `<< /Type /Pages /Kids [${pages.join(' ')}] /Count ${count} >>`;
  let out = '%PDF-1.7\n';
  const offsets = [0];
  objects.forEach((obj, i) => { offsets.push(out.length); out += `${i + 1} 0 obj\n${obj}\nendobj\n`; });
  const xref = out.length;
  out += `xref\n0 ${objects.length + 1}\n0000000000 65535 f \n`;
  offsets.slice(1).forEach(offset => { out += `${String(offset).padStart(10, '0')} 00000 n \n`; });
  out += `trailer\n<< /Size ${objects.length + 1} /Root 1 0 R >>\nstartxref\n${xref}\n%%EOF\n`;
  return Buffer.from(out);
}
before(async () => {
  server = http.createServer(async (req, res) => {
    try {
      let pathname = decodeURIComponent(new URL(req.url, 'http://localhost').pathname).replace(/^\/notes\//, '/');
      if (pathname === '/sample.pdf') { res.setHeader('Content-Type', 'application/pdf'); res.end(pdfFixture()); return; }
      if (pathname.endsWith('/')) pathname += 'index.html';
      const source = pathname.startsWith('/assets/') ? 'files' : 'site';
      const filename = path.join(root, source, pathname);
      res.setHeader('Content-Type', ({'.js':'text/javascript','.mjs':'text/javascript','.css':'text/css','.html':'text/html','.wasm':'application/wasm'})[path.extname(filename)] || 'application/octet-stream');
      res.end(await fs.readFile(filename));
    } catch { res.statusCode = 404; res.end('Not found'); }
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  base = `http://127.0.0.1:${server.address().port}/notes`;
  browser = await chromium.launch({headless:true});
});
after(async () => { if (browser) await browser.close(); if (server) await new Promise(resolve => server.close(resolve)); });
async function setup(t) {
  const context = await browser.newContext({ viewport: {width:1440,height:1000} });
  const page = await context.newPage();
  page.setDefaultTimeout(8000);
  const errors = [];
  page.on('pageerror', e => errors.push(e.message));
  t.after(async () => { await context.close(); assert.deepEqual(errors, [], 'no unhandled browser errors'); });
  await page.goto(`${base}/paper-library/`);
  await page.waitForFunction(() => document.querySelector('[data-paper-status]').textContent.includes('准备好'));
  return page;
}
async function importPdf(page, name = 'research.pdf', buffer = pdfFixture()) {
  await page.locator('[data-paper-file-input]').setInputFiles({name, mimeType:'application/pdf', buffer});
  await page.waitForFunction(() => !document.querySelector('[data-paper-reader-view]').hidden && document.querySelector('[data-paper-reader-placeholder]').hidden);
}
async function counts(page) {
  return page.evaluate(async () => {
    const db = await window.AtlasPaperLibrary.openPaperDatabase();
    try { return await Promise.all(['papers','files','annotations','notes'].map(name => new Promise((resolve,reject) => { const r=db.transaction(name).objectStore(name).count(); r.onsuccess=()=>resolve(r.result); r.onerror=()=>reject(r.error); }))); }
    finally { db.close(); }
  });
}
test('rejects corrupt PDF without persisting an invalid library entry', async t => {
  const page = await setup(t);
  await page.locator('[data-paper-file-input]').setInputFiles({name:'broken.pdf',mimeType:'application/pdf',buffer:Buffer.from('not a PDF')});
  await page.waitForTimeout(1200);
  assert.deepEqual(await counts(page), [0,0,0,0]);
  assert.equal(await page.locator('[data-paper-status]').getAttribute('data-tone'), 'error');
});
test('restores scrolling progress and flushes notes when returning to library', async t => {
  const page = await setup(t);
  await importPdf(page);
  assert.equal(await page.locator('[data-paper-title-input]').inputValue(), 'research');
  await page.locator('[data-paper-note-editor]').fill('Saved immediately before leaving');
  await page.locator('[data-paper-reader-back]').click();
  await page.locator('.atlas-paper-card').click();
  await page.waitForFunction(() => document.querySelector('[data-paper-reader-placeholder]').hidden);
  assert.equal(await page.locator('[data-paper-note-editor]').inputValue(), 'Saved immediately before leaving');
  await page.locator('[data-paper-document]').evaluate(el => { el.scrollTop = el.querySelector('[data-page="3"]').offsetTop; });
  await page.waitForFunction(() => document.querySelector('[data-paper-page-input]').value === '3');
  await page.reload();
  await page.locator('.atlas-paper-card').click();
  await page.waitForFunction(() => document.querySelector('[data-paper-reader-placeholder]').hidden);
  assert.equal(await page.locator('[data-paper-page-input]').inputValue(), '3');
});
test('delete removes the file, metadata, annotations AND notes', async t => {
  const page = await setup(t);
  await importPdf(page);
  await page.locator('[data-paper-note-editor]').fill('Must be deleted');
  page.once('dialog', d => d.accept('page note'));
  await page.locator('[data-paper-page-note]').click();
  await page.waitForFunction(() => document.querySelector('[data-paper-annotation-count]').textContent === '1');
  page.once('dialog', d => d.accept());
  await page.locator('[data-paper-delete]').click();
  await page.waitForFunction(() => document.querySelector('[data-paper-reader-view]').hidden);
  assert.deepEqual(await counts(page), [0,0,0,0]);
});
test('imports a PDF URL and explains failed requests', async t => {
  const page = await setup(t);
  assert.equal(await page.locator('[data-paper-url-input]').count(), 1);
  await page.locator('[data-paper-url-input]').fill(`${base}/missing.pdf`);
  await page.locator('[data-paper-url-import]').click();
  await page.waitForFunction(() => document.querySelector('[data-paper-status]').dataset.tone === 'error');
  assert.match(await page.locator('[data-paper-status]').textContent(), /HTTP 404/);
  assert.deepEqual(await counts(page), [0,0,0,0]);
  await page.route('**/cors-blocked.pdf', route => route.abort());
  await page.locator('[data-paper-url-input]').fill('https://blocked.invalid/cors-blocked.pdf');
  await page.locator('[data-paper-url-import]').click();
  await page.waitForFunction(() => document.querySelector('[data-paper-status]').textContent.includes('CORS'));
  await page.locator('[data-paper-url-input]').fill(`${base}/sample.pdf`);
  await page.locator('[data-paper-url-import]').click();
  await page.waitForFunction(() => !document.querySelector('[data-paper-reader-view]').hidden && document.querySelector('[data-paper-reader-placeholder]').hidden);
  assert.deepEqual((await counts(page)).slice(0,2), [1,1]);
});
test('text geometry, highlight, zoom, search, linked Markdown and JSON export', async t => {
  const page = await setup(t);
  await importPdf(page);
  const span = page.locator('.atlas-paper-text-layer span').first();
  assert.equal(Math.round(await span.evaluate(el => parseFloat(getComputedStyle(el).fontSize))), 20);
  await span.scrollIntoViewIfNeeded();
  await span.evaluate(el => { const range = document.createRange(); range.selectNodeContents(el); const s=window.getSelection(); s.removeAllRanges(); s.addRange(range); el.parentElement.dispatchEvent(new MouseEvent('mouseup', {bubbles:true})); });
  await page.locator('[data-paper-highlight-color]').first().click();
  await page.waitForFunction(() => document.querySelector('[data-paper-annotation-count]').textContent === '1');
  await page.locator('.atlas-paper-annotation-main').click();
  assert.equal(await page.locator('[data-paper-note-editor]').inputValue(), '', 'jumping must not silently edit notes');
  await page.locator('[data-paper-insert-quote]').click();
  assert.match(await page.locator('[data-paper-note-editor]').inputValue(), /\[第1页\]\(.+[&#]page=1\)/);
  await page.locator('[data-paper-zoom-in]').click();
  await page.waitForFunction(() => document.querySelector('[data-paper-zoom-label]').textContent === '110%' && document.querySelector('[data-paper-reader-placeholder]').hidden);
  assert.equal(await page.locator('.atlas-paper-highlight').count(), 1);
  await page.locator('[data-paper-pdf-search]').fill('page 3');
  await page.locator('[data-paper-pdf-search]').press('Enter');
  await page.waitForFunction(() => document.querySelector('[data-paper-page-input]').value === '3');
  const markdownDownload = page.waitForEvent('download');
  await page.locator('[data-paper-export-markdown]').click();
  const md = await markdownDownload;
  assert.match(md.suggestedFilename(), /-\d{4}-\d{4}\.md$/);
  assert.match(await fs.readFile(await md.path(), 'utf8'), /第1页/);
  const jsonDownload = page.waitForEvent('download');
  await page.locator('[data-paper-export-json]').click();
  const json = JSON.parse(await fs.readFile(await (await jsonDownload).path(), 'utf8'));
  assert.equal(json.annotations.length, 1);
  assert.match(json.note.markdown, /Research page/);
  const link = json.note.markdown.match(/\[第1页\]\(([^)]+)\)/)[1];
  await page.goto(link);
  await page.reload();
  await page.waitForFunction(() => !document.querySelector('[data-paper-reader-view]').hidden && document.querySelector('[data-paper-reader-placeholder]').hidden);
  assert.equal(await page.locator('[data-paper-page-input]').inputValue(), '1');
  assert.equal(await page.locator('.atlas-paper-highlight').count(), 1);
  assert.match(await page.locator('[data-paper-note-editor]').inputValue(), /Research page/);
});

test('latest notes can be exported even if browser storage is full', async t => {
  const page = await setup(t);
  await importPdf(page);
  await page.evaluate(() => {
    const original = IDBDatabase.prototype.transaction;
    IDBDatabase.prototype.transaction = function (stores, mode, options) {
      if (mode === 'readwrite') throw new DOMException('Storage is full', 'QuotaExceededError');
      return original.call(this, stores, mode, options);
    };
  });
  await page.locator('[data-paper-note-editor]').fill('Unsaved but recoverable notes');
  const downloaded = page.waitForEvent('download', {timeout:1500});
  await page.locator('[data-paper-export-markdown]').click();
  const file = await downloaded;
  assert.match(await fs.readFile(await file.path(), 'utf8'), /Unsaved but recoverable notes/);
});

test('unavailable IndexedDB falls back to explicit temporary reading', async t => {
  const context = await browser.newContext();
  t.after(() => context.close());
  await context.addInitScript(() => Object.defineProperty(window, 'indexedDB', {value:undefined}));
  const page = await context.newPage();
  await page.goto(`${base}/paper-library/`);
  await page.waitForFunction(() => document.querySelector('[data-paper-status]').textContent.includes('准备好'));
  assert.equal(await page.locator('[data-paper-storage-warning]').isVisible(), true);
  await importPdf(page);
  await page.locator('[data-paper-note-editor]').fill('Temporary note');
  await page.locator('[data-paper-reader-back]').click();
  assert.equal(await page.locator('.atlas-paper-card').count(), 1);
  await page.reload();
  assert.equal(await page.locator('.atlas-paper-card').count(), 0);
});

test('scan-only pages allow page notes and mobile layout stays in viewport', async t => {
  const page = await setup(t);
  await page.setViewportSize({width:390,height:844});
  await importPdf(page, 'scan.pdf', pdfFixture(1, false));
  assert.equal(await page.locator('.atlas-paper-text-layer span').count(), 0);
  assert.match(await page.locator('.atlas-paper-page-label').textContent(), /无文字层/);
  page.once('dialog', d => d.accept('Scan page note'));
  await page.locator('[data-paper-page-note]').click();
  await page.waitForFunction(() => document.querySelector('[data-paper-annotation-count]').textContent === '1');
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth), true);
});

test('ordinary articles never load the reader or PDF.js', async t => {
  const page = await setup(t);
  const resources = [];
  page.on('request', request => resources.push(request.url()));
  await page.goto(`${base}/prerequisite-math/`);
  assert.equal(await page.evaluate(() => typeof window.AtlasPaperLibrary), 'undefined');
  assert.equal(resources.some(url => /paper-library\.js|pdf(?:\.worker)?\.min\.mjs/.test(url)), false);
});


test('failed deletion reports an error and preserves the paper', async t => {
  const page = await setup(t);
  await importPdf(page);
  await page.evaluate(() => {
    const original = IDBObjectStore.prototype.delete;
    IDBObjectStore.prototype.delete = function () {
      if (this.name === 'papers') throw new DOMException('Delete denied', 'UnknownError');
      return original.apply(this, arguments);
    };
  });
  page.once('dialog', d => d.accept());
  await page.locator('[data-paper-delete]').click();
  await page.waitForFunction(() => document.querySelector('[data-paper-status]').dataset.tone === 'error');
  assert.deepEqual(await counts(page), [1,1,0,0]);
  assert.equal(await page.locator('[data-paper-reader-view]').isVisible(), true);
});

test('long PDFs render visible pages and rapid zoom leaves a single page tree', async t => {
  const page = await setup(t);
  await importPdf(page, 'long.pdf', pdfFixture(60));
  assert.equal(await page.locator('.atlas-paper-page').count(), 60);
  assert.ok(await page.locator('.atlas-paper-canvas').count() < 10);
  await page.locator('[data-paper-zoom-in]').evaluate(el => { el.click(); el.click(); el.click(); });
  await page.waitForFunction(() => !document.querySelector('[data-paper-zoom-in]').disabled);
  assert.equal(await page.locator('.atlas-paper-page').count(), 60);
  assert.equal(await page.locator('[data-paper-zoom-label]').textContent(), '110%');
});

test('fractional and out-of-range page numbers cannot corrupt saved progress', async t => {
  const page = await setup(t);
  await importPdf(page);
  await page.locator('[data-paper-page-input]').fill('2.5');
  await page.locator('[data-paper-page-input]').press('Tab');
  assert.equal(await page.locator('[data-paper-page-input]').inputValue(), '2');
  await page.locator('[data-paper-page-input]').fill('999');
  await page.locator('[data-paper-page-input]').press('Tab');
  assert.equal(await page.locator('[data-paper-page-input]').inputValue(), '3');
  await page.locator('[data-paper-reader-back]').click();
  await page.locator('.atlas-paper-card').click();
  await page.waitForFunction(() => !document.querySelector('[data-paper-reader-view]').hidden && document.querySelector('[data-paper-reader-placeholder]').hidden);
  assert.equal(await page.locator('[data-paper-page-input]').inputValue(), '3');
});
