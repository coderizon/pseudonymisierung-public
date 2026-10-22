(function () {
  const $ = (s) => document.querySelector(s);
  const $$ = (s) => [...document.querySelectorAll(s)];
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const fmtSize = (b) => b < 1024 ? b + " B" : b < 1048576 ? (b / 1024).toFixed(0) + " KB" : (b / 1048576).toFixed(1) + " MB";
  const fmtTime = (t) => new Date(t * 1000).toLocaleString("de-DE", { dateStyle: "short", timeStyle: "short" });
  const fmtDur = (s) => { s = Math.round(s); const m = Math.floor(s / 60); return m ? `${m} min ${String(s % 60).padStart(2, "0")} s` : `${s} s`; };

  let logOffset = 0, wasRunning = false, lastJobKey = null, seenOutput = null;

  // ---- tabs
  const TABS = ["daten", "starten", "pruefen", "ergebnisse"];
  function showTab(name, push = true) {
    if (!TABS.includes(name)) name = "daten";
    $$(".step").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
    $$(".panel").forEach((p) => {
      const show = p.dataset.panel === name;
      if (show && p.hidden) { p.classList.remove("enter"); void p.offsetWidth; p.classList.add("enter"); }
      p.hidden = !show;
    });
    if (push && location.hash !== "#" + name) history.replaceState(null, "", "#" + name);
    if (name === "pruefen") rvEnsure();
  }
  $$(".step").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));
  $$("[data-goto]").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.goto)));
  window.addEventListener("hashchange", () => showTab(location.hash.slice(1), false));

  // info modal
  const info = $("#info");
  const openInfo = () => { info.hidden = false; $("#info-close").focus(); };
  const closeInfo = () => { info.hidden = true; };
  $("#info-open").addEventListener("click", openInfo);
  $$(".mtab").forEach((b) => b.addEventListener("click", () => {
    $$(".mtab").forEach((x) => x.classList.toggle("active", x === b));
    $$(".mpanel").forEach((p) => { p.classList.toggle("active", p.dataset.mpanel === b.dataset.mtab); });
  }));
  $("#info-close").addEventListener("click", closeInfo);
  info.addEventListener("click", (ev) => { if (ev.target === info) closeInfo(); });
  document.addEventListener("keydown", (ev) => { if (ev.key === "Escape" && !info.hidden) closeInfo(); });

  async function api(path, opts) {
    const r = await fetch(path, opts);
    if (r.status === 401) { location.href = "/login"; return null; }
    return r.json();
  }
  const post = (path, body) => api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });

  function renderInputs(items) {
    const ul = $("#inputs"), sel = $("#input-select"), cur = sel.value;
    $("#in-count-2").textContent = items.length || ""; $("#in-count-2").hidden = !items.length;
    $(".step[data-tab=daten]").classList.toggle("done", items.length > 0);
    if (!items.length) { ul.innerHTML = '<li class="empty">Noch keine Dokumente hochgeladen.</li>'; sel.innerHTML = '<option value="">Zuerst ein Dokument hochladen</option>'; return; }
    ul.innerHTML = items.map((i) => `<li><button type="button" class="name link-plain" data-use="${esc(i.name)}" title="Für den Durchlauf auswählen">${esc(i.name)}</button>
      <span class="meta">${fmtSize(i.size)} · ${fmtTime(i.mtime)} <button class="tiny" data-del="${esc(i.name)}">löschen</button></span></li>`).join("");
    sel.innerHTML = items.filter((i) => i.ok).map((i) => `<option value="${esc(i.name)}">${esc(i.name)}</option>`).join("");
    if ([...sel.options].some((o) => o.value === cur)) sel.value = cur;
  }

  function renderOutputs(items) {
    const ul = $("#outputs");
    $(".step[data-tab=ergebnisse]").classList.toggle("done", items.length > 0);
    rvOptions(items);
    if (!items.length) { ul.innerHTML = '<li class="empty">Noch keine Ergebnisse.</li>'; return; }
    ul.innerHTML = items.map((o) => {
      const meta = [fmtTime(o.time), o.model, o.total ? fmtDur(o.total) : null, o.entities != null ? `${o.entities} Entitäten` : null].filter(Boolean).join(" · ");
      const d = encodeURIComponent(o.name);
      const rv = o.review;
      const state = rv ? (rv.status === "approved" ? `<span class="state ok">Freigegeben ${fmtTime(rv.approved)}</span>` : `<span class="state todo">Freigabe ausstehend</span>`) : "";
      const links = [
        o.files.includes("findings.json") ? `<button type="button" class="pill small" data-review="${esc(o.name)}">${rv && rv.status === "approved" ? "Erneut prüfen" : "Prüfen"}</button>` : "",
        `<a class="pill small" href="/outputs/${d}.zip">ZIP herunterladen</a>`,
        o.files.includes("pseudonymized.txt") ? `<a class="pill small" href="/outputs/${d}/pseudonymized.txt" target="_blank">Text ansehen</a>` : "",
        o.files.includes("timing.md") ? `<a class="pill small" href="/outputs/${d}/timing.md" target="_blank">Timing</a>` : "",
        o.files.includes("findings.json") ? `<a class="pill small" href="/outputs/${d}/findings.json" target="_blank">Findings</a>` : "",
      ].join("");
      return `<li><span class="name">${state}${esc(o.source)}</span><span class="actions">${links}</span><span class="meta">${esc(meta)}</span></li>`;
    }).join("");
  }
  $("#outputs").addEventListener("click", (ev) => {
    const name = ev.target.dataset && ev.target.dataset.review;
    if (!name) return;
    rv.dir = name; $("#rv-dir").value = name;
    showTab("pruefen");
    rvLoad();
  });

  function renderStatus(st) {
    const wrap = $("#statusbox"), box = $("#status"), fill = $("#bar-fill"), btn = $("#run-btn"), pct = $("#pct"), line = $("#progress-line"), dot = $("#run-dot"), next = $("#status-next");
    btn.disabled = st.running;
    btn.hidden = st.running;
    $("#run-foot").hidden = st.running;
    dot.hidden = !st.running;
    $(".step[data-tab=starten]").classList.toggle("done", !!(st.job && !st.running && st.job.rc === 0));
    if (!st.job) { wrap.hidden = true; fill.style.width = "0"; return; }
    wrap.hidden = false;
    const j = st.job, p = st.progress || {};
    let percent = 0, text = "", state = "";
    if (st.running) {
      const chunkFrac = p.chunk ? p.chunk.done / p.chunk.total : 0;
      percent = Math.min(99, ((p.step - 1 + chunkFrac) / (p.steps || 6)) * 100);
      if (p.step === 0) percent = 3;
      text = p.step ? `Schritt ${p.step}/${p.steps}: ${esc(p.label)}${p.chunk ? ` · Chunk ${p.chunk.done}/${p.chunk.total}` : ""}` : esc(p.label || "Startet …");
      state = "Durchlauf läuft …";
    } else {
      percent = j.rc === 0 ? 100 : 0;
      text = j.rc === 0 ? "Alle Schritte abgeschlossen. Der Vorschlag wartet auf Ihre Prüfung." : `Exit-Code ${j.rc}. Details im Protokoll.`;
      state = j.rc === 0 ? '<span class="ok">Fertig.</span>' : '<span class="fail">Abgebrochen.</span>';
    }
    box.innerHTML = `<p class="muted">${fmtDur(j.elapsed)} · ${state}</p>`;
    line.innerHTML = text;
    pct.textContent = Math.round(percent) + " %";
    fill.style.width = percent + "%";
    next.hidden = !(j.rc === 0 && !st.running);
    if (!st.running && j.rc === 0 && j.output && j.output !== seenOutput) {
      seenOutput = j.output;
      rv.dir = j.output;
      refresh().then(() => { $("#rv-dir").value = j.output; rvLoad(); });
    }
  }

  async function refresh() {
    const s = await api("/api/state");
    if (!s) return;
    renderInputs(s.inputs); renderOutputs(s.outputs); renderStatus(s.status);
    return s;
  }

  async function pollLog() {
    const r = await api(`/api/log?offset=${logOffset}`);
    if (!r) return;
    const jobKey = r.status.job ? r.status.job.started : null;
    if (jobKey !== lastJobKey) { lastJobKey = jobKey; $("#log").textContent = ""; logOffset = 0; if (jobKey) { const again = await api("/api/log?offset=0"); if (again) { r.text = again.text; r.offset = again.offset; } } }
    if (r.text) {
      const log = $("#log"), stick = log.scrollTop + log.clientHeight >= log.scrollHeight - 8;
      log.textContent += r.text; logOffset = r.offset;
      if (stick) log.scrollTop = log.scrollHeight;
    }
    renderStatus(r.status);
    if (wasRunning && !r.status.running) refresh();
    wasRunning = r.status.running;
  }

  // upload
  const drop = $("#upload"), fileInput = $("#file"), upStatus = $("#upload-status");
  $("#pick").addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", () => upload(fileInput.files));
  ["dragenter", "dragover"].forEach((e) => drop.addEventListener(e, (ev) => { ev.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((e) => drop.addEventListener(e, (ev) => { ev.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", (ev) => upload(ev.dataTransfer.files));

  async function upload(files) {
    if (!files || !files.length) return;
    const fd = new FormData();
    [...files].forEach((f, i) => fd.append("file" + i, f, f.name));
    upStatus.textContent = "Lade hoch …";
    try {
      const r = await api("/api/upload", { method: "POST", body: fd });
      if (!r) return;
      const parts = [];
      if (r.saved && r.saved.length) parts.push(`Hochgeladen: ${r.saved.join(", ")}`);
      if (r.errors && r.errors.length) parts.push(r.errors.join("; "));
      if (r.error) parts.push(r.error);
      upStatus.textContent = parts.join(" · ");
      await refresh();
      if (r.saved && r.saved.length) $("#input-select").value = r.saved[r.saved.length - 1];
    } catch (e) { upStatus.textContent = "Upload fehlgeschlagen: " + e; }
    fileInput.value = "";
  }

  // run
  $("#run").addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const file = $("#input-select").value, project = $("#project").value.trim() || "default";
    if (!file) { $("#run-msg").textContent = "Bitte zuerst ein Dokument hochladen."; return; }
    const r = await post("/api/run", { file, project });
    if (!r) return;
    $("#run-msg").textContent = r.error || "";
    if (r.error) return;
    logOffset = 0; $("#log").textContent = "";
    pollLog();
  });

  // pick / delete input
  $("#inputs").addEventListener("click", async (ev) => {
    const use = ev.target.dataset && ev.target.dataset.use;
    if (use) { $("#input-select").value = use; showTab("starten"); return; }
    const name = ev.target.dataset && ev.target.dataset.del;
    if (!name || !confirm(`„${name}“ aus den Eingaben löschen?`)) return;
    const r = await post("/api/delete-input", { file: name });
    if (r && r.error) alert(r.error);
    refresh();
  });

  // ---- review (Prüfen)
  const rv = { dir: null, data: null, edits: {}, added: [], outputs: [], loading: null };
  const SOURCE_TAG = { projekt: '<span class="tag proj">aus Projekt</span>', manuell: '<span class="tag man">angepasst</span>' };

  function rvOptions(outputs) {
    rv.outputs = outputs.filter((o) => o.files.includes("findings.json"));
    const sel = $("#rv-dir");
    sel.innerHTML = rv.outputs.map((o) => `<option value="${esc(o.name)}">${esc(o.source)} · ${fmtTime(o.time)}${o.review && o.review.status === "approved" ? " · freigegeben" : ""}</option>`).join("");
    if (rv.dir && rv.outputs.some((o) => o.name === rv.dir)) sel.value = rv.dir;
    else rv.dir = rv.outputs.length ? rv.outputs[0].name : null;
    const pending = rv.outputs.filter((o) => o.review && o.review.status === "pending").length;
    $("#rv-count").textContent = pending || ""; $("#rv-count").hidden = !pending;
    $(".step[data-tab=pruefen]").classList.toggle("done", rv.outputs.some((o) => o.review && o.review.status === "approved"));
    $("#rv-empty").hidden = rv.outputs.length > 0;
    $("#rv-body").hidden = !rv.outputs.length;
  }
  $("#rv-dir").addEventListener("change", () => { rv.dir = $("#rv-dir").value; rvLoad(); });

  function rvEnsure() {
    if (!rv.dir) return;
    if (!rv.data || rv.data.dir !== rv.dir) rvLoad();
  }

  async function rvLoad() {
    if (!rv.dir) return;
    const dir = rv.dir;
    $("#rv-msg").textContent = "Lade …";
    const d = await api(`/api/review?dir=${encodeURIComponent(dir)}`);
    if (!d) return;
    if (dir !== rv.dir) return;
    if (d.error) { $("#rv-msg").textContent = d.error; $("#rv-groups").innerHTML = ""; rv.data = null; return; }
    d.dir = dir;
    rv.data = d; rv.edits = {}; rv.added = [];
    $("#rv-msg").textContent = "";
    $("#rv-done").hidden = true;
    $("#rv-preview").textContent = ""; $("#rv-prevbox").open = false;
    $("#rv-filter").value = "";
    rvRender();
  }

  const cur = (e) => Object.assign({ category: e.category, replacement: e.replacement, action: e.action }, rv.edits[e.key] || {});
  const catLabel = (id) => { const c = rv.data && rv.data.categories.find((c) => c.id === id); return c ? c.label : id; };
  const catOptions = (sel) => rv.data.categories.map((c) => `<option value="${c.id}"${c.id === sel ? " selected" : ""}>${esc(c.label)}</option>`).join("");

  function rowHtml(e, v, idx) {
    const isNew = idx != null;
    const ref = isNew ? `data-new="${idx}"` : `data-key="${esc(e.key)}"`;
    const byKey = Object.fromEntries(rv.data.entities.map((x) => [x.key, x]));
    const sim = (e.similar || []).filter((k) => byKey[k]).map((k) => `<button type="button" data-like="${esc(k)}" title="Kategorie und Ersetzung von „${esc(byKey[k].text)}“ übernehmen">wie „${esc(byKey[k].text)}“</button>`).join(", ");
    const tags = [isNew ? '<span class="tag new">neu</span>' : (SOURCE_TAG[e.source] || ""), !isNew && e.count === 0 && !e.rejected ? '<span class="tag">vom System nicht ersetzt</span>' : ""].join("");
    return `<div class="rv-row${v.action === "keep" ? " kept" : ""}" ${ref}>
      <div class="rv-orig"><b>${esc(e.text)}</b>${e.count ? `<span class="n">×${e.count}</span>` : ""}${tags}
        ${e.context ? `<div class="ctx">${esc(e.context)}</div>` : ""}${sim ? `<div class="sim">ähnlich: ${sim}</div>` : ""}</div>
      <select class="rv-cat">${catOptions(v.category)}</select>
      <input type="text" class="rv-rep${!isNew && v.replacement !== (e.suggested || "") ? " changed" : ""}" list="rv-descs" value="${esc(v.replacement || "")}" placeholder="${v.category === "PERSON" ? "wird vergeben" : "Ersetzung"}" ${v.action === "keep" ? "disabled" : ""}>
      <label class="rv-keep"><input type="checkbox" class="rv-keepbox" ${v.action === "keep" ? "checked" : ""}> nicht ersetzen</label>
    </div>`;
  }

  function rvRender() {
    const d = rv.data; if (!d) return;
    $("#rv-project").value = d.project || "";
    $("#rv-sub").textContent = d.status === "approved"
      ? `Freigegeben am ${fmtTime(d.approved)}. Weitere Änderungen sind möglich, erneutes Freigeben erzeugt den Text neu.`
      : "Vorschlag des Systems, noch nicht freigegeben. Beschreibungen anpassen, Einträge ausnehmen oder ergänzen, dann freigeben.";
    const warn = $("#rv-warn");
    warn.innerHTML = (d.warnings || []).map((w) => `<p>${esc(w)}</p>`).join("");
    warn.hidden = !(d.warnings && d.warnings.length);
    const live = d.entities.filter((e) => !e.rejected);
    $("#rv-info").textContent = [`${live.length + rv.added.length} Einträge`, d.mode === "aligned" ? "mit Originaldokument abgeglichen" : "Zuordnung über Kontext", d.project_applied ? `${d.project_applied} aus dem Projekt übernommen` : null].filter(Boolean).join(" · ");

    const groups = new Map();
    d.entities.forEach((e) => {
      const v = cur(e);
      const g = e.rejected ? "__rejected" : v.category;
      if (!groups.has(g)) groups.set(g, []);
      groups.get(g).push(rowHtml(e, v));
    });
    rv.added.forEach((a, i) => {
      if (!groups.has(a.category)) groups.set(a.category, []);
      groups.get(a.category).push(rowHtml({ text: a.text, count: 0, context: "", similar: [], suggested: "" }, { category: a.category, replacement: a.replacement, action: "replace" }, i));
    });
    const order = d.categories.map((c) => c.id).concat(["__rejected"]);
    const head = '<div class="rv-head"><span>Original</span><span>Kategorie</span><span>Ersetzung im Text</span><span></span></div>';
    $("#rv-groups").innerHTML = order.filter((g) => groups.has(g)).map((g) => {
      const rows = groups.get(g);
      const open = !(g === "PERSON" || g === "__rejected" || g === "CONTACT");
      const title = g === "__rejected" ? "Vom System verworfen" : catLabel(g);
      const sum = g === "PERSON" ? "Kennungen vergibt das System, pro Projekt gleichbleibend" : g === "__rejected" ? "Kandidaten, die die Prüfung nicht bestanden haben. Bei Bedarf aktivieren." : "";
      return `<details class="rv-group" data-group="${esc(g)}"${open ? " open" : ""}><summary>${esc(title)}<span class="badge">${rows.length}</span>${sum ? `<span class="gsum">${esc(sum)}</span>` : ""}</summary>${head}${rows.join("")}</details>`;
    }).join("");
    $("#rv-descs").innerHTML = [...new Set(d.descriptions.concat(Object.values(rv.edits).map((x) => x.replacement).filter(Boolean)))].map((x) => `<option value="${esc(x)}">`).join("");
    $("#add-cat").innerHTML = catOptions("ORG_CLIENT");
    rvFilter();
  }

  function rvFilter() {
    const q = $("#rv-filter").value.trim().toLowerCase();
    $$("#rv-groups .rv-row").forEach((row) => {
      const hit = !q || row.textContent.toLowerCase().includes(q) || (row.querySelector(".rv-rep").value || "").toLowerCase().includes(q);
      row.hidden = !hit;
      row.classList.toggle("hit", !!q && hit);
    });
    if (q) $$("#rv-groups .rv-group").forEach((g) => { g.open = [...g.querySelectorAll(".rv-row")].some((r) => !r.hidden); });
  }
  $("#rv-filter").addEventListener("input", rvFilter);

  function rowTarget(el) {
    const row = el.closest(".rv-row");
    if (!row) return null;
    if (row.dataset.new != null) return { row, added: rv.added[+row.dataset.new] };
    const e = rv.data.entities.find((x) => x.key === row.dataset.key);
    return e ? { row, e } : null;
  }
  function setEdit(e, patch) { rv.edits[e.key] = Object.assign({}, cur(e), patch); }

  $("#rv-groups").addEventListener("change", (ev) => {
    const t = rowTarget(ev.target); if (!t) return;
    const patch = {};
    if (ev.target.classList.contains("rv-cat")) patch.category = ev.target.value;
    if (ev.target.classList.contains("rv-rep")) patch.replacement = ev.target.value.trim();
    if (ev.target.classList.contains("rv-keepbox")) patch.action = ev.target.checked ? "keep" : "replace";
    if (t.added) { Object.assign(t.added, patch); if (patch.action === "keep") { rv.added.splice(+t.row.dataset.new, 1); } rvRender(); return; }
    setEdit(t.e, patch);
    if (patch.category !== undefined) { rvRender(); return; }
    const v = cur(t.e);
    t.row.classList.toggle("kept", v.action === "keep");
    const rep = t.row.querySelector(".rv-rep");
    rep.disabled = v.action === "keep";
    rep.classList.toggle("changed", v.replacement !== (t.e.suggested || ""));
  });
  $("#rv-groups").addEventListener("click", (ev) => {
    const like = ev.target.dataset && ev.target.dataset.like;
    if (!like) return;
    const t = rowTarget(ev.target); if (!t || !t.e) return;
    const src = rv.data.entities.find((x) => x.key === like); if (!src) return;
    const sv = cur(src);
    setEdit(t.e, { category: sv.category, replacement: sv.category === "PERSON" ? "" : sv.replacement, action: sv.action });
    rvRender();
  });

  $("#add-btn").addEventListener("click", () => {
    const text = $("#add-text").value.trim(), category = $("#add-cat").value, replacement = $("#add-rep").value.trim();
    if (!text) { $("#add-text").focus(); return; }
    rv.added.push({ text, category, replacement });
    $("#add-text").value = ""; $("#add-rep").value = "";
    rvRender();
    $("#rv-addbox").open = true;
  });

  function rvPayload() {
    const edits = rv.data.entities.map((e) => Object.assign({ key: e.key }, cur(e)));
    return { dir: rv.dir, project: $("#rv-project").value.trim() || "default", edits, added: rv.added };
  }

  $("#rv-prev").addEventListener("click", async () => {
    if (!rv.data) return;
    $("#rv-msg").textContent = "Erzeuge Vorschau …";
    const r = await post("/api/review/preview", rvPayload());
    if (!r) return;
    if (r.error) { $("#rv-msg").textContent = r.error; return; }
    const reste = r.stats?.restnamen || [];
    $("#rv-msg").textContent = reste.length
      ? `Freigabe gesperrt: Namen stehen noch im Text (${reste.map((e) => `${e.text}: ${e.anzahl}×`).join(", ")}). Bitte Ersetzungen prüfen.`
      : `${r.stats?.searched || 0} fehlende Ersetzungen ergänzt. Keine Resttreffer der zum Ersetzen ausgewählten Personen. Vorschau bitte prüfen.`;
    $("#rv-preview").innerHTML = esc(r.text).replace(/\[[^\[\]\n]{1,160}\]/g, (m) => `<mark>${m}</mark>`);
    $("#rv-prevbox").open = true;
    $("#rv-prevbox").scrollIntoView({ behavior: "smooth", block: "start" });
  });

  $("#rv-approve").addEventListener("click", async () => {
    if (!rv.data) return;
    const btn = $("#rv-approve"); btn.disabled = true;
    $("#rv-msg").textContent = "Erzeuge Text …";
    const r = await post("/api/review/approve", rvPayload());
    btn.disabled = false;
    if (!r) return;
    if (r.error) { $("#rv-msg").textContent = r.error; return; }
    const st = r.stats || {};
    r.dir = rv.dir; rv.data = r; rv.edits = {}; rv.added = [];
    rvRender();
    $("#rv-msg").innerHTML = `<span class="ok" style="color:var(--ok);font-weight:700">Freigegeben.</span> ${st.replaced || 0} Ersetzungen${st.restored ? `, ${st.restored} im Original belassen` : ""}${st.searched ? `, ${st.searched} per Textsuche ergänzt` : ""}${st.unchanged ? `, ${st.unchanged} unverändert (nicht zuordenbar)` : ""}. Korrekturen im Projekt „${esc(r.project)}“ gemerkt.`;
    $("#rv-done").hidden = false;
    refresh();
  });

  $("#rv-reset").addEventListener("click", async () => {
    if (!rv.data || !confirm("Alle Anpassungen verwerfen und zum Vorschlag des Systems zurückkehren?")) return;
    const r = await post("/api/review/reset", { dir: rv.dir, project: $("#rv-project").value.trim() || null });
    if (!r) return;
    if (r.error) { $("#rv-msg").textContent = r.error; return; }
    r.dir = rv.dir; rv.data = r; rv.edits = {}; rv.added = [];
    $("#rv-preview").textContent = ""; $("#rv-prevbox").open = false;
    rvRender();
    $("#rv-msg").textContent = "Zurückgesetzt.";
    refresh();
  });

  // initial tab: hash, otherwise Status while a run is active, otherwise Daten
  (async () => {
    const s = await refresh();
    const h = location.hash.slice(1);
    showTab(TABS.includes(h) ? h : (s && s.status.running ? "starten" : "daten"), false);
    pollLog();
  })();
  setInterval(pollLog, 2000);
  setInterval(refresh, 30000);
})();
