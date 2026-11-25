// Small behaviours: loading state on submit, example readings, machine-type hint, what-if analysis.
// No inline scripts are used (the Content-Security-Policy forbids them); user-visible text is always set with textContent.
(function () {
  const $ = (sel, root = document) => root.querySelector(sel);
  let data = null;
  try { const el = document.getElementById("page-data"); data = el ? JSON.parse(el.textContent) : null; } catch (e) { data = null; }

  const fmtPct = (p) => (p < 0.0005 ? "<0.1%" : p > 0.9995 ? ">99.9%" : (p * 100).toFixed(1) + "%");

  // ---- 1. loading state on forms that have a submit button with id="submit-btn"
  document.querySelectorAll("form").forEach((form) => {
    form.addEventListener("submit", () => {
      const btn = $("#submit-btn", form);
      if (!btn) return;
      btn.dataset.label = btn.textContent;
      btn.disabled = true;
      btn.textContent = "Analyzing...";
      const note = $("#busy-note");
      if (note) note.hidden = false;
    });
  });
  window.addEventListener("pageshow", (e) => {  // coming back with the browser Back button
    if (!e.persisted) return;
    document.querySelectorAll("#submit-btn").forEach((b) => { b.disabled = false; if (b.dataset.label) b.textContent = b.dataset.label; });
    const note = $("#busy-note"); if (note) note.hidden = true;
  });

  // ---- 2. Analyze page: example readings and machine-type hint
  const analyzeForm = $("#analyze-form");
  if (analyzeForm && data) {
    const setField = (name, value) => { const f = $(`[name="${name}"]`, analyzeForm); if (f) f.value = value; };
    document.querySelectorAll(".example-btn").forEach((btn) => btn.addEventListener("click", () => {
      const ex = data.examples[Number(btn.dataset.example)];
      Object.entries(ex.values).forEach(([k, v]) => setField(k, v));
      const id = $("#machine_id");
      if (!id.value.trim()) { id.value = ex.machine_id; id.dispatchEvent(new Event("input")); }
      $("#air_temp").focus();
    }));
    const idInput = $("#machine_id"), typeSel = $("#type"), note = $("#type-note");
    const original = note ? note.textContent : "";
    const sync = () => {
      const known = (data.machines || []).find((m) => m.code.toLowerCase() === idInput.value.trim().toLowerCase());
      if (known) { typeSel.value = known.type; if (note) note.textContent = `Existing machine: its type is ${known.type} and cannot change.`; }
      else if (note) note.textContent = original;
    };
    idInput.addEventListener("input", sync);
    sync();
  }

  // ---- 3. Result page: what-if analysis (explore only, nothing is saved)
  const wf = $("#whatif-form");
  if (wf) {
    const out = $("#whatif-result");
    const show = (nodes) => { out.replaceChildren(...nodes); };
    const el = (tag, cls, text) => { const n = document.createElement(tag); if (cls) n.className = cls; if (text !== undefined) n.textContent = text; return n; };
    wf.addEventListener("submit", async (ev) => {
      ev.preventDefault();
      const changes = {};
      wf.querySelectorAll("input[name]").forEach((i) => { if (i.value.trim() !== "") changes[i.name] = i.value.trim(); });
      const btn = $("#whatif-btn"); btn.disabled = true; btn.textContent = "Calculating...";
      try {
        const resp = await fetch(wf.dataset.url, { method: "POST", headers: { "Content-Type": "application/json" },
                                                    body: JSON.stringify({ prediction_id: Number(wf.dataset.prediction), changes }) });
        const body = await resp.json();
        if (!resp.ok) {
          const msgs = Object.values(body.errors || {}); if (!msgs.length) msgs.push(body.error || "Something went wrong.");
          show([el("div", "notice notice-warning", msgs.join(" "))]);
        } else {
          const box = el("div", "whatif-out");
          const row = el("div", "d-flex align-items-center gap-3 flex-wrap");
          row.append(el("span", "prob " + body.risk_level.toLowerCase(), fmtPct(body.probability)), el("span", "badge-risk lg " + body.risk_level.toLowerCase(), body.risk_level));
          const delta = body.delta * 100;
          box.append(row, el("div", "mt-2", `Health score ${body.health_score}/100. Compared with the actual analysis (${fmtPct(body.baseline_probability)}): ${delta >= 0 ? "+" : ""}${delta.toFixed(1)} percentage points. ${body.action}.`));
          body.warnings.forEach((w) => box.append(el("div", "small-note mt-1", w)));
          show([box]);
        }
      } catch (e) {
        show([el("div", "notice notice-danger", "The what-if analysis could not be reached. Check that the application is running and try again.")]);
      } finally { btn.disabled = false; btn.textContent = "Calculate"; }
    });
    $("#whatif-reset").addEventListener("click", () => {
      wf.querySelectorAll("input[name]").forEach((i) => { i.value = i.dataset.original; });
      out.replaceChildren();
    });
  }
})();
