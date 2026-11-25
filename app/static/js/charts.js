// Charts for Machine trend, Model Insights and About. All data comes from JSON embedded by the server.
(function () {
  const css = getComputedStyle(document.documentElement);
  const v = (name) => css.getPropertyValue(name).trim();
  const COLOR = { ok: v("--ok"), warn: v("--warn-fill"), crit: v("--crit"), primary: v("--primary"), muted: v("--muted"),
                  grid: "#e3e6ea", lr: "#8a94a1", rf: "#2a5a87", xgb: "#6a5d9a", ens: "#2f8f83" };
  Chart.defaults.font.family = v("--font");
  Chart.defaults.font.size = 12;
  Chart.defaults.color = COLOR.muted;
  Chart.defaults.maintainAspectRatio = false;
  Chart.defaults.plugins.legend.labels.boxWidth = 10;
  Chart.defaults.plugins.legend.labels.boxHeight = 10;
  Chart.defaults.animation = false;

  const el = (id) => document.getElementById(id);
  const dataEl = el("page-data");
  if (!dataEl) return;
  const data = JSON.parse(dataEl.textContent);
  const page = document.body.dataset.page;
  const axis = (title, extra = {}) => ({ title: { display: !!title, text: title }, grid: { color: COLOR.grid }, ...extra });

  // ---------------------------------------------------------------- machine risk trend
  if (page === "machine_detail" && data.trend && el("chart-trend")) {
    const t = data.trend;
    const level = (p) => (p >= 0.7 ? COLOR.crit : p >= 0.3 ? COLOR.warn : COLOR.ok);
    const flat = (y, label) => ({ label, data: t.map(() => y), borderColor: "#aab2bd", borderDash: [5, 4], borderWidth: 1.2, pointRadius: 0 });
    new Chart(el("chart-trend"), {
      type: "line",
      data: { labels: t.map((r) => r.created_at.slice(5, 16)),
              datasets: [{ label: "Failure probability", data: t.map((r) => r.probability * 100), borderColor: COLOR.primary, borderWidth: 2,
                           pointBackgroundColor: t.map((r) => level(r.probability)), pointRadius: 5, tension: 0 },
                         flat(30, "Warning (30%)"), flat(70, "Critical (70%)")] },
      options: { plugins: { legend: { display: false }, tooltip: { callbacks: { afterLabel: (c) => (c.datasetIndex === 0 ? `${t[c.dataIndex].risk_level}, ${t[c.dataIndex].model_label}` : "") } } },
                 scales: { x: axis("Analysis time", { grid: { display: false }, offset: true }), y: axis("Failure probability (%)", { min: 0, max: 100 }) } },
    });
  }

  // ---------------------------------------------------------------- model insights
  if (page === "insights") {
    const names = { air_temp: "Air temperature", process_temp: "Process temperature", rpm: "Rotational speed", torque: "Torque",
                    tool_wear: "Tool wear", temp_diff: "Temperature difference", power_w: "Power (W)", wear_torque: "Tool wear x torque",
                    type_H: "Type H", type_L: "Type L", type_M: "Type M" };
    const nice = (f) => names[f] || f;
    const line = (key, pts, extra = {}) => ({ label: data.labels[key], data: pts, showLine: true, pointRadius: 0, borderWidth: 2,
                                               borderColor: COLOR[key], backgroundColor: COLOR[key], tension: 0, ...extra });
    const xy = (xs, ys) => xs.map((x, i) => ({ x, y: ys[i] }));
    const unit = (title) => ({ type: "linear", min: 0, max: 1, title: { display: true, text: title }, grid: { color: COLOR.grid } });
    const diag = (label) => ({ label, data: [{ x: 0, y: 0 }, { x: 1, y: 1 }], showLine: true, pointRadius: 0, borderColor: "#aab2bd", borderDash: [5, 4], borderWidth: 1.5 });

    new Chart(el("chart-pr"), { type: "scatter",
      data: { datasets: [...data.order.map((k) => line(k, xy(data.test[k].pr_curve.recall, data.test[k].pr_curve.precision))),
        { label: "Random model", data: [{ x: 0, y: data.baseline_rate }, { x: 1, y: data.baseline_rate }], showLine: true, pointRadius: 0, borderColor: "#aab2bd", borderDash: [5, 4], borderWidth: 1.5 }] },
      options: { scales: { x: unit("Recall"), y: unit("Precision") } } });
    new Chart(el("chart-roc"), { type: "scatter",
      data: { datasets: [...data.order.map((k) => line(k, xy(data.test[k].roc_curve.fpr, data.test[k].roc_curve.tpr))), diag("Random model")] },
      options: { scales: { x: unit("False positive rate"), y: unit("True positive rate (recall)") } } });

    if (data.type_curves && el("chart-pr-types-cv")) {
      const tcol = { TWF: "#c4651c", HDF: "#2a5a87", PWF: "#2f8f83", OSF: "#6a5d9a" };
      ["cv", "test"].forEach((which) => {
        const sets = Object.entries(data.type_curves).filter(([, v]) => v[which]).map(([code, v]) => ({
          label: code, data: xy(v[which].recall, v[which].precision), showLine: true, pointRadius: 0, borderWidth: 2,
          borderColor: tcol[code] || COLOR.primary, backgroundColor: tcol[code] || COLOR.primary, tension: 0 }));
        new Chart(el(which === "cv" ? "chart-pr-types-cv" : "chart-pr-types-test"), { type: "scatter", data: { datasets: sets },
          options: { scales: { x: unit("Recall"), y: unit("Precision") } } });
      });
    }
    const bars = new Chart(el("chart-bars"), { type: "bar",
      data: { labels: ["Precision", "Recall", "F1"], datasets: data.order.map((k) => ({ label: data.labels[k], data: [], backgroundColor: COLOR[k] })) },
      options: { scales: { x: axis(null, { grid: { display: false } }), y: axis(null, { min: 0, max: 1 }) } } });
    function setThreshold(thr) {
      bars.data.datasets.forEach((ds, i) => { const t = data.test[data.order[i]].thresholds[thr]; ds.data = [t.precision, t.recall, t.f1]; });
      bars.update();
      el("bars-hint").textContent = `threshold ${Math.round(thr * 100)}%`;
      document.querySelectorAll(".thr-block").forEach((b) => (b.hidden = b.dataset.thr !== thr));
      document.querySelectorAll("#thr-toggle button").forEach((b) => b.classList.toggle("active", b.dataset.thr === thr));
    }
    document.querySelectorAll("#thr-toggle button").forEach((b) => b.addEventListener("click", () => setThreshold(b.dataset.thr)));
    setThreshold("0.3");

    new Chart(el("chart-cal"), { type: "scatter",
      data: { datasets: [diag("Perfect calibration"),
        ...data.order.map((k) => line(k, data.test[k].reliability.map((r) => ({ x: r.mean_predicted, y: r.observed, n: r.n })), { showLine: false, pointRadius: 5 }))] },
      options: { scales: { x: unit("Predicted probability"), y: unit("Observed failure rate") },
                 plugins: { tooltip: { callbacks: { label: (c) => `${c.dataset.label}: predicted ${(c.raw.x * 100).toFixed(1)}%, observed ${(c.raw.y * 100).toFixed(1)}%` + (c.raw.n ? ` (${c.raw.n} machines)` : "") } } } } });

    let permChart, nativeChart;
    function drawImportance(key) {
      const imp = data.test[key].importance;
      const nativeSrc = imp.native || data.test[data.default].importance.native;   // the ensemble has no importance of its own
      const nat = nativeSrc.values.slice(0, 8);
      const mk = (canvas, rows, valueKey, axisTitle, errKey) => new Chart(el(canvas), { type: "bar",
        data: { labels: rows.map((r) => nice(r.feature)), datasets: [{ data: rows.map((r) => r[valueKey]), backgroundColor: COLOR[imp.native ? key : data.default] }] },
        options: { indexAxis: "y", plugins: { legend: { display: false },
                   tooltip: { callbacks: { label: (c) => `${c.parsed.x.toFixed(4)}` + (errKey ? ` (+/- ${rows[c.dataIndex][errKey].toFixed(4)})` : "") } } },
                   scales: { x: axis(axisTitle), y: { grid: { display: false } } } } });
      if (permChart) permChart.destroy();
      if (nativeChart) nativeChart.destroy();
      permChart = mk("chart-imp-perm", imp.permutation, "mean", "Drop in PR-AUC when shuffled", "std");
      nativeChart = mk("chart-imp-native", nat, "value", nativeSrc.kind, null);
      el("native-kind").textContent = nativeSrc.kind + (imp.native ? "" : ` (shown for ${data.labels[data.default]}, since the ensemble has none of its own)`) + (key === "lr" ? ". Positive raises predicted risk, negative lowers it." : ".");
      document.querySelectorAll("#imp-toggle button").forEach((b) => b.classList.toggle("active", b.dataset.model === key));
    }
    document.querySelectorAll("#imp-toggle button").forEach((b) => b.addEventListener("click", () => drawImportance(b.dataset.model)));
    drawImportance(data.default);
  }

  // ---------------------------------------------------------------- about / EDA
  if (page === "about" && data.eda) {
    const fmt = (bin) => { const m = bin.match(/(-?[\d.]+), (-?[\d.]+)/); return m ? `${Math.round(m[1])}-${Math.round(m[2])}` : bin; };
    const draw = (id, rows, xTitle) => new Chart(el(id), { type: "bar",
      data: { labels: rows.map((r) => fmt(r.bin)),
              datasets: [{ type: "bar", label: "Failure rate", data: rows.map((r) => r.failure_rate_pct), backgroundColor: COLOR.primary, categoryPercentage: 0.9 },
                         { type: "line", label: "Overall rate", data: rows.map(() => data.eda.overall), borderColor: "#8a94a1", borderDash: [5, 4], borderWidth: 1.5, pointRadius: 0 }] },
      options: { plugins: { tooltip: { callbacks: { afterLabel: (c) => c.datasetIndex === 0 ? `${rows[c.dataIndex].failures} failures in ${rows[c.dataIndex].count} records` : "" } } },
                 scales: { x: axis(xTitle, { grid: { display: false } }), y: axis("Failure rate (%)", { beginAtZero: true }) } } });
    draw("chart-eda-wear", data.eda.rate_by_tool_wear, "Tool wear (min)");
    draw("chart-eda-rpm", data.eda.rate_by_rpm, "Rotational speed (rpm)");
    draw("chart-eda-torque", data.eda.rate_by_torque, "Torque (Nm)");
    draw("chart-eda-power", data.eda.rate_by_power, "Power (W)");
  }
})();
