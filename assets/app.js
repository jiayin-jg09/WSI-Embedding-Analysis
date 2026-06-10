/* ============================================================
   WSI Pan-Cancer site — interactivity
   Vanilla JS. Chart.js (optional) loaded via CDN before this file.
   ============================================================ */
(function () {
  "use strict";

  /* ---------- THEME ---------- */
  function applyTheme(theme) {
    if (theme === "light") {
      document.documentElement.setAttribute("data-theme", "light");
    } else {
      document.documentElement.removeAttribute("data-theme");
    }
  }
  // apply stored theme ASAP (this script is loaded in <head> with defer; the
  // inline pre-paint snippet in each page handles flash-of-wrong-theme)
  function initTheme() {
    var stored = null;
    try { stored = localStorage.getItem("wsi-theme"); } catch (e) {}
    if (stored) applyTheme(stored);

    var btn = document.querySelector(".theme-toggle");
    if (!btn) return;
    btn.addEventListener("click", function () {
      var isLight = document.documentElement.getAttribute("data-theme") === "light";
      var next = isLight ? "dark" : "light";
      applyTheme(next);
      try { localStorage.setItem("wsi-theme", next); } catch (e) {}
      document.dispatchEvent(new CustomEvent("themechange", { detail: { theme: next } }));
    });
  }

  /* ---------- NAV ACTIVE LINK ---------- */
  function initNav() {
    var here = location.pathname.split("/").pop() || "index.html";
    document.querySelectorAll(".nav-links a").forEach(function (a) {
      var href = (a.getAttribute("href") || "").split("/").pop();
      if (href === here) a.classList.add("active");
    });
  }

  /* ---------- LIGHTBOX ---------- */
  function initLightbox() {
    var zoomables = Array.prototype.slice.call(document.querySelectorAll("img[data-zoom]"));
    if (!zoomables.length) return;

    var box = document.createElement("div");
    box.className = "lightbox";
    box.setAttribute("role", "dialog");
    box.setAttribute("aria-modal", "true");
    box.innerHTML =
      '<button class="lb-close" aria-label="Close">&times;</button>' +
      '<button class="lb-prev" aria-label="Previous">&#8249;</button>' +
      '<img alt="">' +
      '<button class="lb-next" aria-label="Next">&#8250;</button>' +
      '<div class="lb-caption"></div>';
    document.body.appendChild(box);

    var imgEl = box.querySelector("img");
    var capEl = box.querySelector(".lb-caption");
    var current = 0;

    function captionFor(node) {
      if (node.getAttribute("data-caption")) return node.getAttribute("data-caption");
      var fig = node.closest("figure");
      var cap = fig && fig.querySelector("figcaption");
      return cap ? cap.textContent.trim() : (node.getAttribute("alt") || "");
    }
    function show(i) {
      current = (i + zoomables.length) % zoomables.length;
      var node = zoomables[current];
      imgEl.src = node.getAttribute("data-full") || node.src;
      imgEl.alt = node.alt || "";
      capEl.textContent = captionFor(node);
    }
    function open(i) { show(i); box.classList.add("open"); }
    function close() { box.classList.remove("open"); imgEl.src = ""; }

    zoomables.forEach(function (node, i) {
      node.addEventListener("click", function () { open(i); });
    });
    box.querySelector(".lb-close").addEventListener("click", close);
    box.querySelector(".lb-prev").addEventListener("click", function (e) { e.stopPropagation(); show(current - 1); });
    box.querySelector(".lb-next").addEventListener("click", function (e) { e.stopPropagation(); show(current + 1); });
    box.addEventListener("click", function (e) { if (e.target === box) close(); });
    document.addEventListener("keydown", function (e) {
      if (!box.classList.contains("open")) return;
      if (e.key === "Escape") close();
      else if (e.key === "ArrowLeft") show(current - 1);
      else if (e.key === "ArrowRight") show(current + 1);
    });
  }

  /* ---------- SORTABLE TABLES ---------- */
  function initSortableTables() {
    document.querySelectorAll("table.sortable").forEach(function (table) {
      var headers = table.querySelectorAll("thead th");
      headers.forEach(function (th, idx) {
        th.setAttribute("aria-sort", "none");
        if (!th.querySelector(".sort-ind")) {
          var ind = document.createElement("span");
          ind.className = "sort-ind";
          th.appendChild(ind);
        }
        th.addEventListener("click", function () {
          var tbody = table.tBodies[0];
          var rows = Array.prototype.slice.call(tbody.rows);
          var asc = th.getAttribute("aria-sort") !== "ascending";
          headers.forEach(function (h) { h.setAttribute("aria-sort", "none"); });
          th.setAttribute("aria-sort", asc ? "ascending" : "descending");

          rows.sort(function (a, b) {
            var av = cellValue(a.cells[idx]);
            var bv = cellValue(b.cells[idx]);
            if (typeof av === "number" && typeof bv === "number") {
              return asc ? av - bv : bv - av;
            }
            return asc ? String(av).localeCompare(String(bv))
                       : String(bv).localeCompare(String(av));
          });
          rows.forEach(function (r) { tbody.appendChild(r); });
        });
      });
    });

    function cellValue(cell) {
      if (!cell) return "";
      var raw = cell.getAttribute("data-sort");
      if (raw === null) raw = cell.textContent.trim();
      // pull a number out of things like "0.843 [0.803, 0.885]" or "< 1e-30"
      var cleaned = raw.replace(/[<>≈~]/g, "").trim();
      var m = cleaned.match(/-?\d+(\.\d+)?([eE][-+]?\d+)?/);
      if (m && /^[\s\d.,eE+\-<>≈~]+$/.test(cleaned)) {
        var num = parseFloat(m[0]);
        if (!isNaN(num)) return num;
      }
      return raw.toLowerCase();
    }
  }

  /* ---------- COUNT-UP STATS ---------- */
  function initCounters() {
    var nodes = document.querySelectorAll("[data-count]");
    if (!nodes.length || !("IntersectionObserver" in window)) {
      nodes.forEach(function (n) { n.textContent = n.getAttribute("data-count"); });
      return;
    }
    var obs = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (!entry.isIntersecting) return;
        animate(entry.target);
        obs.unobserve(entry.target);
      });
    }, { threshold: 0.4 });
    nodes.forEach(function (n) { obs.observe(n); });

    function animate(el) {
      var target = parseFloat(el.getAttribute("data-count"));
      var decimals = (el.getAttribute("data-decimals") || "0") | 0;
      var prefix = el.getAttribute("data-prefix") || "";
      var suffix = el.getAttribute("data-suffix") || "";
      var dur = 1100, start = null;
      function fmt(v) {
        return prefix + v.toLocaleString(undefined, {
          minimumFractionDigits: decimals, maximumFractionDigits: decimals
        }) + suffix;
      }
      function frame(ts) {
        if (start === null) start = ts;
        var p = Math.min((ts - start) / dur, 1);
        var eased = 1 - Math.pow(1 - p, 3);
        el.textContent = fmt(target * eased);
        if (p < 1) requestAnimationFrame(frame);
        else el.textContent = fmt(target);
      }
      requestAnimationFrame(frame);
    }
  }

  /* ---------- SCROLL REVEAL ---------- */
  function initReveal() {
    var nodes = document.querySelectorAll(".reveal");
    if (!nodes.length || !("IntersectionObserver" in window)) {
      nodes.forEach(function (n) { n.classList.add("in"); });
      return;
    }
    var obs = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) {
          entry.target.classList.add("in");
          obs.unobserve(entry.target);
        }
      });
    }, { threshold: 0.12 });
    nodes.forEach(function (n) { obs.observe(n); });
  }

  /* ---------- CHARTS ---------- */
  function cssVar(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }
  var registeredCharts = [];

  function initCharts() {
    if (!window.Chart) return;
    Chart.defaults.font.family =
      "-apple-system, BlinkMacSystemFont, 'Segoe UI', Inter, system-ui, sans-serif";

    buildDoughnut();
    buildSurvivalBars();
    buildClassifierBars();
    buildTargetBars();

    document.addEventListener("themechange", function () {
      registeredCharts.forEach(function (c) { recolor(c); c.chart.update(); });
    });
  }

  function themeColors() {
    return {
      text: cssVar("--text"),
      muted: cssVar("--muted"),
      accent: cssVar("--accent"),
      accentDim: cssVar("--accent-dim"),
      warn: cssVar("--warn"),
      border: cssVar("--border"),
      panel2: cssVar("--panel-2")
    };
  }

  function recolor(entry) {
    var c = entry.chart, col = themeColors();
    if (entry.type === "doughnut") {
      c.data.datasets[0].borderColor = cssVar("--panel");
      c.options.plugins.legend.labels.color = col.text;
    } else {
      if (c.options.scales.x) {
        c.options.scales.x.ticks.color = col.muted;
        c.options.scales.x.grid.color = col.border;
      }
      if (c.options.scales.y) {
        c.options.scales.y.ticks.color = col.muted;
        c.options.scales.y.grid.color = col.border;
      }
      if (c.options.plugins.legend) c.options.plugins.legend.labels.color = col.text;
      if (entry.recolorData) entry.recolorData(c, col);
    }
  }

  function register(chart, type, recolorData) {
    var entry = { chart: chart, type: type, recolorData: recolorData };
    registeredCharts.push(entry);
    return entry;
  }

  // Cohort composition doughnut (patients per cancer)
  function buildDoughnut() {
    var el = document.getElementById("cohortChart");
    if (!el) return;
    var col = themeColors();
    var labels = ["COAD", "STAD", "LIHC", "CESC", "ESCA", "READ", "ACC", "CHOL"];
    var data = [369, 343, 300, 242, 135, 133, 50, 36];
    var palette = ["#5eead4", "#3fb8a5", "#2dd4bf", "#0d9488", "#67e8f9",
                   "#38bdf8", "#a7f3d0", "#f8a07b"];
    var chart = new Chart(el, {
      type: "doughnut",
      data: { labels: labels,
        datasets: [{ data: data, backgroundColor: palette,
          borderColor: cssVar("--panel"), borderWidth: 2 }] },
      options: {
        responsive: true, maintainAspectRatio: false, cutout: "58%",
        plugins: {
          legend: { position: "right", labels: { color: col.text, boxWidth: 12, padding: 10 } },
          tooltip: { callbacks: { label: function (c) {
            var total = data.reduce(function (a, b) { return a + b; }, 0);
            var pct = Math.round((c.parsed / total) * 100);
            return " " + c.label + ": " + c.parsed + " patients (" + pct + "%)";
          } } }
        }
      }
    });
    register(chart, "doughnut");
  }

  // Per-cancer Age+sex vs best WSI C-index
  function buildSurvivalBars() {
    var el = document.getElementById("survivalChart");
    if (!el) return;
    var col = themeColors();
    var labels = ["ACC", "LIHC", "CESC", "COAD", "STAD", "ESCA", "READ", "CHOL"];
    var ageSex = [0.507, 0.478, 0.562, 0.552, 0.526, 0.609, 0.727, 0.454];
    var wsi    = [0.843, 0.688, 0.700, 0.604, 0.597, 0.594, 0.443, 0.548];
    var chart = new Chart(el, {
      type: "bar",
      data: { labels: labels, datasets: [
        { label: "Age + sex (clinical only)", data: ageSex, backgroundColor: col.muted, borderRadius: 4 },
        { label: "Best WSI model", data: wsi, backgroundColor: col.accent, borderRadius: 4 }
      ] },
      options: {
        responsive: true, maintainAspectRatio: false,
        scales: {
          x: { ticks: { color: col.muted }, grid: { color: col.border } },
          y: { min: 0.4, max: 0.9, ticks: { color: col.muted }, grid: { color: col.border },
               title: { display: true, text: "C-index", color: col.muted } }
        },
        plugins: {
          legend: { labels: { color: col.text } },
          tooltip: { callbacks: { afterBody: function () { return "Chance = 0.50"; } } },
          annotation: undefined
        }
      }
    });
    register(chart, "bar", function (c, col) {
      c.data.datasets[0].backgroundColor = col.muted;
      c.data.datasets[1].backgroundColor = col.accent;
    });
  }

  // Tumor-grade classifier AUCs (horizontal)
  function buildClassifierBars() {
    var el = document.getElementById("classifierChart");
    if (!el) return;
    var col = themeColors();
    var labels = ["KNN-5", "LogReg (L1)", "LinearSVC", "Bagging (LR)",
                  "LogReg (L2)", "SGD (Huber)", "KNN-3", "GaussianNB"];
    var auc = [0.799, 0.734, 0.722, 0.716, 0.704, 0.692, 0.686, 0.686];
    var chart = new Chart(el, {
      type: "bar",
      data: { labels: labels, datasets: [{
        label: "AUC", data: auc,
        backgroundColor: labels.map(function (_, i) { return i === 0 ? col.accent : col.accentDim; }),
        borderRadius: 4
      }] },
      options: {
        indexAxis: "y", responsive: true, maintainAspectRatio: false,
        scales: {
          x: { min: 0.5, max: 0.85, ticks: { color: col.muted }, grid: { color: col.border },
               title: { display: true, text: "AUC (LOO CV)", color: col.muted } },
          y: { ticks: { color: col.muted }, grid: { color: col.border } }
        },
        plugins: { legend: { display: false } }
      }
    });
    register(chart, "bar", function (c, col) {
      c.data.datasets[0].backgroundColor = c.data.labels.map(function (_, i) {
        return i === 0 ? col.accent : col.accentDim;
      });
    });
  }

  // Classification targets explored (AUC)
  function buildTargetBars() {
    var el = document.getElementById("targetChart");
    if (!el) return;
    var col = themeColors();
    var labels = ["Tumor vs normal", "Tumor grade", "BAP1 mutation", "PBRM1 mutation"];
    var auc = [1.0, 0.799, 0.639, 0.380];
    var colors = [col.muted, col.accent, col.warn, col.warn];
    var chart = new Chart(el, {
      type: "bar",
      data: { labels: labels, datasets: [{ label: "Best AUC", data: auc, backgroundColor: colors, borderRadius: 4 }] },
      options: {
        responsive: true, maintainAspectRatio: false,
        scales: {
          x: { ticks: { color: col.muted }, grid: { color: col.border } },
          y: { min: 0, max: 1, ticks: { color: col.muted }, grid: { color: col.border },
               title: { display: true, text: "Best AUC", color: col.muted } }
        },
        plugins: { legend: { display: false },
          tooltip: { callbacks: { afterBody: function () { return "Chance = 0.50"; } } } }
      }
    });
    register(chart, "bar", function (c, col) {
      c.data.datasets[0].backgroundColor = [col.muted, col.accent, col.warn, col.warn];
    });
  }

  /* ---------- ACCORDION (open via hash) ---------- */
  function initAccordionHash() {
    if (!location.hash) return;
    var target = document.querySelector(location.hash);
    if (target && target.tagName === "DETAILS") target.open = true;
  }

  /* ---------- BOOT ---------- */
  function boot() {
    initTheme();
    initNav();
    initLightbox();
    initSortableTables();
    initCounters();
    initReveal();
    initCharts();
    initAccordionHash();
  }
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
