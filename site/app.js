// NT P3 Dashboard V1 — leaderboard page logic
(async function () {
  const DATA_ROOT = 'data/latest';
  let allRows = [];
  let ranges = {};
  let sortState = { key: 'lens', dir: 'desc' };
  let activePresetIds = new Set(['overall']);

  const COLUMNS = [
    { key: 'rank', field: 'display_rank', type: 'num', defaultDir: 'asc' },
    { key: 'model', field: 'model_id', type: 'string', defaultDir: 'asc' },
    { key: 'lens', field: 'lens_score', type: 'num', defaultDir: 'desc' },
    { key: 'thai', field: 'thai_score_rate', type: 'num', defaultDir: 'desc' },
    { key: 'math', field: 'math_score_rate', type: 'num', defaultDir: 'desc' },
    { key: 'overall', field: 'overall_score_rate', type: 'num', defaultDir: 'desc' },
    { key: 'p50', field: 'latency_p50_ms', type: 'num', defaultDir: 'asc' },
    { key: 'p95', field: 'latency_p95_ms', type: 'num', defaultDir: 'asc' },
    { key: 'qmin', field: 'questions_per_min', type: 'num', defaultDir: 'desc' },
    { key: 'cmin', field: 'correct_per_min', type: 'num', defaultDir: 'desc' },
    { key: 'n', field: 'item_count', type: 'num', defaultDir: 'desc' },
  ];

  const PRESET_GROUPS = [
    {
      label: 'Metric',
      presets: [
        { id: 'overall', label: 'Best Overall', weights: { balanced_quality_score: 0.65, overall_score_rate: 0.35 } },
        { id: 'thai', label: 'Best Thai', weights: { thai_score_rate: 0.8, balanced_quality_score: 0.2 } },
        { id: 'math', label: 'Best Math', weights: { math_score_rate: 0.8, balanced_quality_score: 0.2 } },
        { id: 'fastest', label: 'Fastest', weights: { questions_per_min: 1.0 } },
        { id: 'practical', label: 'Best Practical', weights: { overall_score_rate: 0.38, parseable_rate: 0.22, answer_only_compliance_rate: 0.18, questions_per_min: 0.22 } },
        { id: 'small', label: 'Best Small Model', weights: { fit_score: 0.55, balanced_quality_score: 0.3, questions_per_min: 0.15 } },
      ],
    },
    {
      label: 'Scenario',
      presets: [
        { id: 'thai-homework', label: 'Thai homework', weights: { thai_score_rate: 0.52, overall_score_rate: 0.2, parseable_rate: 0.16, answer_only_compliance_rate: 0.08, questions_per_min: 0.04 } },
        { id: 'math-reasoning', label: 'Math reasoning', weights: { math_score_rate: 0.55, overall_score_rate: 0.2, parseable_rate: 0.12, answer_only_compliance_rate: 0.05, questions_per_min: 0.08 } },
        { id: 'classroom', label: 'Fast classroom use', weights: { questions_per_min: 0.45, overall_score_rate: 0.24, parseable_rate: 0.16, answer_only_compliance_rate: 0.15 } },
        { id: 'macmini', label: 'Best on 16GB Mac mini or similar', weights: { fit_score: 0.35, balanced_quality_score: 0.4, questions_per_min: 0.25 } },
      ],
    },
  ];

  const PRESETS = PRESET_GROUPS.flatMap(group => group.presets);
  const LOWER_IS_BETTER = new Set(['latency_p50_ms', 'latency_p95_ms']);

  const FIELD_LABELS = {
    balanced_quality_score: 'BQS',
    overall_score_rate: 'Overall',
    parseable_rate: 'Parseable',
    thai_score_rate: 'Thai',
    math_score_rate: 'Math',
    questions_per_min: 'Q/min',
    correct_per_min: 'Correct/min',
    latency_p50_ms: 'p50',
    latency_p95_ms: 'p95',
    answer_only_compliance_rate: 'Compliance',
    fit_score: '16GB fit',
  };

  const BADGE_CLASS = {
    'Best Quality': 'badge-gold',
    'Best Thai': 'badge-blue',
    'Best Math': 'badge-violet',
    'Fastest on Testbed': 'badge-green',
    'Best Small Model': 'badge-teal',
  };

  async function loadJSON(filename) {
    const resp = await fetch(`${DATA_ROOT}/${filename}`);
    if (!resp.ok) throw new Error(`Failed to load ${filename}: ${resp.status}`);
    return resp.json();
  }

  try {
    const current = await loadJSON('current.json');
    const manifest = current.manifest;
    const leaderboard = current.leaderboard;

    renderMeta(current, manifest, leaderboard);
    renderMethodology(manifest, leaderboard);
    allRows = leaderboard.rows;
    ranges = computeRanges(allRows);
    initPresetControls();
    initNerdMode();
    initSorting();
    renderLeaderboard();
  } catch (err) {
    document.getElementById('leaderboard-body').innerHTML =
      `<tr><td colspan="13" style="text-align:center;padding:40px;color:#991b1b;">
        Failed to load data. Make sure snapshot files exist in ${DATA_ROOT}/.
        <br><small>${err.message}</small>
      </td></tr>`;
  }

  function renderMeta(current, manifest, leaderboard) {
    const label = document.getElementById('benchmark-label');
    label.textContent = manifest.benchmark_label + ' \u2014 ' + manifest.benchmark_scope;

    const modelCount = leaderboard.rows.length;
    const itemCount = leaderboard.rows.length > 0 ? leaderboard.rows[0].item_count : 0;
    const published = new Date(manifest.published_at);
    const publishedStr = published.toLocaleDateString('en-GB', {
      year: 'numeric', month: 'short', day: 'numeric',
      hour: '2-digit', minute: '2-digit', timeZoneName: 'short',
    });

    const items = [
      ['Snapshot', manifest.snapshot_id],
      ['Published', publishedStr],
      ['Testbed', manifest.testbed.host_label],
      ['Models', modelCount],
      ['Items/model', itemCount],
    ];

    document.getElementById('meta-bar').innerHTML = items.map(([k, v]) =>
      `<div class="meta-item"><dt>${k}:</dt><dd>${v}</dd></div>`
    ).join('');

    const latestSource = Array.isArray(current.sources) && current.sources.length
      ? current.sources[current.sources.length - 1].snapshot_id
      : manifest.snapshot_id;
    document.getElementById('snapshot-id-label').textContent =
      'Current aggregate: ' + current.current_id + ' | latest source snapshot: ' + latestSource;
  }

  function renderMethodology(manifest, leaderboard) {
    const el = document.getElementById('methodology-card');
    if (!el) return;

    const itemCount = leaderboard.rows.length > 0 ? leaderboard.rows[0].item_count : 0;
    const modelCount = leaderboard.rows.length;

    el.innerHTML = `
      <h2>Methodology</h2>
      <p class="section-copy">
        This leaderboard evaluates ${modelCount} local model${modelCount === 1 ? '' : 's'} on ${itemCount} NT Grade 3 text-only multiple-choice items per model.
        Presets are client-side ranking lenses over the same published metrics. They do not change benchmark scores or factual badges.
      </p>
      <dl class="definition-list">
        <div>
          <dt>BQS</dt>
          <dd>Balanced Quality Score, the simple average of Thai score rate and Math score rate.</dd>
        </div>
        <div>
          <dt>Lens Score</dt>
          <dd>A 0-100 blended score from the selected presets. Combine presets to express tradeoffs such as Thai quality plus speed.</dd>
        </div>
        <div>
          <dt>Q/min</dt>
          <dd>Questions per minute on the testbed. This is the main speed signal for presets.</dd>
        </div>
        <div>
          <dt>p50 / p95</dt>
          <dd>Median and tail latency per question. Lower is faster; p95 catches slow responses.</dd>
        </div>
      </dl>
    `;
  }

  function fmtRate(value) {
    return (value * 100).toFixed(1) + '%';
  }

  function fmtScore(value) {
    return Math.round(value).toString();
  }

  function fmtNum(value) {
    return typeof value === 'number' ? value.toLocaleString() : value;
  }

  function fitScore(row) {
    if (row.ram_fit_class === 'fits_comfortably_16gb') return 1;
    if (row.ram_fit_class === 'fits_tightly_16gb') return 0.62;
    return 0.35;
  }

  function fieldValue(row, field) {
    return field === 'fit_score' ? fitScore(row) : row[field];
  }

  function formatField(field, row) {
    if (field === 'fit_score') {
      if (row.ram_fit_class === 'fits_comfortably_16gb') return 'comfortable';
      if (row.ram_fit_class === 'fits_tightly_16gb') return 'tight';
      return row.ram_fit_class || 'unknown';
    }
    const value = row[field];
    if (field.endsWith('_rate') || field === 'balanced_quality_score' || field === 'overall_score_rate') return fmtRate(value);
    if (field === 'questions_per_min' || field === 'correct_per_min') return Number(value).toFixed(1);
    if (field === 'latency_p50_ms' || field === 'latency_p95_ms') return fmtNum(value) + 'ms';
    return fmtNum(value);
  }

  function computeRanges(rows) {
    const fields = new Set();
    PRESETS.forEach(preset => Object.keys(preset.weights).forEach(field => fields.add(field)));
    const next = {};
    fields.forEach(field => {
      const values = rows.map(row => fieldValue(row, field)).filter(value => typeof value === 'number' && !Number.isNaN(value));
      next[field] = { min: Math.min(...values), max: Math.max(...values) };
    });
    return next;
  }

  function normalize(row, field) {
    const value = fieldValue(row, field);
    const range = ranges[field];
    if (typeof value !== 'number' || !range || range.max === range.min) return 0.5;
    const raw = (value - range.min) / (range.max - range.min);
    return LOWER_IS_BETTER.has(field) ? 1 - raw : raw;
  }

  function activePresets() {
    return PRESETS.filter(preset => activePresetIds.has(preset.id));
  }

  function scoreRow(row) {
    const contributions = {};
    let weightedScore = 0;
    let totalWeight = 0;

    activePresets().forEach(preset => {
      Object.entries(preset.weights).forEach(([field, weight]) => {
        const contribution = normalize(row, field) * weight;
        contributions[field] = (contributions[field] || 0) + contribution;
        weightedScore += contribution;
        totalWeight += weight;
      });
    });

    const lensScore = totalWeight ? (weightedScore / totalWeight) * 100 : row.balanced_quality_score * 100;
    const reasonFields = Object.entries(contributions)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 3)
      .map(([field]) => field);

    return {
      ...row,
      lens_score: lensScore,
      lens_reason: reasonFields.map(field => `${FIELD_LABELS[field]} ${formatField(field, row)}`).join(' · '),
      display_rank: 0,
    };
  }

  function scoreRows(rows) {
    const ranked = rows.map(scoreRow).sort((a, b) =>
      b.lens_score - a.lens_score ||
      b.balanced_quality_score - a.balanced_quality_score ||
      a.rank - b.rank
    );
    ranked.forEach((row, idx) => {
      row.display_rank = idx + 1;
    });
    return ranked;
  }

  function initPresetControls() {
    const root = document.getElementById('preset-groups');
    root.innerHTML = PRESET_GROUPS.map(group => {
      const buttons = group.presets.map(preset =>
        `<button type="button" class="preset-chip" data-preset-id="${preset.id}" aria-pressed="${activePresetIds.has(preset.id)}">${preset.label}</button>`
      ).join('');
      return `<fieldset class="preset-group"><legend>${group.label}</legend><div class="preset-buttons">${buttons}</div></fieldset>`;
    }).join('');

    root.querySelectorAll('.preset-chip').forEach(button => {
      button.addEventListener('click', () => {
        const id = button.dataset.presetId;
        if (activePresetIds.has(id)) {
          activePresetIds.delete(id);
        } else {
          activePresetIds.add(id);
        }
        if (activePresetIds.size === 0) activePresetIds.add('overall');
        sortState = { key: 'lens', dir: 'desc' };
        updatePresetControls();
        renderLeaderboard();
      });
    });

    document.getElementById('reset-presets').addEventListener('click', () => {
      activePresetIds = new Set(['overall']);
      sortState = { key: 'lens', dir: 'desc' };
      updatePresetControls();
      renderLeaderboard();
    });
  }

  function updatePresetControls() {
    document.querySelectorAll('.preset-chip').forEach(button => {
      const active = activePresetIds.has(button.dataset.presetId);
      button.classList.toggle('active', active);
      button.setAttribute('aria-pressed', active ? 'true' : 'false');
    });
  }

  function initNerdMode() {
    const checkbox = document.getElementById('nerd-mode');
    const wrapper = document.getElementById('leaderboard-wrapper');
    checkbox.addEventListener('change', () => {
      wrapper.classList.toggle('nerd-mode', checkbox.checked);
    });
  }

  function sortRows(rows) {
    const column = COLUMNS.find(c => c.key === sortState.key);
    if (!column) return rows;
    const mult = sortState.dir === 'asc' ? 1 : -1;
    return [...rows].sort((a, b) => {
      const av = a[column.field];
      const bv = b[column.field];
      if (column.type === 'string') return mult * String(av ?? '').localeCompare(String(bv ?? ''));
      const an = typeof av === 'number' ? av : 0;
      const bn = typeof bv === 'number' ? bv : 0;
      return mult * (an - bn) || a.rank - b.rank;
    });
  }

  function onHeaderClick(key) {
    const column = COLUMNS.find(c => c.key === key);
    if (!column) return;

    if (sortState.key === key) {
      sortState = sortState.dir === column.defaultDir
        ? { key, dir: column.defaultDir === 'asc' ? 'desc' : 'asc' }
        : { key: 'lens', dir: 'desc' };
    } else {
      sortState = { key, dir: column.defaultDir };
    }
    renderLeaderboard();
  }

  function updateSortIndicators() {
    document.querySelectorAll('th.sortable').forEach(th => {
      const active = sortState.key === th.dataset.sortKey;
      th.classList.toggle('sort-active', active);
      const ind = th.querySelector('.sort-ind');
      if (ind) ind.textContent = active ? (sortState.dir === 'asc' ? ' ↑' : ' ↓') : '';
    });
  }

  function initSorting() {
    document.querySelectorAll('th.sortable').forEach(th => {
      th.addEventListener('click', () => onHeaderClick(th.dataset.sortKey));
    });
  }

  function tierClass(field, value) {
    if (value == null || typeof value !== 'number' || Number.isNaN(value)) return '';
    if (field === 'latency_p50_ms' || field === 'latency_p95_ms') {
      if (value <= 600) return 'tier-strong';
      if (value <= 1200) return 'tier-good';
      if (value <= 2500) return 'tier-ok';
      return 'tier-weak';
    }
    if (field === 'questions_per_min' || field === 'correct_per_min') {
      const score = normalize({ [field]: value }, field);
      if (score >= 0.75) return 'tier-strong';
      if (score >= 0.5) return 'tier-good';
      if (score >= 0.25) return 'tier-ok';
      return 'tier-weak';
    }
    if (value >= 0.70) return 'tier-strong';
    if (value >= 0.50) return 'tier-good';
    if (value >= 0.30) return 'tier-ok';
    return 'tier-weak';
  }

  function renderBadges(row) {
    return (row.badges || []).map(badge => {
      const cls = BADGE_CLASS[badge] || '';
      return `<span class="badge ${cls}">${badge}</span>`;
    }).join('') || '—';
  }

  function renderDetailRow(row) {
    return `<tr class="detail-row" id="details-${row.display_rank}">
      <td colspan="13">
        <div class="detail-grid">
          <div><span>Thai</span><strong>${fmtRate(row.thai_score_rate)}</strong></div>
          <div><span>Math</span><strong>${fmtRate(row.math_score_rate)}</strong></div>
          <div><span>Overall</span><strong>${fmtRate(row.overall_score_rate)}</strong></div>
          <div><span>p50</span><strong>${fmtNum(row.latency_p50_ms)}ms</strong></div>
          <div><span>p95</span><strong>${fmtNum(row.latency_p95_ms)}ms</strong></div>
          <div><span>Q/min</span><strong>${row.questions_per_min}</strong></div>
          <div><span>Correct/min</span><strong>${row.correct_per_min}</strong></div>
          <div><span>16GB fit</span><strong>${formatField('fit_score', row)}</strong></div>
        </div>
      </td>
    </tr>`;
  }

  function renderLeaderboard() {
    const sorted = sortRows(scoreRows(allRows));
    const tbody = document.getElementById('leaderboard-body');
    tbody.innerHTML = sorted.map(row => {
      const rowHtml = `<tr class="leader-row">
        <td class="num rank-cell">${row.display_rank}</td>
        <td class="model-link">
          <a href="model.html?model=${encodeURIComponent(row.model_id)}">${row.model_id}</a>
          <div class="mobile-badges">${renderBadges(row)}</div>
        </td>
        <td class="lens-cell">
          <strong>${fmtScore(row.lens_score)}</strong>
          <span>${row.lens_reason}</span>
        </td>
        <td class="badge-col">${renderBadges(row)}</td>
        <td class="num metric-col ${tierClass('thai_score_rate', row.thai_score_rate)}">${fmtRate(row.thai_score_rate)}</td>
        <td class="num metric-col ${tierClass('math_score_rate', row.math_score_rate)}">${fmtRate(row.math_score_rate)}</td>
        <td class="num metric-col ${tierClass('overall_score_rate', row.overall_score_rate)}">${fmtRate(row.overall_score_rate)}</td>
        <td class="num metric-col ${tierClass('latency_p50_ms', row.latency_p50_ms)}">${fmtNum(row.latency_p50_ms)}</td>
        <td class="num metric-col ${tierClass('latency_p95_ms', row.latency_p95_ms)}">${fmtNum(row.latency_p95_ms)}</td>
        <td class="num metric-col ${tierClass('questions_per_min', row.questions_per_min)}">${row.questions_per_min}</td>
        <td class="num metric-col ${tierClass('correct_per_min', row.correct_per_min)}">${row.correct_per_min}</td>
        <td class="num metric-col">${row.item_count}</td>
        <td class="details-col"><button type="button" class="details-button" aria-expanded="false" aria-controls="details-${row.display_rank}">Details</button></td>
      </tr>`;
      return rowHtml + renderDetailRow(row);
    }).join('');

    tbody.querySelectorAll('.details-button').forEach(button => {
      button.addEventListener('click', () => {
        const expanded = button.getAttribute('aria-expanded') === 'true';
        button.setAttribute('aria-expanded', expanded ? 'false' : 'true');
        document.getElementById(button.getAttribute('aria-controls')).classList.toggle('open', !expanded);
      });
    });
    updateSortIndicators();
  }
})();
