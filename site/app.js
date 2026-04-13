// NT P3 Dashboard V1 — Leaderboard page logic
(async function () {
  const DATA_ROOT = 'data/latest';
  let allRows = [];
  let currentFilteredRows = [];
  let bounds = {};
  let sortState = { key: null, dir: null };

  const COLUMNS = [
    { key: 'rank',       field: 'rank',                        type: 'num',    defaultDir: 'asc'  },
    { key: 'model',      field: 'model_id',                    type: 'string', defaultDir: 'asc'  },
    { key: 'bqs',        field: 'balanced_quality_score',      type: 'num',    defaultDir: 'desc' },
    { key: 'thai',       field: 'thai_score_rate',             type: 'num',    defaultDir: 'desc' },
    { key: 'math',       field: 'math_score_rate',             type: 'num',    defaultDir: 'desc' },
    { key: 'overall',    field: 'overall_score_rate',          type: 'num',    defaultDir: 'desc' },
    { key: 'parseable',  field: 'parseable_rate',              type: 'num',    defaultDir: 'desc' },
    { key: 'compliance', field: 'answer_only_compliance_rate', type: 'num',    defaultDir: 'desc' },
    { key: 'p50',        field: 'latency_p50_ms',              type: 'num',    defaultDir: 'asc'  },
    { key: 'p95',        field: 'latency_p95_ms',              type: 'num',    defaultDir: 'asc'  },
    { key: 'qmin',       field: 'questions_per_min',           type: 'num',    defaultDir: 'desc' },
    { key: 'cmin',       field: 'correct_per_min',             type: 'num',    defaultDir: 'desc' },
    { key: 'n',          field: 'item_count',                  type: 'num',    defaultDir: 'desc' },
  ];

  const BADGE_CLASS = {
    'Best Quality':   'badge-gold',
    'Best Thai':      'badge-blue',
    'Best Math':      'badge-violet',
    'Fastest':        'badge-green',
    'Most Parseable': 'badge-teal',
  };

  async function loadJSON(filename) {
    const resp = await fetch(`${DATA_ROOT}/${filename}`);
    if (!resp.ok) throw new Error(`Failed to load ${filename}: ${resp.status}`);
    return resp.json();
  }

  try {
    const [manifest, leaderboard] = await Promise.all([
      loadJSON('manifest.json'),
      loadJSON('leaderboard.json'),
    ]);

    renderMeta(manifest, leaderboard);
    renderMethodology(manifest, leaderboard);
    allRows = leaderboard.rows;
    bounds = computeBounds(allRows);
    currentFilteredRows = allRows;
    initModelFilter(allRows);
    initSorting();
    renderLeaderboard(currentFilteredRows);
  } catch (err) {
    document.getElementById('leaderboard-body').innerHTML =
      `<tr><td colspan="14" style="text-align:center;padding:40px;color:#991b1b;">
        Failed to load data. Make sure snapshot files exist in ${DATA_ROOT}/.
        <br><small>${err.message}</small>
      </td></tr>`;
  }

  function renderMeta(manifest, leaderboard) {
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

    const bar = document.getElementById('meta-bar');
    bar.innerHTML = items.map(([k, v]) =>
      `<div class="meta-item"><dt>${k}:</dt><dd>${v}</dd></div>`
    ).join('');

    document.getElementById('snapshot-id-label').textContent =
      'Snapshot: ' + manifest.snapshot_id;
  }

  function fmtRate(v) {
    return (v * 100).toFixed(1) + '%';
  }

  function fmtNum(v) {
    return typeof v === 'number' ? v.toLocaleString() : v;
  }

  function initModelFilter(rows) {
    const select = document.getElementById('model-filter');
    const modelIds = [...new Set(rows.map(row => row.model_id))].sort();
    select.innerHTML = '<option value="">All models</option>' + modelIds.map(
      modelId => `<option value="${modelId}">${modelId}</option>`
    ).join('');

    select.addEventListener('change', () => {
      const selected = select.value;
      currentFilteredRows = selected ? allRows.filter(row => row.model_id === selected) : allRows;
      renderLeaderboard(currentFilteredRows);
    });
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
        Each item expects a single choice from 1-4. Rankings exclude image-required and human-checked tasks.
      </p>
      <dl class="definition-list">
        <div>
          <dt>BQS</dt>
          <dd>Balanced Quality Score, the simple average of Thai score rate and Math score rate. It prevents a model from looking strong by overperforming in only one subject.</dd>
        </div>
        <div>
          <dt>Overall Score</dt>
          <dd>Total correct answers divided by all evaluated items in the published batch.</dd>
        </div>
        <div>
          <dt>Parseable</dt>
          <dd>How often the parser could recover a valid answer choice from the model output.</dd>
        </div>
        <div>
          <dt>Compliance</dt>
          <dd>How often the raw output was exactly one digit, 1-4, with no extra explanation.</dd>
        </div>
        <div>
          <dt>p50 / p95</dt>
          <dd>Median and tail latency per question. Lower is faster.</dd>
        </div>
      </dl>
    `;
  }

  function computeBounds(rows) {
    const b = {};
    for (const field of ['questions_per_min', 'correct_per_min']) {
      const vals = rows.map(r => r[field])
        .filter(v => typeof v === 'number')
        .sort((x, y) => x - y);
      if (!vals.length) {
        b[field] = { p25: 0, p50: 0, p75: 0 };
        continue;
      }
      const pick = (p) => vals[Math.min(vals.length - 1, Math.floor(p * vals.length))];
      b[field] = { p25: pick(0.25), p50: pick(0.5), p75: pick(0.75) };
    }
    return b;
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
      const b = bounds[field] || { p25: 0, p50: 0, p75: 0 };
      if (value >= b.p75) return 'tier-strong';
      if (value >= b.p50) return 'tier-good';
      if (value >= b.p25) return 'tier-ok';
      return 'tier-weak';
    }
    // rates (0-1)
    if (value >= 0.70) return 'tier-strong';
    if (value >= 0.50) return 'tier-good';
    if (value >= 0.30) return 'tier-ok';
    return 'tier-weak';
  }

  function rankDisplay(rank) {
    if (rank === 1) return { medal: '\u{1F947}', cls: 'rank-top3' }; // gold
    if (rank === 2) return { medal: '\u{1F948}', cls: 'rank-top3' }; // silver
    if (rank === 3) return { medal: '\u{1F949}', cls: 'rank-top3' }; // bronze
    return { medal: '', cls: '' };
  }

  function sortRows(rows) {
    if (!sortState.key) return rows;
    const col = COLUMNS.find(c => c.key === sortState.key);
    if (!col) return rows;
    const mult = sortState.dir === 'asc' ? 1 : -1;
    return [...rows].sort((a, b) => {
      const av = a[col.field];
      const bv = b[col.field];
      if (col.type === 'string') {
        return mult * String(av ?? '').localeCompare(String(bv ?? ''));
      }
      const an = typeof av === 'number' ? av : 0;
      const bn = typeof bv === 'number' ? bv : 0;
      return mult * (an - bn);
    });
  }

  function onHeaderClick(key) {
    const col = COLUMNS.find(c => c.key === key);
    if (!col) return;

    if (sortState.key === key) {
      if (sortState.dir === col.defaultDir) {
        sortState = { key, dir: col.defaultDir === 'asc' ? 'desc' : 'asc' };
      } else {
        sortState = { key: null, dir: null };
      }
    } else {
      sortState = { key, dir: col.defaultDir };
    }
    renderLeaderboard(currentFilteredRows);
    updateSortIndicators();
  }

  function updateSortIndicators() {
    document.querySelectorAll('th.sortable').forEach(th => {
      const key = th.dataset.sortKey;
      const active = sortState.key === key;
      th.classList.toggle('sort-active', active);
      const ind = th.querySelector('.sort-ind');
      if (!ind) return;
      if (active) {
        ind.textContent = sortState.dir === 'asc' ? ' \u2191' : ' \u2193';
      } else {
        ind.textContent = '';
      }
    });
  }

  function initSorting() {
    document.querySelectorAll('th.sortable').forEach(th => {
      th.addEventListener('click', () => onHeaderClick(th.dataset.sortKey));
    });
  }

  function renderLeaderboard(rows) {
    const sorted = sortRows(rows);
    const tbody = document.getElementById('leaderboard-body');
    tbody.innerHTML = sorted.map(row => {
      const badges = (row.badges || []).map(b => {
        const cls = BADGE_CLASS[b] || '';
        return `<span class="badge ${cls}">${b}</span>`;
      }).join('');

      const rd = rankDisplay(row.rank);
      const medal = rd.medal ? `<span class="rank-medal">${rd.medal}</span>` : '';
      const cBqs = tierClass('balanced_quality_score', row.balanced_quality_score);
      const cThai = tierClass('thai_score_rate', row.thai_score_rate);
      const cMath = tierClass('math_score_rate', row.math_score_rate);
      const cOverall = tierClass('overall_score_rate', row.overall_score_rate);
      const cParse = tierClass('parseable_rate', row.parseable_rate);
      const cComp = tierClass('answer_only_compliance_rate', row.answer_only_compliance_rate);
      const cP50 = tierClass('latency_p50_ms', row.latency_p50_ms);
      const cP95 = tierClass('latency_p95_ms', row.latency_p95_ms);
      const cQmin = tierClass('questions_per_min', row.questions_per_min);
      const cCmin = tierClass('correct_per_min', row.correct_per_min);

      return `<tr>
        <td class="num ${rd.cls}">${medal}${row.rank}</td>
        <td class="model-link"><a href="model.html?model=${encodeURIComponent(row.model_id)}">${row.model_id}</a></td>
        <td class="num ${cBqs}"><strong>${fmtRate(row.balanced_quality_score)}</strong></td>
        <td class="num ${cThai}">${fmtRate(row.thai_score_rate)}</td>
        <td class="num ${cMath}">${fmtRate(row.math_score_rate)}</td>
        <td class="num ${cOverall}">${fmtRate(row.overall_score_rate)}</td>
        <td class="num ${cParse}">${fmtRate(row.parseable_rate)}</td>
        <td class="num ${cComp}">${fmtRate(row.answer_only_compliance_rate)}</td>
        <td class="num ${cP50}">${fmtNum(row.latency_p50_ms)}</td>
        <td class="num ${cP95}">${fmtNum(row.latency_p95_ms)}</td>
        <td class="num ${cQmin}">${row.questions_per_min}</td>
        <td class="num ${cCmin}">${row.correct_per_min}</td>
        <td class="num">${row.item_count}</td>
        <td>${badges || '\u2014'}</td>
      </tr>`;
    }).join('');
  }
})();
