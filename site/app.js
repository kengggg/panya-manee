// NT P3 Dashboard V1 — Leaderboard page logic
(async function () {
  const DATA_ROOT = 'data/latest';
  let allRows = [];

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
    allRows = leaderboard.rows;
    initModelFilter(allRows);
    renderLeaderboard(allRows);
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
      const filtered = selected ? allRows.filter(row => row.model_id === selected) : allRows;
      renderLeaderboard(filtered);
    });
  }

  function renderLeaderboard(rows) {
    const tbody = document.getElementById('leaderboard-body');
    tbody.innerHTML = rows.map(row => {
      const badges = (row.badges || []).map(b =>
        `<span class="badge">${b}</span>`
      ).join('');

      return `<tr>
        <td class="num">${row.rank}</td>
        <td class="model-link"><a href="model.html?model=${encodeURIComponent(row.model_id)}">${row.model_id}</a></td>
        <td class="num"><strong>${fmtRate(row.balanced_quality_score)}</strong></td>
        <td class="num">${fmtRate(row.thai_score_rate)}</td>
        <td class="num">${fmtRate(row.math_score_rate)}</td>
        <td class="num">${fmtRate(row.overall_score_rate)}</td>
        <td class="num">${fmtRate(row.parseable_rate)}</td>
        <td class="num">${fmtRate(row.answer_only_compliance_rate)}</td>
        <td class="num">${fmtNum(row.latency_p50_ms)}</td>
        <td class="num">${fmtNum(row.latency_p95_ms)}</td>
        <td class="num">${row.questions_per_min}</td>
        <td class="num">${row.correct_per_min}</td>
        <td class="num">${row.item_count}</td>
        <td>${badges || '\u2014'}</td>
      </tr>`;
    }).join('');
  }
})();
