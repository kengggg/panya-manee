// NT P3 Dashboard V1 — Model detail page logic
(async function () {
  const DATA_ROOT = 'data/latest';
  const params = new URLSearchParams(window.location.search);
  const modelId = params.get('model');
  const content = document.getElementById('model-content');

  if (!modelId) {
    content.innerHTML = '<p style="padding:40px;text-align:center;">No model specified. <a href="index.html">Back to leaderboard</a></p>';
    return;
  }

  async function loadJSON(filename) {
    const resp = await fetch(`${DATA_ROOT}/${filename}`);
    if (!resp.ok) throw new Error(`Failed to load ${filename}: ${resp.status}`);
    return resp.json();
  }

  try {
    const [manifest, modelCards, examplesData] = await Promise.all([
      loadJSON('manifest.json'),
      loadJSON('model_cards.json'),
      loadJSON('examples.json'),
    ]);

    document.getElementById('benchmark-label').textContent =
      manifest.benchmark_label + ' \u2014 ' + manifest.benchmark_scope;

    const card = modelCards.models.find(m => m.model_id === modelId);
    if (!card) {
      content.innerHTML = `<p style="padding:40px;text-align:center;">Model "${modelId}" not found. <a href="index.html">Back to leaderboard</a></p>`;
      return;
    }

    const exampleMap = {};
    for (const ex of examplesData.examples) {
      exampleMap[ex.example_id] = ex;
    }

    document.title = `${modelId} — NT P3 Benchmark`;
    render(card, exampleMap, manifest);
  } catch (err) {
    content.innerHTML = `<p style="padding:40px;text-align:center;color:#991b1b;">
      Failed to load data.<br><small>${err.message}</small>
    </p>`;
  }

  function fmtRate(v) {
    return (v * 100).toFixed(1) + '%';
  }

  function fmtMs(v) {
    return v.toLocaleString() + ' ms';
  }

  function render(card, exampleMap, manifest) {
    const m = card.metrics;

    const badgesHtml = card.badges.length > 0
      ? card.badges.map(b => `<span class="badge">${b}</span>`).join(' ')
      : '';

    const allExampleIds = [...(card.example_ids.good || []), ...(card.example_ids.bad || [])];
    const examples = allExampleIds.map(id => exampleMap[id]).filter(Boolean);

    content.innerHTML = `
      <h2 class="model-title">${card.model_id}</h2>
      ${badgesHtml ? `<div class="model-badges">${badgesHtml}</div>` : ''}

      <div class="auto-summary">${card.auto_summary}</div>

      <div class="section methodology-card">
        <h2>How to read this card</h2>
        <dl class="definition-list compact">
          <div>
            <dt>BQS</dt>
            <dd>Average of Thai score rate and Math score rate.</dd>
          </div>
          <div>
            <dt>Overall Score</dt>
            <dd>Total correct answers divided by all evaluated items in this snapshot.</dd>
          </div>
          <div>
            <dt>Parseable</dt>
            <dd>Outputs where the evaluator could recover a valid answer choice.</dd>
          </div>
          <div>
            <dt>Compliance</dt>
            <dd>Outputs that were exactly one digit, 1-4, with no extra text.</dd>
          </div>
          <div>
            <dt>Examples</dt>
            <dd>Representative items from one canonical run in the published snapshot.</dd>
          </div>
        </dl>
      </div>

      <!-- Identity -->
      <div class="section">
        <h2>Identity</h2>
        <div class="metric-grid">
          ${metricItem('Model Family', card.model_family)}
          ${metricItem('Parameter Bucket', card.parameter_bucket)}
          ${metricItem('RAM-Fit Class', card.ram_fit_class.replace(/_/g, ' '))}
          ${metricItem('Backend', manifest.testbed.backend)}
          ${metricItem('Testbed', card.testbed.host_label)}
        </div>
      </div>

      <!-- Quality -->
      <div class="section">
        <h2>Quality</h2>
        <div class="metric-grid">
          ${metricItem('Balanced Quality Score', fmtRate(m.balanced_quality_score))}
          ${metricItem('Thai Score', fmtRate(m.thai_score_rate))}
          ${metricItem('Math Score', fmtRate(m.math_score_rate))}
          ${metricItem('Overall Score', fmtRate(m.overall_score_rate))}
          ${metricItem('Correct / Total', `${m.total_correct} / ${m.item_count}`)}
        </div>
      </div>

      <!-- Reliability -->
      <div class="section">
        <h2>Reliability</h2>
        <div class="metric-grid">
          ${metricItem('Parseable Rate', fmtRate(m.parseable_rate))}
          ${metricItem('Answer-Only Compliance', fmtRate(m.answer_only_compliance_rate))}
          ${metricItem('Average Output Length', (m.average_output_length_chars ?? 0) + ' chars')}
          ${metricItem('Common Failure Types', renderFailureTypes(card.common_failure_types))}
        </div>
      </div>

      <!-- Speed -->
      <div class="section">
        <h2>Speed</h2>
        <div class="metric-grid">
          ${metricItem('p50 Latency', fmtMs(m.latency_p50_ms))}
          ${metricItem('p95 Latency', fmtMs(m.latency_p95_ms))}
          ${metricItem('Questions / min', m.questions_per_min)}
          ${metricItem('Correct / min', m.correct_per_min)}
          ${m.throughput_toks_per_sec ? metricItem('Throughput', m.throughput_toks_per_sec + ' tok/s') : ''}
        </div>
      </div>

      <!-- Strengths / Weaknesses -->
      <div class="section">
        <h2>Strengths / Weaknesses</h2>
        <div class="sw-columns">
          <div>
            <h3>Strengths</h3>
            ${renderSkillList(card.strengths)}
          </div>
          <div>
            <h3>Weaknesses</h3>
            ${renderSkillList(card.weaknesses)}
          </div>
        </div>
      </div>

      <!-- Examples -->
      <div class="section">
        <h2>Examples</h2>
        ${examples.length === 0 ? '<p style="color:var(--text-secondary);">No examples available.</p>' : ''}
        <div id="examples-container">
          ${examples.map(renderExample).join('')}
        </div>
      </div>
    `;

    // Expand buttons are handled by the delegated click handler below.
  }

  function metricItem(label, value) {
    return `<div class="metric-item">
      <span class="metric-label">${label}</span>
      <span class="metric-value">${value}</span>
    </div>`;
  }

  function renderSkillList(skills) {
    if (!skills || skills.length === 0) {
      return '<p style="color:var(--text-secondary);font-size:13px;">None qualified</p>';
    }
    return `<ul class="skill-list">
      ${skills.map(s => `<li>
        <span class="skill-tag-name">${s.skill_tag}</span>
        <span>
          <span class="skill-rate">${fmtRate(s.score_rate)}</span>
          <span class="skill-count">(${s.correct}/${s.total})</span>
        </span>
      </li>`).join('')}
    </ul>`;
  }

  function renderFailureTypes(items) {
    if (!items || items.length === 0) return 'None observed';
    return items.join(', ');
  }

  function renderExample(ex) {
    const typeClass = ex.is_correct ? 'good' : 'bad';
    const typeLabel = ex.is_correct ? 'Correct' : 'Incorrect';
    const skillTags = (ex.skill_tag || []).join(', ') || '\u2014';
    const needsExpand = ex.raw_output_full && ex.raw_output_full.length > 80;
    const choiceEntries = normalizeChoices(ex.choices);
    const modelAnswerLabel = ex.parsed_answer
      ? `${ex.parsed_answer}${ex.model_answer_text ? `. ${escapeHtml(ex.model_answer_text)}` : ''}`
      : '\u2014';
    const correctAnswerLabel = ex.correct_answer
      ? `${ex.correct_answer}${ex.correct_answer_text ? `. ${escapeHtml(ex.correct_answer_text)}` : ''}`
      : '\u2014';

    return `<div class="example-card ${typeClass}">
      <div class="example-header">
        <span><strong>${typeLabel}</strong></span>
        ${ex.year_buddhist ? `<span>Year: <strong>${ex.year_buddhist}</strong></span>` : ''}
        <span>Subject: <strong>${ex.subject}</strong></span>
        <span>Q#${ex.question_id}</span>
        <span>Skills: ${skillTags}</span>
        ${ex.curriculum_standard ? `<span>Standard: ${ex.curriculum_standard}</span>` : ''}
        <span>Latency: ${ex.latency_ms}ms</span>
      </div>

      <div class="example-question-block">
        ${ex.stimulus_text ? `<div class="example-stimulus">${escapeHtml(ex.stimulus_text)}</div>` : ''}
        <div class="example-prompt">${ex.prompt_text ? escapeHtml(ex.prompt_text) : '<span class="muted">Question text unavailable in this snapshot.</span>'}</div>
        ${choiceEntries.length > 0 ? `<ol class="choice-list">${choiceEntries.map(([key, value]) => renderChoice(key, value, ex)).join('')}</ol>` : ''}
      </div>

      <div class="answer-grid">
        <div class="answer-pill model-answer">
          <span class="answer-label">Model answer</span>
          <strong>${modelAnswerLabel}</strong>
        </div>
        <div class="answer-pill correct-answer">
          <span class="answer-label">Correct answer</span>
          <strong>${correctAnswerLabel}</strong>
        </div>
      </div>

      <div class="example-output" data-full="${encodeURIComponent(ex.raw_output_full)}">${escapeHtml(ex.raw_output_truncated)}</div>
      ${needsExpand ? '<button class="expand-btn">Expand</button>' : ''}
    </div>`;
  }

  function normalizeChoices(choices) {
    if (!choices || typeof choices !== 'object') return [];
    return Object.entries(choices).sort((a, b) => Number(a[0]) - Number(b[0]));
  }

  function renderChoice(key, value, ex) {
    const isSelected = String(ex.parsed_answer || '') === String(key);
    const isCorrect = String(ex.correct_answer || '') === String(key);
    return `<li class="choice-item${isSelected ? ' selected' : ''}${isCorrect ? ' correct' : ''}">
      <span class="choice-key">${escapeHtml(String(key))}</span>
      <span class="choice-text">${escapeHtml(String(value || ''))}</span>
      <span class="choice-tags">
        ${isSelected ? '<span class="choice-tag selected-tag">model</span>' : ''}
        ${isCorrect ? '<span class="choice-tag correct-tag">correct</span>' : ''}
      </span>
    </li>`;
  }

  function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str || '';
    return div.innerHTML;
  }

  // Fix expand to show full output
  document.addEventListener('click', function (e) {
    if (!e.target.classList.contains('expand-btn')) return;
    const output = e.target.previousElementSibling;
    if (!output) return;
    const isExpanded = output.classList.toggle('expanded');
    if (isExpanded) {
      output.textContent = decodeURIComponent(output.dataset.full);
    } else {
      // Re-truncate: show truncated version
      const full = decodeURIComponent(output.dataset.full);
      output.textContent = full.slice(0, 200);
    }
    e.target.textContent = isExpanded ? 'Collapse' : 'Expand';
  });
})();
