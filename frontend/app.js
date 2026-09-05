/**
 * ResumeAI frontend — wires the RAG chat UI to the FastAPI backend.
 */
(function () {
  "use strict";

  const API = "";
  const SAMPLE_JD = {
    ai: `Senior AI / Full-Stack Engineer

We are hiring a Senior AI & Full-Stack Engineer to design production RAG systems and high-throughput APIs.

Requirements:
- 5+ years with Python, FastAPI, React, TypeScript
- Hands-on RAG, vector search, LangChain or LlamaIndex, LLMs
- AWS, Docker, Kubernetes, Terraform, CI/CD
- PostgreSQL, Redis, Elasticsearch
- Strong system design, microservices, and measurable impact

Nice to have: PyTorch, Next.js, GCP, interview coaching experience.`,
    devops: `DevOps / Cloud Platform Lead

We need a Senior Cloud & DevOps Architect to own Kubernetes platforms and reliability.

Requirements:
- Kubernetes, Helm, Terraform, GitHub Actions / Jenkins
- GCP or AWS, Linux, Nginx, observability (Prometheus, Grafana)
- Docker, CI/CD, infrastructure as code
- Python or Go scripting
- Incident response and platform SLOs

Nice to have: Ansible, multi-cloud, security hardening.`,
  };

  const state = {
    activeResumeId: null,
    resumes: [],
    geminiActive: false,
    minilmActive: false,
    loadedTabs: new Set(),
  };

  const el = (id) => document.getElementById(id);

  function toast(message, kind) {
    const hub = el("toastHub");
    if (!hub) return;
    const node = document.createElement("div");
    node.className = "toast";
    if (kind === "error") node.style.borderColor = "rgba(244, 63, 94, 0.5)";
    node.textContent = message;
    hub.appendChild(node);
    setTimeout(() => node.remove(), 4200);
  }

  function escapeHtml(str) {
    return String(str ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function renderMarkdown(raw) {
    const escaped = escapeHtml(raw || "");
    const lines = escaped.split("\n");
    const html = [];
    let inList = false;

    const flushList = () => {
      if (inList) {
        html.push("</ul>");
        inList = false;
      }
    };

    for (const line of lines) {
      const heading = line.match(/^(#{1,3})\s+(.+)$/);
      const bullet = line.match(/^[\u2022*\-]\s+(.+)$/) || line.match(/^•\s+(.+)$/);
      const quote = line.match(/^&gt;\s?(.*)$/);

      if (heading) {
        flushList();
        const level = heading[1].length;
        html.push(`<h${level}>${inlineMd(heading[2])}</h${level}>`);
        continue;
      }
      if (bullet) {
        if (!inList) {
          html.push("<ul>");
          inList = true;
        }
        html.push(`<li>${inlineMd(bullet[1])}</li>`);
        continue;
      }
      flushList();
      if (quote) {
        html.push(`<blockquote>${inlineMd(quote[1])}</blockquote>`);
        continue;
      }
      if (!line.trim()) {
        html.push("<br>");
        continue;
      }
      html.push(`<p>${inlineMd(line)}</p>`);
    }
    flushList();
    return html.join("");
  }

  function inlineMd(text) {
    return text
      .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
      .replace(/`([^`]+)`/g, "<code>$1</code>");
  }

  async function api(path, options = {}) {
    const { timeoutMs, ...fetchOpts } = options;
    const controller = timeoutMs ? new AbortController() : null;
    let timer = null;
    if (controller) {
      fetchOpts.signal = controller.signal;
      timer = setTimeout(() => controller.abort(), timeoutMs);
    }
    try {
      const res = await fetch(API + path, fetchOpts);
      let data = null;
      try {
        data = await res.json();
      } catch (_) {
        data = null;
      }
      if (!res.ok) {
        const detail = data && (data.detail || data.message);
        throw new Error(detail || `Request failed (${res.status})`);
      }
      return data;
    } catch (err) {
      if (err && err.name === "AbortError") {
        throw new Error("Gemini validation timed out. Use offline mode, or check your internet connection.");
      }
      throw err;
    } finally {
      if (timer) clearTimeout(timer);
    }
  }

  async function refreshStatus() {
    const data = await api("/api/status");
    state.resumes = data.resumes || [];
    state.activeResumeId = data.active_resume_id;
    state.geminiActive = !!data.gemini_active;
    state.minilmActive = !!data.minilm_active;
    updateGeminiBadge();
    populateResumeSelect();
    updateChatHeader();
    return data;
  }

  function updateGeminiBadge() {
    const dot = el("apiKeyStatusDot");
    const text = el("apiKeyStatusText");
    if (!dot || !text) return;
    dot.classList.toggle("dot-online", state.geminiActive);
    dot.classList.toggle("dot-offline", !state.geminiActive);
    text.textContent = state.geminiActive ? "Gemini On" : "Resume bot";
    const tag = el("ragModeTag");
    if (tag) {
      tag.textContent = state.geminiActive
        ? "Gemini chat"
        : state.minilmActive
          ? "MiniLM RAG"
          : "Resume bot";
    }
  }

  function populateResumeSelect() {
    const select = el("resumeSelect");
    if (!select) return;
    if (!state.resumes.length) {
      select.innerHTML = '<option value="">No resumes loaded</option>';
      return;
    }
    select.innerHTML = state.resumes
      .map((r) => {
        const selected = r.id === state.activeResumeId ? " selected" : "";
        const label = escapeHtml(r.title || r.filename || r.id);
        return `<option value="${escapeHtml(r.id)}"${selected}>${label}</option>`;
      })
      .join("");
  }

  function activeResumeMeta() {
    return state.resumes.find((r) => r.id === state.activeResumeId) || null;
  }

  function resumeLabel() {
    const meta = activeResumeMeta();
    if (!meta) return "the uploaded resume";
    return meta.filename || meta.title || "this resume";
  }

  function botWelcomeHtml() {
    return `<p>Hi — I'm your resume bot for <strong>${escapeHtml(resumeLabel())}</strong>.</p>
<p>Ask me anything about this file: skills, projects, education, internships, or a short overview.</p>`;
  }

  function resetChat(html) {
    const feed = el("chatMessages");
    if (!feed) return;
    feed.innerHTML = "";
    appendMessage({ role: "assistant", html: html || botWelcomeHtml() });
  }

  function updateChatHeader() {
    const title = el("chatActiveDocTitle");
    const meta = activeResumeMeta();
    if (title) {
      title.textContent = meta
        ? `Chat about: ${meta.filename || meta.title}`
        : "Chat about: your resume";
    }
  }

  function switchTab(tabId) {
    document.querySelectorAll(".tab-btn").forEach((btn) => {
      btn.classList.toggle("active", btn.dataset.tab === tabId);
    });
    document.querySelectorAll(".tab-pane").forEach((pane) => {
      pane.classList.toggle("active", pane.id === tabId);
    });
    loadTabData(tabId);
  }

  async function loadTabData(tabId, force) {
    if (!state.activeResumeId) return;
    if (!force && state.loadedTabs.has(tabId)) return;
    try {
      if (tabId === "tab-profile") await loadProfile();
      if (tabId === "tab-audit") await loadAudit();
      if (tabId === "tab-interview") await loadInterview();
      state.loadedTabs.add(tabId);
    } catch (err) {
      toast(err.message, "error");
    }
  }

  function invalidateTabs() {
    state.loadedTabs.clear();
  }

  async function selectResume(resumeId) {
    await api("/api/select-resume", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ resume_id: resumeId }),
    });
    state.activeResumeId = resumeId;
    invalidateTabs();
    updateChatHeader();
    populateResumeSelect();
    resetChat();
    const activeTab = document.querySelector(".tab-pane.active");
    if (activeTab) await loadTabData(activeTab.id, true);
    toast("Now chatting about this resume");
  }

  async function loadSample(sampleId) {
    el("sampleDropdownMenu")?.classList.add("hidden");
    try {
      await api("/api/load-sample", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sample_id: sampleId }),
      });
      await refreshStatus();
      invalidateTabs();
      resetChat();
      const activeTab = document.querySelector(".tab-pane.active");
      if (activeTab) await loadTabData(activeTab.id, true);
      toast("Sample resume loaded — ask the bot about it");
    } catch (err) {
      toast(err.message, "error");
    }
  }

  function initials(name) {
    const parts = String(name || "C").trim().split(/\s+/).slice(0, 2);
    return parts.map((p) => p[0]?.toUpperCase() || "").join("") || "C";
  }

  function setContact(id, value, isLink) {
    const node = el(id);
    if (!node) return;
    const span = node.querySelector("span") || node;
    if (isLink) {
      if (value) {
        node.href = value.startsWith("http") ? value : `https://${value}`;
        span.textContent = value.replace(/^https?:\/\//, "");
        node.style.display = "";
      } else {
        node.href = "#";
        span.textContent = "Not found";
      }
      return;
    }
    span.textContent = value || "Not found";
  }

  async function loadProfile() {
    const profile = await api("/api/analyze/profile");
    el("profName").textContent = profile.candidate_name || "Candidate";
    el("profAvatar").textContent = initials(profile.candidate_name);
    setContact("profEmail", profile.email);
    setContact("profPhone", profile.phone);
    setContact("profLinkedin", profile.linkedin, true);
    setContact("profGithub", profile.github, true);
    el("profSkillsCount").textContent = profile.total_skills_found ?? 0;
    el("profPagesCount").textContent = profile.total_pages ?? 1;
    el("profCompleteness").textContent = `${profile.completeness_score ?? 0}%`;

    const skillsRoot = el("profSkillsCategories");
    const cats = profile.categorized_skills || {};
    const keys = Object.keys(cats);
    if (!keys.length) {
      skillsRoot.innerHTML = '<p class="empty-state">No categorized skills detected.</p>';
    } else {
      skillsRoot.innerHTML = keys
        .map((cat) => {
          const chips = (cats[cat] || [])
            .map((s) => `<span class="skill-chip">${escapeHtml(s)}</span>`)
            .join("");
          return `<div class="skill-cat-card"><div class="skill-cat-title">${escapeHtml(cat)}</div><div class="chips-container">${chips}</div></div>`;
        })
        .join("");
    }
    el("profExperienceText").textContent = profile.experience_text || "No experience section detected.";
    el("profEducationText").textContent = profile.education_text || "No education section detected.";
  }

  function renderSkillSuggestions(suggestions, missing) {
    const list = el("skillSuggestionsList");
    const intro = el("gapPlanIntro");
    if (!list) return;
    if (!suggestions.length) {
      if (intro) {
        intro.textContent = missing && missing.length
          ? "Gaps were detected, but no extra suggestions were generated."
          : "No missing JD skills on this resume. Keep wording aligned with the job post.";
      }
      list.innerHTML = "";
      return;
    }
    if (intro) {
      intro.textContent = `${suggestions.length} required skill${suggestions.length === 1 ? "" : "s"} from this job description are not on your resume. Add them only if you can talk about them in an interview.`;
    }
    list.innerHTML = suggestions
      .map(
        (s) => `<article class="gap-card">
          <div class="gap-card-head">
            <span class="skill-chip chip-missing">${escapeHtml(s.skill)}</span>
            <span class="gap-where">${escapeHtml(s.where_to_add || "Skills section")}</span>
          </div>
          <p class="gap-why"><strong>Why this JD:</strong> ${escapeHtml(s.why || "")}</p>
          <p class="gap-action">${escapeHtml(s.suggestion || "")}</p>
          <blockquote class="gap-example"><strong>Sample bullet:</strong> ${escapeHtml(s.example_bullet || "")}</blockquote>
        </article>`
      )
      .join("");
  }

  async function runJobMatch() {
    const jd = el("jobDescInput").value.trim();
    if (!jd) {
      toast("Paste a job description first", "error");
      return;
    }
    const btn = el("btnRunJobMatch");
    btn.disabled = true;
    try {
      const result = await api("/api/analyze/job-match", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ job_description: jd }),
      });
      const score = result.match_percentage ?? 0;
      el("atsScoreVal").textContent = `${score}%`;
      const ring = document.querySelector(".score-ring-circle");
      if (ring) {
        const color = score >= 75 ? "#10b981" : score >= 50 ? "#f59e0b" : "#f43f5e";
        ring.style.background = `conic-gradient(${color} 0% ${score}%, #e2e8f0 ${score}% 100%)`;
      }
      const matchedCount = result.matched_skills?.length || 0;
      const requiredCount = result.total_required_skills || 0;
      el("atsVerdictTitle").textContent =
        score >= 80 ? "Strong ATS Fit" : score >= 60 ? "Moderate Fit — Customize Resume" : "Low Keyword Overlap";
      el("atsVerdictSubtitle").textContent = requiredCount
        ? `${matchedCount} matched of ${requiredCount} required skills detected in the JD.`
        : "No tech skills detected in this JD. Paste a fuller description that lists tools (Python, React, SQL, etc.).";
      el("countMatched").textContent = matchedCount;
      el("countMissing").textContent = result.missing_skills?.length || 0;
      el("matchedSkillsChips").innerHTML = (result.matched_skills || [])
        .map((s) => `<span class="skill-chip chip-matched">${escapeHtml(s)}</span>`)
        .join("") || '<span class="empty-state">None yet</span>';
      el("missingSkillsChips").innerHTML = (result.missing_skills || [])
        .map((s) => `<span class="skill-chip chip-missing">${escapeHtml(s)}</span>`)
        .join("") || '<span class="empty-state">No gaps found</span>';
      renderSkillSuggestions(result.skill_suggestions || [], result.missing_skills || []);
      el("atsFeedbackContent").innerHTML = renderMarkdown(result.ai_feedback || "");
    } catch (err) {
      toast(err.message, "error");
    } finally {
      btn.disabled = false;
    }
  }

  async function loadAudit() {
    const data = await api("/api/analyze/critique");
    el("auditOverallScore").textContent = data.overall_score ?? "--";
    const pillars = el("auditPillars");
    const scores = data.pillar_scores || {};
    pillars.innerHTML = Object.entries(scores)
      .map(([name, val]) => {
        const n = Number(val) || 0;
        return `<div class="pillar-item"><div class="pillar-meta"><span>${escapeHtml(name)}</span><span>${n}</span></div><div class="pillar-bar-bg"><div class="pillar-bar-fill" style="width:${n}%"></div></div></div>`;
      })
      .join("");
    el("auditVerbsCount").textContent = data.strong_verbs_count ?? 0;
    el("auditVerbsChips").innerHTML = (data.strong_verbs || [])
      .map((v) => `<span class="skill-chip">${escapeHtml(v)}</span>`)
      .join("") || '<span class="empty-state">None detected</span>';
    el("auditMetricsCount").textContent = data.metrics_found_count ?? 0;
    el("auditMetricsChips").innerHTML = (data.sample_metrics || [])
      .map((m) => `<span class="skill-chip">${escapeHtml(m)}</span>`)
      .join("") || '<span class="empty-state">No metrics found</span>';
    el("auditAiSuggestions").innerHTML = data.ai_suggestions
      ? renderMarkdown(data.ai_suggestions)
      : "Connect a Gemini API key for rewrite suggestions. Offline audit still scored action verbs, metrics, and section structure.";
  }

  async function loadInterview() {
    const data = await api("/api/analyze/interview");
    const list = el("interviewQuestionsList");
    const questions = data.questions || [];
    if (!questions.length) {
      list.innerHTML = '<p class="empty-state">No questions generated.</p>';
      return;
    }
    list.innerHTML = questions
      .map((q) => {
        if (q.content && !q.question) {
          return `<div class="question-card"><span class="question-type-tag">${escapeHtml(q.type || "AI")}</span><div class="message-text rich">${renderMarkdown(q.content)}</div></div>`;
        }
        return `<div class="question-card">
          <span class="question-type-tag">${escapeHtml(q.type || "Question")}</span>
          <div class="question-title">${escapeHtml(q.question || "")}</div>
          ${q.look_for ? `<div class="look-for-box"><strong>Look for:</strong> ${escapeHtml(q.look_for)}</div>` : ""}
        </div>`;
      })
      .join("");
  }

  function assistantAvatarSvg() {
    return `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2H2v10l9.29 9.29c.94.94 2.48.94 3.42 0l6.58-6.58c.94-.94.94-2.48 0-3.42L12 2Z"></path><path d="M7 7h.01"></path></svg>`;
  }

  function appendMessage({ role, html, meta, sources }) {
    const feed = el("chatMessages");
    const wrap = document.createElement("div");
    wrap.className = `message-wrapper ${role}`;
    const avatar = document.createElement("div");
    avatar.className = "message-avatar";
    avatar.innerHTML = role === "assistant" ? assistantAvatarSvg() : "You";
    const bubble = document.createElement("div");
    bubble.className = "message-bubble";
      bubble.innerHTML = `
      <div class="message-sender">${role === "assistant" ? "Resume bot" : "You"}</div>
      <div class="message-text rich">${html}</div>
    `;
    if (meta) {
      const metaEl = document.createElement("div");
      metaEl.className = "message-meta";
      metaEl.textContent = meta;
      bubble.appendChild(metaEl);
    }
    if (sources && sources.length) {
      bubble.appendChild(buildSources(sources));
    }
    wrap.appendChild(avatar);
    wrap.appendChild(bubble);
    feed.appendChild(wrap);
    feed.scrollTop = feed.scrollHeight;
    return wrap;
  }

  function buildSources(sources) {
    const box = document.createElement("div");
    box.className = "sources-accordion";
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "sources-toggle-btn";
    btn.textContent = `Citations (${sources.length}) from this file`;
    const list = document.createElement("div");
    list.className = "sources-cards-list hidden";
    list.innerHTML = sources
      .map(
        (s) => `<div class="source-snippet-card">
          <div class="source-snippet-header">
            <span class="source-tag">Page ${s.page_number} • ${escapeHtml(s.section)}</span>
            <span class="source-score-badge">${Math.round((s.score || 0) * 100)}% ${escapeHtml(s.confidence || "")}</span>
          </div>
          <div class="source-snippet-text">${escapeHtml(s.text)}</div>
        </div>`
      )
      .join("");
    btn.addEventListener("click", () => {
      const open = list.classList.toggle("hidden");
      btn.textContent = open
        ? `Citations (${sources.length}) from this file`
        : "Hide citations";
    });
    box.appendChild(btn);
    box.appendChild(list);
    return box;
  }

  function setTyping(show) {
    const existing = document.getElementById("typingRow");
    if (existing) existing.remove();
    if (!show) return;
    const wrap = appendMessage({
      role: "assistant",
      html: '<div class="typing-indicator"><span></span><span></span><span></span></div>',
    });
    wrap.id = "typingRow";
  }

  function updateSuggested(prompts) {
    const bar = el("suggestedPrompts");
    if (!bar || !prompts?.length) return;
    const pills = bar.querySelectorAll(".prompt-pill");
    pills.forEach((pill, i) => {
      if (prompts[i]) {
        pill.dataset.prompt = prompts[i];
        pill.textContent = prompts[i].length > 42 ? prompts[i].slice(0, 40) + "…" : prompts[i];
      }
    });
  }

  async function sendQuery(text) {
    const query = (text || "").trim();
    if (!query) return;
    el("chatInput").value = "";
    autosize(el("chatInput"));
    appendMessage({ role: "user", html: escapeHtml(query) });
    setTyping(true);
    try {
      const data = await api("/api/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, resume_id: state.activeResumeId, top_k: 4 }),
      });
      setTyping(false);
      appendMessage({
        role: "assistant",
        html: renderMarkdown(data.answer || "I couldn't find an answer in this file."),
        sources: data.sources,
      });
      if (data.suggested_questions) updateSuggested(data.suggested_questions);
    } catch (err) {
      setTyping(false);
      appendMessage({
        role: "assistant",
        html: `<p>${escapeHtml(err.message)}</p>`,
      });
    }
  }

  function autosize(textarea) {
    textarea.style.height = "auto";
    textarea.style.height = Math.min(textarea.scrollHeight, 100) + "px";
  }

  function openModal(id) {
    el(id)?.classList.remove("hidden");
  }
  function closeModal(id) {
    el(id)?.classList.add("hidden");
  }

  async function uploadFile(file) {
    if (!file) return;
    const progress = el("uploadProgress");
    const bar = el("uploadProgressBar");
    const status = el("uploadStatusText");
    progress.classList.remove("hidden");
    bar.style.width = "35%";
    status.textContent = "Uploading and parsing...";
    const form = new FormData();
    form.append("file", file);
    try {
      bar.style.width = "70%";
      const result = await api("/api/upload", { method: "POST", body: form });
      bar.style.width = "100%";
      status.textContent = `Indexed ${result.total_chunks} chunks`;
      await refreshStatus();
      invalidateTabs();
      resetChat(`<p>I've loaded <strong>${escapeHtml(result.filename)}</strong> (${result.total_chunks} chunks). Ask me anything about this file.</p>`);
      toast(`Indexed ${result.filename}`);
      setTimeout(() => {
        closeModal("uploadModal");
        progress.classList.add("hidden");
        bar.style.width = "0%";
      }, 700);
    } catch (err) {
      status.textContent = err.message;
      toast(err.message, "error");
    }
  }

  async function saveApiKey() {
    const key = el("apiKeyInput").value.trim();
    const feedback = el("apiKeyFeedback");
    const btn = el("btnSaveApiKey");
    if (!key) {
      toast("Enter an API key or use offline mode", "error");
      return;
    }
    feedback.classList.remove("hidden", "success", "error");
    feedback.textContent = "Validating key (up to 15 seconds)...";
    if (btn) btn.disabled = true;
    try {
      const data = await api("/api/api-key", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ api_key: key }),
        timeoutMs: 18000,
      });
      feedback.textContent = data.message;
      feedback.classList.add(data.success ? "success" : "error");
      state.geminiActive = !!data.gemini_active;
      updateGeminiBadge();
      if (data.success) {
        toast(data.message);
        setTimeout(() => closeModal("apiKeyModal"), 600);
      }
    } catch (err) {
      feedback.classList.add("error");
      feedback.textContent = err.message;
      toast(err.message, "error");
    } finally {
      if (btn) btn.disabled = false;
    }
  }

  function bindEvents() {
    document.querySelectorAll(".tab-btn").forEach((btn) => {
      btn.addEventListener("click", () => switchTab(btn.dataset.tab));
    });

    el("resumeSelect")?.addEventListener("change", (e) => {
      if (e.target.value) selectResume(e.target.value);
    });

    el("btnSampleResumes")?.addEventListener("click", (e) => {
      e.stopPropagation();
      el("sampleDropdownMenu")?.classList.toggle("hidden");
    });
    document.querySelectorAll("#sampleDropdownMenu .dropdown-item").forEach((item) => {
      item.addEventListener("click", () => loadSample(item.dataset.sample));
    });
    document.addEventListener("click", () => el("sampleDropdownMenu")?.classList.add("hidden"));

    el("btnUploadModal")?.addEventListener("click", () => openModal("uploadModal"));
    el("btnCloseUploadModal")?.addEventListener("click", () => closeModal("uploadModal"));
    el("btnApiKeyModal")?.addEventListener("click", () => openModal("apiKeyModal"));
    el("btnCloseApiKeyModal")?.addEventListener("click", () => closeModal("apiKeyModal"));
    el("uploadModal")?.addEventListener("click", (e) => {
      if (e.target.id === "uploadModal") closeModal("uploadModal");
    });
    el("apiKeyModal")?.addEventListener("click", (e) => {
      if (e.target.id === "apiKeyModal") closeModal("apiKeyModal");
    });

    el("btnBrowseFiles")?.addEventListener("click", () => el("fileInput").click());
    el("fileInput")?.addEventListener("change", (e) => uploadFile(e.target.files[0]));

    const zone = el("dropZone");
    if (zone) {
      ["dragenter", "dragover"].forEach((evt) => {
        zone.addEventListener(evt, (e) => {
          e.preventDefault();
          zone.classList.add("drag-over");
        });
      });
      ["dragleave", "drop"].forEach((evt) => {
        zone.addEventListener(evt, (e) => {
          e.preventDefault();
          zone.classList.remove("drag-over");
        });
      });
      zone.addEventListener("drop", (e) => {
        const file = e.dataTransfer?.files?.[0];
        if (file) uploadFile(file);
      });
    }

    el("btnSaveApiKey")?.addEventListener("click", saveApiKey);
    el("btnUseOfflineMode")?.addEventListener("click", () => {
      closeModal("apiKeyModal");
      toast("Using local hybrid RAG (offline)");
    });

    el("chatForm")?.addEventListener("submit", (e) => {
      e.preventDefault();
      sendQuery(el("chatInput").value);
    });
    el("chatInput")?.addEventListener("input", (e) => autosize(e.target));
    el("chatInput")?.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendQuery(el("chatInput").value);
      }
    });
    el("btnClearChat")?.addEventListener("click", () => {
      resetChat("<p>Chat cleared. Ask me anything about this file.</p>");
    });
    document.querySelectorAll(".prompt-pill").forEach((pill) => {
      pill.addEventListener("click", () => sendQuery(pill.dataset.prompt));
    });

    el("btnSampleJdAi")?.addEventListener("click", () => {
      el("jobDescInput").value = SAMPLE_JD.ai;
    });
    el("btnSampleJdDevOps")?.addEventListener("click", () => {
      el("jobDescInput").value = SAMPLE_JD.devops;
    });
    el("btnRunJobMatch")?.addEventListener("click", runJobMatch);
    el("btnRefreshInterview")?.addEventListener("click", async () => {
      state.loadedTabs.delete("tab-interview");
      await loadTabData("tab-interview", true);
      toast("Interview questions refreshed");
    });
  }

  async function init() {
    bindEvents();
    try {
      await refreshStatus();
      resetChat();
      await loadTabData("tab-chat");
    } catch (err) {
      toast("Cannot reach API. Start the FastAPI server first.", "error");
    }
  }

  document.addEventListener("DOMContentLoaded", init);
})();
