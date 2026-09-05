"""
Resume Analyzer
Provides structured profile extraction, ATS Job Match scoring,
bullet point critique, and tailored interview question generation.
"""

import re
import os
from typing import Dict, Any, List, Optional
import google.generativeai as genai

# Common tech skills dictionary for regex & lexical extraction
KNOWN_SKILLS = {
    "Languages": [
        "python", "javascript", "typescript", "java", "c++", "c#", "golang", "rust",
        "ruby", "php", "swift", "kotlin", "scala", "sql", "html", "html5", "css", "css3",
        "js", "r",
    ],
    "Frameworks & Libraries": [
        "react", "react.js", "reactjs", "next.js", "nextjs", "vue", "angular", "node.js",
        "nodejs", "express", "fastapi", "django", "flask", "spring boot", "tailwind",
        "pytorch", "tensorflow", "keras", "pandas", "numpy", "scikit-learn", "langchain",
        "bootstrap", "jquery",
    ],
    "Cloud & DevOps": [
        "aws", "azure", "gcp", "google cloud", "docker", "kubernetes", "k8s", "terraform",
        "ci/cd", "github actions", "jenkins", "ansible", "linux", "nginx", "prometheus",
        "grafana", "github", "git", "xampp", "eclipse",
    ],
    "Databases & Storage": [
        "postgresql", "postgres", "mongodb", "mysql", "redis", "elasticsearch",
        "dynamodb", "cassandra", "sqlite", "snowflake", "neo4j", "pinecone",
    ],
    "Concepts & Tools": [
        "rest api", "rest", "graphql", "microservices", "rag", "llm", "jira", "agile",
        "scrum", "system design", "distributed systems", "oop", "tdd", "machine learning",
        "data structures", "algorithms",
    ],
}

SKILL_ALIASES = {
    "js": "JavaScript",
    "javascript": "JavaScript",
    "react.js": "React",
    "reactjs": "React",
    "react js": "React",
    "nextjs": "Next.js",
    "next.js": "Next.js",
    "nodejs": "Node.js",
    "node.js": "Node.js",
    "postgres": "PostgreSQL",
    "postgresql": "PostgreSQL",
    "k8s": "Kubernetes",
    "golang": "Go",
    "github": "GitHub",
    "git": "Git",
    "ci/cd": "CI/CD",
    "fastapi": "FastAPI",
    "mysql": "MySQL",
    "mongodb": "MongoDB",
    "github actions": "GitHub Actions",
    "scikit-learn": "Scikit-Learn",
    "llm": "LLM",
    "llms": "LLM",
}

JD_STOPWORDS = {
    "the", "and", "for", "with", "you", "our", "will", "are", "this", "that", "from",
    "job", "role", "team", "work", "working", "experience", "years", "year", "plus",
    "must", "have", "able", "strong", "good", "great", "skills", "skill", "required",
    "requirements", "responsibilities", "description", "about", "who", "what", "we",
    "looking", "candidate", "position", "company", "ability", "using", "including",
    "such", "other", "into", "your", "their", "them", "they", "should", "would",
    "preferred", "nice", "etc", "well", "also", "any", "all", "can", "may", "not",
    "be", "or", "to", "in", "on", "of", "as", "an", "a", "at", "by", "is", "it",
}

STRONG_ACTION_VERBS = [
    "architected", "engineered", "spearheaded", "orchestrated", "developed",
    "implemented", "optimized", "accelerated", "designed", "streamlined",
    "scaled", "automated", "launched", "championed", "delivered",
    "reduced", "increased", "maximized", "transformed", "migrated"
]

WEAK_WORDS = [
    "worked on", "helped", "responsible for", "assisted", "did", "participated in",
    "familiar with", "handled", "tasks included"
]

def extract_profile(doc_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract contact info, skills, education, and metadata from parsed resume."""
    full_text = doc_data.get("full_text", "")
    sections = doc_data.get("sections", {})
    
    # Extract Email
    email_match = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", full_text)
    email = email_match.group(0) if email_match else "Not found"

    # Extract Phone
    phone_match = re.search(r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}", full_text)
    phone = phone_match.group(0) if phone_match else "Not found"

    # Extract LinkedIn / GitHub / Links
    linkedin_match = re.search(r"linkedin\.com/in/[\w\-]+", full_text, re.IGNORECASE)
    linkedin = f"https://{linkedin_match.group(0)}" if linkedin_match else None
    
    github_match = re.search(r"github\.com/[\w\-]+", full_text, re.IGNORECASE)
    github = f"https://{github_match.group(0)}" if github_match else None

    # Candidate Name (heuristic: first non-empty line of text or page 1)
    lines = [line.strip() for line in full_text.split("\n") if line.strip()]
    candidate_name = "Candidate"
    if lines:
        first_line = lines[0]
        # If first line looks like a name (2-4 words, no special chars)
        if len(first_line.split()) <= 4 and not re.search(r"[@\d:/\|\\]", first_line) and len(first_line) < 40:
            candidate_name = first_line

    # Extract skills categorized
    categorized_skills: Dict[str, List[str]] = {}
    total_skills_found = 0
    full_text_lower = full_text.lower()
    
    for category, skill_list in KNOWN_SKILLS.items():
        found = []
        seen = set()
        for skill in skill_list:
            if skill_in_text(skill, full_text_lower):
                label = display_skill(skill)
                key = label.lower()
                if key in seen:
                    continue
                seen.add(key)
                found.append(label)
        if found:
            categorized_skills[category] = found
            total_skills_found += len(found)

    # Section completeness
    expected_sections = ["summary", "experience", "education", "skills", "projects", "certifications"]
    sections_present = [sec for sec in expected_sections if sec in sections]
    completeness_score = int((len(sections_present) / len(expected_sections)) * 100)

    return {
        "candidate_name": candidate_name,
        "email": email,
        "phone": phone,
        "linkedin": linkedin,
        "github": github,
        "total_words": doc_data.get("total_words") or len(full_text.split()),
        "total_pages": doc_data.get("total_pages", 1),
        "categorized_skills": categorized_skills,
        "total_skills_found": total_skills_found,
        "sections_detected": list(sections.keys()),
        "completeness_score": completeness_score,
        "summary_text": sections.get("summary", ""),
        "experience_text": sections.get("experience", ""),
        "education_text": sections.get("education", ""),
        "skills_text": sections.get("skills", "")
    }


def skill_in_text(skill: str, text_lower: str) -> bool:
    """Match a skill token without breaking on '.', '+', '#', or '/'."""
    skill = skill.lower().strip()
    if not skill or not text_lower:
        return False
    if skill == "r":
        return bool(re.search(r"(?:^|[\s,;/|])r(?:$|[\s,;/|])", text_lower))
    if skill == "go":
        return bool(re.search(r"(?:^|[\s,;/|])go(?:$|[\s,;/|])", text_lower))
    escaped = re.escape(skill)
    if re.search(r"[^a-z0-9]", skill):
        pattern = rf"(?<![a-z0-9]){escaped}(?![a-z0-9])"
    else:
        pattern = rf"\b{escaped}\b"
    return re.search(pattern, text_lower) is not None


def display_skill(skill: str) -> str:
    key = skill.lower().strip()
    if key in SKILL_ALIASES:
        return SKILL_ALIASES[key]
    if key in {"aws", "sql", "html", "html5", "css", "css3", "php", "gcp", "api", "rag", "llm", "tdd", "oop"}:
        return key.upper()
    return skill.title() if len(skill) > 3 else skill.upper()


def skill_spellings(skill: str) -> List[str]:
    """All common writings of a skill so JS matches JavaScript, React JS matches React, etc."""
    canon = display_skill(skill)
    spellings = {skill.lower().strip(), canon.lower()}
    for alias, label in SKILL_ALIASES.items():
        if label.lower() == canon.lower():
            spellings.add(alias.lower())
            spellings.add(label.lower())
    return [s for s in spellings if s]


def skill_mentioned(skill: str, text_lower: str) -> bool:
    return any(skill_in_text(variant, text_lower) for variant in skill_spellings(skill))


def extract_known_skills(text: str) -> List[str]:
    text_lower = (text or "").lower()
    found: List[str] = []
    seen = set()
    for skill_list in KNOWN_SKILLS.values():
        for skill in skill_list:
            if skill_in_text(skill, text_lower):
                label = display_skill(skill)
                key = label.lower()
                if key not in seen:
                    seen.add(key)
                    found.append(label)
    return found


def extract_jd_requirement_terms(job_description: str) -> List[str]:
    """Pull extra tools/keywords from requirement-style lines when the JD is not a tech laundry list."""
    extras: List[str] = []
    seen = set()
    heading = re.compile(
        r"^(?:requirements?|qualifications?|responsibilities|must have|tech(?:nical)? skills?|tech stack)\s*:?\s*",
        re.I,
    )
    for raw_line in (job_description or "").splitlines():
        line = heading.sub("", raw_line.strip())
        if not line:
            continue
        if not re.search(
            r"requir|qualif|must have|tech(?:nical)? skills?|tech stack|\bskills?\b|experience with|proficien",
            raw_line,
            re.I,
        ):
            if not re.match(r"^[\-\u2022*●]", raw_line.strip()):
                continue
        parts = re.split(r"[,;|•●\u2022]|(?:\s+and\s+)|(?:\s+or\s+)", line)
        for part in parts:
            token = re.sub(r"[^A-Za-z0-9.+#/\- ]", "", part).strip(" .-")
            token = re.sub(r"^(?:experience|knowledge|proficiency|hands-on)\s+(?:in|with|of)?\s*", "", token, flags=re.I)
            words = [w for w in token.split() if w.lower() not in JD_STOPWORDS]
            if not 1 <= len(words) <= 3:
                continue
            if words[0][:1].isdigit():
                continue
            if any(w.lower() in {"engineer", "architect", "developer", "hiring", "candidate", "measurable", "impact", "hands-on"} for w in words):
                continue
            token = " ".join(words)
            if not re.search(r"[A-Za-z]", token):
                continue
            if re.search(r"\d", token) and not re.search(r"[A-Za-z]{2,}", token):
                continue
            key = token.lower()
            if key in seen or len(key) < 2:
                continue
            seen.add(key)
            extras.append(token)
    return extras[:24]


def skill_category_label(skill: str) -> str:
    key = skill.lower().strip()
    canon = display_skill(skill).lower()
    for category, skill_list in KNOWN_SKILLS.items():
        for item in skill_list:
            if item.lower() == key or display_skill(item).lower() == canon:
                return category
    return "Skills"


def _jd_context_for_skill(skill: str, job_description: str) -> str:
    text = job_description or ""
    if not text:
        return "This skill appears in the job description."
    pattern = re.compile(re.escape(skill), re.I)
    for raw in re.split(r"(?<=[.!?\n])\s+", text):
        if pattern.search(raw) or skill_mentioned(skill, raw.lower()):
            clipped = re.sub(r"\s+", " ", raw).strip()
            if 12 < len(clipped) < 180:
                return clipped
    return f"The job description asks for {display_skill(skill)}."


def build_skill_suggestions(missing_skills: List[str], job_description: str) -> List[Dict[str, Any]]:
    """Per-gap advice: where to add the skill and a sample resume bullet."""
    suggestions: List[Dict[str, Any]] = []
    for skill in missing_skills:
        label = display_skill(skill) if skill.lower() in SKILL_ALIASES else skill
        category = skill_category_label(skill)
        if category in {"Cloud & DevOps"}:
            example = (
                f"Deployed and operated a project using {label}, including setup notes and a simple CI or hosting workflow."
            )
            where = "Skills section and one internship/project bullet"
        elif category in {"Databases & Storage"}:
            example = (
                f"Designed or queried data with {label} (schema, CRUD, or reporting) in a course or project."
            )
            where = "Skills section and a project bullet"
        elif category in {"Frameworks & Libraries"}:
            example = (
                f"Built a [web/app] feature with {label} that [users could do X / reduced Y / shipped Z]."
            )
            where = "Skills section and a project or internship bullet"
        elif category in {"Languages"}:
            example = (
                f"Implemented [feature/module] in {label} for [project], focusing on [problem you solved]."
            )
            where = "Skills section (near the top)"
        else:
            example = (
                f"Applied {label} in [project or coursework] to [specific outcome]."
            )
            where = "Skills section; mention it in Experience if you used it on the job"
        suggestions.append({
            "skill": label,
            "category": category,
            "where_to_add": where,
            "why": _jd_context_for_skill(skill, job_description),
            "suggestion": (
                f"If you already know {label}, add it to Skills and prove it in one bullet. "
                f"If you do not know it yet, learn a small project with {label} before listing it — do not fake the keyword."
            ),
            "example_bullet": example,
        })
    return suggestions


def match_job_description(
    doc_data: Dict[str, Any],
    job_description: str,
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Evaluate candidate resume against a target Job Description (ATS match scoring & gap analysis).
    """
    full_text = doc_data.get("full_text", "")
    full_text_lower = full_text.lower()
    jd_text = job_description or ""

    known_in_jd = extract_known_skills(jd_text)
    extra_in_jd = extract_jd_requirement_terms(jd_text)
    # Keep extras that are not already covered by known skills
    known_keys = {s.lower() for s in known_in_jd}
    jd_skills = list(known_in_jd)
    for extra in extra_in_jd:
        if extra.lower() not in known_keys:
            jd_skills.append(extra)
            known_keys.add(extra.lower())

    matched_skills = []
    missing_skills = []
    for skill in jd_skills:
        if skill_mentioned(skill, full_text_lower):
            matched_skills.append(skill)
        else:
            missing_skills.append(skill)

    content_words = lambda text: {
        w for w in re.findall(r"[a-zA-Z][a-zA-Z0-9.+#]{2,}", (text or "").lower())
        if w not in JD_STOPWORDS
    }
    jd_words = content_words(jd_text)
    resume_words = content_words(full_text)
    word_overlap = jd_words.intersection(resume_words)

    if jd_skills:
        skill_score = (len(matched_skills) / max(1, len(jd_skills))) * 75
        keyword_score = (len(word_overlap) / max(1, len(jd_words))) * 25
        match_percentage = min(98, max(8, int(skill_score + keyword_score)))
    elif jd_words:
        match_percentage = min(80, max(10, int((len(word_overlap) / max(1, len(jd_words))) * 100)))
    else:
        match_percentage = 0

    # Try Gemini LLM for detailed ATS feedback if API key is active
    ai_feedback = None
    if api_key:
        try:
            genai.configure(api_key=api_key, transport="rest")
            model = genai.GenerativeModel("gemini-2.5-flash")
            prompt = f"""You are an ATS (Applicant Tracking System) and Senior Tech Recruiter.
Analyze this candidate's resume against the target Job Description.

RESUME CONTENT:
{full_text[:3000]}

JOB DESCRIPTION:
{job_description[:2000]}

Provide a structured ATS Evaluation with:
1. Executive Verdict (Strengths & Fit Assessment)
2. Top 3 Missing Qualifications or Keywords
3. Actionable Recommendations to customize the resume for this exact role.
Format in clean Markdown with bullet points.
"""
            res = model.generate_content(prompt)
            if res and res.text:
                ai_feedback = res.text.strip()
        except Exception:
            pass

    # Fallback recommendations if offline
    if not ai_feedback:
        recs = []
        if missing_skills:
            recs.append(
                f"Priority gaps vs this JD: **{', '.join(missing_skills[:8])}**. "
                "Add only skills you can discuss in an interview."
            )
        if match_percentage < 70:
            recs.append("Mirror the job's exact tool names in your Skills list and in 1–2 project bullets.")
        recs.append("Quantify outcomes (%, time saved, users, pages) so ATS and recruiters both pick them up.")
        ai_feedback = "\n\n".join([f"• {r}" for r in recs])

    return {
        "match_percentage": match_percentage,
        "required_skills": jd_skills,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "total_required_skills": len(jd_skills),
        "skill_suggestions": build_skill_suggestions(missing_skills, jd_text),
        "ai_feedback": ai_feedback
    }

def critique_resume(doc_data: Dict[str, Any], api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Audit resume quality, action verbs, metrics quantification, and ATS readability.
    """
    full_text = doc_data.get("full_text", "")
    full_text_lower = full_text.lower()
    sections = doc_data.get("sections", {})

    # 1. Action verb audit
    verbs_found = []
    for verb in STRONG_ACTION_VERBS:
        if re.search(r"\b" + verb + r"\b", full_text_lower):
            verbs_found.append(verb.capitalize())

    weak_phrases_found = []
    for weak in WEAK_WORDS:
        if weak in full_text_lower:
            weak_phrases_found.append(weak)

    # 2. Quantification / Metric audit (numbers, %, $, x increase)
    metric_matches = re.findall(r"(?:\d+%\b|\$\d+[\d,]*|\b\d+x\b|\b\d+\+?\s*(?:users|clients|engineers|teams|requests|ms|seconds|million|billion|k\b))", full_text, re.IGNORECASE)
    metrics_count = len(metric_matches)

    # 3. Pillar Scores
    action_verb_score = min(100, int((len(verbs_found) / 6) * 100))
    impact_metrics_score = min(100, int((metrics_count / 5) * 100))
    section_score = min(100, int((len(sections) / 5) * 100))
    readability_score = 90 if (doc_data.get("total_words") or len(full_text.split())) > 200 else 60
    
    overall_score = int((action_verb_score * 0.25) + (impact_metrics_score * 0.35) + (section_score * 0.2) + (readability_score * 0.2))

    # AI Critique if Gemini key available
    ai_suggestions = None
    if api_key:
        try:
            genai.configure(api_key=api_key, transport="rest")
            model = genai.GenerativeModel("gemini-2.5-flash")
            prompt = f"""You are a professional resume writer and career coach.
Review this resume and provide 4 high-impact, specific improvements to make this resume top 1%:

RESUME:
{full_text[:3000]}

Format with clear headers and bullet points. Include 'Before vs After' bullet rewrite examples.
"""
            res = model.generate_content(prompt)
            if res and res.text:
                ai_suggestions = res.text.strip()
        except Exception:
            pass

    return {
        "overall_score": overall_score,
        "pillar_scores": {
            "Action Verbs": action_verb_score,
            "Impact & Metrics": impact_metrics_score,
            "Section Structure": section_score,
            "ATS Readability": readability_score
        },
        "strong_verbs_count": len(verbs_found),
        "strong_verbs": verbs_found,
        "weak_phrases_detected": weak_phrases_found,
        "metrics_found_count": metrics_count,
        "sample_metrics": metric_matches[:5],
        "ai_suggestions": ai_suggestions
    }

def generate_interview_questions(doc_data: Dict[str, Any], api_key: Optional[str] = None) -> List[Dict[str, Any]]:
    """Generate role-tailored behavioral and technical interview questions based on resume."""
    full_text = doc_data.get("full_text", "")
    
    # Try Gemini LLM
    if api_key:
        try:
            genai.configure(api_key=api_key, transport="rest")
            model = genai.GenerativeModel("gemini-2.5-flash")
            prompt = f"""Based on this candidate's resume, generate 5 tailored interview questions:
- 2 Technical / Architecture questions challenging specific tools or systems they claimed to build.
- 2 Behavioral questions testing leadership, conflict, or metrics achievements mentioned.
- 1 Curveball / Problem-solving scenario.

RESUME:
{full_text[:3000]}

For each question, output:
- Type (Technical / Behavioral / Scenario)
- Question
- What to look for in the candidate's answer
"""
            res = model.generate_content(prompt)
            if res and res.text:
                return [{"type": "AI Tailored Prep", "content": res.text.strip()}]
        except Exception:
            pass

    # Built-in contextual fallback questions based on sections
    profile = extract_profile(doc_data)
    skills = profile.get("categorized_skills", {})
    all_skills = [s for sublist in skills.values() for s in sublist]
    top_skill = all_skills[0] if all_skills else "Python/JavaScript"
    second_skill = all_skills[1] if len(all_skills) > 1 else "Cloud Architecture"

    return [
        {
            "type": "Technical Deep Dive",
            "question": f"In your resume, you listed expertise with {top_skill} and {second_skill}. Can you walk us through the most complex architectural challenge you solved using these technologies?",
            "look_for": "Look for depth of knowledge, trade-off analysis, edge cases, and performance considerations."
        },
        {
            "type": "Impact & Metrics",
            "question": "Choose one major project or achievement from your work experience and explain the business impact and how you measured its success.",
            "look_for": "Clear alignment with business KPIs, quantified outcomes, and problem ownership."
        },
        {
            "type": "Behavioral / Teamwork",
            "question": "Tell me about a time when you had a technical disagreement with a teammate or stakeholder on system design. How did you resolve it?",
            "look_for": "Empathy, data-driven reasoning, collaboration, and constructive conflict resolution."
        },
        {
            "type": "System Scalability",
            "question": "If the systems or pipelines you built had to handle a 100x surge in traffic or data volume overnight, where would the bottlenecks occur and how would you re-architect them?",
            "look_for": "Caching strategies, database sharding, asynchronous processing, and horizontal scaling patterns."
        }
    ]
