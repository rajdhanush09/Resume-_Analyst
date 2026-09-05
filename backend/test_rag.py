"""
Unit test script to verify PDF/text parsing, RAG chunking, vector retrieval, and resume analysis.
"""

import sys
import os

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.sample_data import SAMPLE_RESUMES
from backend.rag_engine import ResumeRAGStore
from backend.resume_analyzer import extract_profile, match_job_description, critique_resume, generate_interview_questions
from backend.pdf_parser import parse_text

def test_rag_pipeline():
    print("=== 1. Initializing RAG Store ===")
    store = ResumeRAGStore()
    
    print("=== 2. Indexing Sample Resumes ===")
    for sid, sample in SAMPLE_RESUMES.items():
        res = store.add_resume(sid, sample)
        print(f"  Indexed '{sample['title']}': {res['total_chunks']} chunks created. embeddings={res.get('embedding_mode')}")

    print(f"  MiniLM available: {store.minilm.available}")
    if not store.minilm.available:
        print(f"  MiniLM fallback reason: {store.minilm.error}")

    print("\n=== 3. Testing Semantic Retrieval ===")
    test_queries = [
        "What programming languages and frameworks does Alex know?",
        "Tell me about the candidate's experience with RAG and vector search",
        "What education or degrees does Alex have?"
    ]
    for q in test_queries:
        print(f"\nQuery: '{q}'")
        retrieved = store.retrieve("alex_chen_senior_ai_engineer", q, top_k=3)
        assert retrieved, f"No retrieval results for: {q}"
        for idx, match in enumerate(retrieved, 1):
            print(f"  Top {idx} [Score: {match['score']:.2f}, Section: {match['section']}]: {match['text'][:120]}...")
        
        answer_res = store.generate_answer("alex_chen_senior_ai_engineer", q, retrieved)
        print(f"  Generated Answer Mode: {answer_res['mode']}")

    print("\n=== 4. Testing Profile Extractor ===")
    sample = SAMPLE_RESUMES["alex_chen_senior_ai_engineer"]
    profile = extract_profile(sample)
    print(f"  Candidate: {profile['candidate_name']}")
    print(f"  Email: {profile['email']}")
    print(f"  Skills Found ({profile['total_skills_found']}): {profile['categorized_skills']}")

    print("\n=== 5. Testing ATS Job Matcher ===")
    job_desc = """We are looking for a Senior AI & Full-Stack Engineer with strong experience in Python, FastAPI, React, Docker, Kubernetes, AWS, and RAG architectures to build enterprise search platforms."""
    match_result = match_job_description(sample, job_desc)
    print(f"  ATS Match Score: {match_result['match_percentage']}%")
    print(f"  Matched Skills: {match_result['matched_skills']}")
    print(f"  Missing Skills: {match_result['missing_skills']}")
    assert match_result["total_required_skills"] > 0, "JD skill extractor returned 0 required skills"
    assert "Python" in match_result["matched_skills"] or "python" in [s.lower() for s in match_result["matched_skills"]]

    generic_jd = "We want a motivated team player who can communicate well and grow with the company."
    generic = match_job_description(sample, generic_jd)
    print(f"  Generic JD required skills: {generic['total_required_skills']} (score {generic['match_percentage']}%)")

    js_resume = {"full_text": "Skills: JS, React JS, MySQL, PHP", "sections": {"skills": "JS, React JS, MySQL, PHP"}}
    js_jd = "Requirements: JavaScript, React, Python"
    alias_match = match_job_description(js_resume, js_jd)
    print(f"  Alias match: {alias_match['matched_skills']} missing={alias_match['missing_skills']}")
    assert alias_match["total_required_skills"] >= 3
    matched_lower = [s.lower() for s in alias_match["matched_skills"]]
    assert "javascript" in matched_lower
    assert "react" in matched_lower
    assert alias_match["missing_skills"]
    assert alias_match["skill_suggestions"]
    assert alias_match["skill_suggestions"][0]["skill"]
    assert "example_bullet" in alias_match["skill_suggestions"][0]

    print("\n=== 6. Testing Resume Critique & Audit ===")
    critique = critique_resume(sample)
    print(f"  Overall Score: {critique['overall_score']}/100")
    print(f"  Pillars: {critique['pillar_scores']}")
    print(f"  Strong Verbs Detected: {critique['strong_verbs']}")

    print("\n=== 7. Testing Interview Questions ===")
    questions = generate_interview_questions(sample)
    print(f"  Generated {len(questions)} tailored interview questions.")
    for q in questions[:2]:
        print(f"  • [{q.get('type')}]: {q.get('question') or q.get('content')[:100]}")

    print("\n>>> ALL TESTS COMPLETED SUCCESSFULLY! <<<")

if __name__ == "__main__":
    test_rag_pipeline()
