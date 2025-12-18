"""
Streamlit Resume Analyzer (Enhanced)
File: streamlit_resume_analyzer.py

NEW FEATURES ADDED FOR END USERS:
- Resume Strength Meter (visual)
- Skill Match Breakdown (matched vs missing skills)
- Job Readiness Score
- ATS Compatibility Check
- Section Completeness Check
- Career Level Detection
- Keyword Density Insight
- Improvement Priority List

Instructions:
1. Install dependencies:
   pip install streamlit PyPDF2 python-docx scikit-learn

Run:
   streamlit run streamlit_resume_analyzer.py
"""

import streamlit as st
import tempfile
import os
import re

# Text extraction libraries
import PyPDF2
import docx

# Matching
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------- Utilities -------------------------

def extract_text_from_pdf(file_stream):
    try:
        reader = PyPDF2.PdfReader(file_stream)
        return "\n".join([page.extract_text() or "" for page in reader.pages])
    except Exception:
        return ""


def extract_text_from_docx(file_stream):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as tmp:
            tmp.write(file_stream.read())
            path = tmp.name
        doc = docx.Document(path)
        text = "\n".join(p.text for p in doc.paragraphs)
        os.unlink(path)
        return text
    except Exception:
        return ""


def extract_text(uploaded_file):
    if not uploaded_file:
        return ""
    name = uploaded_file.name.lower()
    if name.endswith('.pdf'):
        return extract_text_from_pdf(uploaded_file)
    elif name.endswith('.docx') or name.endswith('.doc'):
        uploaded_file.seek(0)
        return extract_text_from_docx(uploaded_file)
    else:
        try:
            return uploaded_file.getvalue().decode('utf-8')
        except Exception:
            return ""


DEFAULT_SKILLS = [
    "python", "java", "c++", "c#", "javascript", "sql", "html", "css",
    "react", "node", "django", "flask", "fastapi", "aws", "docker",
    "machine learning", "data science", "excel", "communication",
    "teamwork", "leadership"
]


def extract_skills(text):
    text = text.lower()
    return sorted({s for s in DEFAULT_SKILLS if re.search(r"\\b" + re.escape(s) + r"\\b", text)})


def compute_match_score(resume_text, jd_text):
    if not resume_text or not jd_text:
        return 0.0
    vectorizer = TfidfVectorizer(stop_words='english')
    vectors = vectorizer.fit_transform([resume_text, jd_text])
    return cosine_similarity(vectors[0:1], vectors[1:2])[0][0] * 100


def ats_check(text):
    issues = []
    if '|' in text:
        issues.append("Tables detected")
    if any(sym in text for sym in ['★', '✓', '✔']):
        issues.append("Special symbols detected")
    return issues


def section_check(text):
    sections = ["education", "skills", "experience", "projects"]
    missing = [s for s in sections if s not in text.lower()]
    return missing


def career_level(text):
    text = text.lower()
    if 'intern' in text or 'student' in text:
        return "Student / Fresher"
    elif 'year' in text:
        return "Experienced"
    return "Unknown"


def keyword_density(text, keyword):
    return text.lower().count(keyword.lower())


# ------------------------- Streamlit UI -------------------------

st.set_page_config(page_title="AI Resume Analyzer", layout="wide")
st.title("AI Resume Analyzer (Enhanced)")

uploaded_resume = st.file_uploader("Upload Resume (PDF/DOCX)", type=['pdf', 'docx'])
jd_text = st.text_area("Paste Job Description", height=200)

if st.button("Analyze"):
    resume_text = extract_text(uploaded_resume)
    if not resume_text:
        st.warning("Unable to read resume")
    else:
        score = compute_match_score(resume_text, jd_text)
        skills_resume = set(extract_skills(resume_text))
        skills_jd = set(extract_skills(jd_text))

        st.subheader("Resume Strength Meter")
        st.progress(int(score))

        st.subheader("Job Readiness")
        readiness = "Job Ready" if score > 70 else "Partially Ready" if score > 40 else "Not Ready"
        st.write(readiness)

        st.subheader("Skill Match Breakdown")
        st.write("Matched Skills:", list(skills_resume & skills_jd))
        st.write("Missing Skills:", list(skills_jd - skills_resume))

        st.subheader("ATS Compatibility Check")
        ats_issues = ats_check(resume_text)
        st.write("ATS Friendly" if not ats_issues else f"Issues: {ats_issues}")

        st.subheader("Section Completeness")
        missing_sections = section_check(resume_text)
        st.write("Missing Sections:" if missing_sections else "All sections present", missing_sections)

        st.subheader("Career Level Detection")
        st.write(career_level(resume_text))

        st.subheader("Keyword Density Insight")
        st.write("Python keyword count:", keyword_density(resume_text, "python"))

        st.subheader("Improvement Priority")
        if missing_sections:
            st.write("1. Add missing sections")
        if skills_jd - skills_resume:
            st.write("2. Add missing skills")

        st.success("Analysis completed successfully")

st.markdown("---")
st.caption("Enhanced AI Resume Analyzer | Streamlit Project")