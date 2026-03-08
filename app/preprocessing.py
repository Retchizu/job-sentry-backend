import re
from dataclasses import dataclass, field
from typing import List, Optional


SCAM_KEYWORDS = [
    "easy money", "work from home", "no experience needed", "guaranteed",
    "earn thousands", "urgent hiring", "immediate hiring", "contact now",
    "no degree required", "flexible hours", "earn 5000", "quick money",
    "make money fast", "unlimited income", "be your own boss",
    "financial freedom", "get rich", "risk free", "act now",
    "limited time offer", "congratulations", "you have been selected",
]


@dataclass
class PreprocessedInput:
    combined_text: str
    has_salary: bool
    has_company_profile: bool
    has_skills_desc: bool
    job_desc_length: int
    skills_desc_length: int
    company_profile_length: int
    warning_signals: List[str] = field(default_factory=list)


def clean_text(text: Optional[str]) -> str:
    if not text or not isinstance(text, str):
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+\.\S+", " ", text)
    text = re.sub(r"[^\w\s.,!?;:'\"-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_combined_text(
    job_title: str,
    job_desc: str,
    skills_desc: str,
    company_profile: str,
) -> str:
    """Matches thesis pipeline: concatenation of the four core text fields."""
    parts = [
        clean_text(job_title),
        clean_text(job_desc),
        clean_text(skills_desc),
        clean_text(company_profile),
    ]
    return " ".join(p for p in parts if p)


def detect_warning_signals(
    combined_text: str,
    has_salary: bool,
    has_company_profile: bool,
    has_skills_desc: bool,
    job_desc_length: int,
) -> List[str]:
    signals: List[str] = []
    text_lower = combined_text.lower()

    matched = [kw for kw in SCAM_KEYWORDS if kw in text_lower]
    if matched:
        signals.append(f"Suspicious keywords detected: {', '.join(matched[:5])}")

    if not has_company_profile:
        signals.append("Missing company details")
    if not has_salary:
        signals.append("No salary information provided")
    if not has_skills_desc:
        signals.append("No skills or requirements listed")
    if job_desc_length < 50:
        signals.append("Very short job description")

    caps_words = re.findall(r"\b[A-Z]{3,}\b", combined_text)
    if len(caps_words) > 5:
        signals.append("Excessive use of capital letters")

    exclamation_count = combined_text.count("!")
    if exclamation_count > 3:
        signals.append("Excessive exclamation marks")

    return signals


def preprocess_job_post(
    job_title: str,
    job_desc: str,
    company_profile: Optional[str] = None,
    skills_desc: Optional[str] = None,
    salary_range: Optional[str] = None,
    employment_type: Optional[str] = None,
    **kwargs,
) -> PreprocessedInput:
    combined_text = build_combined_text(
        job_title, job_desc, skills_desc or "", company_profile or ""
    )
    has_salary = bool(salary_range and salary_range.strip())
    has_company = bool(company_profile and company_profile.strip())
    has_skills = bool(skills_desc and skills_desc.strip())
    desc_len = len(clean_text(job_desc))
    skills_len = len(clean_text(skills_desc)) if skills_desc else 0
    profile_len = len(clean_text(company_profile)) if company_profile else 0

    warnings = detect_warning_signals(
        combined_text, has_salary, has_company, has_skills, desc_len
    )

    return PreprocessedInput(
        combined_text=combined_text,
        has_salary=has_salary,
        has_company_profile=has_company,
        has_skills_desc=has_skills,
        job_desc_length=desc_len,
        skills_desc_length=skills_len,
        company_profile_length=profile_len,
        warning_signals=warnings,
    )
