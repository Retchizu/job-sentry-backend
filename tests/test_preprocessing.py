from app.preprocessing import (
    build_combined_text,
    clean_text,
    detect_warning_signals,
    preprocess_job_post,
)


class TestCleanText:
    def test_removes_html(self):
        assert "hello world" in clean_text("<p>hello</p> <b>world</b>")

    def test_removes_urls(self):
        result = clean_text("visit https://example.com for details")
        assert "https://" not in result

    def test_removes_emails(self):
        result = clean_text("contact user@example.com now")
        assert "@" not in result

    def test_normalizes_whitespace(self):
        result = clean_text("too   many    spaces")
        assert result == "too many spaces"

    def test_handles_none(self):
        assert clean_text(None) == ""

    def test_handles_empty(self):
        assert clean_text("") == ""


class TestBuildCombinedText:
    def test_combines_fields(self):
        result = build_combined_text("Engineer", "Build things", "Python", "Acme Corp")
        assert "Engineer" in result
        assert "Build things" in result
        assert "Python" in result
        assert "Acme Corp" in result

    def test_handles_empty_fields(self):
        result = build_combined_text("Engineer", "Build things", "", "")
        assert "Engineer" in result
        assert "Build things" in result


class TestDetectWarningSignals:
    def test_detects_scam_keywords(self):
        signals = detect_warning_signals(
            "earn thousands easy money guaranteed", True, True, True, 100
        )
        assert any("Suspicious keywords" in s for s in signals)

    def test_detects_missing_company(self):
        signals = detect_warning_signals("a normal job post", True, False, True, 100)
        assert any("Missing company" in s for s in signals)

    def test_detects_missing_salary(self):
        signals = detect_warning_signals("a normal job post", False, True, True, 100)
        assert any("No salary" in s for s in signals)

    def test_detects_short_description(self):
        signals = detect_warning_signals("short", True, True, True, 5)
        assert any("Very short" in s for s in signals)

    def test_clean_post_no_signals(self):
        signals = detect_warning_signals(
            "We are seeking an experienced software developer to join our team",
            True,
            True,
            True,
            200,
        )
        assert len(signals) == 0


class TestPreprocessJobPost:
    def test_returns_correct_structure(self):
        result = preprocess_job_post(
            job_title="Engineer",
            job_desc="Build software systems with Python and cloud infrastructure.",
            company_profile="Acme Corp is a tech company.",
            skills_desc="Python, AWS",
            salary_range="100k-150k",
        )
        assert isinstance(result.combined_text, str)
        assert result.has_salary is True
        assert result.has_company_profile is True
        assert result.has_skills_desc is True
        assert isinstance(result.warning_signals, list)

    def test_detects_missing_optional_fields(self):
        result = preprocess_job_post(
            job_title="Engineer",
            job_desc="Build software systems with Python and cloud infrastructure.",
        )
        assert result.has_salary is False
        assert result.has_company_profile is False
        assert result.has_skills_desc is False
