"""Scoring failures must not turn potentially suitable jobs into rejections."""
import json

import pytest

from analyzer import scorer
from db import session as db_session
from db.models import Job, JobStatus
from llm import router


@pytest.mark.parametrize("raw", [
    "", "not JSON", '{"score":', '{}',
    json.dumps({"score": -0.1, "explanation": "Invalid score"}),
    json.dumps({"score": 1.1, "explanation": "Invalid score"}),
    json.dumps({"score": float("nan"), "explanation": "Invalid score"}),
    json.dumps({"score": True, "explanation": "Invalid score"}),
    '[{"score": 0, "explanation": "Unrelated", "matched_skills": [], "missing_skills": []}]',
    '{"score": 0.5, "explanation": "", "matched_skills": [], "missing_skills": []}',
    '{"score": 0.5, "explanation": "Relevant", "matched_skills": "Python", "missing_skills": []}',
    '{"score": 0.5, "explanation": "Relevant", "matched_skills": [], "missing_skills": [null]}',
])
async def test_invalid_llm_response_is_not_a_zero_score(monkeypatch, raw):
    async def respond(**kwargs):
        return raw, "test"
    monkeypatch.setattr(router, "call_llm", respond)
    with pytest.raises(ValueError):
        await scorer.llm_score("Python", "Engineer", "Python development")


@pytest.mark.parametrize("fenced", [False, True])
async def test_valid_zero_score_is_a_real_assessment(monkeypatch, fenced):
    async def respond(**kwargs):
        body = json.dumps({"score": 0, "explanation": "Required medical qualifications are absent.",
                           "matched_skills": [], "missing_skills": ["medical license"]})
        return (f"```JSON\n{body}\n```" if fenced else body), "test"
    monkeypatch.setattr(router, "call_llm", respond)
    result = await scorer.llm_score("Python", "Doctor", "Medical license required")
    assert result.score == 0
    assert result.missing_skills == ["medical license"]


def add_job(description, status=JobStatus.NEW, score=None):
    with db_session.get_session() as session:
        job = Job(title="Engineer", company="Acme", location="Zurich", description=description,
                  source="test", url="https://example.test/job", dedup_hash="safety",
                  status=status, match_score=score)
        session.add(job)
        session.flush()
        return job.id


@pytest.mark.parametrize("status,score", [(JobStatus.NEW, None), (JobStatus.CONSIDERING, 0.8)])
def test_failed_scoring_preserves_job_and_existing_assessment(client, monkeypatch, status, score):
    jid = add_job("Python platform engineering and Docker operations. " * 20, status, score)
    async def respond(**kwargs):
        return '{"score":', "test"
    monkeypatch.setattr(router, "call_llm", respond)
    response = client.post("/run/analyze", json={"llm": True, "skip_scored": False})
    assert "error:" in response.text
    with db_session.get_session() as session:
        job = session.get(Job, jid)
        assert job.status == status
        assert job.match_score == score


def test_unknown_keywords_reach_llm_instead_of_archiving(client, monkeypatch):
    jid = add_job("Design reliable enterprise workflows and improve team collaboration. " * 20)
    async def respond(**kwargs):
        return json.dumps({"score": 0.8, "explanation": "Relevant engineering experience.",
                           "matched_skills": ["engineering"], "missing_skills": []}), "test"
    monkeypatch.setattr(router, "call_llm", respond)
    client.post("/run/analyze", json={"llm": True})
    with db_session.get_session() as session:
        job = session.get(Job, jid)
        assert job.status == JobStatus.SHORTLISTED
        assert job.match_score == 0.8
