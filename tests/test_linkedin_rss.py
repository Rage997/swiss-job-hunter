"""Tests for LinkedIn scraper (guest HTML API) — no network calls."""
import pytest
from bs4 import BeautifulSoup
from scrapers.linkedin_rss import LinkedInRssScraper

# Minimal card matching the selectors used in _parse_card
_CARD_FULL = """<li>
  <a class="base-card__full-link"
     href="/jobs/view/Senior-ML-Engineer-Google-1234567890/?trk=rss&amp;refId=xyz"></a>
  <h3 class="base-search-card__title">Senior ML Engineer</h3>
  <h4 class="base-search-card__subtitle">Google</h4>
  <span class="job-search-card__location">Zürich, Switzerland</span>
  <time datetime="2025-01-06"></time>
</li>"""

_CARD_NO_TITLE = """<li>
  <h4 class="base-search-card__subtitle">Google</h4>
</li>"""

_CARD_NO_COMPANY = """<li>
  <a class="base-card__full-link" href="/jobs/view/ML-Engineer-1111111111/"></a>
  <h3 class="base-search-card__title">ML Engineer</h3>
</li>"""

_PAGE = f"""<html><body><ul>
{_CARD_FULL}
<li>
  <a class="base-card__full-link" href="/jobs/view/Data-Scientist-Swiss-Re-9876543210/"></a>
  <h3 class="base-search-card__title">Data Scientist</h3>
  <h4 class="base-search-card__subtitle">Swiss Re</h4>
  <span class="job-search-card__location">Basel, Switzerland</span>
</li>
</ul></body></html>"""


def _card(html: str):
    return BeautifulSoup(html, "lxml").select_one("li")


def test_parse_card_returns_job():
    job = LinkedInRssScraper()._parse_card(_card(_CARD_FULL))
    assert job is not None
    assert job.title == "Senior ML Engineer"
    assert job.company == "Google"
    assert "Zürich" in job.location
    assert job.source == "linkedin.com"
    assert job.source_job_id == "1234567890"
    assert job.url == "https://www.linkedin.com/jobs/view/1234567890/"


def test_parse_card_url_strips_slug_and_tracking():
    job = LinkedInRssScraper()._parse_card(_card(_CARD_FULL))
    assert job is not None
    assert "trk" not in job.url
    assert "refId" not in job.url
    assert job.url == "https://www.linkedin.com/jobs/view/1234567890/"


def test_parse_card_missing_title_returns_none():
    job = LinkedInRssScraper()._parse_card(_card(_CARD_NO_TITLE))
    assert job is None


def test_parse_card_missing_company_defaults():
    job = LinkedInRssScraper()._parse_card(_card(_CARD_NO_COMPANY))
    assert job is not None
    assert job.company == "Unknown"
    assert job.location == "Switzerland"


def test_parse_page_returns_multiple_jobs():
    jobs = LinkedInRssScraper()._parse_page(_PAGE)
    assert len(jobs) == 2
    assert jobs[0].title == "Senior ML Engineer"
    assert jobs[0].source_job_id == "1234567890"
    assert jobs[1].title == "Data Scientist"
    assert jobs[1].source_job_id == "9876543210"


@pytest.mark.asyncio
async def test_short_pages_do_not_skip_vacancies():
    from types import SimpleNamespace
    from urllib.parse import parse_qs, urlparse

    scraper = LinkedInRssScraper()
    ids = [str(1000000000 + n) for n in range(23)]

    async def fetch(url, **kwargs):
        start = int(parse_qs(urlparse(url).query)["start"][0])
        return SimpleNamespace(text="".join(
            _CARD_FULL.replace("1234567890", jid) for jid in ids[start:start + 10]
        ))

    scraper._fetch = fetch
    jobs = [job async for job in scraper.scrape("Engineer", "", 5)]
    assert [job.source_job_id for job in jobs] == ids


@pytest.mark.asyncio
async def test_repeated_pages_do_not_duplicate_results():
    from types import SimpleNamespace
    scraper = LinkedInRssScraper()

    async def fetch(url, **kwargs):
        return SimpleNamespace(text=_PAGE)

    scraper._fetch = fetch
    jobs = [job async for job in scraper.scrape("Engineer", "", 5)]
    assert [job.source_job_id for job in jobs] == ["1234567890", "9876543210"]


@pytest.mark.asyncio
async def test_malformed_card_surfaces_partial_search_failure():
    from types import SimpleNamespace
    from scrapers.base import ScraperError
    scraper = LinkedInRssScraper()

    async def fetch(url, **kwargs):
        return SimpleNamespace(text=_CARD_FULL + _CARD_NO_TITLE)

    scraper._fetch = fetch
    emitted = []
    with pytest.raises(ScraperError, match="Unparseable job card"):
        async for job in scraper.scrape("Engineer", "", 5):
            emitted.append(job.source_job_id)
    assert emitted == ["1234567890"]
