"""Inventory completeness, search boundaries, and safe description enrichment."""
from unittest.mock import AsyncMock

import httpx
import pytest

from scrapers.base import ScraperError
from scrapers.swissdevjobs import SwissDevJobsScraper


def listing(slug="acme-rendering", **changes):
    row = {
        "jobUrl": slug,
        "name": "Senior Graphics Software Engineer",
        "company": "Acme Software AG",
        "actualCity": "Zürich (Enge)",
        "cityCategory": "Zurich",
        "technologies": ["C++", "WebGL"],
        "filterTags": ["Rendering"],
        "techCategory": "GameDev",
        "activeFrom": "2026-10-17T00:00:00+02:00",
        "annualSalaryFrom": 100000,
        "annualSalaryTo": 140000,
        "jobType": "Full-Time",
        "workplace": "hybrid",
        "language": "English",
    }
    row.update(changes)
    return row


def scraper_for(handler):
    scraper = SwissDevJobsScraper()
    scraper._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    scraper._polite_delay = AsyncMock()
    return scraper


async def search(payload, keyword="", location="", pages=1):
    async with scraper_for(lambda request: httpx.Response(200, json=payload)) as scraper:
        return [job async for job in scraper.scrape(keyword, location, pages)]


async def test_inventory_is_complete_beyond_first_page_and_deduplicated():
    rows = [listing(f"company-job-{n}", name=f"Engineer {n}") for n in range(206)]
    jobs = await search([*rows, rows[0], rows[-1]])
    assert [job.source_job_id for job in jobs] == [row["jobUrl"] for row in rows]


@pytest.mark.parametrize("city", ["Zürich", "Zurich", "Zuerich", "ZÜRICH"])
async def test_search_matches_multiword_title_and_city_aliases(city):
    jobs = await search([
        listing(),
        listing("bern", actualCity="Bern", cityCategory="Bern"),
        listing("other", name="Database Administrator", technologies=[], filterTags=[]),
    ], "graphics engineer", city)
    assert [job.source_job_id for job in jobs] == ["acme-rendering"]


@pytest.mark.parametrize("keyword", ["WebGL", "C++", "Rendering", "GameDev", "ACME"])
async def test_search_includes_technology_category_and_company_matches(keyword):
    assert [job.source_job_id for job in await search([listing()], keyword)] == ["acme-rendering"]


async def test_legitimate_no_match_is_empty_not_source_failure():
    assert await search([listing()], "neurosurgeon") == []
    assert await search([listing()], location="Lausanne") == []


async def test_matching_uses_actual_city_and_source_region():
    jobs = await search([listing(actualCity="Winterthur")], location="Zurich")
    assert [job.location for job in jobs] == ["Winterthur"]


async def test_paused_and_disabled_jobs_are_not_returned():
    jobs = await search([
        listing("paused", isPaused=True),
        listing("disabled", isDisabledOrOutdated=True),
        listing(),
    ])
    assert [job.source_job_id for job in jobs] == ["acme-rendering"]


async def test_metadata_comes_from_feed_not_slug_guesses():
    job, = await search([listing("unrelated-slug")])
    assert (job.title, job.company, job.location) == (
        "Senior Graphics Software Engineer", "Acme Software AG", "Zürich (Enge)"
    )
    assert job.source_job_id == "unrelated-slug"
    assert job.url == "https://swissdevjobs.ch/jobs/unrelated-slug"
    assert job.posted_at.isoformat() == "2026-10-17T00:00:00+02:00"
    assert "100,000" in job.salary_raw and "140,000" in job.salary_raw
    assert job.remote_ok is True
    assert job.employment_type == "Full-Time"
    assert job.language_required == "English"
    assert job.description == ""  # Allows the pipeline to request the full JD.


@pytest.mark.parametrize("payload", [[], {}, {"jobs": [listing()]}, None, ["changed schema"]])
async def test_unexpected_inventory_is_a_source_failure(payload):
    with pytest.raises(ScraperError):
        await search(payload)


@pytest.mark.parametrize("field,value", [
    ("jobUrl", ""), ("name", None), ("company", 42),
    ("technologies", "C++"), ("filterTags", [None]),
    ("actualCity", []), ("jobUrl", "jobs/invalid"),
])
async def test_bad_record_fails_before_emitting_partial_results(field, value):
    emitted = []
    async with scraper_for(lambda request: httpx.Response(
        200, json=[listing(), listing("bad", **{field: value})]
    )) as scraper:
        with pytest.raises(ScraperError):
            async for job in scraper.scrape("", "", 1):
                emitted.append(job)
    assert emitted == []


async def test_missing_location_is_not_silently_discarded():
    with pytest.raises(ScraperError):
        await search([listing(actualCity="", cityCategory="")])


@pytest.mark.parametrize("status", [403, 404, 410])
async def test_http_search_failures_are_not_empty_successes(status):
    async with scraper_for(lambda request: httpx.Response(status)) as scraper:
        with pytest.raises(ScraperError):
            _ = [job async for job in scraper.scrape("", "", 1)]


async def test_html_instead_of_json_is_a_source_failure():
    async with scraper_for(lambda request: httpx.Response(200, text="<html>blocked</html>")) as scraper:
        with pytest.raises(ScraperError):
            _ = [job async for job in scraper.scrape("", "", 1)]


async def test_zero_pages_does_not_search():
    def unexpected_request(request):
        pytest.fail("A disabled search must not request the inventory")
    async with scraper_for(unexpected_request) as scraper:
        assert [job async for job in scraper.scrape("", "", 0)] == []


def detail_payload(**changes):
    data = {
        "jobUrl": "acme-rendering",
        "requirementsMustTextArea": "<p>C++ &amp; graphics expertise</p>",
        "requirementsNiceTextArea": "WebGL experience",
        "responsibilitiesTextArea": "Build renderers\nReview performance",
        "description": "<p>Engineering team</p><p>Flexible working</p>",
    }
    data.update(changes)
    return data


async def enrich(payload, status=200):
    async with scraper_for(lambda request: httpx.Response(status, json=payload)) as scraper:
        return await scraper.fetch_full_description("acme-rendering")


async def test_description_keeps_all_sections_without_navigation_or_html():
    text, url = await enrich(detail_payload())
    for content in ["C++ & graphics expertise", "WebGL experience", "Build renderers",
                    "Review performance", "Engineering team", "Flexible working"]:
        assert content in text
    assert "<p>" not in text
    assert url == "https://swissdevjobs.ch/jobs/acme-rendering"


@pytest.mark.parametrize("payload", [
    None, [], "JOB_PLACEHOLDER",
    detail_payload(jobUrl="another-job"),
    detail_payload(requirementsMustTextArea="", requirementsNiceTextArea="",
                   responsibilitiesTextArea="", description=""),
    detail_payload(description={"changed": "schema"}),
])
async def test_bad_details_do_not_mark_a_live_job_expired(payload):
    assert await enrich(payload) is None


@pytest.mark.parametrize("status", [404, 410])
async def test_explicit_http_expiry_is_reported(status):
    assert await enrich("", status) == ()


@pytest.mark.parametrize("flag", ["isPaused", "isDisabledOrOutdated"])
async def test_explicit_source_expiry_is_reported(flag):
    assert await enrich(detail_payload(**{flag: True})) == ()


async def test_blocked_detail_is_unavailable_not_expired():
    assert await enrich("Blocked", 403) is None


async def test_non_json_detail_is_unavailable_not_expired():
    async with scraper_for(lambda request: httpx.Response(200, text="<html>blocked</html>")) as scraper:
        assert await scraper.fetch_full_description("acme-rendering") is None
