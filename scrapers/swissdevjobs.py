"""SwissDevJobs public inventory and full-description APIs."""
from __future__ import annotations

import json
import unicodedata
from datetime import datetime
from typing import AsyncGenerator, Optional
from urllib.parse import quote

from bs4 import BeautifulSoup

from scrapers.base import BaseScraper, ScrapedJob, ScraperError

_BASE_URL = "https://swissdevjobs.ch"


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.casefold())
    return "".join(c for c in text if not unicodedata.combining(c))


def _location_key(text: str) -> str:
    return _normalize(text).replace("zuerich", "zurich")


def _description(data: object, slug: str) -> Optional[str]:
    if not isinstance(data, dict) or data.get("jobUrl") != slug:
        raise ValueError("Missing or mismatched job detail payload")
    if data.get("isDisabledOrOutdated") is True or data.get("isPaused") is True:
        return None
    sections = []
    for field, heading in (
        ("requirementsMustTextArea", "Requirements"),
        ("requirementsNiceTextArea", "Nice to have"),
        ("responsibilitiesTextArea", "Responsibilities"),
        ("description", "About the role and company"),
    ):
        value = data.get(field)
        if value is None:
            continue
        if not isinstance(value, str):
            raise ValueError(f"Invalid detail section: {field}")
        text = BeautifulSoup(value, "html.parser").get_text("\n").strip()
        if text:
            sections.append(f"{heading}\n{text}")
    if not sections:
        raise ValueError("Job detail contains no description sections")
    return "\n\n".join(sections)


class SwissDevJobsScraper(BaseScraper):
    source_name = "swissdevjobs.ch"

    async def scrape(
        self, keyword: str, location: str = "Zürich", max_pages: int = 5
    ) -> AsyncGenerator[ScrapedJob, None]:
        if max_pages <= 0:
            return
        try:
            response = await self._fetch(f"{_BASE_URL}/api/jobsLight")
            rows = response.json()
            # This is the complete, unpaginated inventory used by the website.
            # An empty inventory is not evidence of a legitimate empty search.
            if not isinstance(rows, list) or not rows:
                raise ValueError("Expected a nonempty job inventory")
            jobs = []
            seen = set()
            terms = _normalize(keyword).split()
            city = _location_key(location.strip())
            for row in rows:
                if not isinstance(row, dict):
                    raise ValueError("Invalid job inventory record")
                if row.get("isPaused") is True or row.get("isDisabledOrOutdated") is True:
                    continue
                for field in ("jobUrl", "name", "company"):
                    if not isinstance(row.get(field), str) or not row[field].strip():
                        raise ValueError(f"Job inventory record missing {field}")
                slug = row["jobUrl"]
                if "/" in slug:
                    raise ValueError(f"Invalid job slug: {slug}")
                places = [row.get("actualCity"), row.get("cityCategory")]
                if any(p is not None and not isinstance(p, str) for p in places):
                    raise ValueError(f"Invalid location for {slug}")
                places = [p.strip() for p in places if p and p.strip()]
                if not places:
                    raise ValueError(f"Missing location for {slug}")
                tags = []
                for field in ("technologies", "filterTags"):
                    values = row.get(field, [])
                    if not isinstance(values, list) or any(not isinstance(v, str) for v in values):
                        raise ValueError(f"Invalid {field} for {slug}")
                    tags.extend(values)
                category = row.get("techCategory", "")
                if not isinstance(category, str):
                    raise ValueError(f"Invalid technology category for {slug}")
                searchable = _normalize(" ".join([row["name"], row["company"], category, *tags]))
                if not all(term in searchable for term in terms):
                    continue
                if city and not any(city in _location_key(p) for p in places):
                    continue
                if slug in seen:
                    continue
                seen.add(slug)
                posted_at = None
                if row.get("activeFrom"):
                    posted_at = datetime.fromisoformat(row["activeFrom"])
                low, high = row.get("annualSalaryFrom"), row.get("annualSalaryTo")
                salary = None
                if low is not None and high is not None:
                    salary = f"CHF {low:,}–{high:,} / year"
                jobs.append(ScrapedJob(
                    title=row["name"].strip(),
                    company=row["company"].strip(),
                    location=places[0],
                    description="",  # Enrichment loads the full requirements, not just tags.
                    url=f"{_BASE_URL}/jobs/{quote(slug, safe='')}",
                    source=self.source_name,
                    source_job_id=slug,
                    salary_raw=salary,
                    employment_type=row.get("jobType"),
                    remote_ok=row.get("workplace") in {"remote", "hybrid"},
                    language_required=row.get("language"),
                    posted_at=posted_at,
                    raw_json=json.dumps(row, ensure_ascii=False),
                ))
            # Validate the entire response before emitting jobs: a changed schema
            # must be reported as a source failure, never as a successful partial run.
            for job in jobs:
                yield job
        except Exception as exc:
            raise ScraperError(f"{self.source_name}: {exc}") from exc

    async def fetch_full_description(self, source_job_id: str) -> Optional[tuple]:
        """None means unavailable; () means explicitly expired (safe to archive)."""
        url = f"{_BASE_URL}/jobs/{quote(source_job_id, safe='')}"
        try:
            response = await self._fetch(
                f"{_BASE_URL}/api/jobWithUrl/{quote(source_job_id, safe='')}",
                allow_status={404, 410},
            )
            if response.status_code in {404, 410}:
                return ()
            text = _description(response.json(), source_job_id)
            if text is None:
                return ()
            return text, url
        except Exception as exc:
            print(f"[swissdevjobs] detail fetch error: {exc}")
            return None
