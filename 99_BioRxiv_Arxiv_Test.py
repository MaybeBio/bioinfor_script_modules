# 1.
# Arxiv test

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import httpx

from pyPaperFlow.arxiv_fetcher import ArxivFetcher
from pyPaperFlow.source_utils import safe_filename


ARXIV_FEED = """<?xml version='1.0' encoding='UTF-8'?>
<feed xmlns='http://www.w3.org/2005/Atom' xmlns:arxiv='http://arxiv.org/schemas/atom'>
  <entry>
    <id>http://arxiv.org/abs/2401.12345v1</id>
    <updated>2024-01-02T00:00:00Z</updated>
    <published>2024-01-01T00:00:00Z</published>
    <title>Deep Learning for Demo Papers</title>
    <summary>This paper studies deep learning for biology.</summary>
    <author><name>Alice Smith</name></author>
    <author><name>Bob Lee</name></author>
    <arxiv:doi>10.48550/arXiv.2401.12345</arxiv:doi>
    <arxiv:journal_ref>Demo Journal 2024</arxiv:journal_ref>
    <category term='cs.LG'/>
    <link rel='related' title='pdf' href='https://arxiv.org/pdf/2401.12345v1.pdf' type='application/pdf'/>
  </entry>
</feed>
"""


class FakeHttpClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append({"url": url, "params": params, "headers": headers, "timeout": timeout})
        if not self.responses:
            raise AssertionError("Unexpected extra request")
        return self.responses.pop(0)

    def close(self):
        return None


class FakePaperScraperResult:
    def __init__(self, rows):
        self.rows = rows

    def to_dict(self, orient="records"):
        if orient != "records":
            raise TypeError("FakePaperScraperResult only supports orient='records'")
        return list(self.rows)


class TestArxivFetcher(unittest.TestCase):
    def test_build_query_from_free_text(self):
        fetcher = ArxivFetcher(root_dir="/tmp")
        query = fetcher.build_query("deep learning", start_date="2024-01-01", end_date="2024-12-31")

        self.assertIn("all:deep", query)
        self.assertIn("all:learning", query)
        self.assertIn("submittedDate:[202401010000 TO 202412312359]", query)

    def test_build_query_clamps_future_end_date(self):
        fetcher = ArxivFetcher(root_dir="/tmp")
        today = datetime.now(timezone.utc).strftime("%Y%m%d")

        query = fetcher.build_query("deep learning", start_date="2024-01-01", end_date="2999-12-31")

        self.assertIn(f"submittedDate:[202401010000 TO {today}2359]", query)
        self.assertNotIn("299912312359", query)

    def test_native_fetch_retries_on_429_and_downloads_pdf(self):
        request = httpx.Request("GET", "https://export.arxiv.org/api/query")
        responses = [
            httpx.Response(429, headers={"Retry-After": "0"}, request=request, content=b"rate limited"),
            httpx.Response(200, request=request, content=ARXIV_FEED.encode("utf-8")),
        ]
        fake_client = FakeHttpClient(responses)

        def fake_download(url, path, headers=None, timeout=None):
            Path(path).write_bytes(b"%PDF-1.4\n% native demo pdf\n")
            return True

        with tempfile.TemporaryDirectory() as tmp:
            fetcher = ArxivFetcher(root_dir=tmp)
            with patch.object(fetcher, "_get_http_client", return_value=fake_client), patch(
                "pyPaperFlow.arxiv_fetcher.time.sleep"
            ) as sleep_mock, patch("pyPaperFlow.arxiv_fetcher.download_binary", side_effect=fake_download):
                records = fetcher.fetch_from_query("deep learning", output_dir=tmp, max_results=5)

            self.assertEqual(len(records), 1)
            record = records[0]
            self.assertEqual(record.source, "arxiv")
            self.assertEqual(record.source_id, "2401.12345v1")
            self.assertTrue(record.pdf_downloaded)
            sleep_mock.assert_called_once()
            self.assertEqual(len(fake_client.calls), 2)

            paper_dir = Path(tmp) / "arxiv" / "2024" / "2401.12345v1"
            stem = safe_filename(record.source_id)
            self.assertTrue((paper_dir / f"{stem}.json").exists())
            self.assertTrue((paper_dir / f"{stem}.pdf").exists())

    def test_paperscraper_backend_uses_api_results_and_downloads_pdf(self):
        fake_rows = [
            {
                "title": "Deep Learning for Demo Papers",
                "authors": "Alice Smith, Bob Lee",
                "date": "2024-01-01",
                "abstract": "This paper studies deep learning for biology.",
                "journal": "Demo Journal 2024",
                "doi": "10.48550/arXiv.2401.12345",
                "entry_id": "https://arxiv.org/abs/2401.12345v1",
                "pdf_url": "https://arxiv.org/pdf/2401.12345v1.pdf",
                "updated": "2024-01-02",
                "primary_category": "cs.LG",
                "comment": "",
            }
        ]

        def fake_api(query, fields=None, max_results=None, verbose=None):
            self.assertIn("all:deep", query)
            self.assertIn("all:learning", query)
            return FakePaperScraperResult(fake_rows)

        def fake_download(url, path, headers=None, timeout=None):
            Path(path).write_bytes(b"%PDF-1.4\n% paperscraper demo pdf\n")
            return True

        with tempfile.TemporaryDirectory() as tmp:
            fetcher = ArxivFetcher(root_dir=tmp, backend="paperscraper")
            with patch("pyPaperFlow.arxiv_fetcher.ArxivFetcher._load_paperscraper_api", return_value=fake_api), patch(
                "pyPaperFlow.arxiv_fetcher.download_binary", side_effect=fake_download
            ):
                records = fetcher.fetch_from_query("deep learning", output_dir=tmp, max_results=5)

            self.assertEqual(len(records), 1)
            record = records[0]
            self.assertEqual(record.source, "arxiv")
            self.assertEqual(record.source_id, "2401.12345v1")
            self.assertEqual(record.title, "Deep Learning for Demo Papers")
            self.assertEqual(record.pdf_url, "https://arxiv.org/pdf/2401.12345v1.pdf")
            self.assertTrue(record.pdf_downloaded)

            paper_dir = Path(tmp) / "arxiv" / "2024" / "2401.12345v1"
            stem = safe_filename(record.source_id)
            self.assertTrue((paper_dir / f"{stem}.json").exists())
            self.assertTrue((paper_dir / f"{stem}.pdf").exists())


if __name__ == "__main__":
    unittest.main()


##############################################################################################################################################


# 2.
# BioRxiv test

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pyPaperFlow.biorxiv_fetcher import BioRxivFetcher
from pyPaperFlow.source_utils import basic_boolean_text_match, safe_filename


class FakeResponse:
    def __init__(self, *, text: str = "", content: bytes = b"", json_data=None, status_code: int = 200):
        self.text = text
        self.content = content
        self._json_data = json_data
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        if self._json_data is None:
            raise ValueError("No JSON payload")
        return self._json_data


BIORXIV_PAYLOAD = {
    "messages": [{"status": "ok"}],
    "collection": [
        {
            "title": "AlphaFold improves protein modeling",
            "abstract": "AlphaFold changes the landscape for structure prediction.",
            "authors": [{"name": "Alice Smith"}, {"name": "Bob Lee"}],
            "date": "2026-01-05",
            "doi": "10.1101/2026.01.05.123456",
            "version": "1",
            "category": ["Bioinformatics"],
            "server": "biorxiv",
        }
    ],
}


class TestBioRxivFetcher(unittest.TestCase):
    def test_boolean_text_match_query(self):
        text = BIORXIV_PAYLOAD["collection"][0]["title"] + " " + BIORXIV_PAYLOAD["collection"][0]["abstract"]
        self.assertTrue(basic_boolean_text_match(text, "AlphaFold AND structure"))
        self.assertFalse(basic_boolean_text_match(text, "AlphaFold AND microscopy"))

    def test_search_paginates_past_first_page(self):
        first_page = {
            "message": {
                "items": [
                    {
                        "DOI": f"10.1101/2025.01.01.00000{index}",
                        "title": [f"Unrelated paper {index}"],
                        "publisher": "openRxiv",
                        "type": "posted-content",
                        "prefix": "10.64898",
                        "author": [{"given": "Example", "family": "Author"}],
                        "issued": {"date-parts": [[2026, 1, 5]]},
                    }
                    for index in range(5)
                ],
                "next-cursor": "cursor-2",
            }
        }
        second_page = {
            "message": {
                "items": [
                    {
                        "DOI": "10.1101/2026.01.06.123456",
                        "title": ["AlphaFold resolves structure questions"],
                        "abstract": "<jats:p>A structure-focused analysis of AlphaFold.</jats:p>",
                        "publisher": "openRxiv",
                        "type": "posted-content",
                        "prefix": "10.64898",
                        "author": [{"given": "Alice", "family": "Smith"}],
                        "issued": {"date-parts": [[2026, 1, 6]]},
                    }
                ],
                "next-cursor": "cursor-3",
            }
        }

        def fake_get(url, params=None, headers=None, timeout=None):
            if "api.crossref.org" in url:
                cursor = (params or {}).get("cursor")
                return FakeResponse(json_data=first_page if cursor == "*" else second_page)
            raise AssertionError(f"Unexpected URL: {url}")

        with patch("pyPaperFlow.biorxiv_fetcher.requests.get", side_effect=fake_get):
            fetcher = BioRxivFetcher(root_dir="/tmp", window_days=31)
            records = fetcher.search(
                "AlphaFold AND structure",
                start_date="2026-01-01",
                end_date="2026-01-31",
                max_results=5,
            )

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].title, "AlphaFold resolves structure questions")

    def test_fetch_and_save_query(self):
        def fake_get(url, params=None, headers=None, timeout=None):
            if "api.crossref.org" in url:
                return FakeResponse(
                    json_data={
                        "message": {
                            "items": [
                                {
                                    "DOI": "10.1101/2026.01.05.123456",
                                    "title": ["AlphaFold improves protein modeling"],
                                    "abstract": "<jats:p>AlphaFold changes the landscape for structure prediction.</jats:p>",
                                    "publisher": "openRxiv",
                                    "type": "posted-content",
                                    "prefix": "10.64898",
                                    "author": [{"given": "Alice", "family": "Smith"}, {"given": "Bob", "family": "Lee"}],
                                    "issued": {"date-parts": [[2026, 1, 5]]},
                                    "subject": ["Bioinformatics"],
                                }
                            ],
                            "next-cursor": "",
                        }
                    }
                )
            if url.endswith(".pdf"):
                return FakeResponse(content=b"%PDF-1.4\n% biorxiv demo pdf\n")
            if "biorxiv.org/content" in url:
                html = """
                <html>
                  <head>
                    <meta name='citation_pdf_url' content='https://www.biorxiv.org/content/10.1101/2026.01.05.123456v1.full.pdf' />
                  </head>
                </html>
                """
                return FakeResponse(text=html)
            raise AssertionError(f"Unexpected URL: {url}")

        with tempfile.TemporaryDirectory() as tmp:
            fetcher = BioRxivFetcher(root_dir=tmp, window_days=31)
            with patch("pyPaperFlow.biorxiv_fetcher.requests.get", side_effect=fake_get), patch(
                "pyPaperFlow.source_utils.requests.get", side_effect=fake_get
            ):
                records = fetcher.fetch_from_query(
                    "AlphaFold AND structure",
                    output_dir=tmp,
                    start_date="2026-01-01",
                    end_date="2026-01-31",
                    max_results=5,
                    download_pdf=True,
                )

            self.assertEqual(len(records), 1)
            record = records[0]
            self.assertEqual(record.source, "biorxiv")
            self.assertTrue(record.source_id.startswith("10.1101/2026.01.05.123456"))
            self.assertTrue(record.pdf_downloaded)

            paper_dir = Path(tmp) / "biorxiv" / "2026" / record.source_id.replace("/", "_")
            stem = safe_filename(record.source_id)
            self.assertTrue((paper_dir / f"{stem}.json").exists())
            self.assertTrue((paper_dir / f"{stem}.pdf").exists())


if __name__ == "__main__":
    unittest.main()
