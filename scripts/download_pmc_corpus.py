"""Download CC-licensed PMC article packages listed in a BioRAG corpus manifest."""

from __future__ import annotations

import argparse
import json
import re
import urllib.request
import xml.etree.ElementTree as etree
from pathlib import Path
from urllib.parse import urljoin

OAI_RECORD = (
    "https://pmc.ncbi.nlm.nih.gov/api/oai/v1/mh/"
    "?verb=GetRecord&identifier=oai:pubmedcentral.nih.gov:{numeric_id}&metadataPrefix=pmc"
)
CLOUD_PDF = "https://pmc-oa-opendata.s3.amazonaws.com/{pmcid}.{version}/{pmcid}.{version}.pdf"
PMC_ARTICLE = "https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/"
USER_AGENT = "BioRAG research prototype (open-access corpus downloader)"
PDF_LINK = re.compile(r"href=[\"']([^\"']+\.pdf(?:\?[^\"']*)?)[\"']", re.IGNORECASE)


def fetch(url: str) -> bytes:
    # PMC's OA service often returns an ftp:// link. HTTPS is more reliable on
    # current Windows networks and points to the same NCBI public archive.
    url = url.replace("ftp://ftp.ncbi.nlm.nih.gov/", "https://ftp.ncbi.nlm.nih.gov/")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def article_pdf_url(pmcid: str) -> str | None:
    """Resolve the article PDF from PMC's current OAI-PMH JATS metadata."""
    numeric_id = pmcid.upper().removeprefix("PMC")
    root = etree.fromstring(fetch(OAI_RECORD.format(numeric_id=numeric_id)))
    article_url = PMC_ARTICLE.format(pmcid=pmcid)
    xlink_href = "{http://www.w3.org/1999/xlink}href"
    for element in root.iter():
        if not element.tag.endswith("self-uri"):
            continue
        content_type = (element.get("content-type") or "").lower()
        href = element.get(xlink_href) or element.get("href")
        if href and ("pdf" in content_type or href.lower().endswith(".pdf")):
            return href if href.startswith("http") else urljoin(article_url + "pdf/", href)
    # Defensive fallback for records whose JATS omits a self-uri.
    html = fetch(article_url).decode("utf-8", errors="replace")
    candidates = [urljoin(article_url, match) for match in PDF_LINK.findall(html)]
    return next((url for url in candidates if "/pdf/" in url), None)


def download_paper(pmcid: str, output_dir: Path, version: int = 1) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"{pmcid}.{version}.pdf"
    if target.exists() and target.read_bytes()[:5] == b"%PDF-":
        return [target]
    # The official post-August-2026 distribution path is the anonymous PMC
    # Cloud Service bucket. OAI-PMH remains the metadata/full-text XML source.
    pdf_url = CLOUD_PDF.format(pmcid=pmcid, version=version)
    try:
        payload = fetch(pdf_url)
    except OSError:
        # A small number of records have a later article version. Resolve the
        # browser-facing filename only as a compatibility fallback.
        fallback = article_pdf_url(pmcid)
        if not fallback:
            raise RuntimeError(f"PMC did not expose a PDF for {pmcid}")
        payload = fetch(fallback)
    if not payload.startswith(b"%PDF"):
        raise RuntimeError(f"PMC PDF link returned a non-PDF response for {pmcid}")
    target.write_bytes(payload)
    return [target]


def main() -> None:
    parser = argparse.ArgumentParser(description="Download open-access PMC PDFs for a BioRAG corpus.")
    parser.add_argument("--manifest", type=Path, default=Path("data/corpus/manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("data/corpus/pdfs"))
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    downloaded = 0
    failed: list[str] = []
    for paper in manifest["papers"]:
        pmcid = paper["pmcid"]
        try:
            files = download_paper(pmcid, args.output, int(paper.get("version", 1)))
        except (OSError, RuntimeError, etree.ParseError) as exc:
            failed.append(pmcid)
            print(f"{pmcid}: skipped ({exc})")
            continue
        downloaded += len(files)
        print(f"{pmcid}: downloaded {len(files)} PDF(s)")
    if not downloaded:
        raise RuntimeError(f"No PDFs downloaded. Unavailable PMC IDs: {', '.join(failed)}")


if __name__ == "__main__":
    main()
