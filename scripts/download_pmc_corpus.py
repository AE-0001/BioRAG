"""Download CC-licensed PMC article packages listed in a BioRAG corpus manifest."""

from __future__ import annotations

import argparse
import json
import re
import tarfile
import tempfile
import urllib.request
import xml.etree.ElementTree as etree
from pathlib import Path
from urllib.parse import urljoin

OA_LOOKUP = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id={pmcid}"
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


def package_url(pmcid: str) -> str | None:
    root = etree.fromstring(fetch(OA_LOOKUP.format(pmcid=pmcid)))
    link = root.find(".//link[@format='tgz']")
    return link.get("href") if link is not None else None


def article_pdf_url(pmcid: str) -> str | None:
    """Find PMC's real, filename-bearing PDF URL (the generic /pdf/ URL is HTML)."""
    article_url = PMC_ARTICLE.format(pmcid=pmcid)
    html = fetch(article_url).decode("utf-8", errors="replace")
    candidates = [urljoin(article_url, match) for match in PDF_LINK.findall(html)]
    return next((url for url in candidates if "/pdf/" in url), None)


def download_paper(pmcid: str, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    archive = package_url(pmcid)
    if archive:
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as temporary:
            archive_path = Path(temporary.name)
        try:
            archive_path.write_bytes(fetch(archive))
            with tarfile.open(archive_path, "r:gz") as bundle:
                members = [member for member in bundle.getmembers() if member.name.lower().endswith(".pdf")]
                files: list[Path] = []
                for member in members:
                    source = bundle.extractfile(member)
                    if source is None:
                        continue
                    target = output_dir / f"{pmcid}_{Path(member.name).name}"
                    target.write_bytes(source.read())
                    files.append(target)
                if files:
                    return files
        except (OSError, tarfile.TarError):
            pass
        finally:
            archive_path.unlink(missing_ok=True)
    pdf_url = article_pdf_url(pmcid)
    if not pdf_url:
        raise RuntimeError(f"PMC did not expose a PDF link for {pmcid}")
    payload = fetch(pdf_url)
    if not payload.startswith(b"%PDF"):
        raise RuntimeError(f"PMC PDF link returned a non-PDF response for {pmcid}")
    target = output_dir / f"{pmcid}.pdf"
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
            files = download_paper(pmcid, args.output)
        except (OSError, RuntimeError, tarfile.TarError) as exc:
            failed.append(pmcid)
            print(f"{pmcid}: skipped ({exc})")
            continue
        downloaded += len(files)
        print(f"{pmcid}: downloaded {len(files)} PDF(s)")
    if not downloaded:
        raise RuntimeError(f"No PDFs downloaded. Unavailable PMC IDs: {', '.join(failed)}")


if __name__ == "__main__":
    main()
