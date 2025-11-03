#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ICLR 2026 OpenReview scraper (two-stage: discover -> download)

Stage 1 (discover):
  - Crawls the public submissions listing for ICLR 2026 active submissions.
  - Visits each /forum?id=... page.
  - Keeps ONLY submissions that expose Supplementary Material.
  - Writes a metadata JSON file with entries:
      {
        "forum_id": str,
        "forum_url": str,
        "title": str,
        "status": str,
        "pdf_url": str or null,
        "supplementary_url": str,
        "code_links": [str, ...]
      }

Stage 2 (download):
  - Reads the metadata JSON and downloads missing paper PDFs and supplementary files.
  - Optional --dry-run to just list planned actions.
  - Optional --clone-code to shallow-clone Git repos; non-git code links are saved to links.txt.

Notes:
  - Uses a polite delay + basic retry/backoff.
  - Skips withdrawn submissions.
  - Robust supplementary detection (labels, attachment endpoints, archive heuristics).
  - List view URL used: https://openreview.net/submissions?venue=ICLR.cc/2026/Conference
"""

import argparse
import os
import re
import sys
import json
import time
import subprocess
from urllib.parse import urljoin, urlparse, parse_qs

import requests
from bs4 import BeautifulSoup

BASE_LIST_URL = "https://openreview.net/submissions?venue=ICLR.cc/2026/Conference"
BASE_URL = "https://openreview.net"

CODE_DOMAINS = ("github.com", "gitlab.com", "bitbucket.org", "huggingface.co")

HEADERS = {
    "User-Agent": "iclr2026-scraper/1.1 (+https://openreview.net/)",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

def sleep(delay: float):
    if delay and delay > 0:
        time.sleep(delay)

def req_get(session, url, delay=0.6, max_retries=4, timeout=30):
    """HTTP GET with retry/backoff."""
    backoff = 1.0
    for i in range(max_retries):
        try:
            resp = session.get(url, headers=HEADERS, timeout=timeout)
            if 200 <= resp.status_code < 300:
                sleep(delay)
                return resp
            if resp.status_code in (429, 500, 502, 503, 504):
                time.sleep(backoff)
                backoff *= 1.7
                continue
            resp.raise_for_status()
        except Exception:
            if i == max_retries - 1:
                raise
            time.sleep(backoff)
            backoff *= 1.7
    raise RuntimeError(f"Failed GET after retries: {url}")

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def sanitize_filename(name: str) -> str:
    name = re.sub(r"[\\/:*?\"<>|]", "_", name or "")
    name = re.sub(r"\s+", " ", name).strip()
    return name[:200] if name else "untitled"

def extract_forum_links_from_list_html(html):
    """Parse the submissions list page and return [(title, forum_url), ...]."""
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for a in soup.select("a[href^='/forum?id=']"):
        title_text = a.get_text(strip=True)
        href = a.get("href") or ""
        if title_text and href.startswith("/forum?id="):
            links.append((title_text, urljoin(BASE_URL, href)))
    # dedupe
    seen = set()
    uniq = []
    for t, u in links:
        if u not in seen:
            seen.add(u)
            uniq.append((t, u))
    return uniq

def derive_forum_id_from_url(forum_url):
    qs = parse_qs(urlparse(forum_url).query)
    return qs.get("id", [None])[0]

def find_pdf_link(soup: BeautifulSoup):
    # Common patterns
    # 1) any link to /pdf?id=...
    a = soup.select_one('a[href*="/pdf?id="]')
    if a and a.get("href"):
        href = a["href"]
        return urljoin(BASE_URL, href) if href.startswith("/") else href
    # 2) anchors with "Download PDF"
    for a in soup.find_all("a"):
        text = (a.get_text() or "").strip().lower()
        href = a.get("href") or ""
        if "download pdf" in text or ("/pdf" in href):
            if href.startswith("/"):
                href = urljoin(BASE_URL, href)
            return href
    return None

def find_supplementary_link(soup: BeautifulSoup):
    """
    Try multiple strategies to locate a 'Supplementary Material' link.
    Returns absolute URL or None.
    """
    # 1) Literal label like "Supplementary Material:" with a nearby link
    label = soup.find(string=re.compile(r"^\s*Supplementary Material\s*:", re.I))
    if label:
        parent = getattr(label, "parent", None)
        if parent:
            a = parent.find("a", href=True)
            if a:
                href = a["href"]
                return urljoin(BASE_URL, href) if href.startswith("/") else href

    # 2) Any link where the surrounding context mentions supplementary/supplemental/appendix
    for a in soup.find_all("a", href=True):
        href = a["href"]
        parent = next(a.parents, None)
        ctx_text = parent.get_text(" ", strip=True) if parent else a.get_text(" ", strip=True)
        is_archive = re.search(r"\.(zip|tgz|tar\.gz|rar|7z)$", href, re.I) is not None
        if re.search(r"\bsupp(lementary|lemental)?\b|appendix", ctx_text, re.I) or is_archive:
            return urljoin(BASE_URL, href) if href.startswith("/") else href

    # 3) Attachment endpoints with a name param containing 'supp'
    for a in soup.select('a[href*="/attachment?id="]'):
        href = a.get("href") or ""
        text = a.get_text(" ", strip=True).lower()
        if "supp" in text or "appendix" in text:
            return urljoin(BASE_URL, href) if href.startswith("/") else href
        # look into query string
        if "supp" in href.lower():
            return urljoin(BASE_URL, href) if href.startswith("/") else href

    return None

def find_code_links(soup: BeautifulSoup):
    found = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        netloc = urlparse(href).netloc.lower()
        if not netloc and href.startswith("/"):
            continue
        if any(netloc.endswith(d) for d in CODE_DOMAINS):
            if href not in found:
                found.append(href)
    return found

def scrape_forum(session, forum_url, delay):
    resp = req_get(session, forum_url, delay=delay)
    soup = BeautifulSoup(resp.text, "html.parser")

    # Title: often in h1/h2 near the top
    title = None
    for tag in ("h1", "h2"):
        node = soup.find(tag)
        if node and node.get_text(strip=True):
            title = node.get_text(strip=True)
            break

    # Quick status check — skip withdrawn
    page_text = soup.get_text(" ", strip=True)
    if re.search(r"\bWithdrawn Submission\b", page_text, re.I):
        return None

    pdf_url = find_pdf_link(soup)
    supp_url = find_supplementary_link(soup)
    code_links = find_code_links(soup)

    return {
        "forum_id": derive_forum_id_from_url(forum_url),
        "forum_url": forum_url,
        "title": title or forum_url,
        "status": "Active",
        "pdf_url": pdf_url,
        "supplementary_url": supp_url,
        "code_links": code_links,
    }

def iterate_list_pages(session, start_page=1, max_pages=None, delay=0.6):
    page = start_page
    pages_seen = 0
    while True:
        if max_pages is not None and pages_seen >= max_pages:
            return
        url = BASE_LIST_URL + ("" if page == 1 else f"&page={page}")
        resp = req_get(session, url, delay=delay)
        links = extract_forum_links_from_list_html(resp.text)
        if not links:
            return
        for item in links:
            yield item
        page += 1
        pages_seen += 1

def download_file(session, url, dest_path, delay=0.6, chunk_size=1 << 14):
    ensure_dir(os.path.dirname(dest_path))
    if os.path.exists(dest_path):
        return "exists"
    resp = req_get(session, url, delay=delay, timeout=60)
    cd = resp.headers.get("content-disposition", "")
    # If content-disposition suggests a filename, prefer it
    m = re.search(r'filename\*?=(?:UTF-8\'\')?"?([^";]+)"?', cd)
    if m:
        filename = sanitize_filename(m.group(1))
        dest_path = os.path.join(os.path.dirname(dest_path), filename)
        if os.path.exists(dest_path):
            return "exists"
    with open(dest_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
    return "downloaded"

def safe_subdir_name(title, forum_id):
    base = sanitize_filename(title)
    if forum_id:
        base = f"{base} [{forum_id}]"
    return base

def maybe_clone_repo(repo_url, dest_dir, dry_run=False):
    ensure_dir(dest_dir)
    repo_name = sanitize_filename(repo_url.rstrip("/").split("/")[-1] or "repo")
    repo_path = os.path.join(dest_dir, repo_name)
    if os.path.exists(os.path.join(repo_path, ".git")):
        return "exists"
    if dry_run:
        return "would-clone"
    try:
        subprocess.run(["git", "clone", "--depth", "1", repo_url, repo_path], check=True)
        return "cloned"
    except Exception as e:
        return f"clone-failed: {e}"

# -------------------- CLI stages --------------------

def stage_discover(args):
    """
    Crawl list -> forums, keep entries that have supplementary material,
    write metadata JSON.
    """
    session = requests.Session()
    results = []
    seen = set()
    kept = 0

    for title, forum_url in iterate_list_pages(session, max_pages=args.max_pages, delay=args.delay):
        if forum_url in seen:
            continue
        seen.add(forum_url)

        try:
            meta = scrape_forum(session, forum_url, delay=args.delay)
        except Exception as e:
            print(f"[WARN] Failed to parse forum: {forum_url} ({e})", file=sys.stderr)
            continue

        if not meta:
            continue

        # Keep only if supplementary is present
        if not meta.get("supplementary_url"):
            continue

        results.append(meta)
        kept += 1
        if args.verbose:
            print(f"[KEEP] {meta['title']} ({meta['forum_id']})")

        if args.max_papers and kept >= args.max_papers:
            break

    ensure_dir(os.path.dirname(os.path.abspath(args.meta_file)) or ".")
    with open(args.meta_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"[OK] Wrote metadata for {len(results)} submissions with supplementary material -> {args.meta_file}")

def stage_download(args):
    """
    Read metadata JSON and download PDF + Supplementary for each entry if missing.
    """
    with open(args.meta_file, "r", encoding="utf-8") as f:
        entries = json.load(f)

    if not isinstance(entries, list):
        print("[ERR] Metadata file does not contain a list.", file=sys.stderr)
        sys.exit(1)

    session = requests.Session()
    total = len(entries)
    print(f"[INFO] Loaded {total} metadata entries.")

    for i, meta in enumerate(entries, 1):
        title = meta.get("title") or meta.get("forum_url")
        forum_id = meta.get("forum_id")
        pdf_url = meta.get("pdf_url")
        supp_url = meta.get("supplementary_url")
        code_links = meta.get("code_links") or []

        subdir = os.path.join(args.outdir, safe_subdir_name(title, forum_id))
        pdf_path = os.path.join(subdir, "paper.pdf") if pdf_url else None

        supp_ext = None
        if supp_url:
            path = urlparse(supp_url).path
            m = re.search(r"\.(zip|tgz|tar\.gz|rar|7z)$", path, re.I)
            supp_ext = ("." + m.group(1)) if m else ".supp"
        supp_path = os.path.join(subdir, f"supplementary{supp_ext}") if supp_url else None

        print(f"\n[{i}/{total}] {title}")
        print(f"Forum: {meta.get('forum_url')}")
        print(f"PDF: {pdf_url or '-'}")
        print(f"Supplementary: {supp_url or '-'}")
        if code_links:
            print("Code links:")
            for c in code_links:
                print(f"  - {c}")

        if args.dry_run:
            print(f"[DRY-RUN] Would create: {subdir}")
            if pdf_url:
                print(f"[DRY-RUN] Would download -> {pdf_path}")
            if supp_url:
                print(f"[DRY-RUN] Would download -> {supp_path}")
            if args.clone_code and code_links:
                print(f"[DRY-RUN] Would clone code repos (git) or record links (hf) into {os.path.join(subdir, 'code')}")
            continue

        ensure_dir(subdir)
        # Persist the meta alongside the files (for provenance)
        meta_out = os.path.join(subdir, "meta.json")
        with open(meta_out, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        if pdf_url:
            st = download_file(session, pdf_url, pdf_path, delay=args.delay)
            print(f"[PDF] {st} -> {pdf_path}")

        if supp_url:
            st = download_file(session, supp_url, supp_path, delay=args.delay)
            print(f"[SUPP] {st} -> {supp_path}")

        if args.clone_code and code_links:
            code_dir = os.path.join(subdir, "code")
            ensure_dir(code_dir)
            for c in code_links:
                netloc = urlparse(c).netloc.lower()
                if any(netloc.endswith(d) for d in ("github.com", "gitlab.com", "bitbucket.org")):
                    st = maybe_clone_repo(c, code_dir, dry_run=False)
                    print(f"[CODE] {c} -> {st}")
                else:
                    with open(os.path.join(code_dir, "links.txt"), "a", encoding="utf-8") as lf:
                        lf.write(c + "\n")
                    print(f"[CODE] recorded (non-git): {c}")

def build_parser():
    p = argparse.ArgumentParser(description="ICLR 2026 OpenReview scraper (discover -> download)")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_dis = sub.add_parser("discover", help="Discover submissions with supplementary material and write metadata JSON")
    p_dis.add_argument("--meta-file", default="iclr2026_meta.json", help="Path to write metadata JSON")
    p_dis.add_argument("--max-pages", type=int, default=None, help="Max list pages to scan")
    p_dis.add_argument("--max-papers", type=int, default=None, help="Stop after discovering this many qualifying papers")
    p_dis.add_argument("--delay", type=float, default=0.8, help="Delay (s) between requests")
    p_dis.add_argument("--verbose", action="store_true", help="Print kept titles as they are found")

    p_dl = sub.add_parser("download", help="Download PDFs and supplementary using a metadata JSON file")
    p_dl.add_argument("--meta-file", default="iclr2026_meta.json", help="Metadata JSON to read")
    p_dl.add_argument("--outdir", default="iclr2026_downloads", help="Destination directory")
    p_dl.add_argument("--dry-run", action="store_true", help="List what would be downloaded without creating files")
    p_dl.add_argument("--clone-code", action="store_true", help="Shallow-clone any git repos in code_links")
    p_dl.add_argument("--delay", type=float, default=0.8, help="Delay (s) between requests")

    return p

def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.cmd == "discover":
        stage_discover(args)
    elif args.cmd == "download":
        stage_download(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
