#!/usr/bin/env python3
import csv, io, json, os, random, re, shutil, subprocess, time, pathlib, ssl, urllib.request

# -------- CONFIG --------
COMP_SLUG = os.getenv("KAGGLE_COMP", "titanic").strip()
TARGET_N  = int(os.getenv("TARGET_N", "100"))
OUT_DIR   = pathlib.Path(os.getenv("OUT_DIR", f"kaggle-{COMP_SLUG}")).resolve()

# candidate pool knobs
PAGE_SIZE      = int(os.getenv("KAGGLE_PAGE_SIZE", "100"))
PAGES_BY_COMP  = int(os.getenv("KAGGLE_PAGES_COMP", "10"))   # plain competition listing
PAGES_BY_QUERY = int(os.getenv("KAGGLE_PAGES_QUERY", "10"))  # per search keyword
SLEEP_BETWEEN  = float(os.getenv("SLEEP_BETWEEN", "0.25"))

SEARCH_KEYWORDS = ["torch", "pytorch", "jax", "flax"]

# score scraping
SCORE_PATTERNS = [
    r"(?:^|[^A-Za-z])LB\s*[:\-]?\s*(\d\.\d{3,6})",
    r"(?:leaderboard|public\s*LB)\s*[:\-]?\s*(\d\.\d{3,6})",
    r"(\d{2,3}\.\d)\s*%",
]
SCORE_RE = re.compile("|".join(f"(?:{p})" for p in SCORE_PATTERNS), re.IGNORECASE)

def run(cmd):
    return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)

def list_by_competition():
    refs = []
    for page in range(1, PAGES_BY_COMP+1):
        out = run([
            "kaggle","kernels","list",
            "--competition", COMP_SLUG,
            "--page", str(page),
            "--page-size", str(PAGE_SIZE),
            "--sort-by","hotness",
            "--csv",
        ])
        rdr = csv.DictReader(io.StringIO(out))
        for row in rdr:
            ref = (row.get("ref") or "").strip()
            if ref and ref.count("/") == 1:
                refs.append(ref)
        time.sleep(SLEEP_BETWEEN)
    return refs

def list_by_search():
    refs = []
    for kw in SEARCH_KEYWORDS:
        for page in range(1, PAGES_BY_QUERY+1):
            out = run([
                "kaggle","kernels","list",
                "--competition", COMP_SLUG,
                "-s", kw,                 # keyword search
                "--page", str(page),
                "--page-size", str(PAGE_SIZE),
                "--sort-by","voteCount",  # “hotness” or “dateCreated” are fine too
                "--csv",
            ])
            rdr = csv.DictReader(io.StringIO(out))
            for row in rdr:
                ref = (row.get("ref") or "").strip()
                if ref and ref.count("/") == 1:
                    refs.append(ref)
            time.sleep(SLEEP_BETWEEN)
    return refs

def unique_shuffle(refs):
    seen, out = set(), []
    for r in refs:
        if r not in seen:
            seen.add(r); out.append(r)
    random.shuffle(out)
    return out

def pull_kernel(ref, dest):
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(["kaggle","kernels","pull", ref, "-p", str(dest), "-m"],
                          stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

def read(path: pathlib.Path, limit=800_000):
    try:
        if path.stat().st_size > limit: return ""
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""

def nb_sources(nb: pathlib.Path, max_cells=200):
    try:
        obj = json.loads(nb.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return ""
    cells = obj.get("cells") or []
    parts = []
    for c in cells[:max_cells]:
        if c.get("cell_type") == "code":
            src = c.get("source", "")
            parts.append("".join(src) if isinstance(src, list) else str(src))
    return "\n".join(parts)

def detect_framework(dirpath: pathlib.Path):
    blobs = []
    for p in dirpath.rglob("*.py"): blobs.append(read(p))
    for p in dirpath.rglob("*.ipynb"): blobs.append(nb_sources(p))
    blob = "\n".join(blobs[:2000])
    if not blob: return None
    if "import torch" in blob or "from torch" in blob: return "pytorch"
    if "import jax" in blob or "from jax" in blob or "from flax" in blob or "import flax" in blob:   return "jax"
    return None

def in_competition(dirpath: pathlib.Path) -> bool:
    meta = dirpath / "kernel-metadata.json"
    if not meta.exists(): return False
    try:
        obj = json.loads(meta.read_text(encoding="utf-8", errors="ignore"))
        comps = [s.strip().lower() for s in (obj.get("competition_sources") or [])]
        return COMP_SLUG.lower() in comps
    except Exception:
        return False

def http_get(url, timeout=20, max_bytes=300_000):
    try:
        ctx = ssl.create_default_context()
        with urllib.request.urlopen(url, timeout=timeout, context=ctx) as r:
            return r.read(max_bytes).decode("utf-8", errors="ignore")
    except Exception:
        return ""

def parse_score(text):
    if not text: return None, None
    m = SCORE_RE.search(text)
    if not m: return None, None
    for g in m.groups():
        if g:
            try:
                v = float(g)
                if "%" in text and re.fullmatch(r"\d{2,3}\.\d", g):
                    return round(v/100.0, 6), "percent"
                return round(v, 6), "decimal"
            except: pass
    return None, None

def scrape_score(user, slug):
    url = f"https://www.kaggle.com/code/{user}/{slug}"
    html = http_get(url)
    if not html: return None, None
    # title first
    t = re.search(r"<title>(.*?)</title>", html, re.I | re.S)
    if t:
        sc, src = parse_score(t.group(1))
        if sc is not None: return sc, "title_score"
    # body snippet
    sc, _ = parse_score(html[:25000])
    if sc is not None: return sc, "body_score"
    return None, None

def main():
    base = OUT_DIR / "notebooks"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Build a large candidate list: comp listing + keyword search
    pool = unique_shuffle(list_by_search())

    kept, manifest = [], []

    for ref in pool:
        if len(kept) >= TARGET_N: break
        user, slug = ref.split("/")
        nb_dir = base / user / slug

        pulled_now = False
        if not nb_dir.exists() or (not any(nb_dir.glob("*.ipynb")) and not any(nb_dir.glob("*.py"))):
            try:
                pull_kernel(ref, nb_dir)
                pulled_now = True
            except subprocess.CalledProcessError:
                shutil.rmtree(nb_dir, ignore_errors=True)
                shutil.rmtree(base / user, ignore_errors=True)
                continue

        # Must belong to the target competition (from metadata)
        if not in_competition(nb_dir):
            if pulled_now:
                shutil.rmtree(nb_dir, ignore_errors=True)
                shutil.rmtree(base / user, ignore_errors=True)
            continue

        # Must use PyTorch or JAX
        fw = detect_framework(nb_dir)
        if fw not in ("pytorch","jax"):
            if pulled_now:
                shutil.rmtree(nb_dir, ignore_errors=True)
                shutil.rmtree(base / user, ignore_errors=True)
            continue

        # Scrape score
        score, score_src = scrape_score(user, slug)

        kept.append(ref)
        manifest.append({
            "competition": COMP_SLUG,
            "kernel_ref": ref,
            "kernel_url": f"https://www.kaggle.com/code/{user}/{slug}",
            "author": user,
            "local_path": str(nb_dir),
            "framework": fw,
            "detected_score": score,
            "score_source": score_src
        })

        time.sleep(SLEEP_BETWEEN)

    # write manifest
    out_path = OUT_DIR / "dataset.pytorch_or_jax.scored.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for row in manifest[:TARGET_N]:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    if len(kept) < TARGET_N:
        print(f"[WARN] Collected {len(kept)} (< {TARGET_N}). "
              f"Try raising PAGES or adding keywords to SEARCH_KEYWORDS.")
    else:
        print(f"Kept exactly {TARGET_N} notebooks -> {out_path}")

if __name__ == "__main__":
    main()

