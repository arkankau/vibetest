"""Analysis tools for large text corpora inside the sandbox."""

from __future__ import annotations

import json
import uuid
from textwrap import dedent
from typing import Annotated

from inspect_ai.tool import tool
from inspect_ai.util import sandbox

_ALLOWED_SCANNER_MODELS = {"gpt-5-mini"}

_EMBED_TEXTS_SCRIPT = dedent(
    r"""
    import glob
    import json
    import os
    import sys
    from pathlib import Path

    from openai import OpenAI


    def main() -> int:
        with open(sys.argv[1], "r", encoding="utf-8") as f:
            args = json.load(f)

        input_glob = args["input_glob"]
        output_path = args["output_path"]
        model = args["model"]
        batch_size = int(args["batch_size"])
        max_chars = int(args["max_chars"])

        paths = sorted(
            p for p in glob.glob(input_glob, recursive=True) if os.path.isfile(p)
        )
        if not paths:
            raise SystemExit(f"No files matched glob: {input_glob}")

        client = OpenAI()
        items = []
        texts = []
        for path in paths:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            clipped = text[:max_chars]
            texts.append(clipped)
            items.append(
                {
                    "id": os.path.basename(path),
                    "path": path,
                    "num_chars": len(text),
                    "num_chars_embedded": len(clipped),
                    "text_preview": clipped[:240],
                }
            )

        embeddings = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            response = client.embeddings.create(model=model, input=batch)
            embeddings.extend([row.embedding for row in response.data])

        for item, embedding in zip(items, embeddings, strict=True):
            item["embedding"] = embedding

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "kind": "text_embeddings",
            "model": model,
            "input_glob": input_glob,
            "count": len(items),
            "items": items,
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f)

        print(
            json.dumps(
                {
                    "output_path": output_path,
                    "count": len(items),
                    "model": model,
                }
            )
        )
        return 0


    if __name__ == "__main__":
        raise SystemExit(main())
    """
)


_CLUSTER_EMBEDDINGS_SCRIPT = dedent(
    r"""
    import json
    import math
    import os
    import sys
    from pathlib import Path

    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score


    def choose_k(x: np.ndarray, min_clusters: int, max_clusters: int) -> int:
        n_items = len(x)
        if n_items <= 2:
            return 1
        max_k = min(max_clusters, n_items - 1)
        min_k = min(min_clusters, max_k)
        if max_k <= 1:
            return 1
        if min_k == max_k:
            return min_k

        best_k = min_k
        best_score = -1.0
        for k in range(min_k, max_k + 1):
            model = KMeans(n_clusters=k, random_state=0, n_init=10)
            labels = model.fit_predict(x)
            if len(set(labels)) < 2:
                continue
            score = silhouette_score(x, labels)
            if score > best_score:
                best_score = score
                best_k = k
        return best_k


    def main() -> int:
        with open(sys.argv[1], "r", encoding="utf-8") as f:
            args = json.load(f)

        embeddings_path = args["embeddings_path"]
        output_path = args["output_path"]
        n_clusters = int(args["n_clusters"])
        min_clusters = int(args["min_clusters"])
        max_clusters = int(args["max_clusters"])
        representative_count = int(args["representative_count"])

        with open(embeddings_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        items = payload["items"]
        if not items:
            raise SystemExit("No items found in embeddings payload.")

        x = np.array([item["embedding"] for item in items], dtype=float)
        if len(items) == 1:
            labels = np.array([0], dtype=int)
            centroids = x
            chosen_k = 1
        else:
            chosen_k = n_clusters if n_clusters > 0 else choose_k(x, min_clusters, max_clusters)
            chosen_k = max(1, min(chosen_k, len(items)))
            model = KMeans(n_clusters=chosen_k, random_state=0, n_init=10)
            labels = model.fit_predict(x)
            centroids = model.cluster_centers_

        cluster_rows = []
        for idx, item in enumerate(items):
            item["cluster_id"] = int(labels[idx])

        for cluster_id in sorted(set(int(v) for v in labels.tolist())):
            member_ix = np.where(labels == cluster_id)[0]
            centroid = centroids[cluster_id]
            distances = [
                (float(np.linalg.norm(x[i] - centroid)), int(i))
                for i in member_ix.tolist()
            ]
            distances.sort(key=lambda row: row[0])
            reps = []
            for _, i in distances[:representative_count]:
                reps.append(
                    {
                        "id": items[i]["id"],
                        "path": items[i]["path"],
                        "text_preview": items[i].get("text_preview", ""),
                    }
                )
            cluster_rows.append(
                {
                    "cluster_id": int(cluster_id),
                    "size": int(len(member_ix)),
                    "representatives": reps,
                }
            )

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        out = {
            "kind": "text_clusters",
            "embeddings_path": embeddings_path,
            "n_clusters": int(chosen_k),
            "count": len(items),
            "clusters": cluster_rows,
            "items": [
                {
                    "id": item["id"],
                    "path": item["path"],
                    "cluster_id": item["cluster_id"],
                    "text_preview": item.get("text_preview", ""),
                }
                for item in items
            ],
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(out, f)

        print(
            json.dumps(
                {
                    "output_path": output_path,
                    "count": len(items),
                    "n_clusters": int(chosen_k),
                }
            )
        )
        return 0


    if __name__ == "__main__":
        raise SystemExit(main())
    """
)


_PARALLEL_LLM_SCANNER_SCRIPT = dedent(
    r"""
    import glob
    import json
    import os
    import sys
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from pathlib import Path

    from openai import OpenAI


    def main() -> int:
        with open(sys.argv[1], "r", encoding="utf-8") as f:
            args = json.load(f)

        input_glob = args["input_glob"]
        output_path = args["output_path"]
        prompt_template = args["prompt_template"]
        model = args["model"]
        max_workers = int(args["max_workers"])
        max_chars = int(args["max_chars"])
        property_text = args.get("property_text", "")

        paths = sorted(
            p for p in glob.glob(input_glob, recursive=True) if os.path.isfile(p)
        )
        if not paths:
            raise SystemExit(f"No files matched glob: {input_glob}")

        traces = []
        for path in paths:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                transcript = f.read()[:max_chars]
            traces.append((os.path.basename(path), path, transcript))

        client = OpenAI()

        def scan_one(trace_id: str, path: str, transcript: str) -> dict:
            prompt = (
                prompt_template.replace("{property_text}", property_text)
                .replace("{trace_id}", trace_id)
                .replace("{path}", path)
                .replace("{transcript}", transcript)
            )
            response = client.responses.create(model=model, input=prompt)
            output_text = response.output_text.strip()
            row = {
                "trace_id": trace_id,
                "path": path,
                "raw_output": output_text,
            }
            try:
                parsed = json.loads(output_text)
                if isinstance(parsed, dict):
                    row.update(parsed)
                else:
                    row["parsed_output"] = parsed
            except Exception:
                row["parse_error"] = "output_was_not_json"
            return row

        rows = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(scan_one, trace_id, path, transcript)
                for trace_id, path, transcript in traces
            ]
            for future in as_completed(futures):
                rows.append(future.result())

        rows.sort(key=lambda row: (row.get("path", ""), row.get("trace_id", "")))
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

        print(
            json.dumps(
                {
                    "output_path": output_path,
                    "count": len(rows),
                    "model": model,
                    "max_workers": max_workers,
                }
            )
        )
        return 0


    if __name__ == "__main__":
        raise SystemExit(main())
    """
)


async def _run_sandbox_python(
    script: str,
    args: dict,
    *,
    timeout: int = 1800,
) -> str:
    env = sandbox()
    token = uuid.uuid4().hex
    script_path = f"/tmp/vibetest_tool_{token}.py"
    args_path = f"/tmp/vibetest_tool_{token}.json"

    await env.write_file(script_path, script)
    await env.write_file(args_path, json.dumps(args))
    result = await env.exec(
        ["python", script_path, args_path],
        cwd="/workspace/repo",
        timeout=timeout,
    )
    try:
        await env.exec(["rm", "-f", script_path, args_path], timeout=30)
    except Exception:
        pass

    if not result.success:
        stderr = (result.stderr or "").strip()
        stdout = (result.stdout or "").strip()
        details = stderr or stdout or f"returncode={result.returncode}"
        raise RuntimeError(details)

    return (result.stdout or "").strip()


@tool
def embed_text_corpus():
    async def execute(
        input_glob: Annotated[str, "Glob for text files inside the sandbox, e.g. /workspace/repo/traces/*.txt"],
        output_path: Annotated[str, "Output JSON path inside the sandbox, e.g. /workspace/repo/analysis/embeddings.json"],
        model: Annotated[str, "Embedding model name"] = "text-embedding-3-small",
        batch_size: Annotated[int, "Batch size for embeddings API"] = 64,
        max_chars: Annotated[int, "Maximum characters to embed per file"] = 40000,
    ) -> str:
        """Embed a corpus of text files and save embeddings inside the sandbox.

        Args:
            input_glob: Glob for text files inside the sandbox.
            output_path: Output JSON path inside the sandbox.
            model: Embedding model name.
            batch_size: Batch size for the embeddings API.
            max_chars: Maximum characters to embed per file.

        Returns:
            Summary of the embedding run and output path.
        """
        stdout = await _run_sandbox_python(
            _EMBED_TEXTS_SCRIPT,
            {
                "input_glob": input_glob,
                "output_path": output_path,
                "model": model,
                "batch_size": batch_size,
                "max_chars": max_chars,
            },
        )
        meta = json.loads(stdout)
        return (
            f"Embedded {meta['count']} files with {meta['model']} and wrote "
            f"{meta['output_path']}"
        )

    return execute


@tool
def cluster_text_embeddings():
    async def execute(
        embeddings_path: Annotated[str, "Path to embeddings JSON produced by embed_text_corpus"],
        output_path: Annotated[str, "Output JSON path for cluster assignments and representatives"],
        n_clusters: Annotated[int, "Number of clusters. Use 0 to choose automatically."] = 0,
        min_clusters: Annotated[int, "Minimum number of clusters when auto-selecting"] = 2,
        max_clusters: Annotated[int, "Maximum number of clusters when auto-selecting"] = 12,
        representative_count: Annotated[int, "Number of representative samples per cluster"] = 3,
    ) -> str:
        """Cluster previously computed text embeddings and save assignments.

        Args:
            embeddings_path: Path to embeddings JSON produced by embed_text_corpus.
            output_path: Output JSON path for cluster assignments and representatives.
            n_clusters: Number of clusters, or 0 to auto-select.
            min_clusters: Minimum number of clusters when auto-selecting.
            max_clusters: Maximum number of clusters when auto-selecting.
            representative_count: Number of representative samples per cluster.

        Returns:
            Summary of the clustering run and output path.
        """
        stdout = await _run_sandbox_python(
            _CLUSTER_EMBEDDINGS_SCRIPT,
            {
                "embeddings_path": embeddings_path,
                "output_path": output_path,
                "n_clusters": n_clusters,
                "min_clusters": min_clusters,
                "max_clusters": max_clusters,
                "representative_count": representative_count,
            },
        )
        meta = json.loads(stdout)
        return (
            f"Clustered {meta['count']} items into {meta['n_clusters']} clusters and wrote "
            f"{meta['output_path']}"
        )

    return execute


@tool
def query_text_clusters():
    async def execute(
        cluster_path: Annotated[str, "Path to cluster JSON produced by cluster_text_embeddings"],
        cluster_ids: Annotated[str, "Comma-separated cluster ids to inspect; leave empty for all"] = "",
        representative_limit: Annotated[int, "Maximum representatives to show per cluster"] = 3,
        include_assignments: Annotated[bool, "Whether to include per-item assignments"] = False,
    ) -> str:
        """Inspect cluster assignments and representative samples from a saved cluster file.

        Args:
            cluster_path: Path to cluster JSON produced by cluster_text_embeddings.
            cluster_ids: Comma-separated cluster ids to inspect; empty means all clusters.
            representative_limit: Maximum representatives to show per cluster.
            include_assignments: Whether to include per-item assignments.

        Returns:
            Human-readable cluster summary.
        """
        env = sandbox()
        raw = await env.read_file(cluster_path)
        payload = json.loads(raw)
        wanted = None
        if cluster_ids.strip():
            wanted = {
                int(part.strip())
                for part in cluster_ids.split(",")
                if part.strip()
            }

        lines: list[str] = [
            f"Clusters: {payload.get('n_clusters', 0)}",
            f"Items: {payload.get('count', 0)}",
        ]
        for cluster in payload.get("clusters", []):
            cluster_id = int(cluster["cluster_id"])
            if wanted is not None and cluster_id not in wanted:
                continue
            lines.append(f"\nCluster {cluster_id} (size={cluster['size']}):")
            for rep in cluster.get("representatives", [])[:representative_limit]:
                lines.append(f"- {rep['path']}")
                preview = (rep.get("text_preview") or "").strip()
                if preview:
                    lines.append(f"  preview: {preview[:160]}")

        if include_assignments:
            lines.append("\nAssignments:")
            for item in payload.get("items", []):
                cluster_id = int(item["cluster_id"])
                if wanted is not None and cluster_id not in wanted:
                    continue
                lines.append(f"- cluster={cluster_id} path={item['path']}")

        return "\n".join(lines)

    return execute


@tool
def run_parallel_llm_scanner():
    async def execute(
        input_glob: Annotated[str, "Glob for text files inside the sandbox, e.g. /workspace/repo/traces/*.txt"],
        prompt_template: Annotated[str, "Prompt template with placeholders {property_text}, {trace_id}, {path}, and {transcript}"],
        output_path: Annotated[str, "Output JSONL path inside the sandbox"],
        property_text: Annotated[str, "Target property text inserted into the prompt template"] = "",
        model: Annotated[str, "Model for the scanner. Only gpt-5-mini is allowed."] = "gpt-5-mini",
        max_workers: Annotated[int, "Maximum concurrent per-trace scanner requests"] = 32,
        max_chars: Annotated[int, "Maximum characters to send per trace"] = 40000,
    ) -> str:
        """Run a parallel per-trace LLM scanner and save JSONL rows inside the sandbox.

        Args:
            input_glob: Glob for text files inside the sandbox.
            prompt_template: Prompt template with placeholders for property, path, trace id, and transcript.
            output_path: Output JSONL path inside the sandbox.
            property_text: Target property text inserted into the prompt template.
            model: Model for the scanner.
            max_workers: Maximum concurrent per-trace scanner requests.
            max_chars: Maximum characters to send per trace.

        Returns:
            Summary of the scanner run and output path.
        """
        if model not in _ALLOWED_SCANNER_MODELS:
            allowed = ", ".join(sorted(_ALLOWED_SCANNER_MODELS))
            raise ValueError(f"Unsupported scanner model: {model}. Allowed models: {allowed}")
        stdout = await _run_sandbox_python(
            _PARALLEL_LLM_SCANNER_SCRIPT,
            {
                "input_glob": input_glob,
                "prompt_template": prompt_template,
                "output_path": output_path,
                "property_text": property_text,
                "model": model,
                "max_workers": max_workers,
                "max_chars": max_chars,
            },
            timeout=3600,
        )
        meta = json.loads(stdout)
        return (
            f"Scanned {meta['count']} files with {meta['model']} in parallel "
            f"(max_workers={meta['max_workers']}) and wrote {meta['output_path']}"
        )

    return execute
