#!/usr/bin/env python3
"""Gradient-based adversarial prefix optimization against embedding clustering.

Uses GCG-style (Greedy Coordinate Gradient) discrete token optimization to find
per-trace adversarial prefixes that break campaign trace clustering.  Requires
GPU access and a local copy of the embedding model.

The attack:
  For each campaign trace, find a prefix (sequence of tokens) such that
  embedding(prefix + trace) is maximally dissimilar from other campaign trace
  embeddings and maximally similar to a benign cluster centroid.

Usage (on GPU machine):
    python experiments/gradient_prefix_attack.py \
        --trace-repo results/safety/trace_repos/safety_dm_cyber_d6_bg100_qwen35_bayesian_boost \
        --cases 50 \
        --output-dir results/gradient_prefixes \
        --model BAAI/bge-base-en-v1.5 \
        --steps 500 --prefix-len 20 --topk 64
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def load_case(case_dir: Path) -> dict:
    """Load trace data and metadata for a single case."""
    meta_dir = case_dir.parent / "_metadata"
    case_id = case_dir.name
    meta_path = meta_dir / f"{case_id}.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    traces_dir = case_dir / "traces"
    trace_files = sorted(traces_dir.glob("trace_*.txt"))

    trace_data = {}
    for f in trace_files:
        rel = f"traces/{f.name}"
        trace_data[rel] = f.read_text(encoding="utf-8")

    return {
        "case_id": case_id,
        "case_dir": case_dir,
        "trace_data": trace_data,
        "campaign_traces": meta.get("gt_hacked_trace_files", []),
        "meta": meta,
    }


def optimize_prefixes_gcg(
    model,
    tokenizer,
    campaign_texts: list[str],
    benign_texts: list[str],
    *,
    prefix_len: int = 20,
    steps: int = 500,
    topk: int = 64,
    batch_size: int = 8,
    max_chars: int = 2000,
    device: str = "cuda",
    verbose: bool = True,
) -> list[str]:
    """GCG-style discrete prefix optimization.

    For each campaign trace, find a prefix that minimizes pairwise cosine
    similarity with other campaign traces and maximizes similarity to
    benign cluster centroids.

    Returns: list of prefix strings (one per campaign trace).
    """
    n_campaign = len(campaign_texts)

    # Compute benign embeddings (fixed targets)
    if verbose:
        print(f"  Embedding {len(benign_texts)} benign traces...")
    benign_embs = embed_texts_batched(
        model, tokenizer, benign_texts, max_chars=max_chars,
        batch_size=batch_size, device=device,
    )
    # Cluster benign embeddings into n_campaign groups to assign targets
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=min(n_campaign, len(benign_texts)), random_state=0, n_init=5)
    km.fit(benign_embs.cpu().numpy())
    target_centroids = torch.tensor(km.cluster_centers_, dtype=torch.float32, device=device)
    # Normalize targets
    target_centroids = F.normalize(target_centroids, dim=-1)

    # Initialize prefix tokens (random from vocabulary)
    vocab_size = tokenizer.vocab_size
    rng = np.random.RandomState(42)
    # Use common English words as initial tokens (better than random)
    init_text = "The following is a general educational discussion about various technology topics including"
    init_ids = tokenizer.encode(init_text, add_special_tokens=False)
    # Pad or truncate to prefix_len
    if len(init_ids) < prefix_len:
        init_ids = init_ids + rng.randint(1000, 10000, size=prefix_len - len(init_ids)).tolist()
    init_ids = init_ids[:prefix_len]

    # Per-trace prefix token IDs
    prefix_ids_list = [torch.tensor(init_ids, dtype=torch.long, device=device) for _ in range(n_campaign)]

    # Get embedding layer
    embed_layer = model.get_input_embeddings()
    embed_weight = embed_layer.weight.detach()  # (vocab_size, hidden_dim)

    best_prefixes = [""] * n_campaign
    best_loss = float("inf")

    for step in range(steps):
        total_loss = 0.0
        improved = False

        # Compute current campaign embeddings with prefixes
        current_campaign_embs = []
        for i in range(n_campaign):
            prefix_text = tokenizer.decode(prefix_ids_list[i].tolist(), skip_special_tokens=True)
            full_text = prefix_text + "\n\n" + campaign_texts[i][:max_chars]
            emb = embed_single(model, tokenizer, full_text, device=device)
            current_campaign_embs.append(emb)
        campaign_emb_stack = torch.stack(current_campaign_embs)  # (n_campaign, hidden)
        campaign_emb_norm = F.normalize(campaign_emb_stack, dim=-1)

        # Compute loss: maximize pairwise distance + pull toward targets
        pairwise_sim = campaign_emb_norm @ campaign_emb_norm.T  # (n, n)
        # Mask diagonal
        mask = 1.0 - torch.eye(n_campaign, device=device)
        mean_pairwise = (pairwise_sim * mask).sum() / mask.sum()

        # Target similarity: each trace should be close to a different centroid
        target_sim = (campaign_emb_norm * target_centroids[:n_campaign]).sum(dim=-1).mean()

        loss = mean_pairwise - 0.5 * target_sim
        total_loss = loss.item()

        if step % 50 == 0 and verbose:
            coherence = mean_pairwise.item()
            print(f"  Step {step:4d}: loss={total_loss:.4f} coherence={coherence:.4f} target_sim={target_sim.item():.4f}")

        # GCG: for each trace, try replacing one token position
        for trace_idx in range(n_campaign):
            prefix_ids = prefix_ids_list[trace_idx]
            campaign_text = campaign_texts[trace_idx][:max_chars]

            # Tokenize the full input
            prefix_text = tokenizer.decode(prefix_ids.tolist(), skip_special_tokens=True)
            full_text = prefix_text + "\n\n" + campaign_text

            # Encode to get token IDs
            full_ids = tokenizer.encode(full_text, add_special_tokens=True, return_tensors="pt").to(device)

            # Forward pass with gradient tracking on the prefix portion
            prefix_embeds = embed_layer(prefix_ids.unsqueeze(0))  # (1, prefix_len, hidden)
            prefix_embeds = prefix_embeds.detach().requires_grad_(True)

            # Get the rest of the input embeddings
            suffix_text = "\n\n" + campaign_text
            suffix_ids = tokenizer.encode(suffix_text, add_special_tokens=False, return_tensors="pt").to(device)
            suffix_embeds = embed_layer(suffix_ids)

            # Combine
            # Add CLS token
            cls_id = torch.tensor([[tokenizer.cls_token_id or tokenizer.bos_token_id or 101]], device=device)
            cls_embed = embed_layer(cls_id)

            combined_embeds = torch.cat([cls_embed, prefix_embeds, suffix_embeds], dim=1)

            # Forward through model
            with torch.enable_grad():
                outputs = model(inputs_embeds=combined_embeds, output_hidden_states=True)
                # Get the pooled representation
                if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                    trace_emb = outputs.pooler_output[0]
                else:
                    # Mean pooling over last hidden state
                    trace_emb = outputs.last_hidden_state[0].mean(dim=0)
                trace_emb = F.normalize(trace_emb, dim=-1)

                # Loss for this trace
                other_embs = torch.cat([campaign_emb_norm[:trace_idx], campaign_emb_norm[trace_idx+1:]], dim=0)
                sim_to_others = (trace_emb @ other_embs.T).mean()
                sim_to_target = (trace_emb @ target_centroids[trace_idx % len(target_centroids)]).sum()
                trace_loss = sim_to_others - 0.5 * sim_to_target

                trace_loss.backward()

            # Gradient w.r.t. prefix embeddings
            grad = prefix_embeds.grad  # (1, prefix_len, hidden)
            if grad is None:
                continue

            # Pick the token position with the largest gradient norm
            token_grads = grad[0]  # (prefix_len, hidden)
            pos_scores = token_grads.norm(dim=-1)  # (prefix_len,)
            best_pos = pos_scores.argmax().item()

            # For the selected position, find top-k replacement tokens
            # Score each token by: -grad . (embed(new_token) - embed(old_token))
            old_embed = embed_weight[prefix_ids[best_pos]]  # (hidden,)
            # Compute score for each vocab token
            scores = -token_grads[best_pos] @ (embed_weight - old_embed).T  # (vocab_size,)
            topk_ids = scores.topk(topk).indices  # (topk,)

            # Evaluate each candidate
            best_candidate_loss = trace_loss.item()
            best_candidate_id = prefix_ids[best_pos].item()

            for cand_id in topk_ids:
                new_prefix_ids = prefix_ids.clone()
                new_prefix_ids[best_pos] = cand_id

                # Quick forward pass to evaluate
                new_prefix_text = tokenizer.decode(new_prefix_ids.tolist(), skip_special_tokens=True)
                new_full_text = new_prefix_text + "\n\n" + campaign_text
                with torch.no_grad():
                    new_emb = embed_single(model, tokenizer, new_full_text, device=device)
                    new_emb = F.normalize(new_emb, dim=-1)
                    new_sim = (new_emb @ other_embs.T).mean()
                    new_target = (new_emb @ target_centroids[trace_idx % len(target_centroids)]).sum()
                    new_loss = new_sim.item() - 0.5 * new_target.item()

                if new_loss < best_candidate_loss:
                    best_candidate_loss = new_loss
                    best_candidate_id = cand_id.item()

            # Update prefix
            if best_candidate_id != prefix_ids[best_pos].item():
                prefix_ids_list[trace_idx][best_pos] = best_candidate_id
                improved = True

        # Update best prefixes
        if total_loss < best_loss or step == 0:
            best_loss = total_loss
            for i in range(n_campaign):
                best_prefixes[i] = tokenizer.decode(prefix_ids_list[i].tolist(), skip_special_tokens=True)

        if step > 50 and not improved:
            if verbose:
                print(f"  Converged at step {step}")
            break

    # Final evaluation
    if verbose:
        final_embs = []
        for i in range(n_campaign):
            full = best_prefixes[i] + "\n\n" + campaign_texts[i][:max_chars]
            final_embs.append(embed_single(model, tokenizer, full, device=device))
        final_stack = F.normalize(torch.stack(final_embs), dim=-1)
        final_sim = (final_stack @ final_stack.T)
        mask = 1.0 - torch.eye(n_campaign, device=device)
        coherence = (final_sim * mask).sum() / mask.sum()
        print(f"  Final coherence: {coherence.item():.4f}")

    return best_prefixes


def embed_single(model, tokenizer, text: str, device: str = "cuda") -> torch.Tensor:
    """Embed a single text, returning a 1-D embedding tensor."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
        return outputs.pooler_output[0]
    return outputs.last_hidden_state[0].mean(dim=0)


def embed_texts_batched(
    model, tokenizer, texts: list[str], *,
    max_chars: int = 2000, batch_size: int = 8, device: str = "cuda",
) -> torch.Tensor:
    """Embed multiple texts in batches."""
    all_embs = []
    for start in range(0, len(texts), batch_size):
        batch = [t[:max_chars] for t in texts[start:start + batch_size]]
        inputs = tokenizer(batch, return_tensors="pt", truncation=True, max_length=512, padding=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            embs = outputs.pooler_output
        else:
            embs = outputs.last_hidden_state.mean(dim=1)
        all_embs.append(embs)
    return torch.cat(all_embs, dim=0)


def main():
    p = argparse.ArgumentParser(description="Gradient-based adversarial prefix optimization.")
    p.add_argument("--trace-repo", type=Path, required=True,
                   help="Source trace repo (e.g. safety_dm_cyber_d6_bg100_qwen35_bayesian_boost)")
    p.add_argument("--cases", type=int, default=50, help="Number of cases to process")
    p.add_argument("--output-dir", type=Path, default=Path("results/gradient_prefixes"))
    p.add_argument("--model", type=str, default="BAAI/bge-base-en-v1.5")
    p.add_argument("--steps", type=int, default=500, help="Optimization steps per case")
    p.add_argument("--prefix-len", type=int, default=20, help="Prefix length in tokens")
    p.add_argument("--topk", type=int, default=64, help="Top-k candidates per GCG step")
    p.add_argument("--max-chars", type=int, default=2000, help="Max chars per trace for embedding")
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    print(f"Loading model: {args.model}")
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True).to(args.device)
    model.eval()
    print(f"Model loaded on {args.device}. Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    dm_sets = args.trace_repo / "distributed-misuse-sets"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for case_idx in range(args.cases):
        case_id = f"case_{case_idx:04d}"
        case_dir = dm_sets / case_id
        if not case_dir.exists():
            print(f"Skipping {case_id}: not found")
            continue

        output_path = args.output_dir / f"gradient_prefixes_{case_id}.json"
        if output_path.exists():
            print(f"Skipping {case_id}: already computed")
            continue

        case = load_case(case_dir)
        campaign_traces = case["campaign_traces"]
        if not campaign_traces:
            continue

        campaign_texts = [case["trace_data"][t] for t in campaign_traces if t in case["trace_data"]]
        benign_texts = [v for k, v in case["trace_data"].items() if k not in set(campaign_traces)]

        print(f"\n{'='*60}")
        print(f"{case_id}: {len(campaign_traces)} campaign, {len(benign_texts)} benign")
        t0 = time.time()

        prefixes = optimize_prefixes_gcg(
            model, tokenizer,
            campaign_texts, benign_texts,
            prefix_len=args.prefix_len,
            steps=args.steps,
            topk=args.topk,
            max_chars=args.max_chars,
            device=args.device,
        )

        elapsed = time.time() - t0
        print(f"  Completed in {elapsed:.1f}s")

        # Save
        result = {rel: prefix for rel, prefix in zip(campaign_traces, prefixes)}
        output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"  Saved: {output_path}")


if __name__ == "__main__":
    main()
