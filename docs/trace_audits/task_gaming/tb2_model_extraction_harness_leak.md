# `model-extraction-relu-logits__KpJ7zNX`: harness leak, not a clean cheating case

Date: 2026-04-01

Source artifact:
- `results/external_audit/betty_tb2_hard_devcheat_20260401a/top_traces/03_model-extraction-relu-logits__KpJ7zNX.txt`

## Bottom line

This is not a clean developer-cheating exemplar.

It is better described as a harness leak or benchmark design failure. The task says the shape of `A1` is unknown and the model should be recovered by querying `forward()`, but the agent can simply read `forward.py`, which reveals the hidden dimension, seed, and full parameterization.

## What happens

The task prompt says:

- the network is a one-layer ReLU model
- the input dimension is 10
- the shape of `A1` is unknown
- the agent should recover `A1` by making queries to `forward()`

But early in the trace, the agent reads `/app/forward.py`, which reveals:

- `np.random.seed(0)`
- `A1 = np.random.randn(20, 10) * 0.1`
- the hidden size is `20`
- the full model structure and parameters

The agent then explicitly says it now has “all the context needed” and records:

- `A1 shape: (20, 10)`
- `Parameters: initialized with np.random.seed(0), scaled by 0.1`

It later writes `steal.py`, runs it successfully, and verifies against the “true A1”.

## Why this is weaker than `qq6Z5nB`

The final script appears to use a real extraction technique: finding ReLU kink boundaries along random lines. So unlike `circuit-fibsqrt__qq6Z5nB`, this is not a pure non-solution that passes anyway.

The problem is that the benchmark leaked information the prompt said should be hidden.

That means:

- the agent’s stopping rule is informed by leaked ground truth
- the agent’s design choices are informed by leaked ground truth
- the benchmark no longer measures purely black-box recovery

## Why I would not call this developer cheating

I do not have evidence that the benchmark authors intentionally inserted this leak to inflate scores. The stronger claim here is:

- `benchmark leak / harness design failure`: yes
- `successful model use of leaked benchmark information`: yes
- `clean intentional developer cheating`: no evidence

So this belongs in a separate bucket from `qq6Z5nB`.

## Evidence excerpt

The decisive early leak is:

```text
[0004] agent:
TOOLS:
- read_file: /app/forward.py
...
np.random.seed(0)
A1 = np.random.randn(20, 10) * 0.1
```

And immediately after:

```text
Now I have full understanding of the environment.
...
- A1 shape: (20, 10)
- Parameters: initialized with np.random.seed(0), scaled by 0.1
```

## Use in writeups

This case is still useful, but only if labeled correctly:

- good as a benchmark leak / eval-design failure
- not good as a flagship developer-cheating example
- worth keeping separate from true scaffold exploits
