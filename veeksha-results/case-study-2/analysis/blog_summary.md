# Workload Shape Case Study (0.18 sessions/s)

Selected shared session arrival rate: `0.18` sessions/s.

## Matched fresh-token budget

- Linear total new input tokens: `75000`
- DAG total new input tokens: `75000`
- Linear total output tokens: `45000`
- DAG total output tokens: `45000`
- DAG / linear effective-input ratio from the trace design: `1.203175`
- DAG / linear cacheable-history ratio from the trace design: `1.266667`

## Main finding

The workloads present the same fresh-token budget, but the DAG run forces the system to operate at longer effective context lengths and different cache behavior. That changes the reported performance even though user-visible work is matched.

## Observed comparison

- Mean fresh input tokens per request: linear `500.0`, DAG `500.0`
- Mean total prompt tokens per request: linear `2094.3`, DAG `2517.5`
- P95 total prompt tokens: linear `3700.0`, DAG `3699.0`
- TTFC p99: linear `0.205s`, DAG `0.444s`
- E2E p95: linear `3.054s`, DAG `5.410s`
- Decode-window TBC p99: linear `11.1 ms`, DAG `52.4 ms`
- vLLM prefix-cache hit rate: linear `0.495`, DAG `0.690`
- vLLM prompt-cache token ratio: linear `1.000`, DAG `1.000`
- vLLM KV-cache usage: linear `0.000`, DAG `0.000`

## Interpretation

- DAG mean total prompt length is `20.2%` higher than linear.
- DAG P95 total prompt length is `-0.0%` higher than linear.
- TTFC p99 shifts by `116.7%` between shapes at the same selected rate.
- E2E p95 shifts by `77.2%` between shapes at the same selected rate.

## If we optimized for A but traffic is really B

Assume we sized the deployment, batching policy, or autoscaling thresholds from
workload A. That would calibrate the system to a mean prompt length of roughly
`2094` tokens/request, TTFC p99 of `0.205s`, E2E p95 of `3.054s`, and decode
windows that peak around `6` active requests. If production traffic is actually
closer to workload B, that tuning leaves money on the table for three distinct
reasons:

- The DAG workload keeps more of the traffic in the long-context regime. Its
  P95 prompt length is not higher, but its mean prompt length is `20.2%` higher
  and its mean cacheable history is `26.5%` higher. So this is not just a rare
  tail event: a larger share of requests routinely decode against longer
  effective contexts.
- Fan-out and fan-in reshape concurrency. At the same external arrival rate of
  `0.18` sessions/s, the decode-window analysis sees `41` active requests in the
  DAG run versus `6` in the linear run. A fleet tuned on linear traces will
  therefore under-estimate queueing and decode contention once branches become
  ready together.
- The higher prefix-cache hit rate in the DAG run does not mean the workload is
  cheaper. Prefix-cache hit rate improves from `0.495` to `0.690`, and the
  prompt-cache token ratio is `1.0` in both runs, so caching is working. The
  loss comes from the topology creating more long-context decode work and more
  bursty overlap, not from the cache failing.

Operationally, this means the same deployment would deliver substantially worse
latency on DAG-shaped traffic: TTFC p99 more than doubles (`0.205s` to
`0.444s`), E2E p95 rises by `77.2%` (`3.054s` to `5.410s`), and decode-window
TBC p99 degrades by about `4.7x` (`11.1 ms` to `52.4 ms`). If we instead insist
on holding the latency target fixed, the throughput estimates imply that we
would need about `1.64x` more steady-state decode capacity by TPOT
(`124.3` to `75.9` tokens/s) and up to `2.81x` more burst decode capacity by
the decode-window throughput estimate (`39.5` to `14.1` tokens/s). That is the
money-loss mechanism: optimizing on workload A systematically under-prices the
compute footprint of workload B even though the fresh-token budget is identical.

## Artifacts

- `latency_comparison.png`
- `prompt_context_comparison.png`
- `cache_ratio_comparison.png`
- `fresh_prompt_ecdf.png`
- `total_prompt_ecdf.png`
