# Source Resolution

The imported M1 documents are immutable snapshots and retain their original
relative paths. This resolver records stable public locations and the exact
local PDFs used during the research audit without modifying those snapshots.

| Local source | Public source | SHA-256 |
| --- | --- | --- |
| `../../2510.18586v2.pdf` | <https://arxiv.org/abs/2510.18586> | `2a1f88f896afb82a2f84d4e2c19f140d62a862157fd51ab5148ef0acc0eead11` |
| `../../2511.02230v6.pdf` | <https://arxiv.org/abs/2511.02230> | `ec60c3a7f4edb94c2a527da704cf901f0f44fd004a91c5a523b9c006ae11f2fc` |
| `../../2605.06472v1.pdf` | <https://arxiv.org/abs/2605.06472> | `eb2cc2a7750f972a228519376ad27214cc4d674686c2e7209207c8d47b0da301` |

## First-Order Papers

All three local files are arXiv preprints in the audited versions. A template
conference header is not treated as publication evidence.

Citation-ready records:

- Zhuohang Bian, Feiyang Wu, Teng Ma, and Youwei Zhuo. *Tokencake: A
  KV-Cache-centric Serving Framework for LLM-based Multi-Agent Applications*.
  arXiv:2510.18586 [cs.DC], version 2, October 31, 2025.
  <https://doi.org/10.48550/arXiv.2510.18586>.
- Hanchen Li, Runyuan He, Qiuyang Mang, Qizheng Zhang, Huanzhi Mao, Xiaokun
  Chen, Hangrui Zhou, Alvin Cheung, Joseph Gonzalez, and Ion Stoica.
  *Continuum: Efficient and Robust Multi-Turn LLM Agent Scheduling with KV
  Cache Time-to-Live*. arXiv:2511.02230 [cs.OS], first posted in 2025; version
  6 revised May 25, 2026. <https://doi.org/10.48550/arXiv.2511.02230>.
- Haoyu Zheng, Fangcheng Fu, Jia Wu, Binhang Yuan, Yongqiang Zhang, Hao Wang,
  Yuanyuan Zhu, Xiao Yan, and Jiawei Jiang. *Efficient Serving for Dynamic
  Agent Workflows with Prediction-based KV-Cache Management*.
  arXiv:2605.06472 [cs.LG], version 1, May 7, 2026.
  <https://doi.org/10.48550/arXiv.2605.06472>.

| Work | Mechanism already covered | Boundary imposed on DAGKV |
| --- | --- | --- |
| Bian et al., *Tokencake: A KV-Cache-centric Serving Framework for LLM-based Multi-Agent Applications*, arXiv:2510.18586v2, 2025 | Explicit multi-agent DAG; critical-agent GPU reservation; function-call-triggered proactive D2H; developer-hint plus EWMA prediction; predictive H2D; CPU block buffering; gradual GPU reservation. | DAG-aware offload, critical-path protection, predictive upload, CPU block pools, and gradual reservation cannot be primary novelty claims. Tokencake does not establish DAGKV's canonical shared-owner lifetime, generation/ABA safety, transactional failure cleanup, or independently replayable conservation contract. |
| Li et al., *Continuum: Efficient and Robust Multi-Turn LLM Agent Scheduling with KV Cache Time-to-Live*, arXiv:2511.02230v6, 2025 (revised 2026) | Per-tool duration distributions; a cost-derived GPU KV TTL using reload/prefill and accumulated queueing delay; program-level FCFS; expiry and forced-victim handling for sequential ReAct turns. | TTL retention, history-based tool-return prediction, queue-aware pinning, and bounded expiry cannot be primary novelty claims. Continuum names speculative branches and asynchronous multi-agent coordination as future extensions; DAGKV must demonstrate dependency-correct lease aggregation over fork/join consumers, explicit shared ownership, and failure invariants. |
| Zheng et al., *Efficient Serving for Dynamic Agent Workflows with Prediction-based KV-Cache Management* (PBKV), arXiv:2605.06472v1, 2026 | Dynamic call graph; multi-step prediction from history and request semantics; cross-workflow value aggregation on shared radix nodes; lifecycle-first hierarchical eviction; conservative prefetch under idle GPU space and PCIe bandwidth. | Broad prediction-guided DAG eviction/prefetch and independent future-access aggregation are occupied. Any DAGKV policy claim must distinguish dependence between branches/owners, derive its resource constraints, or add a materially different transfer protocol with direct evidence. |

Reported performance figures remain paper claims until reproduced in DAGKV's
controlled environment. Tokencake reports more than `47.06%` end-to-end
latency reduction and up to `16.9%` effective GPU-memory-utilization gain over
vLLM. Continuum reports `1.12-3.66x` lower trace-replay delay,
`1.10-3.22x` higher throughput, and up to `8.18x` lower delay in its real
SWE-agent testbed. PBKV reports up to `1.85x` speedup over LRU on dynamic
workflows and up to `1.26x` over KVFlow on a static workflow.

The detailed mechanism comparison and exact C1-C3 claim boundaries live in
[`imported/RELATED_WORK_MATRIX.md`](imported/RELATED_WORK_MATRIX.md). That
matrix is the novelty authority; this file resolves the exact local sources.

## Mainline Guardrails

1. M2 remains a lifecycle-correctness stage: canonical identity, shared-owner
   isolation, TTL enforcement, transfer integrity, failure cleanup, independent
   ledger replay, and real vLLM token/logit equivalence. It makes no policy
   performance claim.
2. M3 may evaluate only the narrowed hypotheses already recorded in the matrix:
   dependency-aware probabilistic lease aggregation, a controller derived from
   explicit capacity/bandwidth constraints, and deadline-aware partial-prefix
   single-flight. Each mechanism needs an independent switch and a matched
   baseline from the three papers above.
3. Generic DAG awareness, proactive offload, predictive upload, TTL retention,
   future-access scoring, shared-prefix value aggregation, hierarchical
   eviction, and conservative prefetch are prior art for this project.
4. A B-tier systems-paper target requires a smaller falsifiable claim with real
   GPU evidence, adversarial failure tests, paired baselines, and a frozen
   artifact. Combining known signals under a new score is insufficient.
