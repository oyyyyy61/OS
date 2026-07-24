# M0 Reproduction Manifest

## Material Passport

| Field | Frozen value |
|---|---|
| Artifact ID | `offload-m0-repro-20260723` |
| Stage | M0 research and reproduction freeze |
| Freeze date | 2026-07-23 |
| Observation window | 2026-07-23 12:55-13:49 CST (`UTC+08:00`) |
| Verification status | `ANALYZED`; M0 payloads are recoverable, scoped v2 live GPU traces are audited, and raw-physical-line v2 control ABBA execution is accepted for measurement validity |
| Experiment class | Environment-sensitive, stochastic GPU serving benchmark |
| Primary project | `/home/data/25_oyzx/cagent-work/offload` |
| Historical evidence root | `offload/experiments/results/reproduction_headroom_full_20260723` |

This manifest freezes the state that was available on 2026-07-23. A hash marked
`tree-manifest` is computed by sorting relative file names, hashing each file
with SHA-256, then hashing that ordered manifest. It identifies local content;
it does not itself provide a recoverable copy of uncommitted files. The later
`experiments/source_freezes/m0_20260723` bundle supplies the recoverable M0
source, patch, untracked-file, and dependency payloads described below.

## Repository Provenance

### Main checkout

| Field | Frozen value |
|---|---|
| Repository | `git@github.com:oyyyyy61/Cagent.git` |
| Checkout | `/home/data/25_oyzx/cagent-work` |
| Branch | `kv-offloading` |
| HEAD | `089b122770af81dbca40b63f23de344902fdb432` |
| Remote relation | `HEAD == origin/kv-offloading` at observation time |
| Commit time | 2026-07-03 14:07:34 `+08:00` |
| Recorded Agentrix gitlink | `c357601ee37ff386958e82546b7caa9113c843e6` |
| Full porcelain status | 3 worktree modifications including the Agentrix gitlink, 2,502 tracked deletions, 11 untracked entries |
| Full status SHA-256 | `a98ee276397f3ae341ea15f0cf4971f54d52065720712467a98bbb061b2d9803` |

The status digest is over `git status --porcelain=v1 -z`. The large deletion
count includes 2,495 paths under the main checkout's tracked `vllm/` tree. The
top-level dirty set also contains the modified `Agentrix` submodule and three
untracked local PDFs. These paths were preserved.

The `offload/` status at freeze time was:

```text
 M offload/README.md
 M offload/__init__.py
 D offload/cpu_block_buffer.py
 D offload/gpu_allocator.py
 D offload/location_table.py
 D offload/manager.py
 D offload/test_v0_manager.py
 D offload/transfer.py
 D offload/types.py
?? offload/benchmarks/
?? offload/compat/
?? offload/core/
?? offload/experiments/
?? offload/integrations/
?? offload/research/
?? offload/runtime/
?? offload/tests/
```

`HEAD` contains only nine `offload` files: `README.md`, `__init__.py`, and the
seven deleted top-level Python files above. Current code cannot be reconstructed
from the main commit alone. The current non-result, non-research source tree had
170 files and tree-manifest SHA-256
`9ffd18538a261c401f5fd66b3a75468cb31969cdadca75b68bdb7b0029635a9f`.
The digest excludes `offload/experiments/results`, `offload/research`,
`__pycache__`, and `.pytest_cache`.

### Agentrix and vLLM checkouts

| Layer | Branch / commit | Recorded parent gitlink | Dirty evidence |
|---|---|---|---|
| Agentrix | `overlap` / `7190d7814841aab1fc0eaee1504d1671084d3e8d` | main repo records `c357601ee37ff386958e82546b7caa9113c843e6` | `.gitmodules` modified, `vllm` dirty, one untracked FlashAgents PDF; status SHA-256 `64aa895edb8a46445a37bd6a3bd64298622f8902281709376a82e8e57ae90f9e` |
| vLLM | `codex/forkattention-offload-adapt` / `2e5d72fa44e29cdae21023d6cd4de03bdba9bfbd` | Agentrix records `e139c755bd39cf34ef1f2a4280c68a7c4952dccf` | 34 modified files and 10 untracked entries across scheduler, connector, block-pool, engine, tests, CMake and FlashAgents; status SHA-256 `4da61d5496eb7fcc60966b31debda05469f08519d4cca0114dea982cf271b67b` |

Agentrix remote is `git@github.com:oyyyyy61/Agentrix.git`; vLLM remote is
`https://github.com/T4t4KAU/vllm.git`. For the vLLM working tree, the tracked
binary-patch SHA-256 is
`4e9599eb660a6f57493096ec3a2a49f321ee06e3d9126851438bd9d545cb2d48`,
and the sorted untracked-file tree-manifest SHA-256 is
`c664e71745714386b7c3905c7c05a12b6b36f8660fd7fa423ab162c985f2f958`.
These observation-time digests identify the pre-archive inputs. The recoverable
bundle stores its own final vLLM patch and untracked archive with the hashes in
the next section.

### Recoverable M0 source bundle

`experiments/source_freezes/m0_20260723` preserves the dirty M0 runtime without
overwriting historical results:

| Payload | SHA-256 |
|---|---|
| `offload_source.tar.gz` | `7ea6ae83c0667d0ac83bd3b1c25d799cec701b600e85fcddf8940514dfb96c08` |
| `vllm_tracked.patch` | `0ce45c0e2d87b33fa599fd8f494d97fb5f612ef59dc6c84cd66a491ebf7c4a4d` |
| `vllm_untracked.tar.gz` | `ecc3cc2ce4a04e896047f70682fb0f54d32c4b77a024f56f7d64d9844d6118e5` |
| `agentrix_tracked.patch` | `e4dfdf7521ea3a14107d15ac4d79c975e2ab54b0074689b4f18abe8e52269299` |
| `requirements.freeze.txt` | `16450cc1d680cda638fc14bb814f323114e89391a689aa0f3b086f5e89254328` |
| Bundle `README.md` | `b02ead3bddb280826aafc810bebfe6e6c1cdc1a737d79d10a58fd43734a09225` |

Both tracked patches passed reverse-apply checks at freeze time; both gzip
archives passed integrity checks. The files are mode `0664` in a mode `0775`
directory. Their hashes detect modification, while filesystem permissions do
not enforce write protection. This bundle reconstructs M0 and predates the
latest v2 lifecycle, reset, request-detach, coverage, and control-runner work.

## Host and Runtime Environment

| Field | Frozen value |
|---|---|
| Host | `lab501-gpu4090` |
| OS | Ubuntu 22.04.5 LTS |
| Kernel | `6.8.0-124-generic`, x86_64 |
| libc | glibc 2.35 |
| Time zone | Asia/Shanghai, `UTC+08:00` |
| GPU | GPU 0, NVIDIA GeForce RTX 4090, UUID `GPU-988a25f8-7e0b-b374-8619-f9dade40cc6a` |
| GPU memory | 24,564 MiB reported by `nvidia-smi` |
| Compute capability | 8.9 |
| Driver | 580.159.03 |
| Driver CUDA capability | 13.0 |
| CUDA toolkit | 13.0, `nvcc V13.0.88` |
| PyTorch CUDA | 13.0; cuDNN 91900 |
| GPU occupancy snapshot | 432 MiB total display use, 0% utilization, no compute application reported at 13:49 CST |
| Port snapshot | unavailable: `ss -ltnp` could not open the netlink socket in this managed environment |

The user-supplied interpreter path
`/home/data/25_oyzx/Agentrix/vllm/offload/bin/python` does not exist. Commands
below use the available project interpreter:

```text
/home/data/25_oyzx/Agentrix/vllm/.venv/bin/python
Python 3.12.13 (Clang 21.1.4)
```

The interpreter is a symlink into uv's CPython installation. `uv` is version
`0.10.11`. PyTorch reports `cuda_available=True` and one CUDA device.

### Installed dependency identity

`pip freeze --all | LC_ALL=C sort` contained 239 lines and had SHA-256
`6bd235ab581cfaadc3aafe11ee2166fc27ffc1c05e3088257ad32c71885f55b6`.
Key installed distributions were:

```text
vllm==0.1.dev18192+g173016d22
torch==2.11.0+cu130
transformers==5.13.1
tokenizers==0.22.2
safetensors==0.8.0
numpy==2.2.6
msgspec==0.21.1
pydantic==2.13.4
pytest==9.1.1
```

The editable vLLM distribution version embeds `g173016d22`, while the active
source checkout is `2e5d72f...`. Import-path checks point at
`/home/data/25_oyzx/Agentrix/vllm/vllm/__init__.py`; the version suffix still
constitutes provenance drift and must be rebuilt or explained before a final
run.

| Dependency input | SHA-256 |
|---|---|
| `Agentrix/vllm/pyproject.toml` | `eaf27ad259878f18ad961f9d892f8c102fc3bd7f156df0ea8f4c5fb91d3fb820` |
| `Agentrix/vllm/requirements/common.txt` | `ef5490222caa41c6841e194e93ce970e3045efca07a8a6da4a0e41ebf340c2ec` |
| `Agentrix/vllm/requirements/cuda.txt` | `da696481856e448e3ec80be49ac9c718fc8e764de8fb34949a8f681c422e605f` |
| `Agentrix/vllm/requirements/test/cuda.txt` | `c0322395675e91d08746ccdc810d2295161522bd7ee94e39c6f14a6a21fbf0ae` |
| `Agentrix/vllm/requirements/kv_connectors.txt` | `ee25ee817f373d04d3b5b182ccc011d30512d5528be89617d0e2db0548690923` |

No `uv.lock` exists in the vLLM checkout. The installed-environment digest is
therefore part of the freeze and the requirements files alone do not lock every
transitive dependency.

## Model and Tokenizer

Model root:
`/home/data/25_oyzx/moqae_runtime_gpu/modelscope/Qwen/Qwen3-8B`.
The model directory is a clean ModelScope Git checkout at
`26028140be3ee69b82b1d1450179ab71bb1121b9`, remote
`https://www.modelscope.cn/Qwen/Qwen3-8B.git`.

Frozen configuration: `Qwen3ForCausalLM`, BF16, 36 layers, hidden size 4096,
32 attention heads, 8 KV heads, head dimension 128, vocabulary 151,936, and
maximum configured position 40,960. The weight index reports 16,381,470,720
bytes of tensor payload.

| Model file | Bytes | SHA-256 |
|---|---:|---|
| `model-00001-of-00005.safetensors` | 3,996,250,744 | `31d6a825ae35f11fb85b195b4c42c146c051e446433125a215336abdf95cbf5f` |
| `model-00002-of-00005.safetensors` | 3,993,160,032 | `5991236cea6fe21f3d43cab0f0e84448734fbbe0789816202989f2ddc9d18282` |
| `model-00003-of-00005.safetensors` | 3,959,604,768 | `c5185c4794be2d8a9784d5753c9922db38df478ce11f9ed0b415b7304d896836` |
| `model-00004-of-00005.safetensors` | 3,187,841,392 | `b5ee7de71fbf17db3d5704e0c8f2bc7d005ca9e1d7ca2aeb19827b0cfcaa917a` |
| `model-00005-of-00005.safetensors` | 1,244,659,840 | `20c2d6366ab85c90786ccdd829cd2b9e7d30ef3b2ebbb998280e7e4014b542ff` |
| `model.safetensors.index.json` | 32,878 | `f9fdbcb91c23971c13ec5d5f2573d2349e8f61f2f049371ec699281748fdb1bc` |
| `config.json` | 728 | `f7c4eadfbbf522470667b797a3c89be2524832d2d599797248dc304fff447c30` |
| `generation_config.json` | 239 | `2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2` |

Tokenizer class is `Qwen2Tokenizer`; its chat template is embedded in
`tokenizer_config.json`. Model `eos_token_id` is 151645; tokenizer EOS is
`<|im_end|>` and pad is `<|endoftext|>`.

| Tokenizer file | Bytes | SHA-256 |
|---|---:|---|
| `tokenizer.json` | 11,422,654 | `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4` |
| `tokenizer_config.json` | 9,732 | `d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101` |
| `vocab.json` | 2,776,833 | `ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910` |
| `merges.txt` | 1,671,853 | `8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5` |

The model metadata names Transformers 4.51.0, while the live environment uses
5.13.1. Output and logprob guardrails remain mandatory for every final matrix.

## Agentrix Data

Data root: `/home/data/25_oyzx/Agentrix/benchmark/data`. All ten regular files
were hashed; no sampling or normalization was applied during this freeze.

| Relative path | Bytes | SHA-256 |
|---|---:|---|
| `agencybench_v2.jsonl` | 255,194 | `24cc1cf9a867d9b2dc5cc8e504902669c73a5df62ed4d844f6886cea8cceffc1` |
| `agencybench_v2/data-00000-of-00001.arrow` | 249,400 | `5e3e091a9db83b6a98be09415a23eab064a9dd34bf50fc8213837944c20536ec` |
| `agencybench_v2/dataset_info.json` | 1,503 | `4842b95373273b818fe0200ef416d15f74ddf571b41906a0dfac48226757ace2` |
| `agencybench_v2/state.json` | 250 | `6f9b9e20e50f8b5779168a8a701ca6d80cb9636793bcdf1b2b0ff6b894255476` |
| `agentboard.jsonl` | 70,753 | `798903f28699111ec3819331d35d7704fe196cd52712a9f3550dac8355365690` |
| `appworld.jsonl` | 63,566 | `81b17c099c888f2bfd37717801adb2c36599e9adfe560b9541a9b2748e1fa7b0` |
| `swebench_verified.jsonl` | 8,222,254 | `889bccf7ada1a43d211050ac666f3b31032997209afb10dccdc6ea52128a8435` |
| `swebench_verified/data-00000-of-00001.arrow` | 7,782,360 | `0d119efe73413554335bd410a04d82fd4a586bfd312cee677ee40af5de2ac46e` |
| `swebench_verified/dataset_info.json` | 1,518 | `338f326bee5f38e6a42fb1ccc7c94348dd4f40f5383dad8008d0836f9b43b161` |
| `swebench_verified/state.json` | 249 | `cd778b1a4a9cc8d5e21e445f4c4c6137a5ba5818a427833d635f4ce807360f07` |

Dataset license, upstream revision, and leakage-resistant workflow/template/time
splits are not yet frozen. These hashes identify the local bytes only.

## Paper Inputs

| Scope | Local file | Bytes | SHA-256 |
|---|---|---:|---|
| TokenCake | `cagent-work/2510.18586v2.pdf` | 17,321,529 | `2a1f88f896afb82a2f84d4e2c19f140d62a862157fd51ab5148ef0acc0eead11` |
| Continuum | `cagent-work/2511.02230v6.pdf` | 1,008,345 | `ec60c3a7f4edb94c2a527da704cf901f0f44fd004a91c5a523b9c006ae11f2fc` |
| PBKV claim boundary | `cagent-work/2605.06472v1.pdf` | 1,635,439 | `eb2cc2a7750f972a228519376ad27214cc4d674686c2e7209207c8d47b0da301` |
| Present in Agentrix, outside current M0 claim matrix | `Agentrix/Fang et al. - FlashAgents Accelerating Multi-Agent LLM Systems via Streaming Prefill Overlap.pdf` | 724,975 | `d1bdf3250abd30601ae2673f259ace18e3b73454416b92671fea95d45b55f86a` |

The PBKV PDF was added during M0 because it materially supersedes the original
C1/C2 boundary. The related-work matrix records web sources for the remaining
systems; a local immutable copy has not been frozen for each web source.

### Companion documents

The archive and working tree are reported separately so later evidence-ledger
amendments do not rewrite M0 history.

| Artifact | M0 archive SHA-256 | Current working-file SHA-256 |
|---|---|---|
| `offload/research/RESEARCH_CONTRACT.md` | `3f6da0dd2418fb413937cbedb3d9406832a8965f741b35ed7811775130db2543` | `3f6da0dd2418fb413937cbedb3d9406832a8965f741b35ed7811775130db2543` |
| `offload/research/RELATED_WORK_MATRIX.md` | `31e6ed5e6f615093bd717f40418aeb8e2609bbabe7eb34ff09908df954655e51` | `31e6ed5e6f615093bd717f40418aeb8e2609bbabe7eb34ff09908df954655e51` |
| `offload/research/CLAIM_EVIDENCE_LEDGER.md` | `dba79242830129c0ca3e1581e0258de3632f61315aa826712651240c1aaf24eb` | `f26744a478e1aa06354dd711b558690f030fdeb32a3ac9f867ac17258c464447` |
| `offload/research/M1_CANONICAL_LEDGER_SPEC.md` | not part of M0 companion trio | `b17fc4ec3a1c7ecc2c07bde2f2832f9625c7f7e951f9894dd70402036e60d812` |

This manifest omits its own hash because embedding a file's SHA-256 inside that
same file is not a stable self-contained construction. Hash the completed file
externally when packaging the four-document M0 bundle.

## Historical Artifacts and Recomputed Baseline

### Artifact integrity

The primary historical directory contains 386 files and 157,094,937 bytes:
76 JSON, 234 JSONL, 75 logs, and one Markdown report. All JSON documents and
all JSONL records parsed successfully with `jq` on 2026-07-23.

| Artifact | SHA-256 |
|---|---|
| `reproduction_headroom_full_20260723/REPORT.md` | `ea49d3153ccbe7035fb2e90d4b088da8b273212b5a83502cd5efcc59a8d17ad2` |
| `reproduction_headroom_full_20260723/aggregate.json` | `aefa60e09918ed8810534757f239966ee98466edd979204e8412ef55de9223f1` |
| Complete historical directory, tree-manifest | `f0738ed52a747643bf4f072d71e55330a1c24ba7bb4585586a6d537ffd9dcfc3` |
| Full-run launcher `run_full_headroom_audit.sh` | `a8f6bf4dc6cf09a4dac7a1d4be03849af63bb997e043430e8ea16bc1f0492799` |
| Aggregator `headroom_audit/aggregate.py` | `7c4a78d8169e82686ed53eaa10f1a1c661918b177b1f700678bdc1fc7b62ec16` |
| `TOKENCAKE_SPACE_CLOSEOUT_20260723.md` | `112ebe44100b0f5a2cf935d18c13e27df750337ec57b8a3d713754ca04793f06` |
| Space closeout `aggregate.json` | `155867e726997070b5875dc81addc8a5483fc83257995bceb5dc1239d011a99a` |

The broader `offload/experiments/results` directory held 624 files and
378,750,582 bytes. Only the primary directory and named supporting artifacts
above are part of this M0 content freeze.

### Historical arithmetic

The primary directory contains 36 TokenCake runs, 36 Continuum main-matrix
runs, and three Continuum boundary runs. Its aggregate records:

| Result | Frozen value | Interpretation at M0 |
|---|---:|---|
| TokenCake Time overall | `1.024569x`, 95% CI `[1.008791, 1.040348]` | descriptive historical arithmetic |
| TokenCake Time, 100 ms | `0.990039x` | regression in all 12 runs |
| TokenCake Time, 300 ms | `0.995151x` | near neutral |
| TokenCake Time, 1000 ms | `1.088518x` | positive long-tool regime |
| TokenCake Space closeout | `0.9132x` | frozen negative ablation |
| Continuum connector-only / GPU-only | `1.142780x`, CI `[1.067909, 1.217650]` | fixed-order exploratory evidence |
| Continuum adaptive / connector-only | `1.026098x`, CI `[0.972829, 1.079366]` | performance gate failed |
| Continuum boundary adaptive / connector-only | `0.996532x`, CI `[0.982892, 1.010173]` | neutral boundary |
| Repeated control relative range | mean `17.8311%`, max `40.6122%` | variance/order warning |

All 36 main Continuum runs report identical cache-hit ratio and computed-token
counts across connector-only, static TTL, and adaptive TTL. Adaptive native
load-completed keys decline from 85,542 to 83,260 (`2.6677%`), yet lower native
loads do not align consistently with speedup.

Lifecycle counters disagree across layers: the high-level adapter reports zero
`expired_unpins` and zero `forced_unpins`, while the block-pool audit totals
110,137 expired blocks and 65,213 forced-eviction blocks for the main adaptive
runs. This is an accounting-layer mismatch, so neither set can close an M1
conservation equation without per-block terminal IDs.

The launcher fixes phase order. TokenCake uses
`baseline_vllm,tokencake_time,oracle_keep,oracle_prefetch,oracle_lifecycle`;
Continuum uses
`program_fcfs,program_fcfs_offload,static_ttl_offload,adaptive_ttl_offload`.
Only three seeds are reused across 12 cells. The reported 36-run intervals do
not represent 36 independent workload clusters.

### Test baselines

Three separate offload counts exist and must remain distinguishable:

- task handoff historical claim: `226 passed`;
- `offload/README.md` at freeze time: `224 passed, 17 warnings`;
- local no-cache run during this audit: `233 passed, 14 warnings in 4.97s`.

The local collection initially reported 233 tests. Concurrent M0/M1 work added
strict accounting, history-audit, and phase-order tests during the observation
window. The final no-cache run for source tree-manifest `9ffd1853...` reported
`245 passed, 14 warnings in 5.05s`. This is the observation-window baseline.

An earlier vLLM command collected 216 tests and finished with
`215 passed, 1 failed, 15 warnings in 357.06s`. The only failure was
`test_async_scheduling_pp_allows_rescheduling_with_output_placeholders`:
`pipeline_parallel_size=2` is rejected because this host exposes one GPU. The
other scheduler, TTL retention, and offloading-connector cases passed. This run
preceded the final canonical trace patch and is retained as a diagnostic
baseline.

For the final vLLM patch, the worker-metadata plus connector-scheduler offline
regression passed `92 passed, 15 warnings in 12.92s`; the focused canonical
trace/accounting subset passed 10 tests. A clean managed invocation of the
connector scheduler alone independently passed `86 passed, 15 warnings in
12.14s`. The final cagent measurement trio passed 19 tests, and Ruff passed on
the changed accounting paths.

### Post-freeze v2 validation amendment

Later v2 work is tracked separately from the M0 counts above:

- the CPU manager/lifecycle/factory/tiering regression, including producer
  request detach before allocation removal, passed `278` tests;
- the offloading-connector directory passed `187` tests with `2` skipped;
- the current complete `offload/tests` run passed `323` tests with `14`
  warnings; control-targeted tests passed `24`, and related accounting tests
  passed `64` with one NVML warning.

All changed and directly relevant files pass scoped Ruff and format checks. A
separate full-tree scan reports 443 lint diagnostics and 73 files that would be
reformatted, mostly in earlier M2/M3 prototypes. The broader historical active
tree remains an open lint baseline; it was not mechanically rewritten during
this M1 work.

The reset-fixed GPU result is
`m1_gpu_canonical_dag_qwen3_8b_s1_b2_t256_o1_g128_resetfix_20260723_2113.json`
(SHA-256 `23213421befe7055dfa8d84b54c4e41ecfad8aa626d78603932534966007713d`).
All six phase traces pass all 12 strict v2 gates in their declared observed
action scopes with zero issues.

The real-prefetch coverage result is
`m1_gpu_canonical_dag_qwen3_8b_s1_b2_t256_o1_g128_prefetchcov4_20260723_215312.json`
(SHA-256 `1816b355af23311dddb4bcd71823966c70264b6e238d08d67e07d694f46ca38f`).
Its coverage trace has SHA-256
`6dd17876b1b42f98d903068ad86b802944cfcf4ca7bad85e9f58e491ea2d55b3`
and records three prefetch operations, all scheduled and completed. The phase
declares `lifecycle_action_coverage_only` and
`performance_claim_eligible=false`; it is excluded from performance evidence.
The main policy-pressure audit remains invalid.

The accepted executable control protocol is
`experiments/results/m1_measurement_control_abba_12seed_rawindex_v2_20260723/protocol.json`
(file SHA-256
`cdbe5476061b1e8d12bb6d018f203155f36974daa86ab3feb50b1546bf8ca614`).
It freezes 48 balanced ABBA rows over 12 raw-physical-line samples with
identical `control_a` and `control_b` bindings. Internal
protocol/configuration/schedule digests are
`15fc69536f975287cb4e88c9a978a46d5f84dffe33584e9fc737e194fac4778e`,
`34d1c5124226f8c1a05be4b7266051552cc034d9c3d65e2cb07ff49dc81d6fe1`,
and `b54d10860149fdb00d46851936d4bf5cb01696d7cdd52db4071238446ee76307`.
Static validation passes and `core_payloads_frozen=true`.

The accepted run completes all 48 rows, records 96 journal events, and has no
failed rows. All report gates pass over 12 independent clusters. The paired
identical-control geometric ratio is `1.0004591871` with 95% CI
`[0.9928798409, 1.0080963917]`, within the frozen `[0.95, 1.05]` equivalence
interval. The relative 95% CI half-width is `1.890365%` for `control_a` and
`1.096603%` for `control_b`, each below 5%. The result contains 75,776
canonical rows with zero audit issues, zero live objects at end, and conserved
payload bytes. The report SHA-256 is
`b6d9e2033c4439724f32690cfd109d017a0e12f8a922fbf02b32a7decf74c1d3`; the
journal SHA-256 is
`730fe31210862e57afb015d763072ea4a8574389cbdc8e0d0979ded98a3da712`.

The accepted root's `SHA256SUMS` SHA-256 is
`feaca3dc974afbdba95c7007cbacb21501868528b614f8fa1517e7f5f64bf1d2` and
its manifest SHA-256 is
`230abc0d4f4a5213a7f353ffe49576fabc0b54201962be1e362cf5ad218c6ce2`.
Files are mode `0444`; directories are mode `0555`. Because the two labels
execute identical controls, this artifact establishes measurement fidelity and
lifecycle closure only. It carries no policy-benefit claim.

## Reproduction Commands

Run from `/home/data/25_oyzx/cagent-work` unless a command changes directory.
These checks avoid pytest cache and Python bytecode writes.

### Provenance and environment

```bash
git status --short --branch
git rev-parse HEAD
git ls-tree HEAD Agentrix
git -C /home/data/25_oyzx/Agentrix rev-parse HEAD
git -C /home/data/25_oyzx/Agentrix ls-tree HEAD vllm
git -C /home/data/25_oyzx/Agentrix/vllm rev-parse HEAD
nvidia-smi
nvcc --version
/home/data/25_oyzx/Agentrix/vllm/.venv/bin/python --version
/home/data/25_oyzx/Agentrix/vllm/.venv/bin/python -m pip freeze --all | LC_ALL=C sort | sha256sum
```

### Model, tokenizer, data, and paper hashes

```bash
cd /home/data/25_oyzx/moqae_runtime_gpu/modelscope/Qwen/Qwen3-8B
find . -maxdepth 1 -type f -print0 | sort -z | xargs -0 sha256sum

cd /home/data/25_oyzx/Agentrix/benchmark/data
find . -type f -print0 | sort -z | xargs -0 sha256sum

cd /home/data/25_oyzx/cagent-work
sha256sum 2510.18586v2.pdf 2511.02230v6.pdf 2605.06472v1.pdf
```

### Historical artifact validation

```bash
cd /home/data/25_oyzx/cagent-work/offload/experiments/results/reproduction_headroom_full_20260723
find . -type f -name '*.json' -print0 | xargs -0 -n 1 jq empty
find . -type f -name '*.jsonl' -print0 | xargs -0 jq -e . >/dev/null
find . -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum
```

Run the independent M1 history auditor from the project root:

```bash
PYTHONDONTWRITEBYTECODE=1 \
  /home/data/25_oyzx/Agentrix/vllm/.venv/bin/python \
  -m offload.benchmarks.measurement.history_audit \
  offload/experiments/results/reproduction_headroom_full_20260723
```

The CLI intentionally returns a failed gate for this history. Its code SHA-256
is `a432abe17e8e3503b4310a4aa573d907c3de70a30c80b768a10b264a6b62d1fe`;
the corresponding test SHA-256 is
`46f2afc4e96f0e2175ac06a45a2287a0cf4b8945a274e3632286377498f3fe14`.
It found 336 unique trace references: 225 exist, 111 are missing, and nine
discovered trace files are orphaned. The 81,443 readable events contain no
canonical row, no canonical `byte_count`, `block_count`, or `block_ids`, and no
release, prefetch, or expiry event family.

To reaggregate without touching the historical directory:

```bash
cd /home/data/25_oyzx/cagent-work
AUDIT_TMP=$(mktemp -d /tmp/offload-reaggregate.XXXXXX)
PYTHONDONTWRITEBYTECODE=1 \
  /home/data/25_oyzx/Agentrix/vllm/.venv/bin/python \
  -m offload.benchmarks.headroom_audit.aggregate \
  --tokencake-dir offload/experiments/results/reproduction_headroom_full_20260723/tokencake \
  --continuum-dir offload/experiments/results/reproduction_headroom_full_20260723/continuum \
  --output "${AUDIT_TMP}/aggregate.json" \
  --report "${AUDIT_TMP}/REPORT.md"
sha256sum "${AUDIT_TMP}/aggregate.json" "${AUDIT_TMP}/REPORT.md"
```

The existing full launcher reproduces the historical fixed-order protocol. A
fresh output directory is required:

```bash
cd /home/data/25_oyzx/cagent-work
NEW_OUTPUT=/home/data/25_oyzx/cagent-work/offload/experiments/results/reproduction_headroom_rerun_20260723
test ! -e "${NEW_OUTPUT}"
HEADROOM_RESUME=0 \
  bash offload/compat/current/run_full_headroom_audit.sh \
  /home/data/25_oyzx/moqae_runtime_gpu/modelscope/Qwen/Qwen3-8B \
  /home/data/25_oyzx/Agentrix/benchmark/data \
  "${NEW_OUTPUT}"
```

This launcher is suitable for historical replication and remains outside M1
confirmatory evidence because its phase order is fixed.

### CPU and targeted vLLM tests

```bash
cd /home/data/25_oyzx/cagent-work
PYTHONDONTWRITEBYTECODE=1 \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /home/data/25_oyzx/Agentrix/vllm/.venv/bin/python \
  -m pytest -p no:cacheprovider -q offload/tests

cd /home/data/25_oyzx/Agentrix/vllm
PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venv/bin/python \
  -m pytest -p no:cacheprovider -q \
  tests/v1/kv_connector/unit/offloading_connector/test_scheduler.py \
  tests/v1/kv_connector/unit/offloading_connector/test_worker_metadata.py

# Broader scheduler regression; the PP=2 case requires at least two GPUs.
PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venv/bin/python \
  -m pytest -p no:cacheprovider -q \
  tests/v1/core/test_scheduler.py \
  tests/v1/core/test_ttl_prefix_cache_retention.py
```

### M0 archive and M1 live-ledger validation

```bash
cd /home/data/25_oyzx/cagent-work/offload
sha256sum experiments/source_freezes/m0_20260723/*
gzip -t experiments/source_freezes/m0_20260723/offload_source.tar.gz
gzip -t experiments/source_freezes/m0_20260723/vllm_untracked.tar.gz

cd /home/data/25_oyzx/cagent-work
PYTHONDONTWRITEBYTECODE=1 \
  /home/data/25_oyzx/Agentrix/vllm/.venv/bin/python \
  -m offload.benchmarks.measurement.accounting \
  --trace offload/experiments/results/native_offload_traces/cpu_dram_prefetch_coverage_1784814836502232042.jsonl \
  --required-actions allocate,map,bind,release,lease,save,load,prefetch,unmap,evict
```

The final command reports 1,898 canonical rows, three scheduled and three
completed prefetch operations, zero issues, and all 12 strict gates passing.
The action list is the declared scope for this coverage-only trace.

The historical PBKV/Stateful-PBKV schedule remains byte-for-byte unchanged.
Its retirement sidecar is machine-readable:

```bash
cd /home/data/25_oyzx/cagent-work/offload
sha256sum \
  experiments/results/m1_protocol_audit_20260723/confirmatory_abba_schedule.json \
  experiments/results/m1_protocol_audit_20260723/confirmatory_abba_schedule.retirement.json
jq -e '.status == "superseded_non_executable"' \
  experiments/results/m1_protocol_audit_20260723/confirmatory_abba_schedule.retirement.json

cd /home/data/25_oyzx/cagent-work
/home/data/25_oyzx/Agentrix/vllm/.venv/bin/python \
  -m offload.benchmarks.measurement.control_abba validate \
  --protocol offload/experiments/results/m1_measurement_control_abba_12seed_rawindex_v2_20260723/protocol.json

cd /home/data/25_oyzx/cagent-work/offload/experiments/results/m1_measurement_control_abba_12seed_rawindex_v2_20260723
sha256sum -c SHA256SUMS
jq -e '
  .status == "accepted" and
  .execution.completed_rows == 48 and
  .execution.failed_rows == 0 and
  .execution.all_report_gates_passed and
  .statistics.precision_gate_passed and
  .lifecycle_accounting.issues == 0 and
  .lifecycle_accounting.payload_bytes_conserved
' MANIFEST.json
```

The original file SHA-256 is `32c3389f0277d6ec2b723c0238b61a95b0d828a2a7b21f8d42a28b8bd5367cbe`;
its internal canonical schedule digest is
`e864f9395c65772e61b33865fad07a0f59761b3526265bf072d3dfb89d8753eb`.
The sidecar SHA-256 is
`902951435bc85b7dae0ff0e34c6aecc018c1c1b75148e3801b175c288a36333c`.
The v2 protocol validator returns `valid=true` and no errors. The accepted
root passes `sha256sum -c SHA256SUMS` and the manifest predicate above.

## Known Drift and Limitations

1. The M0 runtime is recoverable from `source_freezes/m0_20260723`. The accepted
   post-M0 M1 state is independently frozen at
   `source_freezes/m1_lifecycle_v2_20260724_001249`. Later working-tree changes
   require a new no-overwrite freeze before they can support paper-facing runs.
2. Parent gitlinks disagree with both nested checkout commits. Checking out the
   recorded submodules produces different source.
3. The requested Python path is absent; the available `.venv` was substituted.
   The editable vLLM version suffix also disagrees with the source commit.
4. Dependency resolution lacks a complete lock file. The 239-line pip-freeze
   payload is stored in the M0 bundle and detects drift, while it does not carry
   wheels or guarantee that every distribution remains installable.
5. Model metadata targets Transformers 4.51.0 and the environment contains
   5.13.1. Final correctness checks must cover tokens, outputs, and logprobs.
6. The hardware claim is limited to one consumer RTX 4090. The PP=2 unit case
   fails its one-GPU validation; multi-GPU and data-center GPU claims have no
   evidence.
7. The historical matrix uses three reused seeds and fixed phase order. The
   independent audit measures a repeated-control relative-range 95% CI
   half-width of `8.9972%`, above the frozen `5%` target.
8. Historical connector traces lack canonical job IDs, byte counts, terminal
   failure/cancel states, allocation generations, and one reconciled lease
   lifecycle. High-level and block-pool expiry/forced-release counters conflict.
9. Historical results use 3 GiB pageable host KV memory, deterministic lognormal
   tool delays, no ForkAttention, and one local dirty vLLM tree. Absolute DMA
   timing and performance claims are confined to that setup.
10. Agentrix data bytes are frozen, while licenses, upstream revisions, trace
    field completeness, and leakage-resistant splits remain open.
11. Port enumeration was denied by the managed environment. No compute process
    was visible in the GPU snapshot, yet system-wide service occupancy was not
    fully observable.
12. The historical result directory and M0 source bundle are writable and
    unversioned. Their hashes detect later changes; filesystem permissions do
    not enforce immutability.
13. Test initialization depends on platform environment. One restricted
    invocation attempted Hugging Face DNS for `facebook/opt-125m`; another
    inherited a `disable_hybrid_kv_cache_manager` configuration that failed 56
    cases before assertions. The same final source passed all 92 connector and
    worker-metadata tests in a clean explicit-offline environment. Final test
    commands must record environment variables and cache identity.
14. The v2 live proof is scoped to one process and the primary CPU tier. A
    global run-end barrier, cross-manager slot namespace, process/file failure
    recovery, and secondary-tier allocation ledger remain open.
15. Changed and directly relevant files pass scoped Ruff/format checks. The
    broader historical active tree reports 443 lint diagnostics and 73 files
    that would be reformatted; no full-tree-clean claim is made.
16. The first v1 ABBA execution is retained as
    `superseded_failed_diagnostic`: it completed 36 of 48 rows before row 36
    failed with `no Agentrix record at filtered index 26`. Its frozen indices
    were raw JSONL physical lines, while execution treated them as positions
    after minimum-subtask filtering; only sample indices 0, 1, and 3 conformed
    to the frozen snapshot. Its `SHA256SUMS` hash is
    `746bbe58a583dac8d4cf4ec34f22803378516e7510b5369a27c016baca4c80d6`;
    its status-manifest hash is
    `1b9862be6e4372270f9acf59cdfd9028b4baf6247167441b9a2cf45f7ea180fa`.
    The v2 raw-physical-line protocol corrects this identity mismatch and was
    separately executed from row zero.

## M0 and M1 Gates

| Gate | Status on 2026-07-24 | Evidence / remaining condition |
|---|---|---|
| M0 repository, environment, model, tokenizer, data, PDF, result audit | `PASS for identification` | Exact local identities and hashes are recorded above. |
| M0 companion documents | `PASS for archived identity` | The archive contains `RELATED_WORK_MATRIX` `31e6ed5e...`, `RESEARCH_CONTRACT` `3f6da0dd...`, and `CLAIM_EVIDENCE_LEDGER` `dba79242...`; the amended working ledger is `f26744a4...`. |
| M0 claim boundary | `PASS with supersession` | PBKV creates critical overlap; original C1/C2 are superseded. Revised claims focus on re-admission-aware value, derived GPU/DMA dual control, and deadline-aware partial-prefix single-flight. |
| M0 source reconstructability | `PASS for recoverable local snapshot` | `source_freezes/m0_20260723` stores offload source, both tracked patches, vLLM untracked files, and the 239-line dependency payload. Hashes provide integrity detection; permission-level immutability is not claimed. |
| **M0 overall** | **`PASS with integrity caveat`** | The archived M0 state can be reconstructed locally. It does not identify the newer post-M0 working tree. |
| M1 canonical event implementation | `PASS for scoped substrate` | v2 implements physical allocation, content mapping, logical owner binding, TTL lease, and physical transfer state machines. CPU request detach and deferred reset/tombstone semantics have direct tests. |
| M1 canonical event conservation | `PASS for observed primary-tier live scope` | Six reset-fixed DAG traces and the real-prefetch coverage trace each pass all 12 strict gates with zero issues. Multi-manager, crash-recovery, and secondary-tier boundaries remain open. |
| M1 control ABBA protocol | `PASS for frozen executable protocol` | Raw-physical-line v2 validates: 48 balanced rows, 12 samples, identical executable controls, frozen core payloads, and complete protocol/config/schedule digests. |
| M1 paired randomized or ABBA execution | `PASS for measurement control` | The accepted v2 execution is `48/48`, with 96 journal events, zero failures, exact row/order/hash/ledger conformance, 12 clusters, and all gates true. Its identical labels provide no policy-effect estimate. The v1 `36/48` run is excluded as `superseded_failed_diagnostic` after its raw-versus-filtered index mismatch. |
| M1 preregistered metrics/statistics | `PASS for contract` | Primary paired workflow JCT log ratio, seed-cluster inference, no outlier deletion, and secondary/mechanism metrics are frozen in `RESEARCH_CONTRACT.md`. |
| M1 independent paired sample count | `PASS for measurement control` | The accepted v2 result has 12 independent workload-seed clusters. Historical performance evidence still has only three clusters. |
| M1 control 95% CI half-width at most 5% | `PASS for measurement control` | V2 repeated-control half-widths are `1.890365%` and `1.096603%`; both meet the 5% gate. |
| M1 mechanism coverage | `PASS for observed actions` | The coverage-only trace closes three real prefetches alongside save/load/allocation/mapping/binding/lease events. Fault terminals remain synthetic evidence, and coverage latency is excluded from performance claims. |
| Current source identity | `PASS for scoped reconstruction` | `source_freezes/m1_lifecycle_v2_20260724_001249` binds deterministic runtime source, main/Agentrix/vLLM patches, exhaustive inventories, exact dependencies, research documents, accepted result hashes, and test/style evidence. Its `SHA256SUMS` SHA-256 is `b2937a6c...f3166`; manifest SHA-256 is `ed9ce038...c6dfb`. Clean-base patch application, archive extraction, checksum, dependency, inventory, jq, and read-only permission checks pass. |
| **M1 overall** | **`PASS for declared scope`** | Lifecycle substrate, primary-tier live accounting, 12-cluster measurement-control ABBA, precision, raw identity, and source reconstruction pass for one process, one RTX 4090, and the primary CPU-DRAM tier. Multi-manager, secondary-tier, crash-recovery, and multi-GPU claims remain excluded. M2 correctness work is open. |

## M1 Closure and M2 Entry

The M1 decision is bound by
`experiments/results/m1_acceptance_20260724`. Every later accepted run must
continue to carry source, model, tokenizer, data, configuration, schedule,
result, and raw-ledger hashes, and its ledger audit must pass before
aggregation. The prefetch-coverage phase remains outside performance
aggregation. M2 now targets one canonical block/lease/workflow schema and a
live orchestrator with ownership, release, use-after-free, isolation, and
output-correctness gates.
