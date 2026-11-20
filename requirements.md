# requirements.md — SEAFL vs Libra: Semi-Asynchronous FL Comparison

> Living document. Update the **Changelog** at the bottom every time something changes.
> Status tags: `[ ]` todo · `[~]` in progress · `[x]` done · Priority: **P0** must-have for the POC demo · **P1** needed for the final report · **P2** stretch.

---

## 0. Context and deadlines

| Item | Date | What it needs from the code |
|---|---|---|
| 2nd presentation | 23–28 Nov 2026 | **Working POC**: both algorithms running in one simulator, at least one comparison plot |
| Final submission | 30 Nov 2026 | Full code (documented, indented — graded), IEEE-format report with the comparison |

Course rule (Option 1): implement ≥2 *related* algorithms, build an application to test them, compare their performance. Code documentation and indentation carry marks.

**Papers**
- **SEAFL** — Islam et al., IPDPS 2025 (arXiv:2503.05755). Fixes stale updates by *weighting* them (staleness + importance), plus SEAFL² partial training.
- **Libra** — Wang et al., ICDCS 2024. Fixes unfair participation by *gating* who contributes (fast-device limits, participation-based stale-model admission, biased client selection).

**Core research question:** under identical data, device heterogeneity and metrics, how do "weight the updates" (SEAFL) and "control who participates" (Libra) compare on accuracy, convergence time, overhead and fairness?

---

## 1. Scope

### 1.1 Algorithms

| ID | Algorithm | Type | Source | Priority |
|---|---|---|---|---|
| A1 | **SEAFL** (Alg. 1) | semi-async, buffer-triggered | SEAFL paper | **P0** |
| A2 | **SEAFL²** (Alg. 2, partial training) | semi-async | SEAFL paper | P1 |
| A3 | **Libra** (all 3 mechanisms) | semi-async, deadline-triggered | Libra paper | **P0** |
| B1 | FedAvg | synchronous | baseline in both papers | **P0** |
| B2 | FedBuff (= SEAFL with uniform weights, β = ∞) | semi-async | SEAFL baseline | P1 (nearly free) |
| B3 | FedAsync (K = 1) | asynchronous | baseline in both papers | P1 |
| B4 | SAFA | semi-async | Libra baseline | P2 |

Ablations (P1/P2): SEAFL with β = ∞; SEAFL vs SEAFL²; Libra-NoL (no fast-device limit), Libra-RS (random selection), Libra-SS (stale models chosen by staleness, not participation).

### 1.2 Out of scope
- Real phone traces (Libra's "Tracer" data is not public) → replaced by a synthetic heterogeneity model (§3.3).
- Real hardware deployment / Raspberry Pis.
- Privacy mechanisms (DP, secure aggregation).

---

## 2. Architecture — one simulator, pluggable strategies

Both algorithms must run on **the same simulator**, same data split, same device-speed draws and same seed. Only the server/client strategy differs. That is the whole point of the comparison.

### 2.1 Execution model — DECIDED (decisions.md D1–D3)
**Real multi-process message-passing system on one JarvisLabs L4 machine** (no virtual-clock simulator).
- 1 server process + 20 client processes, TCP via `multiprocessing.connection` (authkey set).
- Clients on CPU (1 thread each); server on GPU. Heterogeneity = injected seeded Pareto delays, `time_scale` knob.
- Server time (`time.monotonic()`) is the only clock. Eval every 5 s; fixed wall-time budget per run.
- Moving to 2 machines = change `server_host` (P2, no code planned).

### 2.2 Components and 2.3 interfaces
See **CLAUDE.md → Layout / Contracts** (single source of truth). Summary: `config, data, model, transport, client, server, metrics, strategies/{fedavg,seafl,libra}, run`; strategies are pure state machines `start / on_update / on_tick → list[(cid, msg)]` (decisions.md D12).

---

## 3. Functional requirements

### 3.1 SEAFL (A1) — P0
- [ ] Server samples a random subset S_t of N_active clients and broadcasts w_t^g.
- [ ] Buffer uploads; aggregate when **K** updates are in the buffer.
- [ ] Staleness S_k = t − t_k; enforce **S_k ≤ β** (server waits for any client that would exceed β).
- [ ] Staleness factor γ = α·β / (S_k + β).
- [ ] Importance factor s = μ·(cos(Δ_k, w_t^g) + 1)/2. *(Ambiguity: paper says cosine between the update and the global model — confirm whether Δ = w_k − w_{t_k} or w_k itself; log both early on, see §9 Q4.)*
- [ ] Weight p_k = (|D_k|/|D_buffer|)·(γ + s), normalised over the K updates.
- [ ] w_new = Σ p_k w_k; server mixing w_{t+1} = (1 − ϑ)·w_t + ϑ·w_new, ϑ = 0.8.
- [ ] New global model sent only to the K clients that just reported.
- Defaults: K = 10, β = 10, α = 3, μ = 1, ϑ = 0.8, E = 5.

### 3.2 SEAFL² (A2) — P1
- [ ] Server sends a **notify** message to a client whose update is about to exceed β.
- [ ] Client finishes its current epoch, uploads immediately, skips remaining epochs.
- [ ] Count notification messages in communication overhead.

### 3.3 Device heterogeneity model (shared) — P0
- [ ] Per-client compute speed drawn from a **Pareto** distribution (SEAFL's heavy-tailed setup); fixed per seed. Delay = `base × min(1 + Pareto(1.5), 20) × time_scale` (cap: D6).
- [ ] Per-client network latency (upload/download) — simple per-client constant + jitter.
- [ ] Optional availability: clients go idle for random periods (Zipf idle time, as in SEAFL's preliminary study) — P1.
- [ ] **Resource-quality label** for Libra's biased selection: good = top-X% by speed (stand-in for the ContextFL signal Libra uses). Document this substitution.

### 3.4 Libra (A3) — P0
Round structure: broadcast to selected clients → clients train → server collects until deadline **T** → aggregate (standard sample-weighted averaging) → every 10 rounds run Local Model Selection + Biased Client Selection.
- [ ] **(a) Limit fast devices:** a client may upload more than once per round. Uploads 1–2 are free; for upload 3..σ_max, compute **Sinkhorn distance** between output distributions of the round's global model and the new local model on a small D_val; if ≥ L, stop training until next round. Hard stop at σ_max = 5.
- [ ] **(b) Stale-model admission by participation count P:** sort clients by P descending; top N·(1−M) get ϑ = 0 (stale uploads rejected), bottom M get ϑ = 1. M = 0.5. Early-drop flag ρ = 1 if (τ_max − v_i) < (P_i − P^M); also drop when v_i = τ_max.
- [ ] **(c) Biased client selection:** candidates = good-resource clients; sort by reported local loss (descending); pick min(N·C, |A|). C = 0.6.
- [ ] Clients report scalar local loss with each upload.
- [ ] Selection logic runs every 10 rounds.
- Defaults: N = 20 (paper: 100), C = 0.6, τ_max = 5, σ_max = 5, M = 0.5, E = 5, batch 32, SGD lr 0.01.
- **Unspecified in paper → chosen in decisions.md D6:** T = median expected round time; L = pilot median; Sinkhorn on mean softmax over D_val, cost 1 − I, ε = 0.1, 100 iters; D_val = 500.

### 3.5 Baselines — P0/P1
- [ ] FedAvg: select fraction of clients, wait for all, sample-weighted average.
- [ ] FedBuff (P1): flag on SEAFL — p_k = 1/K, β = ∞, ϑ = 1.
- [ ] FedAsync (P1): flag value — FedBuff with K = 1.
- ~~SAFA~~ — cut (D4).

---

## 4. Experiment plan

Shared fixed settings: N = 20 clients, E = 5 local epochs, batch 32, SGD, 3 seeds per config (P1; 1 seed for POC). Same wall-time budget for every run (D8).

| Exp | Purpose | Data / model | Non-IID β | Algorithms | Priority |
|---|---|---|---|---|---|
| E1 | Headline comparison | FMNIST + CNN | 0.5 | SEAFL, Libra, FedAvg | **P0 (POC)** |
| E2 | Each method's "home turf" | CIFAR-10 + widened small CNN | 0.5 (Libra's) and 5 (SEAFL's) | SEAFL, Libra, FedAvg, FedBuff, FedAsync | P1 |
| E3 | Heterogeneity sweep | FMNIST + CNN | 0.5 | SEAFL, Libra | P1 |
| E4 | Sensitivity | FMNIST + CNN | 0.5 | SEAFL: K ∈ {3,5,10}, β ∈ {3,5,10,∞}; Libra: τ_max, σ_max, M | P1 |
| E5 | Ablations | FMNIST + CNN | 0.5 | SEAFL vs SEAFL² vs β=∞; Libra vs -NoL/-RS/-SS | P1 |
| ~~E6~~ | ~~Hybrid~~ | cut (D4) | | | |

Compute: JarvisLabs L4 (28 vCPU, 124 GB, 24 GB VRAM). ResNet-18 cut (D5). Pilot run first: picks `paper` vs `scaled` preset per algorithm on validation, plus L, T, budget, target (D6, D8).

---

## 5. Metrics (all logged per aggregation event)

Use all four axes so neither paper wins just because of the metric choice.

| Axis | Metric | From |
|---|---|---|
| Accuracy | final / best test accuracy; accuracy fluctuation (std over last X evals) | both |
| Convergence | **virtual wall-clock time to target accuracy**; number of aggregations to target | SEAFL |
| Communication | # uploads, # downloads, # control messages (notify, ϑ/ρ/χ flags), total bytes (model size × messages) | Libra |
| Computation | total client samples processed / epochs; **wasted compute** (training thrown away: rejected stale updates, early drops); server aggregation time | Libra |
| Fairness | **variance of participation count P**; Jain's fairness index over P; % clients that never / rarely contribute | Libra + ours |
| Staleness | distribution of staleness of aggregated updates | ours |

Target accuracies per dataset: to fix after the first runs (e.g. 95% MNIST, 80% FMNIST, 50% CIFAR-10).

---

## 6. Outputs needed for the report / presentation

- [ ] Accuracy vs virtual time curves (all algorithms, per experiment).
- [ ] Accuracy vs # communications curve.
- [ ] Bar chart: time-to-target.
- [ ] Table: comm/compute overhead normalised to FedAvg.
- [ ] Participation-count histogram per algorithm + fairness table (variance, Jain's).
- [ ] Sensitivity heatmaps / line plots (E4) and ablation table (E5).
- [ ] Architecture diagram of the simulator (for report + slides).

---

## 7. Non-functional requirements (graded: code documentation and indentation)

- [ ] Python 3.10+, PyTorch; `pyproject.toml` pinned.
- [ ] Formatter/linter: **ruff** only (`ruff format` + `ruff check`), 4-space indentation; gate command in CLAUDE.md (D11).
- [ ] Docstrings for every module, class and public function (Google or NumPy style); each strategy file starts with a header mapping code to the paper's equations/algorithm lines (e.g. "Eq. 4 → `staleness_factor()`").
- [ ] Type hints throughout.
- [ ] Deterministic: global seed controls data split, device speeds, client sampling, torch.
- [ ] One command per experiment: `python -m flsim.run --algo seafl --preset scaled --seed 0 --out results/e1/seafl/0` (argparse, no YAML — D9); matrix in `experiments.sh`.
- [ ] Unit tests (pytest) for: Dirichlet partition, SEAFL weights (γ at S=0 is α, at S=β is α/2; weights sum to 1), Libra participation sort / ϑ / ρ logic, Sinkhorn distance on known inputs, message send/recv round-trip.
- [ ] README: setup, how to reproduce each figure, design overview, mapping of paper → code, known deviations.
- [ ] Results saved as CSV/JSON under `results/<exp>/<algo>/<seed>/`; plots regenerated from those files only.

---

## 8. Repo layout
See **CLAUDE.md → Layout** (single source of truth).

---

## 9. Open decisions — all resolved, see `decisions.md`

| Old # | Resolved in |
|---|---|
| Q1 execution model | D1 (real multi-process, 1 machine, no sim) |
| Q2 datasets/models | D5 (FMNIST + CNN P0; CIFAR-10 P1; no ResNet) |
| Q3 Libra unspecified values | D6 |
| Q4 SEAFL importance | D7 (delta; log both) |
| Q5 round alignment | D8 (wall time + #messages; timer eval) |
| Q6 team split | D10 (solo) |
| Q7 framework | D1/D2 (own code, stdlib `multiprocessing.connection`) |

Still open until the pilot run: final L, T, wall-time budget, target accuracy, preset per algorithm.

---

## 10. Milestones (proposed)

| By | Milestone |
|---|---|
| 21 Nov | System core (transport, data, model, client, server loop) + FedAvg running on Jarvis |
| 22 Nov | SEAFL and Libra running end-to-end on FMNIST (P0) |
| 23 Nov | E1 plots + metrics → **POC ready for 2nd presentation** |
| 26 Nov | Baselines + E2–E4 runs, SEAFL² |
| 28 Nov | Ablations, docs, tests, README |
| 30 Nov | Report figures frozen, final submission |

---

## Changelog
- **2026-11-19** — v0.1 first draft: scope, architecture, per-algorithm requirements, experiment matrix, metrics, code-quality rules, open questions.
- **2026-11-19** — v0.2 Q&A planning (decisions.md D1–D12): virtual-clock sim → real multi-process (20 clients, JarvisLabs L4); `multiprocessing.connection` transport; FMNIST P0, ResNet cut; N = 20 with `paper`/`scaled` presets; delta importance; wall-time + #msgs axes, timer eval; argparse not YAML; ruff only; solo; SAFA/E6 cut, FedBuff/FedAsync as flags. Layout + contracts moved to CLAUDE.md.
- **2026-11-20** — v0.3 after pilot (decisions.md D16–D18): SEAFL preset `scaled` (K = 5, all active); target acc 0.80; budget 1200 s; Libra T = 4 × fastest local round (≈ 74 s), selection every 3 rounds (paper: 10), good-resource set rotates per refresh (`--resource-noise 0.5`); listener backlog fix.
