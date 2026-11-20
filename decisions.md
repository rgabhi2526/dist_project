# decisions.md — Design Q&A log

> Every planning decision, asked and answered one at a time. Newest at the bottom.
> Format: **Question** → options considered → **Decision** → why. Overrides anything in `requirements.md` that conflicts; requirements.md §9 gets updated from here.

---

## D1 — Execution model (2026-11-19)

**Q:** Virtual-clock discrete-event simulator, real multi-process system, or both?

**Options discussed**
- (a) Virtual-clock sim only — reproducible, fast; risk: looks like a single-process ML script, weak "distributed" story.
- (b) Real multi-process across 2 laptops / cloud — clearly distributed; but 100 PyTorch processes won't fit in RAM, heterogeneity must still be faked with `sleep()`, real-time runs make 50+ experiment runs take days, campus Wi-Fi may block peer traffic, async ordering is non-reproducible.
- (c) Both transports — same message-driven strategy code on a sim and a real transport.
- (d) **Real multi-process on ONE machine, ~20 clients** ← chosen.

**Decision:** 1 server process + 20 client processes, communicating only by messages over TCP (localhost). Device heterogeneity injected as Pareto-distributed delays with a `time_scale` knob. **No virtual-clock simulator.**

**Why**
- Genuinely distributed: separate processes, no shared memory, message passing only — demoable live.
- 20 processes fit comfortably in RAM/CPU; injected delays ≫ compute time, so OS scheduling noise stays small.
- One engine instead of two → less code before the 23 Nov POC.
- Algorithm correctness is covered by pytest on pure functions (weights, ϑ/ρ, Sinkhorn), which need no clock.
- Moving to 2 machines later = change `server_host` in config (P2, no code planned).

**Accepted costs:** results at 20-client scale (papers use 100) → parameters scaled down and documented; run-to-run noise → 3 seeds, mean ± std.

---

## D2 — Messaging layer (2026-11-19)

**Q:** How do server and clients talk?

**Options:** gRPC · ZeroMQ · raw `socket` + `pickle` · `multiprocessing.Queue` · **`multiprocessing.connection` (Listener/Client)** ← chosen.

**Decision:** stdlib `multiprocessing.connection`. Messages are plain dicts (with `state_dict` tensors) sent via `conn.send()` / `conn.recv()`. `authkey` set so only our processes can connect (pickle is unsafe on untrusted input). Server: one reader thread per client connection → one `queue.Queue` → single main loop is the only code that touches strategy state (no locks in algorithm code).

**Why:** real TCP (cross-machine capable), framing + serialisation built in, zero new dependencies. gRPC = proto/codegen/tensor-bytes plumbing for no benefit to the comparison; raw sockets = re-implementing framing; `Queue` = single-machine only.

**Known ceiling:** thread-per-client is fine at 20; switch to `selectors` only if N grows to hundreds.

---

## D3 — Hardware (2026-11-19)

**Q:** Where do the runs happen?

**Options:** A30 instance (16 vCPU, 64 GB RAM, 24 GB VRAM) vs **L4 instance (28 vCPU, 124 GB RAM, 24 GB VRAM)** on JarvisLabs ← chosen.

**Decision**
- **JarvisLabs L4.** Clients run on CPU, so vCPU count is the bottleneck, not GPU.
- **N = 20 clients**, each `torch.set_num_threads(1)`, on **CPU**.
- **Server on GPU** (aggregation, per-aggregation test eval, Libra Sinkhorn).
- `--device cuda` client flag kept only for CIFAR (P1).
- Develop + unit-test on the Mac; sync via git; Jarvis only for real runs (billed hourly).
- Runs launched in `tmux`/`nohup`; metrics flushed to disk after every aggregation so crashes keep partial results.

**Why:** 20 processes sharing one GPU get time-sliced → timing distortion; a small CNN on FMNIST is fast on 1 CPU thread. 28 vCPU leaves ~7 cores spare for the server and OS → cleaner timings. CUDA context per process (~300–500 MB VRAM) also rules out GPU clients at larger N.

---

## D4 — Scope (2026-11-19)

**Q:** Which algorithms / baselines / experiments survive?

| Item | Decision | Why |
|---|---|---|
| SEAFL | **P0** | core |
| Libra (all 3 mechanisms) | **P0** | core |
| FedAvg | **P0** | sync reference, ~20 lines |
| FedBuff | **P1**, flag on SEAFL | = SEAFL with uniform weights, β = ∞ |
| FedAsync | **P1**, flag value | = FedBuff with K = 1, no new code |
| SEAFL² | **P1** | SEAFL paper's own contribution; `NOTIFY` is a real control message |
| Libra ablations -NoL / -RS / -SS | **P1**, 3 boolean flags | each turns one mechanism off |
| SAFA | **cut** | outside the research question |
| E6 hybrid | **cut** | novelty only scores under Option 2; we are Option 1 |
| 2-machine demo | **cut** | config change only if ever wanted |

**Result:** 3 strategy files (`fedavg`, `seafl`, `libra`) + flags → 8 configurations.

---

## D5 — Dataset / model (2026-11-19)

**Q:** Which datasets and models, given CPU clients?

**Decision**
- **P0:** FMNIST + small CNN (2 conv + 2 FC, ~100k params, ~400 KB per message).
- **P1:** CIFAR-10 + same CNN widened, for E2 at Dirichlet β = 0.5 (Libra's setting) and β = 5 (SEAFL's).
- **Cut:** ResNet-18 (too slow on CPU clients; GPU clients contradict D3). Plain MNIST (≈98% for every method → no separation).
- Loading via `torchvision.datasets`; own ~10-line numpy Dirichlet partitioner.
- Server: FMNIST test set for evaluation; 500 samples carved out of it as Libra's D_val (excluded from the test metric).

**Why:** FMNIST at β = 0.5 is hard enough to separate methods, cheap enough for 1 CPU thread per client.

---

## D6 — Parameters at N = 20 (2026-11-19)

**Q:** Keep paper values or scale them for 20 clients? What do we pick where Libra's paper is silent?

**Decision:** run **both** presets in a pilot, keep whichever works better, document the choice.

| Param | `paper` preset | `scaled` preset | Note |
|---|---|---|---|
| E / batch / lr | 5 / 32 / 0.01 SGD | same | |
| Device delay | `base × min(1 + Pareto(a=1.5), 20)` | same | cap at 20× — uncapped tail can make a client that never finishes |
| SEAFL active clients | subset (paper) | all 20 | |
| SEAFL K | 10 | 5 | 5 keeps K/N ≈ paper ratio |
| SEAFL β | 10 | 10 | in aggregations, N-independent; E4 sweeps {3, 5, 10, ∞} |
| SEAFL α / μ / ϑ | 3 / 1 / 0.8 | same | |
| Libra C | 0.6 → 12 clients/round | same | |
| Libra τ_max / σ_max / M | 5 / 5 / 0.5 | same | |
| FedAvg clients/round | — | 12 | matches Libra's C → fair |

**Not given by Libra's paper — ours only, documented in README "known deviations":**
- Deadline **T** = median expected round time over all clients (from seeded delays).
- Sinkhorn input = mean softmax over D_val for global vs local model (one 10×10 problem).
- Cost matrix = 1 − I (0/1 label cost), ε = 0.1, 100 iterations. *(ponytail: approximates total variation; semantic cost only if L can't be tuned.)*
- Threshold **L** = median distance observed at upload ≥ 3 in a pilot run.

**Fairness rule for "whatever works better":**
1. Selection metric = time-to-target **on D_val / a validation split**, never on the test set (else we're tuning on test).
2. Same tuning budget for every algorithm — each gets the same two presets, 1 seed each.
3. Chosen preset per algorithm is recorded here and in the report.

Implemented as `--preset paper|scaled` (argparse), individual flags still override.

---

## D7 — SEAFL importance factor (2026-11-19)

**Q:** In s = μ·(cos(·, w_g) + 1)/2, is the cosine taken on full local weights w_k or on the update Δ_k = w_k − w_{t_k}?

**Decision:** **delta version** (cos(Δ_k, w_g)) is the implementation. Both cosines are computed and **logged** for every update (one extra dot product).

**Why:** full weights of two models from a shared start are nearly parallel → cos ≈ 0.99 → s ≈ μ constant → importance term is a no-op. The delta varies with update direction, matching the paper's intent. Logged "weights" cosine gives data-backed justification in the report. `--importance weights` flag added only if a reviewer asks.

---

## D8 — Alignment, stopping, evaluation (2026-11-19)

**Q:** SEAFL is buffer-triggered, Libra/FedAvg deadline/round-triggered — how are runs compared and when do they stop?

**Decision**
- **Never compare on "rounds".** X-axes: (1) wall-clock seconds since server start, (2) cumulative messages (uploads + downloads + control).
- **Stop:** fixed wall-time budget, identical for all algorithms (≈ 20 min at `time_scale` = 1; final value after pilot). Not "stop at target" — plateau and fluctuation are results too.
- **Eval on a timer:** server evaluates the current global model every 5 s wall time (GPU, ~0.1 s each). Same sampling rate for every algorithm; aggregation-heavy methods (FedAsync) aren't penalised by extra evals.
- Logged per eval: `t_wall, n_msgs, acc, loss`; per aggregation: staleness of each update, participation counts, both cosines (D7).
- **Target accuracy** for time-to-target = ≈ 90% of best accuracy any method reaches in the pilot (FMNIST ~80%).
- **Single clock:** server `time.monotonic()` only; clients never send timestamps → no cross-process clock skew.

**Correction during discussion:** first proposal "eval after every aggregation, excluded from budget" rejected — clients keep training in real time during eval, so its time can't be excluded cleanly. Timer-based eval fixes it.

---

## D9 — Config mechanism (2026-11-19)

**Q:** YAML config files per experiment (requirements.md §7) or something lighter?

**Decision:** **argparse only.** One `config.py`: every hyperparameter is a flag with the D6 default; `--preset paper|scaled` overrides a group. Experiment matrix = one `experiments.sh`, one line per run. Each run dumps its fully-resolved args to `results/<exp>/<algo>/<seed>/config.json`.

**Why:** the shell script *is* the experiment matrix, readable in one screen; `config.json` gives exact provenance. Skipped PyYAML, merge logic and a `configs/` dir — add only if a run ever needs > ~15 flags.

---

## D10 — Team split (2026-11-19)

**Q:** Who builds what?

**Decision:** **solo.** Build order follows the dependency seam:
1. System: transport → data/model → client → server loop → FedAvg end-to-end on Jarvis (target 21 Nov).
2. Algorithms: SEAFL + Libra as pure functions with pytest, then wired into the loop (target 22 Nov).
3. Metrics + plots → E1 POC (target 23 Nov).

Message schema and strategy hooks are fixed in CLAUDE.md before coding so later pieces plug in without rework.

---

## D11 — Code quality & commenting (2026-11-19)

**Q:** Which tooling, and what does "well commented" mean concretely? (graded)

**Decision**
- **`ruff` only** (`ruff format` = black, `ruff check` = flake8 + isort). Gate: `ruff format . && ruff check . && pytest`. No pre-commit hook (solo; gate lives in CLAUDE.md).
- Commenting rules → CLAUDE.md "Code style" section (module docstrings, paper → code map, Google docstrings with paper symbols, why-comments, `# ponytail:` markers, type hints).
- Tests = only the requirements.md §7 list.

---

## D12 — Strategy/message contracts (2026-11-19, set while writing CLAUDE.md)

**Decision:** strategies are plain classes (no ABC — 3 implementations, duck typing suffices) acting as pure state machines: `start / on_update / on_tick` return `list[(cid, msg)]`; the server does all I/O. Algorithm math in module-level pure functions for pytest. Message types: `HELLO, UPDATE, TRAIN, NOTIFY, HALT, SHUTDOWN` (fields in CLAUDE.md).

**Why:** no I/O in strategies → unit-testable without processes; one main loop → no locks.

---

## D13 — Libra gaps + shared contract details (2026-11-19, set before coding; veto-able)

**Libra paper gaps → our defaults (README "known deviations"):**
- Good-resource set A = fastest **70%** of clients (`--good-frac 0.7` → 14 of 20, so selecting 12 by loss still has a choice).
- Multiple uploads per round: **all** accepted uploads are aggregated, sample-weighted (that's exactly the fast-device influence Libra's limit controls).
- P^M = participation count of the client at the M-quantile boundary of the sorted list.
- Early drop ρ is evaluated **server-side**: on each aggregation the server sends `HALT` to in-flight stale clients with ρ = 1 or v_i ≥ τ_max (saves their compute, counts as a control message); stale uploads that still arrive are rejected and counted as wasted.
- Round end: clients that already uploaded this round are re-sent `TRAIN` (if selected) or `HALT`; clients with zero uploads keep training and become stale.
- Libra-NoL removes both the Sinkhorn limit and σ_max (the deadline still bounds uploads).

**Shared contract additions (also in CLAUDE.md):**
- `DISCONNECT` — server-internal message pushed by a reader thread on EOF.
- Wire format: `send_bytes(pickle.dumps(msg))` so every send/recv returns its byte count for free (communication metric).
- Every strategy keeps `self.events: list[dict]` (`kind` ∈ agg / reject / halt / notify / select, plus fields). The server drains it into CSV. This is the one source of truth for metrics.
- Heterogeneity: `client_delays(n, seed, base, cap=20)` lives in `config.py` so server (Libra deadline T, good-resource set) and clients use the same draws.

---

## D14 — Interpretations made while implementing wave 1 (2026-11-19)

**SEAFL** (`flsim/strategies/seafl.py`)
- Blocking rule: aggregation waits while any *in-flight* client has `version + 1 − base > β`; buffered clients never block. Buffer may exceed K while waiting; all of it is aggregated.
- S_k = `version − base_version` measured before the version bump. Δ_k = w_k − snapshot(w_{base}); cosine against current w_t.
- SEAFL² NOTIFY only fires when ≥ K updates are buffered *and* blocked; each blocking client notified once per wait.
- `K > n_active` raises (buffer could never fill). Non-float buffers are copied, not averaged.
- `paper` preset `n_active = N/2` is a placeholder — **verify against the SEAFL paper**.

**Libra** (`flsim/strategies/libra.py`)
- Ties in ϑ assignment and selection broken by smaller cid; #rejecting clients = `round(N·(1−M))`.
- Round 0: random selection within good set A (no losses yet); RS ablation random every refresh. Unseen clients' loss = +∞ (picked first).
- Sinkhorn computed for upload_idx 3…σ_max−1 (1-based); at σ_max → HALT. L = None → distances logged only.
- After any stale upload (accepted or not) the client is HALTed and idle; old-base models are not re-uploaded.
- P_i += 1 per accepted upload (stale included). Empty round keeps the old model.
- Libra-SS: stale accepted iff v_i ≤ τ_max // 2 (our reading).
- TRAIN policy = `{multi_upload, sigma_max, theta}`; `upload_idx` is 1-based and resets on each TRAIN.

**Config / data / model**
- Delay draw: `base · min(1 + Lomax(1.5), cap)` = classical Pareto(min 1); callers multiply by `time_scale`.
- FMNIST CNN = 105,866 params (width 16); CIFAR width 32. No BatchNorm (clean averaging).
- `train_epoch` returns the sample-weighted mean loss, which is what Libra reports as local loss.

---

## D15 — Fixes and tuning after wave-2 smoke runs (2026-11-19)

**Bugs found and fixed**
- **Server send stall:** a client read its socket only between epochs, so a ~400 KB TRAIN could block the single server loop for a whole epoch plus its delay. → Each client now has a daemon inbox thread that drains the socket all the time. The server keeps a >0.5 s send-blocked warning as a canary. After the fix, no warnings in any smoke run.
- **Libra auto deadline ignored compute:** → `T = 1.2 · E · (median(delay · time_scale) + epoch_compute_s)`. `--epoch-compute-s` defaults to 1.1 s (measured on the Mac). **Re-measure on Jarvis.**

**Tuning defaults (finalised in the pilot)**
- `--base-delay 3.0` (measured epoch is 1.08 s, so injected delay ≈ 3× compute at `time_scale` 1).
- The pilot runs Libra once, because presets only change SEAFL's K / β / n_active.

**Known effects, accepted for now**
- On the Mac CPU, eval takes 2–3 s inside the main loop, which makes Libra deadline ticks late. On the Jarvis GPU it should take ~0.1 s. If ticks are still late on Jarvis, move eval to its own thread.
- At smoke scale (4 clients, `time_scale` 0.05), Libra accepted almost no uploads because T ≈ 1.55 s. This is a pilot tuning issue for T, not a logic bug.
- SEAFL and FedAvg aggregate on CPU (~100k params, cheap). Only Libra's model and Sinkhorn run on the server device. This deviates slightly from D3.
- FMNIST S3 mirror failed DNS on the Mac. The raw `.gz` files were taken from the zalandoresearch GitHub repo into `data/FashionMNIST/raw/`.

---

## D16 — Listener backlog bug + GPU held after Ctrl-C (2026-11-20)

- **Bug:** `transport.listen()` used Python's default `Listener` backlog of **1**. On Linux, 20 clients connecting at once overflowed the accept queue. The kernel silently dropped the extra connections, and the run hung with 13/20 clients connected. macOS refuses immediately, so the retry loop had hidden it. **Fix:** `backlog=128` (`LISTEN_BACKLOG`). Regression test: 20 threads connect at the same moment.
- **GPU memory held outside the container** after the earlier Ctrl-C (22.5 GB used, no visible process, torch could not create a context). Fixed by pausing and resuming the JarvisLabs instance. **Rule:** before every E-series batch, check `torch.cuda.mem_get_info()` shows about 22 GiB free.

---

## D17 — Pilot results → final settings; Libra made to exercise its mechanisms (2026-11-20)

**Pilot results** (Jarvis L4, N = 20, 1 seed, 1200 s):

| run | best acc | t to 95% of own best | acc @300 / 600 / 900 s | Jain | never participated |
|---|---|---|---|---|---|
| fedavg | 0.805 | 380 s | 0.596 / 0.765 / 0.778 | 0.82 | 10% |
| libra | 0.885 | 355 s | 0.840 / 0.868 / 0.868 | 0.60 | 30% |
| seafl_paper | 0.771 | 920 s | 0.100 / 0.635 / 0.723 | 0.50 | 50% |
| seafl_scaled | 0.862 | 365 s | 0.692 / 0.821 / 0.841 | 0.82 | 0% |

**Settled**
- **SEAFL preset = `scaled`** (K = 5, all 20 active). `paper` (n_active = N/2) structurally excludes half the clients.
- **Target accuracy = 0.80** (≈ 90% of the best, 0.885).
- **Budget stays 1200 s.** FedAvg and SEAFL still gain 2–3 points between 900 and 1200 s.
- SEAFL staleness: 43 aggs, mean 2.1, max 10 = β. The bound works.

**Problem found:** Libra's mechanisms barely ran. T ≈ 33 s (median-based) and the fastest local round ≈ 18.6 s, so no client reached upload #3, giving 0 Sinkhorn checks (mechanism a). There were only 2 rejects (b) and about 3 selection refreshes (c). "Libra" was effectively FedAvg with a deadline and multi-upload.

**Decision (user chose option A):**
- **T = 4 × fastest possible local round = 4 · E · (base_delay · time_scale + epoch_compute_s)** ≈ 74 s. This is seed-independent. Percentile-based T was tried and swung 69–153 s across seeds.
- Per seed: 6–10 clients get ≥ 3 uploads per round (Sinkhorn runs), 2–4 go stale (admission runs), ~16 rounds per run.
- **`--select-every` 10 → 3** so biased selection refreshes ~5 times per run instead of once.
- Both are deviations (README "known deviations"): they scale Libra's round structure to our 20-client heterogeneity.
- Metrics gap fixed: round-end and post-stale HALTs are now logged as `halt` events (reasons `round_end`, `stale_done`). The pilot had 52 ctrl msgs but 0 halt events.

**Still needed:** one Libra pilot run with the new T → Sinkhorn distance distribution → set L. Open question: the good-resource set excludes the slowest 30% forever.

---

## D18 — Libra good-resource set rotates over time (2026-11-20)

**Q:** In the pilot, 30% of clients never participated in Libra (Jain 0.60). Keep that?

**Cause:** our substitution, not Libra. Libra's set A comes from ContextFL's *time-varying* resource signal (battery, network, load), so A rotates. Our stand-in was a *fixed* speed rank, which excluded the same slowest 6 clients forever.

**Options:** (a) keep the static set: unfair to Libra's fairness numbers · **(b) noisy resource score per refresh** ← chosen · (c) `good_frac` = 1: removes the resource bias that defines mechanism (c).

**Decision:** at every selection refresh, each client's resource score = delay × lognormal(0, σ), drawn from the strategy's seeded RNG. A = the best-scoring 70%. `--resource-noise 0.5` (0 gives the old static set).

**Effect (seed 0, ~6 refreshes):** 4 of the 6 statically excluded clients enter A at least once. The 2 that never do sit at the 20× delay cap (~60 s per epoch), i.e. genuine extreme stragglers. A client newly in A is picked first, because an unseen loss counts as +∞ (D14).

Deviation → README "known deviations". Test: `test_resource_noise_rotates_good_set`.

---

## D19 — Sinkhorn threshold L from Libra pilot 2 (2026-11-20)

**Pilot 2** (Libra only, T = 74.4 s, rotating A, L unset, 1200 s): best 0.881, final 0.880, 455 uploads, 14 HALTs (11 `round_end`, 3 `stale_done`), 0 rejects, Jain 0.58, 15% never participated (pilot 1: 30%).

**Sinkhorn distances** at uploads ≥ 3: n = 109 (pilot 1: 0) · p25 0.142 · **median 0.181** · p75 0.231.

**Decision:** **L = 0.18** (median, as planned in D6). Roughly half of the 3rd+ uploads in a round get halted. Passed explicitly in `experiments.sh` (`--sinkhorn-L 0.18`) so the matrix records it. The config default stays None so pilots only log.

**Findings to carry into the report**
- With no L, fast clients upload up to 4× per round, so participation is skewed (Jain 0.58). This is the behaviour Libra's limit targets, and E1 shows whether L = 0.18 fixes it.
- Mechanism (b), stale admission, is nearly idle at this scale: 3 stale uploads in 1200 s, all admitted. We don't distort the setup to force it. E3 (more heterogeneity → more stragglers) is where it gets exercised.

---

## Open — next questions
- None blocking E1. After E1: E3 heterogeneity levels, E5 ablations. Revisit after the pilot run: final L, T, wall-time budget, target accuracy, preset choice per algorithm (D6, D8).
