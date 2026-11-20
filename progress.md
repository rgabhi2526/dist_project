# progress.md: What has been built, and how

> Build log and reference. Read this to see **what exists, how it works, how to run it, and what is left**.
> The *why* behind each choice is in `decisions.md` (D1, D2, …). The spec is in `requirements.md`. The rules are in `CLAUDE.md`.
> Update this file at the end of every work session (newest entry at the bottom of the log).

---

## 1. Status at a glance

| Area | Status | Where |
|---|---|---|
| Planning (Q&A, 15 decisions) | ✅ done | `decisions.md` D1–D15 |
| Config, data, model | ✅ built + tested | `flsim/config.py`, `data.py`, `model.py` |
| Transport (TCP messaging) | ✅ built + tested | `flsim/transport.py` |
| SEAFL (+ SEAFL², FedBuff, FedAsync flags) | ✅ built + tested | `flsim/strategies/seafl.py` |
| Libra (+ -NoL / -RS / -SS flags) | ✅ built + tested | `flsim/strategies/libra.py` |
| FedAvg baseline | ✅ built + tested | `flsim/strategies/fedavg.py` |
| Client process, server loop, launcher | ✅ built, smoke-tested on Mac | `flsim/client.py`, `server.py`, `run.py` |
| Metrics + plots | ✅ built + tested | `flsim/metrics.py`, `plots.py` |
| Experiment matrix | ✅ pilot + E1, resumable (skips complete runs) | `experiments.sh` |
| Verbose live output | ✅ banner, per-eval line, per-event line, client lines, end summary | `--log-level INFO\|DEBUG` |
| Sanity run on Jarvis L4 (20 clients, Libra, 120 s) | ✅ passed | see log 2026-11-20 |
| Pilot run (picks L, T, budget, target, preset) | ✅ done → D17 (SEAFL `scaled`, target 0.80, budget 1200 s) | `results/pilot/` |
| Libra pilot 2 (new T, rotating A → pick L) | ✅ done → L = 0.18 (D19) | `results/pilot2/libra/0` |
| E1 headline run (3 seeds) → POC plots | ✅ done (results below, figures on Jarvis) | `results/e1/` |
| README (setup, reproduce, paper→code, deviations) | ⬜ todo | |
| E2–E5 (CIFAR, sweeps, ablations) | ⬜ P1 | requirements.md §4 |
| Report data pack for the teammate | ✅ done | `report/REPORT_DATA.md`, `report/figures`, `report/tables`, `report/data/e1` |
| IEEE report | ⬜ teammate writing it | |

Gate right now: `ruff format . && ruff check . && pytest` → **54 tests pass**, ruff clean. About 4,000 lines.

---

## 2. How the system works (one picture)

```
                     ┌───────────────────────── server process (run.py → server.py) ─────────────────────────┐
                     │                                                                                        │
 client 0 ──TCP──▶ reader thread ─┐                                                                           │
 client 1 ──TCP──▶ reader thread ─┼─▶ queue.Queue ─▶ SINGLE MAIN LOOP ──▶ strategy.on_update(msg, now)        │
   ...                            │                   │   every iteration ──▶ strategy.on_tick(now)          │
 client 19 ─TCP──▶ reader thread ─┘                   │   every 5 s       ──▶ evaluate global model (GPU)    │
                     │                                │   returns (cid, msg) list ──▶ transport.send ──TCP──▶ clients
                     │                                └─▶ drain strategy.events ──▶ events.jsonl / evals.csv  │
                     └────────────────────────────────────────────────────────────────────────────────────────┘

 client process (client.py), ×20, CPU, 1 thread:
   inbox thread drains socket ─▶ local queue
   on TRAIN: load weights → for each epoch: train → sleep(injected delay) → check inbox (NOTIFY/HALT/TRAIN)
             → upload UPDATE → (Libra multi-upload: keep training until HALT / σ_max) → idle
```

**Key ideas**
- **Real distributed system** (D1). There are 21 OS processes. They share no memory and communicate only through messages over TCP (`multiprocessing.connection`, authkey-protected, D2).
- **Strategies are pure state machines** (D12). `start / on_update / on_tick` return the messages to send and do no I/O. That is why every algorithm can be unit-tested without starting any process.
- **One main loop owns all strategy state.** It needs no locks. Reader threads only enqueue (D2).
- **Heterogeneity is injected.** Each client sleeps `base_delay × min(1 + Pareto(1.5), 20) × time_scale` per epoch. The draws are seeded, so server and clients see the same values (D6, D13).
- **One clock.** All timestamps come from the server's `time.monotonic()`. Clients never send time (D8).
- **Comparison axes are wall-clock time and number of messages**, never "rounds". Every algorithm is evaluated on a 5 s timer and gets the same time budget (D8).

---

## 3. What each file does

| File | Lines | What it does | Key functions |
|---|---|---|---|
| `flsim/config.py` | 191 | Every hyperparameter as an argparse flag. `--preset paper\|scaled`. Heterogeneity model. | `parse_args`, `client_delays`, `save_config` |
| `flsim/data.py` | 120 | FMNIST/CIFAR-10 loading, Dirichlet(β) non-IID split, test / D_val split | `load_dataset`, `dirichlet_partition`, `split_test_dval` |
| `flsim/model.py` | 121 | Small CNN (105,866 params for FMNIST, no BatchNorm), training and evaluation | `make_model`, `train_epoch`, `evaluate` |
| `flsim/transport.py` | 281 | TCP messaging. Byte-counting send/recv, connect with retries, HELLO handshake, reader threads, message constructors | `send`, `recv`, `listen`, `connect`, `accept_clients`, `start_readers` |
| `flsim/strategies/seafl.py` | 391 | SEAFL: staleness factor γ, importance s (delta cosine), weights p_k, mixing ϑ, β-wait, SEAFL² NOTIFY. FedBuff and FedAsync are flags. | `staleness_factor`, `importance_factor`, `aggregation_weights`, `SEAFLStrategy` |
| `flsim/strategies/libra.py` | 610 | Libra: Sinkhorn fast-device limit, participation-based stale admission (ϑ, ρ), biased client selection, deadline rounds. Ablation flags. | `sinkhorn_distance`, `assign_admission`, `early_drop`, `biased_selection`, `LibraStrategy` |
| `flsim/strategies/fedavg.py` | 171 | Synchronous FedAvg baseline | `FedAvgStrategy` |
| `flsim/client.py` | 207 | One client process: inbox thread, local training, injected delay, uploads | `client_main` |
| `flsim/server.py` | 143 | The single main loop: dispatch, ticks, timer eval, logging, shutdown | `serve` |
| `flsim/metrics.py` | 169 | evals.csv / events.jsonl / summary.json writers, Jain's index, participation | `EvalLogger`, `jain_index`, `write_summary` |
| `flsim/run.py` | 171 | Launcher. Downloads data once, spawns 20 clients, builds the strategy, resolves Libra's deadline T, cleans up | `main` |
| `plots.py` | 583 | All report figures and tables, built from saved results only | `main`, `time_to_target`, `plot_curves`, … |
| `experiments.sh` | 42 | Experiment matrix: `pilot` and `e1` | |
| `tests/` | 8 files, 53 tests | data 7 · model 4 · transport 8 · SEAFL 7 · Libra 17 · FedAvg 2 · metrics 3 · plots 5 | |

Each strategy file starts with a **paper → code map** (each equation or algorithm line → its function).

---

## 4. How to run things

```bash
# Mac (dev): checks
ruff format . && ruff check . && pytest

# One run (any machine)
python -m flsim.run --algo seafl --preset scaled --seed 0 --epoch-compute-s 0.72 --out results/e1/seafl/0
#   --algo fedavg | seafl | libra
#   FedBuff:  --algo seafl --uniform-weights --stale-bound inf --theta 1
#   FedAsync: same as FedBuff + --K 1
#   SEAFL²:   --algo seafl --partial
#   Libra ablations: --no-limit | --random-select | --staleness-select

# Experiment matrix (Jarvis, inside tmux)
tmux new -s pilot
PY=python EXTRA="--epoch-compute-s 0.72" bash experiments.sh pilot 2>&1 | tee pilot.log
PY=python EXTRA="--epoch-compute-s 0.72" bash experiments.sh e1    2>&1 | tee e1.log

# Figures + tables → results/<exp>/figures/ (PNG + PDF, CSV + MD)
python plots.py results/pilot
```

**Output of every run:** `results/<exp>/<algo>[_<preset>]/<seed>/`
- `config.json`: every resolved argument, including Libra's computed deadline
- `evals.csv`: `t, version, acc, loss, n_up, n_down, n_ctrl, bytes_up, bytes_down` (one row every 5 s)
- `events.jsonl`: every agg / reject / halt / notify / select / sinkhorn event
- `summary.json`: final and best accuracy, participation counts, Jain's index, message totals

**Jarvis setup:**
- Machine: L4, 28 vCPU, 124 GB RAM.
- Environment: uv env `~/myenv`, repo in `~/dist_project`.
- FMNIST: if the S3 mirror fails, take the raw `.gz` files from the zalandoresearch GitHub repo and put them in `data/FashionMNIST/raw/`.

---

## 5. Measured numbers so far

| What | Value | Where measured |
|---|---|---|
| One local epoch (3000 samples, batch 32, 1 thread) | 1.08 s Mac · **0.72 s Jarvis** | `--epoch-compute-s` |
| Eval of the test set | 2–3 s Mac CPU (delays Libra ticks) · fine on the Jarvis GPU | D15 |
| Libra deadline T (20 clients, base delay 3) | ≈ 33 s → 3 rounds in 120 s | Jarvis sanity run |
| Libra sanity, 120 s | 37 uploads, 0 rejects, 4 control msgs, **acc 76.8%** | Jarvis |

---

## 5b. E1 results (FMNIST, N = 20, Dirichlet 0.5, 1200 s, seeds 0–2, mean ± std)

| | FedAvg | SEAFL (scaled) | Libra (L = 0.18) |
|---|---|---|---|
| Best acc | 0.832 ± 0.021 | 0.870 ± 0.013 | **0.882 ± 0.002** |
| Final acc | 0.830 ± 0.019 | 0.859 ± 0.015 | **0.880 ± 0.002** |
| Fluctuation (std last 5 evals) | 0.0000 | 0.0028 | 0.0014 |
| Variance of P | 1.7 | 68 | 240 |
| Jain index | **0.897** | 0.872 | 0.677 |
| % never participating | 3.3 | **0.0** | 10.0 |
| Uploads / downloads / bytes vs FedAvg | 1× / 1× / 1× | 4.29× / 4.30× / 4.30× | 4.51× / 2.03× / 3.26× |
| Control msgs (total, 3 seeds) | 0 | 0 | 56 HALTs |

Libra events over the 3 seeds: 115 Sinkhorn halts, 48 round-end HALTs, 6 stale uploads, 1 ϑ reject.

**Takeaways**
- Libra is the most accurate and the most stable across seeds. SEAFL is 1.2 points behind with 6× the seed variance. Both beat FedAvg by 4–5 points.
- On fairness the ranking flips: SEAFL includes everyone (Jain 0.87), while Libra's fast clients still dominate (Jain 0.68, up from 0.58 without L), because uploads 1–2 per round are always free.
- Libra uses fewer bytes than SEAFL: a repeat upload in the same round needs no new download.
- Libra's stale admission (mechanism b) is almost idle at this heterogeneity, so it gets tested in E3.

## 6. Known limitations / things to watch
- Results use 20 clients (the papers use 100). Parameters are scaled down (D6).
- Libra's paper does not give T, L, the Sinkhorn cost, or the good-resource set. Our choices are in D6 and D13 and go in the README under "known deviations".
- The SEAFL `paper` preset `n_active = N/2` is a placeholder. **Verify it against the SEAFL paper** (D14).
- FedAsync is FedBuff with K = 1, without the paper's staleness-decayed mixing (D4).
- Eval runs inside the main loop. If Libra ticks are late on Jarvis, move eval to its own thread (D15).
- SEAFL and FedAvg aggregate on CPU (small model, so cheap). Only Libra's model goes to the GPU (D15).

---

## 7. Build log (newest at bottom)

### 2026-11-19: Planning
- Interactive Q&A produced decisions D1–D12. Switched from a virtual-clock simulator to a real multi-process system on one JarvisLabs L4. Fixed scope, parameters, metrics and code-style rules.
- Created `decisions.md` and `CLAUDE.md`, and updated `requirements.md` to v0.2.

### 2026-11-19: Wave 1: foundations (4 parallel Sonnet agents)
- Built config/data/model, transport, SEAFL, Libra, each with tests. Gate: 44 tests.
- Implementation interpretations recorded as D13 (Libra gaps, contract details) and D14.

### 2026-11-19: Wave 2: end-to-end system (Sonnet agent)
- Built client, server loop, FedAvg, metrics, launcher, `experiments.sh`.
- Mac smoke runs (4 clients): all algorithms run, accuracy rises, no orphan processes.
- **Bugs found and fixed (D15):** (1) the server could block while sending to a busy client → added a client inbox thread. (2) Libra's automatic deadline ignored compute time → the formula now includes `--epoch-compute-s`. The default base delay was raised to 3.0.

### 2026-11-19: Plots (Sonnet agent)
- `plots.py` produces every figure and table in requirements §6. Gate: 54 tests. Commits: `6ec37b5` initial, `70b3003` plots.

### 2026-11-20: Jarvis bring-up
- Measured epoch time 0.72 s. A 120 s Libra sanity run with 20 clients passed (rounds on time, 76.8% accuracy).
- **Next:** pilot run → fix L, T, budget, target, preset in decisions.md → E1 with 3 seeds → POC plots for 23 Nov.

### 2026-11-20: Pilot (partial) + verbose, resumable runs
- Pilot on Jarvis: fedavg, libra and seafl_paper finished. seafl_scaled was stopped with Ctrl-C at start-up and has no results.
- **Verbose output** (`--log-level INFO`, default). The launcher prints a start banner with all key parameters (incl. Libra's T), "dataset loaded", "N clients connected", and an end summary (accuracy, messages, MB, rejects/halts, Jain).
  - The server prints one line per 5 s eval (`t=…/budget v= acc= loss= | up down ctrl | agg rej halt`) and one line per event (agg, reject, halt, notify, select).
  - Each client prints `client.NN` lines: connected, TRAIN, upload (loss, round time), NOTIFY/HALT, shutdown.
  - `--log-level DEBUG` adds per-epoch, per-message and Sinkhorn lines.
- **Resumable runs:** summary.json has `complete` (true only when the budget ended normally, so a crash or Ctrl-C gives false). `experiments.sh` prints `SKIP` for complete runs, `RERUN` (after deleting) for others, and start/end lines per run.
- `plots.py` ignores dot-dirs (`.ipynb_checkpoints` from Jupyter on the box).
- The agent doing this was stopped part-way; the rest was finished directly. Gate: 54 tests, ruff clean. Mac smoke + skip test verified.

### 2026-11-20: Start-up hang fixed (D16)
- seafl_scaled hung at 13/20 clients connected on Jarvis. Cause: `Listener` backlog defaulted to 1, and Linux silently drops the overflow. Fixed with `backlog=128` plus a 20-client regression test. A 20-client Mac run starts and exits cleanly. Gate: 55 tests.
- The GPU was found full after the earlier Ctrl-C; a pause/resume of the instance freed it. Check free GPU memory before every batch.

### 2026-11-20: Pilot analysed → D17, D18
- Pilot results: libra 0.885, seafl_scaled 0.862, fedavg 0.805, seafl_paper 0.771 (best acc, 1200 s). Chosen: SEAFL `scaled`, target 0.80, budget 1200 s.
- **Libra's mechanisms barely ran in the pilot:** 0 Sinkhorn checks, 2 rejects, ~3 selection refreshes. → Libra T = 4 × fastest local round ≈ 74 s (seed-independent; 6–10 clients/round reach ≥ 3 uploads, 2–4 go stale, ~16 rounds/run), `--select-every 3`, and round-end / post-stale HALTs are now logged as events (D17).
- **30% of clients never participated in Libra** because of our fixed speed-rank stand-in for ContextFL → the good-resource set A now rotates with per-refresh lognormal noise, `--resource-noise 0.5` (D18).
- Gate: 56 tests, ruff clean.

### 2026-11-20: Libra pilot 2 → L (D19)
- With T = 74 s and rotating A: 109 Sinkhorn checks (pilot 1: 0), median distance 0.181 → **L = 0.18**. Libra best 0.881. 15% never participated (was 30%). Jain 0.58, because fast clients upload up to 4×/round when no L is set.
- Stale admission is nearly idle at this scale (3 stale uploads, 0 rejects). It gets tested in E3.
- `experiments.sh` e1: libra gets `--sinkhorn-L 0.18`. All pilot decisions are now closed, so E1 is ready.

### 2026-11-20: E1 done
- 9 runs (3 algorithms × 3 seeds) on the Jarvis L4, ~3.2 h, no failures. Results in §5b. POC figures are in `results/e1/figures/` on Jarvis (to copy to the Mac for the slides).

### 2026-11-21: Report data pack (D20)
- E1 results imported from Jarvis (`results.zip`, kept out of git) into `results/e1`. Tables regenerated at target 0.80.
- `plots.py` (Sonnet agent) adds: fairness in two forms (influence and inclusion), a time-to-target table with messages and MB to target, a hyperparameter table, a raw per-seed table, Libra mechanism counts and `meta.json`. Gate: 60 tests.
- New finding: SEAFL's throughput is tied to the slowest client (43 vs 131 aggregations depending on whether a seed has a capped straggler). Libra is unaffected.
- `report/` = hand-off pack for the report writer: `REPORT_DATA.md` (298 lines, tables pasted verbatim from the generated files), 6 figures (PNG + PDF), 9 tables (MD + CSV), raw E1 data (812 KB).
