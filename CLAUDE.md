# CLAUDE.md — SEAFL vs Libra (semi-async FL comparison)

DS course project, Option 1, **solo**. Compare SEAFL (weight stale updates) vs Libra (gate who participates) on one real message-passing system.
- Spec: `requirements.md` · Every design decision + why: `decisions.md` (wins over requirements.md on conflict).
- Deadlines: POC **23 Nov 2026** (2nd presentation) · final code + IEEE report **30 Nov 2026**.

## Workflow rules
- New design decision → discuss as Q&A, then append a `D<n>` entry to `decisions.md` (question, options, decision, why). Never decide silently in code.
- Scope/spec change → update `requirements.md` and add a Changelog line.
- Before saying anything is done: `ruff format . && ruff check . && pytest` must pass.
- Develop + test on the Mac. Real runs only on the JarvisLabs L4 (billed hourly): launch inside `tmux`, results flush to disk after every eval.
- Never add Claude/AI as `Co-Authored-By` on commits.

## Architecture (locked — see decisions.md D1–D3)
- 1 server process + **N = 20** client processes. They share nothing and talk only by messages over TCP (`multiprocessing.connection` Listener/Client, `authkey` always set).
- Server: one reader thread per client → one `queue.Queue` → a **single main loop** that alone touches strategy state. No locks in algorithm code.
- Clients: CPU, `torch.set_num_threads(1)`. Server: GPU if available (aggregation, eval, Sinkhorn).
- Heterogeneity = injected delays: `base × min(1 + Pareto(a=1.5), 20) × time_scale`, seeded per client.
- Single clock: server `time.monotonic()`. Clients never send timestamps.
- Eval on a 5 s wall-time timer, not per aggregation. Runs stop at a fixed wall-time budget. Never compare on "rounds".

## Layout
```
flsim/
  config.py        argparse, all hyperparams, --preset paper|scaled
  data.py          torchvision FMNIST/CIFAR-10, Dirichlet(β) split, D_val carve-out
  model.py         small CNN
  transport.py     Listener/Client helpers, message constructors
  client.py        client process: train E epochs, sleep injected delay, upload
  server.py        main loop, eval timer, metrics logging
  metrics.py       CSV/JSON writers, Jain's index, participation stats
  strategies/      fedavg.py  seafl.py  libra.py   (FedBuff/FedAsync/ablations = flags)
  run.py           launcher: spawns server + N clients, dumps config.json
plots.py           reads results/ only
experiments.sh     the experiment matrix, one line per run
tests/
results/<exp>/<algo>/<seed>/   config.json, evals.csv, aggregations.csv
```
Don't add files or packages beyond this without a `decisions.md` entry.

## Contracts (fixed before coding — change only via decisions.md)
**Messages** are plain dicts with a `"type"` key:
| Direction | type | fields |
|---|---|---|
| C→S | `HELLO` | `cid` |
| C→S | `UPDATE` | `cid, base_version, weights (state_dict), n_samples, loss, epochs_done, upload_idx` |
| S→C | `TRAIN` | `version, weights, epochs, policy (dict: Libra ϑ/ρ, etc.)` |
| S→C | `NOTIFY` | — (SEAFL²: finish current epoch, upload now) |
| S→C | `HALT` | — (Libra: stop training until the next `TRAIN`) |
| S→C | `SHUTDOWN` | — |
| internal | `DISCONNECT` | `cid` — pushed by a server reader thread on EOF |

Wire format: `conn.send_bytes(pickle.dumps(msg))` / `pickle.loads(conn.recv_bytes())`, so `transport.send/recv` return byte counts (comm metric). Weights are always CPU tensors on the wire.

**Strategy** = plain class, no base class / ABC. Pure state machine, does no I/O, returns the messages to send:
```python
def start(self, now: float) -> list[tuple[int, dict]]                  # initial TRAIN broadcasts
def on_update(self, msg: dict, now: float) -> list[tuple[int, dict]]   # an UPDATE arrived
def on_tick(self, now: float) -> list[tuple[int, dict]]                # timers: deadlines, β waits
global_model  # attribute the server evaluates
events        # list[dict], kind ∈ agg/reject/halt/notify/select; server drains → CSV (D13)
```
Strategies build message dicts literally per the table above (they don't import transport).
Heterogeneity draws: `config.client_delays(n, seed, base, cap=20)`, shared by server and clients.
The algorithm math (γ, s, p_k, ϑ/ρ sort, Sinkhorn, selection) lives in **module-level pure functions** so pytest can check it without processes.

## Code style (graded: documentation + indentation)
1. `ruff format` (4-space indent, line length 100) + `ruff check`. Python 3.10+, type hints on every signature.
2. Every file starts with a module docstring: what it does, and where it sits in the system.
3. Each strategy file starts with a **paper → code map**, e.g. `Alg. 1 line 7 / Eq. 4 → staleness_factor()`.
4. Every class and function gets a Google-style docstring (Args / Returns) that names the paper symbol it implements (γ, ϑ, ρ, σ, τ_max, …).
5. Inline comments explain **why**, not what. Every variable that is a paper symbol gets one: `gamma = alpha * beta / (s + beta)  # γ, SEAFL Eq. 4`.
6. Deliberate simplifications are marked `# ponytail: <what>, <ceiling>, <upgrade path>`.
7. Values the papers don't give (Libra T, L, Sinkhorn cost/ε, D_val size; SEAFL delta-vs-weights) carry a comment that points to the decisions.md entry (D6/D7) and go in the README "known deviations" section.
8. Keep it boring: stdlib + torch/torchvision/numpy/matplotlib only. No new dependency without a decisions.md entry.

## Tests (pytest, only these)
Dirichlet split (sizes and coverage) · SEAFL γ(S=0)=α, γ(S=β)=α/2, weights sum to 1 · Libra participation sort → ϑ/ρ · Sinkhorn on known inputs (identical dists → ~0) · message send/recv round-trip.

## Commands
```bash
ruff format . && ruff check . && pytest                       # gate
python -m flsim.run --algo seafl --preset scaled --seed 0 --out results/e1/seafl/0
bash experiments.sh                                           # full matrix (on Jarvis, in tmux)
python plots.py results/e1                                    # figures from saved results only
```
