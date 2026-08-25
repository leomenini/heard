# Latency report

All numbers from the i5-1335U laptop (no GPU), Arch Linux, PipeWire + Hyprland.
Fill tables by running:

```bash
uv run python scripts/benchmark_latency.py                 # everything
uv run python scripts/benchmark_latency.py --only intent   # fast vs generative
uv run python scripts/benchmark_latency.py --only stt      # finalize + STT
uv run python scripts/benchmark_latency.py --only dispatch # dry + real spawns
uv run python scripts/benchmark_latency.py --audio cmd.wav # STT on a real clip
```

## Pipeline stages (post key-release)

Numbers from i5-1335U unless marked *container*. The classify row in the
first benchmark build double-encoded (measurement artifact); true slot+gate
cost is sub-ms — remeasured by the fixed benchmark.

| Stage | p50 | p95 | notes |
|---|---|---|---|
| audio finalize | _pending_ | _pending_ | concat ring buffer + VAD trim, ~4s clip |
| STT (tail after last pause) | _pending_ | _pending_ | faster-whisper base int8, beam 1 |
| STT (full clip, no streaming) | _pending_ | _pending_ | what pre-streaming v0.1 paid |
| intent: embed query (MiniLM int8) | 53.0ms | 101.1ms* | *p95/max include first-call warmup |
| intent: classify + slots | <1ms* | <1ms* | *remeasure with fixed benchmark |
| intent: generative fallback | 6474.4ms | 6864.0ms | constrained decode; see split below |
| dispatch (dry) | ~0.1ms | ~0.1ms | registry validate + handler entry (container) |
| dispatch spawn (wpctl) | 38.2ms | — | container measurement |
| **end-to-end** | _pending_ | _pending_ | key-release -> feedback line |

Generative tail split (encode vs decode loop): fill from the
"generative tail split" section of `--only intent`.

## Fast-path accuracy (probe set)

Run of 2026-08-21, defaults (accept 0.55 / margin 0.04 / floor 0.35):

| Metric | Value |
|---|---|
| accepted by classifier | 17/20 |
| correct tool when accepted | 17/17 |
| misaccepts (wrong tool dispatched) | **0** |
| wrongly declined (should have fallen back) | **0** |
| fell back to generative | 3/20 (`Launch the calculator`, `Start VS Code`, `Focus on the browser`) |
| fallback accuracy on its share | 3/3 |
| off-topic rejection | 4/4 declined via unknown centroid |
| score range of correct accepts | 0.604–0.950 (median 0.871) |
| scores of the 3 fallbacks | 0.540 / 0.496 / 0.379 |

After the dictionary-corroborated launch gate + antonym prototypes
(local verification): **19/20 accepted**, still zero misaccepts/declines;
only `Focus on the browser` remains on fallback by design.
Confirm with `--only intent` on hardware.

Closest centroid pairs (watch for regressions):
`volume_control:up <-> down` 0.908 -> 0.887 after new prototypes;
next worst `media_control:next <-> previous` ~0.78.

Thresholds live in `heard/classifier.py` (`FLOOR_DECLINE`, `ACCEPT_SCORE`,
`MARGIN`, `DICT_FLOOR`, `DICT_CUTOFF`) — all env-overridable for live tuning.

## Dictionary rescue tuning

Measured on **held-out** launch phrasings ("open <real installed app>", names
absent from `PROTOTYPES`) with fastembed MiniLM: 12 launch probes, 19
off-topic. The benchmark probe set cannot detect `launch_app` regressions on
its own: its English `launch_app` probes are near-verbatim prototype
utterances and pass by construction.

| config | launch accepted | misaccepts | off-topic misaccepts |
|---|---|---|---|
| baseline (rescue off) | 3/12 | 0 | 0 |
| rank<=3, floor 0.35 | 7/12 | 0 | 0 |
| **rank<=3, floor 0.30** | **9/12** | **0** | **0** |
| rank<=3, floor 0.25 | 11/12 | 0 | 2 |
| rank<=5, floor 0.30 | 9/12 | 0 | 1 |

`RESCUE_RANK` is nearly inert; `RESCUE_FLOOR` is the load-bearing knob and the
cliff is sharp. Both failures at 0.25 are the same shape: a sentence
*mentioning* an app read as a command to launch it (`add vlc to my shopping
list` -> `launch_app`). These are fastembed geometry numbers and do not
transfer to a trained Needle head.

## Minimize centroid

Motivated by a real misaccept on Cinnamon/X11: "minimizar ventana actual"
resolved to `window_action:close` at 0.69 (destructive) because no minimize
centroid existed. After adding one (8 bilingual prototypes), fastembed
MiniLM multilingual:

| probe set | result |
|---|---|
| held-out minimize phrasings (4, none in PROTOTYPES) | all -> minimize, scores 0.649–0.954 |
| the former misaccept phrase | close 0.69 -> **minimize 0.935** |
| close regression (4 phrases EN+ES) | all -> close, 0.814–0.956 |
| off-topic | **0 misaccepts** |

("cuánto falta para las cinco" resolves to `system_query:time`, which is the
correct dispatch, not a misaccept.)

## Generative Needle probes

See the CONSTRAINED / UNCONSTRAINED sections of the benchmark output; paste a
trimmed copy here per finetune.

## Spawn costs (reference)

`wpctl get-volume` ~38ms, `hyprctl activewindow` ~19ms in-container; expect
similar on the laptop. MPRIS commands go over one persistent D-Bus connection
(no spawn at all).
