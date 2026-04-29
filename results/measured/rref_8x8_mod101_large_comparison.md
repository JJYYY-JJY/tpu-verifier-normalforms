# RREF 8x8/F_101 Large Measured Comparison

Inputs:

- `results/measured/rref_8x8_mod101_apple_m4_large.json`
- `results/measured/rref_8x8_mod101_colab_v6e1_large.json`

Both runs used shard count 4096, train steps 2000, benchmark count 512,
hidden sizes `[512, 512]`, max steps 72, and checkpoint cadence 50.

## Summary

| run | backend | devices | selected batch | status | total seconds | train seconds | benchmark seconds |
| :--- | :--- | ---: | ---: | :--- | ---: | ---: | ---: |
| Apple M4 | cpu | 1 | 512 | ok | 279.361 | 51.789 | 222.352 |
| Colab v6e-1 | tpu | 1 | 512 | ok | 743.259 | 16.536 | 711.163 |

## Training

| run | final loss | checkpoint step | train samples/sec proxy |
| :--- | ---: | ---: | ---: |
| Apple M4 | 1.784809 | 2000 | 19772.382 |
| Colab v6e-1 | 1.833382 | 2000 | 61927.125 |

Colab v6e-1 training throughput was 3.132x Apple M4 by the harness proxy.

## Benchmark

| run | policy | success rate | status counts | samples/sec |
| :--- | :--- | ---: | :--- | ---: |
| Apple M4 | leftmost | 1.000 | `{"success": 512}` | 2097.588 |
| Apple M4 | neural | 0.000 | `{"max_steps_exceeded": 512}` | 2.313 |
| Colab v6e-1 | leftmost | 1.000 | `{"success": 512}` | 1259.444 |
| Colab v6e-1 | neural | 0.000 | `{"max_steps_exceeded": 512}` | 0.722 |

No hidden teacher fallback: leftmost is an explicit baseline, and neural failures
remain visible as neural status counts. In both measured runs, all 512 neural
benchmark samples reached `max_steps_exceeded` under `max_steps=72`.

## Interpretation

Facts:

- Both measured runs completed with `status: ok`.
- Both selected batch size 512 during calibration.
- Both reached checkpoint step 2000.
- The TPU run trained faster: 16.536s vs 51.789s.
- The TPU run was slower end to end: 743.259s vs 279.361s.
- The TPU run's benchmark stage dominated total time: 711.163s.

Inference:

- For this current harness, Colab v6e-1 accelerates the large training loop but
  does not accelerate the end-to-end measured run, because benchmark rollout is
  the dominant stage and is slower in the Colab run.

Open next step:

- Improve learned RREF rollout quality or benchmark batching before treating TPU
  training speed as end-to-end system speedup.
