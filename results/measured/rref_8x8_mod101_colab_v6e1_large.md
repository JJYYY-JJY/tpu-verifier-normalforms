# RREF 8x8/F_101 Measured Run: colab-v6e1-large

- Status: `ok`
- Backend: `tpu`
- Required backend: `tpu`
- Local devices: `1`
- Selected batch size: `512`
- Train final loss: `1.83338`
- Checkpoint step: `2000`
- Total wall time seconds: `743.259`

## Calibration

| batch | status | samples/sec | seconds |
| ---: | :--- | ---: | ---: |
| 128 | ok | 101.745 | 6.290 |
| 256 | ok | 572.447 | 2.236 |
| 512 | ok | 855.614 | 2.992 |

## Benchmark

| policy | success rate | status counts | samples/sec |
| :--- | ---: | :--- | ---: |
| leftmost | 1.000 | `{"success": 512}` | 1259.444 |
| neural | 0.000 | `{"max_steps_exceeded": 512}` | 0.722 |

No hidden teacher fallback: leftmost is reported only as an explicit baseline; neural failures remain neural status counts.
