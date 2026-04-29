# RREF 8x8/F_101 Measured Run: local-smoke

- Status: `ok`
- Backend: `cpu`
- Required backend: `None`
- Local devices: `1`
- Selected batch size: `4`
- Train final loss: `13.3651`
- Checkpoint step: `2`
- Total wall time seconds: `2.880`

## Calibration

| batch | status | samples/sec | seconds |
| ---: | :--- | ---: | ---: |
| 2 | ok | 1.220 | 1.639 |
| 4 | ok | 9.092 | 0.440 |

## Benchmark

| policy | success rate | status counts | samples/sec |
| :--- | ---: | :--- | ---: |
| leftmost | 1.000 | `{"success": 2}` | 2154.690 |
| neural | 0.000 | `{"max_steps_exceeded": 2}` | 5.688 |

No hidden teacher fallback: leftmost is reported only as an explicit baseline; neural failures remain neural status counts.
