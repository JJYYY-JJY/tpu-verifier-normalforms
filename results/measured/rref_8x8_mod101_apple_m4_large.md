# RREF 8x8/F_101 Measured Run: apple-m4-large

- Status: `ok`
- Backend: `cpu`
- Required backend: `cpu`
- Local devices: `1`
- Selected batch size: `512`
- Train final loss: `1.78481`
- Checkpoint step: `2000`
- Total wall time seconds: `279.361`

## Calibration

| batch | status | samples/sec | seconds |
| ---: | :--- | ---: | ---: |
| 128 | ok | 260.703 | 2.455 |
| 256 | ok | 1730.678 | 0.740 |
| 512 | ok | 2882.348 | 0.888 |

## Benchmark

| policy | success rate | status counts | samples/sec |
| :--- | ---: | :--- | ---: |
| leftmost | 1.000 | `{"success": 512}` | 2097.588 |
| neural | 0.000 | `{"max_steps_exceeded": 512}` | 2.313 |

No hidden teacher fallback: leftmost is reported only as an explicit baseline; neural failures remain neural status counts.
