# Controlled throughput benchmark (cuda, bf16)

Full training step on random tokens, 3 interleaved rounds x 40 steps per setting, median reported.

| setting | batch x context | median step (ms) | tokens/s | peak memory (MB) |
|---|---|---|---|---|
| depth/2 layers | 64 x 128 | 27.0 | 303,744 | 185 |
| depth/4 layers, heads/4 heads, context/ctx 128, position/sinusoidal | 64 x 128 | 44.5 | 183,932 | 331 |
| depth/6 layers | 64 x 128 | 71.4 | 114,739 | 476 |
| heads/2 heads | 64 x 128 | 44.5 | 184,191 | 275 |
| heads/8 heads | 64 x 128 | 39.5 | 207,328 | 472 |
| context/ctx 64 | 128 x 64 | 49.7 | 164,958 | 275 |
| context/ctx 256 | 32 x 256 | 45.3 | 180,987 | 472 |
| position/learned | 64 x 128 | 48.7 | 168,056 | 331 |
| position/none | 64 x 128 | 45.4 | 180,600 | 331 |
