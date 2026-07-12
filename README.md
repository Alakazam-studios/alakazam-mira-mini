# alakazam-mira

Play **MIRA** — a neural world model of car soccer, every frame generated live by the model —
locally on your own GPU. No cloud, no account; after the first weight download everything runs
on your machine (NVIDIA CUDA, or Apple Silicon via MPS).

```
pip install alakazam-mira
mira play
```

Our from-scratch reproduction of the MIRA recipe (General Intuition × Kyutai), distilled
(fewer diffusion steps, smaller student, compact decoder) until it runs on consumer hardware.
