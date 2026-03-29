# Experiments Module

End-to-end pipelines for adversarial example generation and detection.

## STRUCTURE

```
experiments/
├── README.md       # Detailed usage docs
├── run_mnist.py    # MNIST pipeline (argparse: -a, -b)
├── run_toy.py      # Toy 2D dataset pipeline (hardcoded)
└── run_tda.py      # TDA topological analysis (argparse: -d, -n)
```

## WHERE TO LOOK

| Task | Script | Command |
|------|--------|---------|
| Run MNIST | `run_mnist.py` | `python experiments/run_mnist.py -a fgsm` |
| Run Toy | `run_toy.py` | `python experiments/run_toy.py` |
| Run TDA | `run_tda.py` | `python experiments/run_tda.py -d mnist -n 500` |

## OUTPUT FORMAT

Characteristic files (`.npy`) saved with last column = label:
- `1` = adversarial example
- `0` = clean/noisy example

```python
data = np.load('results/mnist/lid_fgsm.npy')
X = data[:, :-1]  # Characteristics
y = data[:, -1]   # Labels (1=adversarial, 0=clean)
```

## CONVENTIONS

- Each script is standalone (uses `sys.path.append` for imports)
- MNIST/TDA use argparse; Toy is hardcoded (runs all attacks)
- Output saved to `results/{dataset}/`
- Ripser required for TDA analysis

## ANTI-PATTERNS

- **DO NOT** import from `visualizer/` in experiment scripts
- **ALWAYS** specify attack with `-a` flag for MNIST
