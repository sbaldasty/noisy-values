# Noise tracking in differentially private data

Work in progress!

https://chatgpt.com/c/69baecdd-f518-832e-8309-62f7a3a23167

## Current prototype API

`NoisyValue` tracks:
- A symbolic random expression (`expr`)
- One realized observation (`observed`)
- Latent symbols (`thetas`) constrained by algebraic equations (`equations`)

### Construction

Use any SymPy distribution-backed random variable:

```python
import sympy as sp
from sympy.stats import Exponential

x = NoisyValue.gaussian(10, 1, provenance="A")
y = NoisyValue.from_distribution(5, Exponential, 2, provenance="B")
# equivalent lower-level API:
# rv = Exponential("E0", 2)
# y = NoisyValue.from_noise_rv(5, rv, provenance="B")
```

### Sampling

`sample_n` supports mixed distributions without per-distribution branching:

```python
z = x * y
samples = z.sample_n(1000, seed=123)
```

By default, sampling uses independent noise draws for:
- latent-theta reconstruction from constraints
- predictive expression evaluation

This keeps uncertainty propagation working for arbitrary SymPy random variables.

### Optional symbolic cloning

If you need symbolic elimination with fresh noise symbols, pass a `noise_cloner`
to `eliminate_thetas` or `sample_n`. The cloner must map one random variable to
one random variable.

### Important semantic note

This is algebraic elimination + forward Monte Carlo, not exact Bayesian
conditioning.
