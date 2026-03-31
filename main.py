from sympy.stats import Normal
from sympy.stats import sample


import sympy as sp
import numpy as np
from sympy.stats.rv import random_symbols

# ---- global registries ----
_theta_counter = 0
_noise_counter = 0

def fresh_theta(tag=None):
    global _theta_counter
    name = f"theta_{_theta_counter}" if tag is None else f"theta_{tag}_{_theta_counter}"
    _theta_counter += 1
    return sp.Symbol(name)

def fresh_normal(mu=0, sigma=1):
    global _noise_counter
    name = f"N{_noise_counter}"
    _noise_counter += 1
    return Normal(name, mu, sigma)


def fresh_noise_name(prefix="R"):
    global _noise_counter
    name = f"{prefix}{_noise_counter}"
    _noise_counter += 1
    return name


class NoisyValue:
    def __init__(self, expr, observed, thetas=None, equations=None):
        self.expr = sp.sympify(expr)
        self.observed = float(observed)

        self.thetas = set() if thetas is None else set(thetas)

        if equations is None:
            self.equations = [self.expr - self.observed]
        else:
            self.equations = equations

    def __repr__(self):
        return f"NoisyValue(expr={self.expr}, observed={self.observed})"

    @classmethod
    def from_noise_rv(cls, true_value, noise_rv, provenance=None, **sample_kwargs):
        """
        Build a NoisyValue from any SymPy random variable.

        Parameters
        ----------
        true_value : float-like
            Latent true value associated with the observation.
        noise_rv : RandomSymbol
            SymPy random variable added to theta.
        provenance : str | None
            Optional tag included in theta naming.
        sample_kwargs : dict
            Forwarded to sympy.stats.sample when generating the observation.
        """
        noise_symbols = random_symbols(noise_rv)
        if len(noise_symbols) != 1 or noise_rv not in noise_symbols:
            raise TypeError("noise_rv must be a single SymPy random variable")

        theta = fresh_theta(provenance)
        expr = theta + noise_rv
        observed_expr = expr.subs({theta: sp.sympify(true_value)})
        observed = float(sample(observed_expr, **sample_kwargs))

        return cls(expr, observed, thetas={theta})

    @classmethod
    def from_distribution(cls, true_value, dist_builder, *dist_args, provenance=None, name_prefix="R", **dist_kwargs):
        """
        Build a NoisyValue from a SymPy distribution constructor.

        Example: NoisyValue.from_distribution(10, Exponential, 2)
        """
        name = fresh_noise_name(name_prefix)
        noise_rv = dist_builder(name, *dist_args, **dist_kwargs)
        return cls.from_noise_rv(true_value, noise_rv, provenance=provenance)
    
    @classmethod
    def gaussian(cls, true_value, sigma, provenance=None):
        """
        provenance: optional tag to reuse the same θ across calls
        """
        noise = fresh_normal(0, sigma)
        return cls.from_noise_rv(true_value, noise, provenance=provenance)

    @staticmethod
    def _combine(a, b, op):
        if isinstance(b, NoisyValue):
            expr = op(a.expr, b.expr)
            observed = op(a.observed, b.observed)
            thetas = a.thetas | b.thetas
            equations = a.equations + b.equations
            return NoisyValue(expr, observed, thetas, equations)
        else:
            expr = op(a.expr, b)
            observed = op(a.observed, b)
            return NoisyValue(expr, observed, a.thetas, a.equations)

    def __add__(self, other):
        return NoisyValue._combine(self, other, lambda a, b: a + b)

    def __radd__(self, other):
        return NoisyValue._combine(self, other, lambda a, b: b + a)

    def __sub__(self, other):
        return NoisyValue._combine(self, other, lambda a, b: a - b)

    def __rsub__(self, other):
        return NoisyValue._combine(self, other, lambda a, b: b - a)

    def __mul__(self, other):
        return NoisyValue._combine(self, other, lambda a, b: a * b)

    def __rmul__(self, other):
        return NoisyValue._combine(self, other, lambda a, b: b * a)

    def __truediv__(self, other):
        return NoisyValue._combine(self, other, lambda a, b: a / b)

    def __rtruediv__(self, other):
        return NoisyValue._combine(self, other, lambda a, b: b / a)

    def _solve_theta_substitutions(self):
        eqs = [sp.Eq(eq, 0) for eq in self.equations]
        thetas = list(self.thetas)

        sol = sp.solve(eqs, thetas, dict=True)
        if not sol:
            raise ValueError("Could not solve for latent variables")

        return sol[0]

    def eliminate_thetas(self, noise_cloner=None):
        expr = self.expr
        sol = self._solve_theta_substitutions()
        substitutions = {}
        clone_cache = {}

        for theta, rhs in sol.items():
            if noise_cloner is not None:
                replacements = {}
                for rv in random_symbols(rhs):
                    if rv not in clone_cache:
                        cloned = noise_cloner(rv)
                        cloned_symbols = random_symbols(cloned)
                        if len(cloned_symbols) != 1 or cloned not in cloned_symbols:
                            raise TypeError("noise_cloner must return a single SymPy random variable")
                        clone_cache[rv] = cloned
                    replacements[rv] = clone_cache[rv]
                rhs = rhs.subs(replacements)

            substitutions[theta] = rhs

        return expr.subs(substitutions)

    def sample_n(self, n=1000, noise_cloner=None, library="scipy", seed=None, **sample_kwargs):
        if n <= 0:
            return np.array([], dtype=float)

        sample_seed = seed
        if isinstance(seed, int):
            sample_seed = np.random.default_rng(seed)

        if noise_cloner is not None:
            expr = self.eliminate_thetas(noise_cloner=noise_cloner)
            if not random_symbols(expr):
                return np.full(n, float(expr), dtype=float)

            try:
                values = sample(expr, size=n, library=library, seed=sample_seed, **sample_kwargs)
                return np.asarray(values, dtype=float)
            except Exception:
                samples = []
                for _ in range(n):
                    samples.append(float(sample(expr, library=library, seed=sample_seed, **sample_kwargs)))
                return np.asarray(samples, dtype=float)

        if not self.thetas:
            expr = self.expr
            if not random_symbols(expr):
                return np.full(n, float(expr), dtype=float)
            values = sample(expr, size=n, library=library, seed=sample_seed, **sample_kwargs)
            return np.asarray(values, dtype=float)

        sol = self._solve_theta_substitutions()
        rhs_noise_vars = list({rv for rhs in sol.values() for rv in random_symbols(rhs)})
        predictive_noise_vars = list(random_symbols(self.expr))

        samples = []
        for _ in range(n):
            rhs_noise_draws = {
                rv: float(sample(rv, library=library, seed=sample_seed, **sample_kwargs))
                for rv in rhs_noise_vars
            }
            theta_values = {
                theta: float(rhs.subs(rhs_noise_draws))
                for theta, rhs in sol.items()
            }
            predictive_noise_draws = {
                rv: float(sample(rv, library=library, seed=sample_seed, **sample_kwargs))
                for rv in predictive_noise_vars
            }

            value = float(self.expr.subs(theta_values).subs(predictive_noise_draws))
            samples.append(value)

        return np.asarray(samples, dtype=float)

if __name__ == "__main__":
    x = NoisyValue.gaussian(10, 1, provenance="A")
    y = NoisyValue.from_distribution(5, sp.stats.Exponential, 2, provenance="B")

    z = x * y

    print(z.expr)
    samples = z.sample_n(1000, seed=123)
    print(f"Mean: {samples.mean()}, Std: {samples.std()}")
