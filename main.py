from sympy import Expr
from sympy.stats import Normal


import sympy as sp
import numpy as np
from sympy.stats import Normal, sample
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
    def gaussian(cls, true_value, sigma, provenance=None):
        """
        provenance: optional tag to reuse the same θ across calls
        """
        theta = fresh_theta(provenance)
        noise = fresh_normal(0, sigma)

        expr = theta + noise
        observed = true_value + float(sample(noise))

        return cls(expr, observed, thetas={theta})

    def _combine(self, other, op):
        if isinstance(other, NoisyValue):
            expr = op(self.expr, other.expr)
            observed = op(self.observed, other.observed)
            thetas = self.thetas | other.thetas
            equations = self.equations + other.equations
            return NoisyValue(expr, observed, thetas, equations)
        else:
            expr = op(self.expr, other)
            observed = op(self.observed, other)
            return NoisyValue(expr, observed, self.thetas, self.equations)

    def __add__(self, other):
        return self._combine(other, lambda a, b: a + b)

    __radd__ = __add__

    def __sub__(self, other):
        return self._combine(other, lambda a, b: a - b)

    def __rsub__(self, other):
        return NoisyValue(other - self.expr,
                          other - self.observed,
                          self.thetas)

    def __mul__(self, other):
        return self._combine(other, lambda a, b: a * b)

    __rmul__ = __mul__

    def __truediv__(self, other):
        return self._combine(other, lambda a, b: a / b)

    def __rtruediv__(self, other):
        return NoisyValue(other / self.expr,
                          other / self.observed,
                          self.thetas)

    def eliminate_thetas(self):
        if not self.thetas:
            return self.expr

        eqs = [sp.Eq(eq, 0) for eq in self.equations]
        thetas = list(self.thetas)

        sol = sp.solve(eqs, thetas, dict=True)

        if not sol:
            raise ValueError("Could not solve for latent variables")

        # take first solution (usually unique in your setting)
        sol = sol[0]

        return self.expr.subs(sol)

    def sample_n(self, n=1000):
        expr = self.eliminate_thetas()

        vars_ = list(random_symbols(expr))

        samples = []
        for _ in range(n):
            print([float(sample(v)) for v in vars_])
            subs = {v: float(sample(v)) for v in vars_}
            samples.append(float(expr.subs(subs)))

        return np.array(samples)    

x = NoisyValue.gaussian(10, 1, provenance="A")
y = NoisyValue.gaussian(5, 2, provenance="B")

z = x * y

print(z.expr)
# (theta_A_0 + N0)*(theta_B_1 + N1)

samples = z.sample_n(1000)
print(f"Mean: {samples.mean()}, Std: {samples.std()}")
