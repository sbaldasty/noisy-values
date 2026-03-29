from sympy import Expr
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

    def eliminate_thetas(self):
        expr = self.expr

        eqs = [sp.Eq(eq, 0) for eq in self.equations]
        thetas = list(self.thetas)

        sol = sp.solve(eqs, thetas, dict=True)

        if not sol:
            raise ValueError("Could not solve for latent variables")

        sol = sol[0]

        new_subs = {}

        for theta, rhs in sol.items():
            # replace any noise variables in rhs with fresh copies
            noise_vars = random_symbols(rhs)
            replacements = {}

            for v in noise_vars:
                replacements[v] = fresh_normal(
                    float(v.pspace.distribution.mean),
                    float(v.pspace.distribution.std)
                )

            new_rhs = rhs.subs(replacements)
            new_subs[theta] = new_rhs

        return expr.subs(new_subs)

    def sample_n(self, n=1000):
        expr = self.eliminate_thetas()

        vars_ = list(random_symbols(expr))

        samples = []
        for _ in range(n):
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
