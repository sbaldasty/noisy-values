import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

from sympy import Symbol
from sympy.stats import sample
from sympy.stats.rv import random_symbols

_theta_counter = 0
_noise_counter = 0


def fresh_theta(tag=None):
    global _theta_counter
    name = f"theta_{_theta_counter}" if tag is None else f"theta_{tag}_{_theta_counter}"
    _theta_counter += 1
    return Symbol(name)


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

        The returned `expr` is the latent value (`theta`), while the measurement
        mechanism is encoded in `equations` as `theta + noise - observed = 0`.
        This makes downstream sampling reflect analyst belief about the true
        quantity rather than release-to-release spread.

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
        measurement_expr = theta + noise_rv
        observed_expr = measurement_expr.subs({theta: sp.sympify(true_value)})
        observed = float(sample(observed_expr, **sample_kwargs))

        equations = [measurement_expr - observed]
        return cls(theta, observed, thetas={theta}, equations=equations)

    @classmethod
    def from_distribution(cls, true_value, dist_builder, *dist_args, provenance=None, name_prefix="R", **dist_kwargs):
        """
        Build a NoisyValue from a SymPy distribution constructor.

        Example: NoisyValue.from_distribution(10, Exponential, 2)
        """
        name = fresh_noise_name(name_prefix)
        noise_rv = dist_builder(name, *dist_args, **dist_kwargs)
        return cls.from_noise_rv(true_value, noise_rv, provenance=provenance)
    
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
        # If they don't want any samples, don't return any
        if n <= 0:
            return np.array([], dtype=float)

        # Random number generator to use for sampling
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


def plot_confidence_heatmap(
    noisy_value,
    theta_range=None,
    observed_range=None,
    grid_size=201,
    n_samples=4000,
    cmap="viridis_r",
    ax=None,
    seed=None,
    library="scipy",
    **sample_kwargs,
):
    """
    Plot a heat map of central-interval thresholds for a single latent value.

    Axes
    -----
    x-axis: candidate true value (theta)
    y-axis: candidate observed value (y)

    Heat value
    ----------
    For each (theta, y), the color encodes the smallest central interval
    probability level whose interval around theta contains y.

    Lower threshold values correspond to narrower intervals. The default colormap
    (`viridis_r`) makes these narrower intervals brighter.
    """
    if len(noisy_value.thetas) != 1:
        raise ValueError("Heatmap requires exactly one latent theta in NoisyValue")
    if len(noisy_value.equations) != 1:
        raise ValueError("Heatmap currently supports exactly one observation equation")
    if grid_size < 2:
        raise ValueError("grid_size must be at least 2")
    if n_samples <= 0:
        raise ValueError("n_samples must be > 0")

    theta = next(iter(noisy_value.thetas))
    measurement_expr = noisy_value.equations[0] + noisy_value.observed

    rng = seed
    if isinstance(seed, int):
        rng = np.random.default_rng(seed)

    if theta_range is None or observed_range is None:
        pilot_expr = measurement_expr.subs({theta: noisy_value.observed})
        try:
            pilot = np.asarray(
                sample(pilot_expr, size=2000, library=library, seed=rng, **sample_kwargs),
                dtype=float,
            )
        except Exception:
            pilot = np.asarray(
                [
                    float(sample(pilot_expr, library=library, seed=rng, **sample_kwargs))
                    for _ in range(2000)
                ],
                dtype=float,
            )

        spread = float(np.std(pilot, ddof=1))
        if not np.isfinite(spread) or spread == 0.0:
            spread = 1.0

        if theta_range is None:
            theta_range = (noisy_value.observed - 4 * spread, noisy_value.observed + 4 * spread)
        if observed_range is None:
            observed_range = (noisy_value.observed - 4 * spread, noisy_value.observed + 4 * spread)

    theta_grid = np.linspace(theta_range[0], theta_range[1], grid_size)
    observed_grid = np.linspace(observed_range[0], observed_range[1], grid_size)

    heat = np.empty((grid_size, grid_size), dtype=float)
    for j, theta_value in enumerate(theta_grid):
        expr_at_theta = measurement_expr.subs({theta: theta_value})

        try:
            y_draws = np.asarray(
                sample(expr_at_theta, size=n_samples, library=library, seed=rng, **sample_kwargs),
                dtype=float,
            )
        except Exception:
            y_draws = np.asarray(
                [
                    float(sample(expr_at_theta, library=library, seed=rng, **sample_kwargs))
                    for _ in range(n_samples)
                ],
                dtype=float,
            )

        dist_from_center = np.abs(y_draws - theta_value)[:, None]
        query_radius = np.abs(observed_grid - theta_value)[None, :]

        # Minimal central-interval level that contains each y on this column.
        heat[:, j] = np.mean(dist_from_center <= query_radius, axis=0)

    created_fig = False
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))
        created_fig = True

    im = ax.imshow(
        heat,
        origin="lower",
        extent=[theta_grid[0], theta_grid[-1], observed_grid[0], observed_grid[-1]],
        aspect="auto",
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
    )

    ax.set_xlabel("True value (theta)")
    ax.set_ylabel("Observed value")
    ax.set_title("Confidence-Interval Threshold Heatmap")
    ax.plot(theta_grid, theta_grid, color="white", linestyle="--", linewidth=1.2, alpha=0.8)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Smallest central CI probability containing observed value")

    if created_fig:
        plt.tight_layout()

    return {
        "ax": ax,
        "heat": heat,
        "theta_grid": theta_grid,
        "observed_grid": observed_grid,
    }

if __name__ == "__main__":
    # x = NoisyValue.from_distribution(10, sp.stats.Normal, 0, 1, provenance="A")
    # y = NoisyValue.from_distribution(5, sp.stats.Exponential, 2, provenance="B")

    # z = x * y

    # print(z.expr)
    # samples = z.sample_n(1000, seed=123)
    # print(f"Mean: {samples.mean()}, Std: {samples.std()}")


    # Without cloning
    import sympy as sp
    import numpy as np

    # One RV object reused for both measurement noise and predictive noise
    eps = sp.stats.Normal("E", 0, 1)
    theta = fresh_theta("demo")
    observed = 10.0

    # expr uses theta + eps (predictive noise)
    # equation uses theta + eps - observed = 0 (measurement noise)
    # These are conceptually different noises, but represented by same RV object.
    nv = NoisyValue(
        expr=theta + eps,
        observed=observed,
        thetas={theta},
        equations=[theta + eps - observed],
    )

    bad_expr = nv.eliminate_thetas()   # becomes 10.0 due to E - E cancellation
    bad_samples = nv.sample_n(5000, seed=0)

    print("bad_expr:", bad_expr)
    print("bad std:", np.std(bad_samples))

    # With cloning
    import sympy as sp
    import numpy as np

    def clone_normal(rv):
        # Demo cloner for Normal RVs: same parameters, fresh name
        d = rv.pspace.distribution
        return sp.stats.Normal(f"{rv.symbol.name}_clone", d.mean, d.std)

    eps = sp.stats.Normal("E", 0, 1)
    theta = fresh_theta("demo")
    observed = 10.0

    nv = NoisyValue(
        expr=theta + eps,
        observed=observed,
        thetas={theta},
        equations=[theta + eps - observed],
    )

    good_expr = nv.eliminate_thetas(noise_cloner=clone_normal)  # 10 - E_clone + E
    good_samples = nv.sample_n(5000, noise_cloner=clone_normal, seed=0)

    print("good_expr:", good_expr)
    print("good std:", np.std(good_samples))
