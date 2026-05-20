from core import _as_noisy_float


def noisy_min(*values):
    if not values:
        raise ValueError("noisy_min requires at least one value")

    result = _as_noisy_float(values[0])
    for value in values[1:]:
        result = result.minimum(_as_noisy_float(value))
    return result


def noisy_max(*values):
    if not values:
        raise ValueError("noisy_max requires at least one value")

    result = _as_noisy_float(values[0])
    for value in values[1:]:
        result = result.maximum(_as_noisy_float(value))
    return result
