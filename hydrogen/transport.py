"""Conservative remapping and implicit upwind transport on constant-area loops."""
import numpy as np
from scipy.linalg import solve_banded
from .profiles import check_populations


def remap(old_edges, values, new_edges):
    """Piecewise-constant cell-average remap, conserving the integral."""
    if not np.allclose(old_edges[[0, -1]], new_edges[[0, -1]], atol=1e-3, rtol=1e-10):
        raise ValueError('Remap domains differ')
    scalar = values.ndim == 1
    a = values[:, None] if scalar else values
    cumulative = np.vstack([np.zeros(a.shape[1]), np.cumsum(a*np.diff(old_edges)[:, None], axis=0)])
    sampled = np.column_stack([np.interp(new_edges, old_edges, cumulative[:, j]) for j in range(a.shape[1])])
    out = np.diff(sampled, axis=0)/np.diff(new_edges)[:, None]
    if np.any(out < -1e-10*np.max(a)):
        raise ValueError('Negative density after conservative remap')
    out = np.maximum(out, 0)
    return out[:, 0] if scalar else out


def advect(edges, density, populations, velocity, dt, boundary):
    """Advance all six species densities with the same conservative flux.

    Open boundaries use explicitly supplied incoming fractions and the adjacent
    carrier density. Outflow uses the interior state. Return fractions and the
    transported carrier, so conditioning on saved density can be diagnosed.
    """
    n = len(density)
    widths = np.diff(edges)
    centers = (edges[:-1]+edges[1:])/2
    v = np.interp(edges, centers, velocity)
    plus, minus = np.maximum(v, 0), np.minimum(v, 0)
    a = np.zeros((3, n))
    a[1] = 1 + dt*(plus[1:]-minus[:-1])/widths
    a[0, 1:] = dt*minus[1:-1]/widths[:-1]
    a[2, :-1] = -dt*plus[1:-1]/widths[1:]
    rhs = density[:, None]*populations
    rhs[0] += dt*plus[0]*density[0]*boundary[0]/widths[0]
    rhs[-1] -= dt*minus[-1]*density[-1]*boundary[1]/widths[-1]
    species = solve_banded((1, 1), a, rhs)
    carrier = species.sum(axis=1)
    if np.any(carrier <= 0):
        raise ValueError('Nonpositive transported hydrogen density')
    return check_populations(species/carrier[:, None]), carrier
