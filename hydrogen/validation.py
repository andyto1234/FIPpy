"""Held-out Hstate comparisons. Reference populations never feed reconstruction."""
from pathlib import Path
import json
import numpy as np
from .profiles import read_profile, read_populations


def validate_history(history, root, *, output=None, threshold=0.05):
    """Report width-weighted errors for Te <= 100,000 K and the full loop.

    A threshold is a user-selected screening criterion, not a scientific accuracy
    guarantee. An equilibrium reference cannot validate time-dependent physics.
    """
    rows = []
    weighted_error = 0.
    total_width = 0.
    all_max = 0.
    worst = None
    for i in history.indices:
        p = read_profile(root, int(i))
        truth = read_populations(root, p)[:, 5]
        predicted = history.fraction(int(i), p.s, root=root, time=p.time)
        err = abs(predicted-truth)
        mask = p.te <= 1e5
        width = p.widths[mask]
        mean = float(np.average(err[mask], weights=width)) if len(width) else None
        maximum = float(err[mask].max()) if len(width) else None
        if len(width):
            weighted_error += float(np.sum(err[mask]*width))
            total_width += float(width.sum())
            if worst is None or maximum > worst['chromosphere_max_abs']:
                worst = dict(index=int(i), time_s=p.time, chromosphere_max_abs=maximum)
        all_max = max(all_max, float(err.max()))
        rows.append(dict(index=int(i), time_s=p.time, cells=len(p.s), chromosphere_mean_abs=mean,
                         chromosphere_max_abs=maximum, full_loop_max_abs=float(err.max())))
    report = dict(mode=history.metadata['mode'], initial=history.metadata['initial'],
                  reference='HYDRAD six-state statistical equilibrium (bundled source)',
                  profiles=len(rows), time_start_s=rows[0]['time_s'], time_end_s=rows[-1]['time_s'],
                  chromosphere_definition='electron temperature <= 100000 K',
                  chromosphere_width_weighted_mean_abs=weighted_error/total_width if total_width else None,
                  full_loop_max_abs=all_max, worst_profile=worst,
                  screening_threshold_max_abs=threshold,
                  screening_passed=(worst is not None and worst['chromosphere_max_abs'] <= threshold),
                  time_dependent_physics_validated=False, profiles_detail=rows,
                  reconstruction=history.metadata)
    if output:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2))
    return report


def plot_validation(history, root, path, indices):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(indices), 2, figsize=(11, 2.8*len(indices)), squeeze=False)
    for row, i in enumerate(indices):
        p = read_profile(root, int(i))
        truth = read_populations(root, p)[:, 5]
        predicted = history.fraction(int(i), p.s)
        for side, ax in enumerate(axes[row]):
            distance = p.s/1e8 if side == 0 else (p.length-p.s)/1e8
            mask = distance < 5
            order = np.argsort(distance[mask])
            ax.plot(distance[mask][order], truth[mask][order], label='Saved HYDRAD H II', color='black')
            ax.plot(distance[mask][order], predicted[mask][order], '--', label=f"Reconstructed ({history.metadata['mode']})", color='tab:blue')
            ax.set(xlabel='Distance from footpoint (Mm)', ylabel='Hydrogen ionised fraction', ylim=(-.02,1.02),
                   title=f"{'Left' if side == 0 else 'Right'} leg, t = {p.time:g} s")
            ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('Hydrogen reconstruction against saved populations\nSource: '+Path(root).name, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
