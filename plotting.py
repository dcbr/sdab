import pathlib
import numpy as np
import matplotlib.pyplot as plt

colors = ("tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:gray")


def _init_figure(rows=1, cols=1, share_x=False, share_y=False, squeeze=True, x_label='', y_label='', title=''):
    f, axes = plt.subplots(rows, cols, sharex=share_x, sharey=share_y, squeeze=squeeze)
    try:
        x_labels = np.full((rows, cols), x_label)
    except ValueError:
        x_labels = np.full((cols, rows), x_label).T
    try:
        y_labels = np.full((rows, cols), y_label)
    except ValueError:
        y_labels = np.full((cols, rows), y_label).T
    for ax, lx, ly in zip(np.ravel(axes), np.ravel(x_labels), np.ravel(y_labels)):
        ax.set_xlabel(lx)
        ax.set_ylabel(ly)
    f.suptitle(title)
    return f, axes


def _handle_figure(f, save_path=None, close=True):
    r = None
    if save_path is not None:
        save_path = pathlib.Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        f.savefig(str(save_path))
    if close:
        plt.close(f)
    else:
        r = f
    return r


def _to_NP(a):
    """Converts an array of shape (P,) to shape (1, P). An array of shape (N, P) or higher dimensionality
    is left untouched. The P dimension can be ragged."""
    if not isinstance(a[0], (list, np.ndarray)):
        a = [a]
    return a


def _to_NPC(a):
    """Converts an array of shape (P,) to shape (1, P, 1) and an array of shape (N, P) to shape (N, P, 1).
    An array of shape (N, P, C) is left untouched. The P and C dimension can be ragged."""
    a = _to_NP(a)  # (P,) -> (1, P)
    if not isinstance(a[0][0], (list, np.ndarray)):
        a = [np.reshape(ai, (-1, 1)) for ai in a]
    return a


def show_all():
    plt.show()


def close_all():
    plt.close('all')


def line_plots(ys, legends=None, x_label='', y_label='', title='', save_path=None, close=True):
    f, axes = _init_figure(rows=ys[0].shape[1], share_x="col", x_label=x_label, y_label=y_label, title=title)
    axes = np.ravel(axes)
    for y, color in zip(ys, colors):
        for comp, ax in enumerate(axes):
            ax.plot(y[:,comp], color=color)
    if legends is not None:
        axes[0].legend(legends, loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=4)
    return _handle_figure(f, save_path, close)


def shaded_line_plot(xs, ys, yms, yMs, legends=None, alpha=0.2, x_label='', y_label='', title='', save_path=None, close=True):
    """xs has shape (N, P) and ys, yms and yMs have shape (N, P, C) where N is the amount of different shaded lines
    to draw in a single plot, P is the number of points to draw the shaded line through, C is the number of components
    (each drawn in a separate subplot).
    an xs with shape (P,) is also accepted (and converted to shape (1, P)); an ys, yms or yMs of shape (P,) is also
    accepted (and converted to shape (1, P, 1)) and shape (N, P) is also accepted (and converted to shape (N, P, 1))"""
    xs = _to_NP(xs)
    ys, yms, yMs = _to_NPC(ys), _to_NPC(yms), _to_NPC(yMs)
    if legends is None:
        legends = [None] * len(xs)
    show_legend = False
    f, axes = _init_figure(rows=ys[0].shape[1], share_x="col", x_label=x_label, y_label=y_label, title=title)
    axes = np.ravel(axes)
    for x, y, ym, yM, label, color in zip(xs, ys, yms, yMs, legends, colors):
        for comp, ax in enumerate(axes):
            ax.fill_between(x, ym[:,comp], yM[:,comp], color=color, alpha=alpha)
            ax.plot(x, y[:,comp], label=label, color=color)
            show_legend |= label is not None
    if show_legend:
        axes[0].legend(loc='best')
    return _handle_figure(f, save_path, close)


def stacked_fill_plot(x, ys, legends, x_label='', y_label='', title='', save_path=None, close=True):
    f, ax = _init_figure(x_label=x_label, y_label=y_label, title=title)
    yp = np.zeros(x.shape)
    for y, label, color in zip(ys, legends, colors):
        yn = yp + y
        ax.fill_between(x, yp, yn, label=label, color=color)
        yp = yn
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.08), ncol=3)
    return _handle_figure(f, save_path, close)


def violin_plot(data, labels, Q=0.8, x_label='', y_label='', title='', save_path=None, close=True):
    f, ax = _init_figure(x_label=x_label, y_label=y_label, title=title)
    ax.violinplot(data, showmeans=False, showextrema=True, showmedians=False)
    q0, medians, q1 = np.quantile(data, [(1-Q)/2, 0.5, (1+Q)/2], axis=1)
    m = np.min(data, axis=1)
    M = np.max(data, axis=1)
    mu = np.mean(data, axis=1)
    std = np.std(data, axis=1)

    ticks = 1 + np.arange(len(labels))
    ax.scatter(ticks, medians, marker='o', color='white', s=10, zorder=3)
    ax.vlines(ticks, q0, q1, color='tab:blue', linestyle='-', lw=6)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    return _handle_figure(f, save_path, close), {"min": m, "q0": q0, "med": medians, "q1": q1, "max": M, "mean": mu, "std": std}
