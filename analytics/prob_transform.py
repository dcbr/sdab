from trunc_norm import TruncatedNormal
import rescaling
import plotting
import numpy as np
import torch

if __name__ == "__main__":
    # Create plot of normalized policy distribution and rescaled policy distribution using following parameters
    x_l, x_0, x_u = -1, 0, 1
    y_l, y_0, y_u = -2, 0, 5
    y_m, y_M = -10, 10
    mu, sigma = 0.8, 0.4
    N = 500
    tags = ["lin", "pwl", "hyp"]
    show_clip = False
    # Setup parameters and distributions
    p = rescaling.Params(y_l, y_u, y_0, x_l, x_u, x_0)
    p_max = rescaling.Params(y_l=y_m, y_u=y_M, x_l=x_l, x_u=x_u)
    tn = TruncatedNormal(mu, sigma, x_l, x_u)

    # Normalized policy distribution
    x = np.concatenate([np.linspace(x_l, x_0, N)[:-1], np.linspace(x_0, x_u, N)])
    pdf_x = tn.log_prob(torch.as_tensor(x)).exp().numpy()

    f = plotting.shaded_line_plot(x, pdf_x, x*0, pdf_x, None, 0.1, r"$\tilde{a}$", r"$\tilde{\pi}(\tilde{a}| \mathbf{s})$", "", None, close=False)
    f.axes[0].plot([x_l, x_l, None, x_u, x_u], [0, np.max(pdf_x), None, 0, np.max(pdf_x)], "r--", linewidth=1)  # Bounds
    plotting._handle_figure(f, "pdf_norm.svg", close=False)
    plotting.show_all()

    # Rescaled policy distributions
    ys = []
    pdfs = []
    legend = []

    for tag in tags:
        r = rescaling.from_tag(tag)
        y = r.rescale(x, p).ravel()
        pdf_y = pdf_x * np.abs(r.inverse_grad(y, p).ravel())
        ys.append(y)
        pdfs.append(pdf_y)
        legend.append(fr"$\sigma_\mathrm{{{tag}}}$")
        # Sanity check: pdf_y integral should be close to 1
        print(f"{np.trapz(pdf_y, y)}")

    if show_clip:
        r = rescaling.Linear()
        y = r.rescale(x, p_max).ravel()
        pdf_y = pdf_x * np.abs(r.inverse_grad(y, p_max).ravel())
        Il, Iu = y < y_l, y > y_u
        p_l = np.trapz(pdf_y[Il], y[Il])  # Approximation of discrete probability at lower bound
        p_u = np.trapz(pdf_y[Iu], y[Iu])  # Approximation of discrete probability at upper bound
        y = y[~Il & ~Iu]
        pdf_y = pdf_y[~Il & ~Iu]

        ys.append(y)
        pdfs.append(pdf_y)
        legend.append(r"$\sigma_\mathrm{clip}$")

    f = plotting.shaded_line_plot(ys, pdfs, [np.zeros_like(pdf) for pdf in pdfs], pdfs, legend, 0.1, r"$a$", r"$\pi(a| \mathbf{s})$", "", None, close=False)
    if show_clip:
        f.axes[0].arrow(y_l, 0, 0, p_l, width=0.02, head_width=0.1, head_length=0.01, length_includes_head=True, color=plotting.colors[len(tags)])
        f.axes[0].arrow(y_u, 0, 0, p_u, width=0.02, head_width=0.1, head_length=0.01, length_includes_head=True, color=plotting.colors[len(tags)])
    f.axes[0].plot([y_l, y_l, None, y_u, y_u], [0, np.max(pdfs), None, 0, np.max(pdfs)], "r--", linewidth=1)  # Bounds
    f.axes[0].legend(loc="upper left", bbox_to_anchor=(0.05, 0.95))
    plotting._handle_figure(f, "pdf_scaled.svg", close=False)
    plotting.show_all()
