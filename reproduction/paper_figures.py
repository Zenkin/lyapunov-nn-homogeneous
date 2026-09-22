"""Paper-style plots computed from saved networks, with explicit audit domains."""

from pathlib import Path
import json
import math
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from example_1.improved import example1 as e1
from example_2.improved import example2 as e2

BLUE, ORANGE, RED, BLACK = "#56B4E9", "#E69F00", "#D55E00", "#171717"
LINE, PURPLE, GRAY = "#0072B2", "#7055AA", "#737D8C"
V = r"V_{\mathrm{comp}}"
DV = r"\dot V_{\mathrm{comp}}"
BOUND = r"$\partial\widehat\Omega_c$"
plt.rcParams.update(
    {
        "font.size": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelsize": 13,
        "legend.fontsize": 10,
        "lines.linewidth": 2,
    }
)


def export(fig, out, name):
    fig.savefig(out / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def panel(ax, label):
    ax.text(-0.12, 1.035, label, transform=ax.transAxes, weight="bold", fontsize=16)


def scientific(value):
    mantissa, exponent = f"{value:.2e}".split("e")
    return rf"{mantissa}\times10^{{{int(exponent)}}}"


def phase(ax, pendulum=False):
    if pendulum:
        ax.set(
            xlim=(-math.pi, math.pi),
            ylim=(-4, 4),
            xlabel=r"Angle, $\theta$ (rad)",
            ylabel=r"Angular velocity, $\dot\theta$",
        )
        ax.set_xticks(
            [-math.pi, -math.pi / 2, 0, math.pi / 2, math.pi],
            [r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"],
        )
        for x in (-math.pi / 2, math.pi / 2):
            ax.axvline(x, color=GRAY, ls=":", lw=1.3)
    else:
        ax.set(
            xlabel=r"Displacement, $z_1=x_1-x_{\mathrm{eq}}$",
            ylabel=r"Velocity, $z_2=x_2$",
        )


def target(ax):
    ax.add_patch(
        Rectangle(
            (-0.5, -0.5), 1, 1, facecolor="#F3F4F6", edgecolor=BLACK, lw=1.7, zorder=5
        )
    )
    ax.scatter(0, 0, marker="*", s=85, c=BLACK, edgecolors="white", zorder=6)


def integrate(field, initial, end, step=0.005, wrapped=False):
    x = torch.tensor(initial, dtype=torch.float64)
    states = [x.numpy().copy()]
    with torch.no_grad():
        for _ in range(round(end / step)):
            a = field(x)
            b = field(x + step * a / 2)
            c = field(x + step * b / 2)
            d = field(x + step * c)
            x = x + step * (a + 2 * b + 2 * c + d) / 6
            if wrapped:
                x[:, 0] = e2.wrapped_angle(x[:, 0])
            states.append(x.numpy().copy())
    return np.arange(len(states)) * step, np.stack(states)


def comparison(
    out,
    name,
    initial,
    inside,
    success,
    paths,
    time,
    values,
    derivatives,
    boundary,
    level,
    note,
    pendulum=False,
    limits=None,
    equilibria=None,
):
    fig, axes = plt.subplot_mosaic(
        [["phase", "value"], ["phase", "derivative"]],
        figsize=(12, 7.5),
        layout="constrained",
    )
    ax = axes["phase"]
    ax.scatter(initial[inside, 0], initial[inside, 1], s=15, c=BLUE, alpha=0.75)
    ax.scatter(
        initial[~inside, 0],
        initial[~inside, 1],
        s=24,
        marker="^",
        facecolors="none",
        edgecolors=ORANGE,
    )
    if (~success).any():
        ax.scatter(initial[~success, 0], initial[~success, 1], marker="x", color=RED)
    boundary(ax)
    for i, (color, style, marker) in enumerate(((LINE, "-", "o"), (ORANGE, "--", "D"))):
        ax.plot(paths[:, i, 0], paths[:, i, 1], color=color, ls=style)
        ax.scatter(
            *paths[0, i],
            s=75,
            marker=marker,
            facecolor="white",
            edgecolor=color,
            lw=2,
            zorder=6,
        )
        axes["value"].plot(time, values[:, i], color=color, ls=style)
        axes["derivative"].plot(time, derivatives[:, i], color=color, ls=style)
    phase(ax, pendulum)
    if pendulum:
        ax.scatter(0, 0, marker="*", c=BLACK, s=80, zorder=6)
        if equilibria is not None:
            ax.scatter(
                equilibria, np.zeros(len(equilibria)), marker="x", s=50, color=GRAY
            )
    else:
        ax.set(xlim=limits[0], ylim=limits[1])
        target(ax)
    ax.text(
        0.025,
        0.965,
        note,
        transform=ax.transAxes,
        va="top",
        fontsize=11,
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "none"},
    )
    axes["value"].axhline(level, color=BLACK, ls=":", lw=1.2)
    axes["value"].text(
        0.98,
        0.95,
        rf"$c={level:.3f}$" if pendulum else rf"$2\kappa={level:.3f}$",
        transform=axes["value"].transAxes,
        ha="right",
        va="top",
    )
    coordinate = "x" if pendulum else "z"
    axes["value"].set(
        ylabel=rf"${V}({coordinate}(t))$", xlim=(0, 10 if pendulum else 4)
    )
    axes["derivative"].axhline(0, color=BLACK, ls=":", lw=1)
    axes["derivative"].set(
        ylabel=rf"${DV}({coordinate}(t))$",
        xlabel=r"Time, $t$",
        xlim=(0, 10 if pendulum else 4),
    )
    for a, label in zip(axes.values(), ("(a)", "(b)", "(c)")):
        panel(a, label)
    first = (
        "reached target; remained in validation domain"
        if pendulum
        else r"$x_0$ inside plotted sublevel"
    )
    second = (
        "reached target after leaving validation domain"
        if pendulum
        else r"$x_0$ outside plotted sublevel"
    )
    fig.legend(
        handles=[
            Line2D([], [], marker="o", ls="", color=BLUE, label=first),
            Line2D(
                [],
                [],
                marker="^",
                ls="",
                markerfacecolor="none",
                color=ORANGE,
                label=second,
            ),
            Line2D([], [], color=LINE, label="selected inside trajectory"),
            Line2D([], [], color=ORANGE, ls="--", label="selected outside trajectory"),
            Line2D([], [], color=BLACK, label=BOUND if pendulum else rf"${V}=2\kappa$"),
        ],
        loc="outside lower center",
        ncol=2,
        frameon=False,
    )
    export(fig, out, name)


def make_figures(out, h, n, p, eq, kappa, l, u, c2, level, a1, a2, t2, equilibria):
    # Audit the plotted composite sublevel separately from the broader
    # mixed-grid validation in the original implementation.
    gx, gy, pts = e1.mesh_rectangle(-3.5, 3.5, -12, 12, 401, midpoint=True)
    val, dv = e1.evaluate_candidate_in_chunks(h, n, pts, p, eq, kappa)
    val = val.reshape(gx.shape)
    dv = dv.reshape(gx.shape)
    xx = pts.numpy()
    outside = ((np.abs(xx[:, 0]) > 0.5) | (np.abs(xx[:, 1]) > 0.5)).reshape(gx.shape)
    audited = (val <= 2 * kappa) & outside
    cats = np.where(dv < -0.05, 0, np.where(dv < 0, 1, 2)).astype(float)
    cats[~audited] = 3
    inner = e1.level_boundary(h, kappa, 2048, offset=0.25).detach().numpy()

    def boundary1(ax):
        ax.contour(gx, gy, val, levels=[2 * kappa], colors=[BLACK], linewidths=2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.9), layout="constrained")
    im = axes[0].contourf(
        gx,
        gy,
        val,
        levels=np.linspace(0, 2.3 * kappa, 31),
        cmap="cividis",
        extend="max",
    )
    fig.colorbar(im, ax=axes[0], label=rf"Composite candidate, ${V}$")
    axes[0].plot(inner[:, 0], inner[:, 1], color="white", ls="--", lw=2)
    axes[1].pcolormesh(
        gx,
        gy,
        cats,
        cmap=ListedColormap([BLUE, ORANGE, RED, "#E5E7EB"]),
        vmin=-0.5,
        vmax=3.5,
        shading="nearest",
        rasterized=True,
    )
    for i, ax in enumerate(axes):
        boundary1(ax)
        phase(ax)
        target(ax)
        ax.axhline(0, color=GRAY, ls=":", lw=1.2)
        ax.set(xlim=(-3.5, 3.5), ylim=(-12, 12))
        panel(ax, "(a)" if i == 0 else "(b)")
    axes[0].legend(
        handles=[
            Line2D([], [], color=BLACK, label=rf"${V}=2\kappa$"),
            Line2D([], [], color=GRAY, ls="--", label=r"$V_\infty=\kappa$"),
            Patch(facecolor="#F3F4F6", edgecolor=BLACK, label=r"target set $X$"),
        ],
        loc="upper right",
    )
    axes[1].legend(
        handles=[
            Patch(color=BLUE, label=rf"${DV}\leq-0.05$"),
            Patch(color=ORANGE, label=rf"$-0.05<{DV}<0$"),
            Patch(color=RED, label=rf"${DV}\geq0$"),
            Patch(color="#E5E7EB", label="outside plotted audit set"),
        ],
        loc="upper right",
    )
    maxdv = float(dv[audited].max())
    violations = int(np.count_nonzero(dv[audited] >= 0))
    axes[1].text(
        0.03,
        0.04,
        rf"Inside $\{{{V}\leq2\kappa\}}\setminus X$: $N_{{\dot V\geq0}}={violations}$"
        + "\n"
        + rf"$\max {DV}={scientific(maxdv)}$",
        transform=axes[1].transAxes,
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "none"},
    )
    export(fig, out, "example1_candidate_and_decay")
    initial1 = a1["initial_states"]
    with torch.no_grad():
        iv = e1.smooth_combined_candidate(h, n, torch.tensor(initial1), kappa).numpy()
    inside1 = iv <= 2 * kappa
    time1, paths1 = integrate(
        lambda z: e1.shifted_field(z, p, eq), [[3.0, -1.0], [4 / 3, 4.0]], 20
    )
    vs1, ds1 = e1.evaluate_candidate_in_chunks(
        h, n, torch.tensor(paths1.reshape(-1, 2)), p, eq, kappa
    )
    vs1 = vs1.reshape(-1, 2)
    ds1 = ds1.reshape(-1, 2)
    comparison(
        out,
        "example1_trajectories",
        initial1,
        inside1,
        a1["success"],
        paths1,
        time1,
        vs1,
        ds1,
        boundary1,
        2 * kappa,
        f"{int(a1['success'].sum())}/625 in X at t = 20\n{inside1.sum()} initialized inside\n{(~inside1).sum()} initialized outside",
        limits=((-4.2, 4.2), (-12.6, 12.6)),
    )
    np.savez_compressed(
        out / "example1/paper_arrays.npz",
        x=gx,
        y=gy,
        value=val,
        derivative=dv,
        audit_mask=audited,
        selected_time=time1,
        selected_paths=paths1,
        selected_values=vs1,
        selected_derivatives=ds1,
    )
    # Example 2: preserve both positive-derivative patches outside the selected
    # origin component and all simulated excursions beyond the validation box.
    count = c2.validation_points_per_axis
    grid = a2["x"].reshape(count, count, 2)
    gx2, gy2 = grid[:, :, 0], grid[:, :, 1]
    v2 = a2["V_NN"].reshape(count, count)
    d2 = a2["dV_NN"].reshape(count, count)
    local = a2["V_local"].reshape(count, count)

    def boundary2(ax):
        ax.contour(gx2, gy2, v2, levels=[level], colors=[BLACK], linewidths=2)

    conditions = np.where(d2 + c2.decay_rate * v2 <= 0, 0, np.where(d2 < 0, 1, 2))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.9), layout="constrained")
    im = axes[0].contourf(gx2, gy2, v2, levels=30, cmap="cividis")
    fig.colorbar(im, ax=axes[0], label=rf"Composite candidate, ${V}$")
    axes[0].contour(
        gx2,
        gy2,
        local,
        levels=[c2.kappa, 2 * c2.kappa],
        colors=["white"],
        linestyles=["--", ":"],
        linewidths=1.8,
    )
    axes[1].pcolormesh(
        gx2,
        gy2,
        conditions,
        cmap=ListedColormap([BLUE, ORANGE, RED]),
        vmin=-0.5,
        vmax=2.5,
        shading="nearest",
        rasterized=True,
    )
    critical = a2["roa_critical_point"]
    axes[1].scatter(
        *critical, marker="*", s=100, facecolor="white", edgecolor=BLACK, zorder=5
    )
    roots = [
        item["theta"] for item in equilibria["equilibria"] if abs(item["theta"]) > 1e-6
    ]
    axes[1].scatter(roots, np.zeros(len(roots)), marker="x", s=60, color=BLACK)
    for i, ax in enumerate(axes):
        boundary2(ax)
        phase(ax, True)
        ax.scatter(
            0,
            0,
            marker="*" if i == 0 else "o",
            c=BLACK,
            edgecolors="white",
            s=60,
            zorder=6,
        )
        panel(ax, "(a)" if i == 0 else "(b)")
    axes[0].legend(
        handles=[
            Line2D([], [], color=BLACK, label=BOUND),
            Line2D([], [], color=GRAY, ls="--", label=r"$V_\ell=\kappa$"),
            Line2D([], [], color=GRAY, ls=":", label=r"$V_\ell=2\kappa$"),
        ],
        loc="upper right",
    )
    axes[1].legend(
        handles=[
            Patch(color=BLUE, label=rf"${DV}+0.1{V}\leq0$"),
            Patch(color=ORANGE, label=rf"$-0.1{V}<{DV}<0$"),
            Patch(color=RED, label=rf"${DV}\geq0$"),
            Line2D([], [], color=BLACK, label=BOUND),
        ],
        loc="upper left",
    )
    mask = a2["roa_component"] & (np.linalg.norm(grid, axis=2) > 1e-12)
    axes[1].text(
        0.025,
        0.035,
        rf"Inside $\widehat\Omega_c\setminus\{{0\}}$: $N_{{\dot V\geq0}}={np.count_nonzero(d2[mask]>=0)}$"
        + "\n"
        + rf"$\max {DV}={scientific(d2[mask].max())}$",
        transform=axes[1].transAxes,
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "none"},
    )
    export(fig, out, "example2_candidate_and_decay")
    k, pm = e2.consistent_corrected_local_design()
    angle = float(t2["initial"][np.argmin(np.abs(t2["initial"][:, 0] - 1.9)), 0])
    time2, paths2 = integrate(
        lambda x: e2.closed_loop_field(x, u, k, pm, c2),
        [[angle, -2], [angle, -1.6]],
        10,
        wrapped=True,
    )
    vv, dd, controls, _, _ = e2.composite_value_derivative(
        l, u, torch.tensor(paths2.reshape(-1, 2)), c2, create_graph=False
    )
    vv = vv.detach().numpy().reshape(-1, 2)
    dd = dd.detach().numpy().reshape(-1, 2)
    controls = controls.detach().numpy().reshape(-1, 2)
    stayed = ~t2["exceeded_velocity_domain"]
    comparison(
        out,
        "example2_trajectories",
        t2["initial"],
        stayed,
        t2["success"],
        paths2,
        time2,
        vv,
        dd,
        boundary2,
        level,
        f"{t2['success'].sum()}/525 reached the target\n"
        + rf"{stayed.sum()} remained in $|\dot\theta|\leq4$"
        + f"\n{(~stayed).sum()} left the validation domain",
        pendulum=True,
        equilibria=roots,
    )
    # First selected trajectory crosses theta=pi/2; interpolate the crossing
    # between adjacent integration samples and evaluate its speed.
    trajectory = paths2[:, 0]
    theta = trajectory[:, 0]
    gravity = np.sin(theta)
    control = np.cos(theta) * controls[:, 0]
    crossings = np.flatnonzero(np.cos(theta[:-1]) * np.cos(theta[1:]) <= 0)
    if not len(crossings):
        raise RuntimeError(
            "Selected trajectory did not cross the loss-of-authority line"
        )
    idx = int(crossings[0])
    frac = (math.pi / 2 - theta[idx]) / (theta[idx + 1] - theta[idx])
    cross_time = float(time2[idx] + frac * (time2[idx + 1] - time2[idx]))
    cross_speed = float(
        trajectory[idx, 1] + frac * (trajectory[idx + 1, 1] - trajectory[idx, 1])
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.9), layout="constrained")
    axes[0].plot(theta, trajectory[:, 1], color=LINE)
    axes[0].scatter(
        *trajectory[0],
        facecolor="white",
        edgecolor=LINE,
        marker="o",
        s=80,
        lw=2,
        label="initial state",
    )
    axes[0].scatter(
        math.pi / 2,
        cross_speed,
        c=ORANGE,
        edgecolor=BLACK,
        marker="D",
        s=75,
        zorder=5,
        label=r"crossing of $\cos\theta=0$",
    )
    axes[0].scatter(0, 0, c=BLACK, marker="*", s=80, label="target")
    phase(axes[0], True)
    axes[0].legend(loc="lower left", frameon=False)
    axes[1].plot(time2, gravity, color=LINE, label=r"$\omega^2\sin\theta\;(\omega=1)$")
    axes[1].plot(time2, control, color=ORANGE, ls="--", label=r"$\cos\theta\,u$")
    axes[1].plot(
        time2, gravity + control, color=PURPLE, ls="-.", label=r"$\ddot\theta$"
    )
    axes[1].axhline(0, color="#D1D5DB", lw=1)
    axes[1].axvline(cross_time, color=BLACK, ls=":", lw=1.2)
    axes[1].scatter([cross_time], [1], color=LINE, s=55, zorder=5)
    axes[1].scatter([cross_time], [0], color=ORANGE, marker="D", s=65, zorder=5)
    axes[1].annotate(
        rf"$|\dot\theta|\approx{abs(cross_speed):.2f}$",
        (cross_time, 1),
        xytext=(10, 10),
        textcoords="offset points",
        fontsize=11,
    )
    axes[1].set(
        xlim=(0, 2),
        xlabel=r"Time, $t$",
        ylabel="Normalized angular-acceleration contribution",
    )
    axes[1].legend(loc="upper right", frameon=False)
    panel(axes[0], "(a)")
    panel(axes[1], "(b)")
    export(fig, out, "example2_loss_of_authority")
    np.savez_compressed(
        out / "example2/selected_trajectories.npz",
        time=time2,
        paths=paths2,
        values=vv,
        derivatives=dd,
        controls=controls,
    )
    if not (vs1[0, 0] <= 2 * kappa < vs1[0, 1]):
        raise RuntimeError(
            "Example 1 selected trajectories do not straddle the plotted sublevel"
        )
    if not (vv[0, 0] <= level < vv[0, 1]):
        raise RuntimeError(
            "Example 2 selected trajectories do not straddle the plotted sublevel"
        )
    for array in (val, dv, vs1, ds1, v2, d2, vv, dd, controls):
        if not np.isfinite(array).all():
            raise RuntimeError("Nonfinite values in plotted arrays")
    return {
        "example1": {
            "plot_grid": "401 x 401 cell midpoints in [-3.5,3.5] x [-12,12]",
            "kappa": kappa,
            "maximum_derivative_in_audit_set": maxdv,
            "nonnegative_derivatives": violations,
            "initial_inside": int(inside1.sum()),
            "initial_outside": int((~inside1).sum()),
            "selected_initial_states": paths1[0].tolist(),
            "selected_initial_values": vs1[0].tolist(),
            "selected_integration_step": 0.005,
        },
        "example2": {
            "level": level,
            "selected_initial_states": paths2[0].tolist(),
            "selected_integration_step": 0.005,
            "selected_initial_values": vv[0].tolist(),
            "crossing_time": cross_time,
            "crossing_speed": abs(cross_speed),
        },
    }


def gallery(out):
    names = [
        ("example1_candidate_and_decay", "Example 1: candidate and derivative"),
        ("example1_trajectories", "Example 1: trajectories"),
        ("example2_candidate_and_decay", "Example 2: candidate and derivative"),
        ("example2_trajectories", "Example 2: trajectories"),
        (
            "example2_loss_of_authority",
            "Example 2: crossing of the loss-of-authority line",
        ),
    ]
    data = json.loads((out / "verification.json").read_text())
    status = (
        "All numerical reproduction checks passed."
        if data["all_checks_passed"]
        else "FAILED: inspect verification.json before using these figures."
    )
    html = '<!doctype html><html lang="en"><meta charset="utf-8"><title>Checkpoint reproduction</title><style>body{font:16px/1.6 system-ui;margin:40px auto;max-width:1200px;padding:0 24px}img{width:100%}figure{margin:35px 0}a{color:#155b91}</style><h1>Checkpoint reproduction</h1>'
    html += (
        f"<p>{status} No retraining.</p><p>Example 1: 625/625 in X at t=20; 250 inside the plotted composite sublevel. Example 2: c=3.466; 525/525 reach the target, 343 remain in the validation domain and 182 leave it.</p>"
        if data["all_checks_passed"]
        else f"<p>{status}</p>"
    )
    html += '<p><a href="verification.json">Numerical checks, parameters and checkpoint hashes</a>. Finite-grid observations do not establish a continuous-domain certificate. PNG and SVG files are generated from the arrays saved beside them.</p>'
    for name, title in names:
        html += f'<figure><figcaption>{title} · <a href="{name}.svg">SVG</a></figcaption><a href="{name}.png"><img src="{name}.png" alt="{title}"></a></figure>'
    (out / "RESULTS.html").write_text(html + "</html>", encoding="utf-8")
