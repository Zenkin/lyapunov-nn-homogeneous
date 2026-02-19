import os
import matplotlib.pyplot as plt

from .viz_cfg import VizCfg, VIZ_DEBUG


def _to_math(text: str, *, use_math: bool) -> str:
    """
    Преобразует строку в MathText (LaTeX-подобный синтаксис Matplotlib).

    Если use_math=True:
    - оборачиваем строку в $...$, чтобы Matplotlib отрисовал как формулу,
      но только если в строке еще нет символа '$' (чтобы не оборачивать дважды).

    Если use_math=False:
    - возвращаем строку как есть.
    """
    if not use_math:
        return text
    if "$" in text:
        return text
    return f"${text}$"


def plot_heatmap(
    X1,
    X2,
    Z,
    *,
    title: str,
    cbar_label: str,
    xlabel: str | None = None,
    ylabel: str | None = None,
    cfg: VizCfg = VIZ_DEBUG,
    save_path: str | None = None,
    show: bool = True,
    # Равновесие: координаты задаём в вызове, а оформление берём из cfg
    eq_point: tuple[float, float] | None = None,
    # Если хочешь иногда менять подпись eq без правки cfg
    eq_label: str | None = None,
):
    """
    Отрисовка двумерного поля Z на сетке (X1, X2) в едином стиле cfg.

    Важная архитектурная договоренность:
    - cfg задаёт стиль и поведение (цветовая карта, контуры, шрифты, легенда...),
    - title/xlabel/ylabel/cbar_label и сами данные (X1,X2,Z) задаёт вызывающий код.
    """

    # Если подписи осей не переданы, используем дефолты из cfg
    if xlabel is None:
        xlabel = cfg.default_xlabel
    if ylabel is None:
        ylabel = cfg.default_ylabel

    fig, ax = plt.subplots(figsize=cfg.figsize)

    # --- заливка ---
    contf = ax.contourf(
        X1,
        X2,
        Z,
        levels=cfg.levels,
        cmap=cfg.cmap,
    )

    # --- контуры ---
    if cfg.show_contour_lines:
        levels_lines = cfg.contour_line_levels
        if levels_lines is None:
            levels_lines = contf.levels

        cs = ax.contour(
            X1,
            X2,
            Z,
            levels=levels_lines,
            colors=cfg.contour_color,
            linewidths=cfg.contour_linewidth,
            alpha=cfg.contour_alpha,
        )

        if cfg.show_contour_labels:
            ax.clabel(
                cs,
                inline=cfg.contour_label_inline,
                inline_spacing=cfg.contour_label_inline_spacing,
                fontsize=cfg.contour_label_fontsize,
                fmt=cfg.contour_label_fmt,
            )

    # --- colorbar ---
    cbar = fig.colorbar(
        contf,
        ax=ax,
        pad=cfg.cbar_pad,
        shrink=cfg.cbar_shrink,
    )
    cbar.set_label(
        _to_math(cbar_label, use_math=cfg.use_math),
        fontsize=cfg.cbar_fontsize,
        labelpad=cfg.cbar_labelpad,
    )
    cbar.ax.tick_params(labelsize=cfg.tick_fontsize)

    # --- подписи/заголовок ---
    ax.set_title(_to_math(title, use_math=cfg.use_math), fontsize=cfg.title_fontsize)
    ax.set_xlabel(_to_math(xlabel, use_math=cfg.use_math), fontsize=cfg.label_fontsize)
    ax.set_ylabel(_to_math(ylabel, use_math=cfg.use_math), fontsize=cfg.label_fontsize)
    ax.tick_params(labelsize=cfg.tick_fontsize)

    ax.set_aspect(cfg.aspect)

    # --- равновесие ---
    if eq_point is not None:
        # Подпись для легенды: приоритет у аргумента, иначе берём из cfg
        label = cfg.eq_label if eq_label is None else eq_label

        ax.scatter(
            [eq_point[0]],
            [eq_point[1]],
            s=cfg.eq_size,
            c=cfg.eq_color,
            marker=cfg.eq_marker,
            edgecolors=cfg.eq_edgecolor if cfg.eq_edgecolor is not None else "none",
            linewidths=cfg.eq_linewidth,
            zorder=cfg.eq_zorder,
            label=_to_math(label, use_math=cfg.use_math) if (label is not None) else None,
        )

    # --- легенда ---
    if cfg.show_legend:
        handles, labels = ax.get_legend_handles_labels()
        if len(handles) > 0:
            if cfg.legend_outside:
                ax.legend(
                    loc=cfg.legend_loc,
                    bbox_to_anchor=cfg.legend_bbox,
                    fontsize=cfg.legend_fontsize,
                    frameon=True,
                )
            else:
                ax.legend(
                    loc=cfg.legend_loc,
                    fontsize=cfg.legend_fontsize,
                    frameon=True,
                )

    if cfg.tight_layout:
        fig.tight_layout()

    # --- сохранение ---
    if save_path is not None:
        directory = os.path.dirname(save_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        fig.savefig(save_path, dpi=cfg.dpi)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax
