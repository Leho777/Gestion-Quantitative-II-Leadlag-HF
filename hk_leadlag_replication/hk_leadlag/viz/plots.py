"""Static matplotlib plots for diagnostic and final figures."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from hk_leadlag.base import LeadLagResult, NonSyncSeries
from hk_leadlag.wavelet import (
    DaubechiesFilter,
    autocorrelation_wavelet,
    level_j_wavelet_filter,
)


@dataclass(slots=True)
class LeadLagPlotter:
    """Render contrast curves, multi-scale histograms and panel summaries."""

    figsize: tuple[float, float] = (10, 6)
    style: str = "seaborn-v0_8-whitegrid"

    def __post_init__(self) -> None:
        try:
            plt.style.use(self.style)
        except OSError:
            pass

    def plot_contrast(self, result: LeadLagResult, save: Path | None = None) -> plt.Figure:
        """One panel per scale showing |Gamma_hat(tau)| with the argmax."""
        if result.contrast is None or result.grid is None:
            raise ValueError("contrast / grid not available in result")
        levels = result.levels
        n = len(levels)
        ncols = 2
        nrows = (n + 1) // 2
        fig, axes = plt.subplots(nrows, ncols, figsize=self.figsize, sharex=True)
        axes = np.atleast_1d(axes).flatten()
        for k, j in enumerate(levels):
            ax = axes[k]
            ax.plot(result.grid, result.contrast[k])
            ax.axvline(result.theta_hat[k], color="crimson", ls="--", lw=1)
            ax.set_title(f"Scale j={int(j)}, θ̂={result.theta_hat[k]:.2f}")
            ax.set_xlabel("τ")
        for ax in axes[n:]:
            ax.axis("off")
        fig.suptitle(f"{result.estimator_name} - multi-scale contrast")
        fig.tight_layout()
        if save:
            Path(save).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save, dpi=150, bbox_inches="tight")
        return fig

    def plot_panel_histograms(
        self,
        panel: np.ndarray,
        levels: np.ndarray,
        save: Path | None = None,
    ) -> plt.Figure:
        """Per-scale histograms in the style of Figure 1 of HK20."""
        n = panel.shape[1]
        ncols = 2
        nrows = (n + 1) // 2
        fig, axes = plt.subplots(nrows, ncols, figsize=self.figsize)
        axes = np.atleast_1d(axes).flatten()
        for k in range(n):
            ax = axes[k]
            ax.hist(panel[:, k], bins=40, edgecolor="white")
            ax.axvline(0, color="black", lw=0.8)
            ax.set_title(f"Scale j={int(levels[k])}")
            ax.set_xlabel("θ̂")
        for ax in axes[n:]:
            ax.axis("off")
        fig.tight_layout()
        if save:
            Path(save).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save, dpi=150, bbox_inches="tight")
        return fig

    def plot_data_overview(
        self,
        times1: np.ndarray,
        prices1: np.ndarray,
        times2: np.ndarray,
        prices2: np.ndarray,
        labels: tuple[str, str] = ("series 1", "series 2"),
        save: Path | None = None,
    ) -> plt.Figure:
        """Sanity plot of the two series plus their inter-tick duration distribution.

        Twin y-axes on the price panel keep assets at very different levels
        (e.g. BTC at 1e5 vs ETH at 2e3) visually comparable.
        """
        # Subsample to ~5000 points per series for fast rendering.
        def _sub(t: np.ndarray, p: np.ndarray, n: int = 5000):
            if len(t) <= n:
                return t, p
            idx = np.linspace(0, len(t) - 1, n).astype(int)
            return t[idx], p[idx]

        t1s, p1s = _sub(times1, prices1)
        t2s, p2s = _sub(times2, prices2)

        fig, axes = plt.subplots(1, 2, figsize=self.figsize)

        # Left: log-prices with twin y-axis.
        ax1 = axes[0]
        ln1 = ax1.plot(t1s, p1s, lw=0.7, color="#1f77b4", label=labels[0])
        ax1.set_ylabel(f"log-price {labels[0]}", color="#1f77b4")
        ax1.tick_params(axis="y", labelcolor="#1f77b4")
        ax1.set_xlabel("time (s from start)")
        ax2 = ax1.twinx()
        ln2 = ax2.plot(t2s, p2s, lw=0.7, color="#d62728", label=labels[1])
        ax2.set_ylabel(f"log-price {labels[1]}", color="#d62728")
        ax2.tick_params(axis="y", labelcolor="#d62728")
        axes[0].set_title("Log-prices (separate y-axes)")
        lines = ln1 + ln2
        axes[0].legend(lines, [l.get_label() for l in lines], loc="best")

        # Right: inter-tick durations histogram.
        dt1 = np.diff(times1)
        dt2 = np.diff(times2)
        bins = np.logspace(-4, 2, 60)
        axes[1].hist(dt1, bins=bins, alpha=0.55, label=labels[0], color="#1f77b4")
        axes[1].hist(dt2, bins=bins, alpha=0.55, label=labels[1], color="#d62728")
        axes[1].set_xscale("log")
        axes[1].set_yscale("log")
        axes[1].set_xlabel("inter-tick duration (s, log)")
        axes[1].set_ylabel("count (log)")
        axes[1].set_title("Inter-tick durations")
        axes[1].legend(loc="best")

        fig.tight_layout()
        if save:
            Path(save).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save, dpi=150, bbox_inches="tight")
        return fig

    def plot_heatmap_2d(
        self,
        result: LeadLagResult,
        normalise: bool = True,
        save: Path | None = None,
    ) -> plt.Figure:
        """2D heatmap of |Γ̂_j(τ)|: scales j on y, lags τ on x.

        Each row is one scale and the white dashed line traces θ̂_j across
        scales, i.e. how the lead-lag moves with the time horizon.
        """
        if result.contrast is None or result.grid is None:
            raise ValueError("contrast / grid not available in result")
        contrast = result.contrast.copy()
        if normalise:
            # Row-normalise so every scale shares the same colour scale.
            row_max = contrast.max(axis=1, keepdims=True)
            row_max[row_max == 0] = 1.0
            contrast = contrast / row_max

        fig, ax = plt.subplots(figsize=self.figsize)
        im = ax.imshow(
            contrast,
            aspect="auto",
            origin="lower",
            extent=[result.grid.min(), result.grid.max(), 0.5, len(result.levels) + 0.5],
            cmap="viridis",
        )
        # Trace the per-scale argmax.
        ax.plot(result.theta_hat, result.levels, color="white", lw=2,
                ls="--", marker="o", markersize=6, markeredgecolor="black",
                label=r"$\hat\theta_j$")
        ax.set_xlabel(r"lag $\tau$")
        ax.set_ylabel("scale index $j$")
        ax.set_title(rf"{result.estimator_name} - |$\hat\Gamma_j(\tau)$| heatmap")
        ax.set_yticks(result.levels)
        ax.legend(loc="upper right", framealpha=0.9)
        fig.colorbar(im, ax=ax, label="normalised contrast" if normalise else "|Γ̂|")
        fig.tight_layout()
        if save:
            Path(save).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save, dpi=150, bbox_inches="tight")
        return fig

    def plot_mc_boxplot(
        self,
        panel: np.ndarray,
        levels: np.ndarray,
        true_theta: np.ndarray | None = None,
        save: Path | None = None,
    ) -> plt.Figure:
        """One box per scale for the Monte Carlo distribution of θ̂_j.

        panel is shape ``(n_paths, j_max)``, levels is the scale indices, and
        true_theta (if given) is overlaid as red crosses.
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        positions = np.asarray(levels, dtype=float)
        bp = ax.boxplot(
            [panel[:, k] for k in range(panel.shape[1])],
            positions=positions,
            widths=0.6,
            patch_artist=True,
            showmeans=True,
            meanprops=dict(marker="D", markerfacecolor="white", markeredgecolor="black"),
        )
        for box in bp["boxes"]:
            box.set(facecolor="#4c72b0", alpha=0.65)
        if true_theta is not None:
            ax.scatter(positions, true_theta, marker="x", color="crimson",
                       s=80, lw=2, zorder=5, label=r"true $\theta_j$")
            ax.legend(loc="upper left")
        ax.set_xlabel("scale index $j$")
        ax.set_ylabel(r"$\hat\theta_j$")
        ax.set_title("Monte Carlo distribution of θ̂_j per scale")
        ax.set_xticks(positions)
        ax.set_xticklabels([str(int(p)) for p in positions])
        fig.tight_layout()
        if save:
            Path(save).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save, dpi=150, bbox_inches="tight")
        return fig

    def plot_scalogram(
        self,
        times: np.ndarray,
        prices: np.ndarray,
        scales: np.ndarray | None = None,
        wavelet: str = "morl",
        save: Path | None = None,
    ) -> plt.Figure:
        """Continuous wavelet scalogram of a single price series.

        Built on PyWavelets' CWT, Morlet by default (smooth and well-localised
        in frequency). Two panels: the log-price and the time-scale heatmap.
        """
        import pywt

        if scales is None:
            scales = np.arange(1, 128)
        # CWT needs an equispaced grid; resample if the input is not regular.
        if not np.allclose(np.diff(times), np.diff(times)[0], rtol=1e-4):
            t_reg = np.linspace(times[0], times[-1], len(times))
            p_reg = np.interp(t_reg, times, prices)
        else:
            t_reg, p_reg = times, prices
        returns = np.diff(p_reg)
        coefs, _ = pywt.cwt(returns, scales, wavelet)
        power = np.abs(coefs)

        fig, axes = plt.subplots(
            2, 1, figsize=self.figsize, sharex=True,
            gridspec_kw={"height_ratios": [1, 3]},
        )
        axes[0].plot(t_reg, p_reg, lw=0.7, color="black")
        axes[0].set_ylabel("log-price")
        axes[0].set_title(f"Scalogram ({wavelet})")
        im = axes[1].imshow(
            power,
            aspect="auto",
            origin="lower",
            extent=[t_reg[0], t_reg[-1], scales[0], scales[-1]],
            cmap="magma",
        )
        axes[1].set_yscale("log")
        axes[1].set_xlabel("time")
        axes[1].set_ylabel("scale (log)")
        fig.colorbar(im, ax=axes[1], label="|W(t, s)|")
        fig.tight_layout()
        if save:
            Path(save).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save, dpi=150, bbox_inches="tight")
        return fig

    def plot_daubechies_filter(
        self,
        wavelet_name: str = "db10",
        levels: list[int] | None = None,
        save: Path | None = None,
    ) -> plt.Figure:
        """Display the Daubechies filter h_p, its QMR g_p, and h_{j,p} for several j."""
        daub = DaubechiesFilter(wavelet_name)
        if levels is None:
            levels = [1, 2, 3, 4]
        fig, axes = plt.subplots(1, 3, figsize=(self.figsize[0] * 1.4, self.figsize[1] * 0.6))
        # Panel 1: h_p and g_p.
        axes[0].stem(daub.h, label="$h_p$ (wavelet)", basefmt=" ")
        axes[0].plot(daub.g, marker="x", ls="--", color="crimson", label="$g_p$ (scaling, QMR)")
        axes[0].set_title(f"{wavelet_name} filters (L={daub.L})")
        axes[0].set_xlabel("p"); axes[0].legend()
        # Panel 2: level-j filters h_{j,p}.
        for j in levels:
            h_j = level_j_wavelet_filter(daub, j)
            axes[1].plot(h_j, label=f"j={j} (L_j={len(h_j)})")
        axes[1].set_title("Recursive level-j filter $h_{j,p}$")
        axes[1].set_xlabel("p"); axes[1].legend()
        # Panel 3: autocorrelation wavelet Psi_j.
        for j in levels:
            psi = autocorrelation_wavelet(daub, j)
            L_j = (len(psi) + 1) // 2
            lags = np.arange(-(L_j - 1), L_j)
            axes[2].plot(lags, psi, label=f"j={j}")
        axes[2].set_title(r"Autocorrelation wavelet $\Psi_j(l)$")
        axes[2].set_xlabel("lag l"); axes[2].legend()
        fig.tight_layout()
        if save:
            Path(save).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save, dpi=150, bbox_inches="tight")
        return fig

    def plot_estimator_comparison(
        self,
        results: dict[str, LeadLagResult],
        save: Path | None = None,
    ) -> plt.Figure:
        """Plot θ̂_j per scale, one line per estimator (HK vs HRY vs DS, etc.).

        Every estimator must have been run on the same ``levels``.
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        for name, res in results.items():
            ax.plot(res.levels, res.theta_hat, marker="o", lw=1.5, label=name)
        ax.set_xlabel("scale index $j$")
        ax.set_ylabel(r"$\hat\theta_j$")
        ax.set_title("Estimator comparison (per-scale lead-lag)")
        ax.legend()
        fig.tight_layout()
        if save:
            Path(save).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save, dpi=150, bbox_inches="tight")
        return fig
