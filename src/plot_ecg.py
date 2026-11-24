import gc
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.layout_engine import TightLayoutEngine
from matplotlib.patches import Rectangle


@dataclass
class ECGPlot:
    """CPU arrays and plot annotations sent to an independent PNG writer."""

    image: npt.NDArray[Any]
    aligned: npt.NDArray[Any]
    signal_prob: npt.NDArray[Any]
    source_points: npt.NDArray[Any]
    lines: npt.NDArray[Any] | None
    xs: list[float]
    ys: list[float]
    width: float
    height: float
    title: str

    def save(self, output_basepath: str) -> None:
        fig = Figure(figsize=(20, 14))
        FigureCanvasAgg(fig)
        axs = fig.subplots(2, 2)
        axs[0, 0].imshow(self.image * 0.999)
        axs[0, 0].scatter(self.source_points[:, 0], self.source_points[:, 1], s=20, c="red")
        axs[0, 1].imshow(self.aligned * 0.999)
        axs[1, 0].imshow(self.signal_prob, interpolation="none", vmin=0, vmax=1)
        for xval in self.xs:
            for yval in self.ys:
                axs[0, 1].add_patch(
                    Rectangle(
                        (xval, yval),
                        width=self.width,
                        height=self.height,
                        edgecolor="red",
                        facecolor="none",
                    )
                )
        if self.lines is not None:
            self.lines -= np.linspace(0, 24_000, num=self.lines.shape[0])[:, None]  # 2 uV offset per lead
            axs[1, 1].plot(self.lines.T, linewidth=0.5)
        if fig.get_layout_engine() is None:
            # Execute the one-shot layout directly to avoid rasterizing twice.
            TightLayoutEngine().execute(fig)
        else:
            fig.tight_layout()
        fig.suptitle(self.title, fontsize=16)
        fig.savefig(output_basepath + ".png", dpi=200)
        # Figures contain cycles that retain the raster buffers between jobs.
        del axs, fig
        gc.collect()
