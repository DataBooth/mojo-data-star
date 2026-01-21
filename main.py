from __future__ import annotations

import base64
import io
from typing import Tuple

from fasthtml.common import *  # type: ignore[import-not-found]
from datastar_py.fasthtml import DatastarStreamingResponse  # type: ignore[import-not-found]
from PIL import Image

try:
    # Import the Mojo-built extension (mandelbrot.mojo → mandelbrot.*)
    from mandelbrot import compute_mandelbrot
except ImportError:  # pragma: no cover - extension not built yet
    compute_mandelbrot = None  # type: ignore[assignment]


WIDTH = 800
HEIGHT = 600


def _default_view() -> Tuple[float, float, float, float]:
    """Initial view window of the Mandelbrot set."""

    xmin, xmax = -2.0, 1.0
    ymin, ymax = -1.5, 1.5
    return xmin, ymin, xmax, ymax


def _colour_map(iters: int, max_iters: int = 256) -> Tuple[int, int, int]:
    if iters >= max_iters:
        return 0, 0, 0
    t = iters / max_iters
    r = int(9 * (1 - t) * t * t * t * 255)
    g = int(15 * (1 - t) * (1 - t) * t * t * 255)
    b = int(8.5 * (1 - t) * (1 - t) * (1 - t) * t * 255)
    return r, g, b


# Include Datastar from CDN as in the official examples
_datastar_script = Script(
    src="https://cdn.jsdelivr.net/gh/starfederation/datastar@v1.0.0-RC.7/bundles/datastar.js",
    type="module",
)

app, rt = fast_app(htmx=False, hdrs=(_datastar_script,))


@rt("/")
async def index(request):
    # Datastar keeps track of signals on the element with data-signals.
    # These signals will be sent back to the backend on each request.
    signals = {
        "xmin": -2.0,
        "xmax": 1.0,
        "ymin": -1.5,
        "ymax": 1.5,
    }

    controls = Div(
        H1("mojo-data-star: Interactive Mandelbrot"),
        P("Use the controls to tweak the view; Datastar streams updates from Mojo."),
        Div(
            Label("x min", Input(type="number", step="0.01", data_bind="xmin")),
            Label("x max", Input(type="number", step="0.01", data_bind="xmax")),
            Label("y min", Input(type="number", step="0.01", data_bind="ymin")),
            Label("y max", Input(type="number", step="0.01", data_bind="ymax")),
            Button(
                "Render",
                data_on_click="@get('/mandelbrot')",
                cls="btn",
            ),
            cls="controls",
        ),
        Div(id="mandel-container", data_on_load="@get('/mandelbrot')"),
    )

    body = Body(
        Div(
            controls,
            data_signals=signals,
        )
    )

    return Html(
        Head(_datastar_script),
        body,
    )


@rt("/mandelbrot")
async def mandelbrot_route(request):
    """Datastar SSE endpoint that renders the current view as an image.

    It reads the current Datastar signals, runs the Mojo Mandelbrot computation,
    and streams back an updated fragment for the mandel-container div.
    """
    if compute_mandelbrot is None:
        async def _error():
            yield DatastarStreamingResponse.merge_fragments(
                Div("Mojo extension module 'mandelbrot' is not built yet.", id="mandel-container")
            )

        return DatastarStreamingResponse(_error())

    data = await request.json()

    xmin = float(data.get("xmin", -2.0))
    xmax = float(data.get("xmax", 1.0))
    ymin = float(data.get("ymin", -1.5))
    ymax = float(data.get("ymax", 1.5))

    iters_tensor = compute_mandelbrot(WIDTH, HEIGHT, xmin, ymin, xmax, ymax)

    img = Image.new("RGB", (WIDTH, HEIGHT))
    pixels = img.load()

    for y in range(HEIGHT):
        for x in range(WIDTH):
            iters = int(iters_tensor[y, x])
            pixels[x, y] = _colour_map(iters)

    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode("ascii")

    fragment = Div(
        Img(
            src=f"data:image/png;base64,{img_b64}",
            alt="Mandelbrot",
            id="mandel-img",
            width=str(WIDTH),
            height=str(HEIGHT),
        ),
        id="mandel-container",
    )

    async def _():
        yield DatastarStreamingResponse.merge_fragments(fragment)

    return DatastarStreamingResponse(_())
