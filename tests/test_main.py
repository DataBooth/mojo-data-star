from __future__ import annotations

from starlette.testclient import TestClient

import main


def test_colour_map_range() -> None:
    r, g, b = main._colour_map(10, max_iters=256)

    for channel in (r, g, b):
        assert 0 <= channel <= 255


def test_mandelbrot_route_renders_image(monkeypatch) -> None:
    class FakeTensor:
        def __getitem__(self, idx):  # type: ignore[override]
            # Always return a modest iteration count inside the range
            return 10

    def fake_compute(width, height, xmin, ymin, xmax, ymax):  # type: ignore[override]
        return FakeTensor()

    monkeypatch.setattr(main, "compute_mandelbrot", fake_compute)

    client = TestClient(main.app)
    response = client.post("/mandelbrot", json={"xmin": -2.0, "xmax": 1.0, "ymin": -1.5, "ymax": 1.5})

    assert response.status_code == 200
    # Response is an SSE stream; for this smoke test just ensure we see PNG data URL
    text = response.text
    assert "data:image/png;base64" in text


def test_mandelbrot_route_reports_error_when_extension_missing(monkeypatch) -> None:
    monkeypatch.setattr(main, "compute_mandelbrot", None)

    client = TestClient(main.app)
    response = client.post("/mandelbrot", json={})

    assert response.status_code == 200
    assert "Mojo extension module" in response.text
