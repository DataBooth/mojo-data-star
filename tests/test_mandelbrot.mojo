from mandelbrot import compute_mandelbrot
from testing import assert_equal, assert_true, TestSuite


def test_dimensions():
    let width = 10
    let height = 6
    let xmin = -2.0
    let ymin = -1.5
    let xmax = 1.0
    let ymax = 1.5

    let iters = compute_mandelbrot(width, height, xmin, ymin, xmax, ymax)

    # Tensor is indexed [row, col] = [y, x]
    assert_equal(iters.shape[0], height)
    assert_equal(iters.shape[1], width)


def test_values_in_range():
    let width = 8
    let height = 8
    let xmin = -2.0
    let ymin = -1.5
    let xmax = 1.0
    let ymax = 1.5

    let iters = compute_mandelbrot(width, height, xmin, ymin, xmax, ymax)

    for y in range(height):
        for x in range(width):
            let v = iters[y, x]
            assert_true(v >= 0)
            # MAX_ITERS is 256 in the implementation; keep the bound in sync.
            assert_true(v <= 256)


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
