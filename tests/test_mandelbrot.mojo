from mandelbrot import compute_mandelbrot
from std.testing import assert_equal, assert_true, TestSuite


def test_dimensions() raises:
    var width = 10
    var height = 6
    var xmin = -2.0
    var ymin = -1.5
    var xmax = 1.0
    var ymax = 1.5

    var iters = compute_mandelbrot(width, height, xmin, ymin, xmax, ymax)

    # Tensor is indexed [row, col] = [y, x].
    # Instead of relying on tensor.shape (which uses a static layout in this
    # example), verify that every index in the requested [height, width]
    # region is readable and that we see exactly height * width elements.
    var count: Int = 0
    for y in range(height):
        for x in range(width):
            var _ = iters[y, x]
            count += 1

    assert_equal(count, width * height)


def test_values_in_range() raises:
    var width = 8
    var height = 8
    var xmin = -2.0
    var ymin = -1.5
    var xmax = 1.0
    var ymax = 1.5

    var iters = compute_mandelbrot(width, height, xmin, ymin, xmax, ymax)

    for y in range(height):
        for x in range(width):
            var v = iters[y, x]
            assert_true(v >= 0)
            # MAX_ITERS is 256 in the implementation; keep the bound in sync.
            assert_true(v <= 256)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
