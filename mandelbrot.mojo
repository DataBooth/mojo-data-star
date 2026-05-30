from std.collections import List
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from std.os import abort

# Interactive Mandelbrot kernel and host-side computation.
# NOTE: This version targets the Mojo 1.0 beta toolchain.
# It currently runs on the CPU but is structured so a GPU kernel
# can be slotted in following the gpu.host fundamentals docs.

comptime MAX_ITERS: Int32 = 256


struct MandelbrotGrid(Copyable, Movable):
    var width: Int
    var height: Int
    var values: List[Int32]

    def __init__(out self, width: Int, height: Int, values: List[Int32]):
        self.width = width
        self.height = height
        self.values = values.copy()

    def copy(self) -> Self:
        var copied = List[Int32]()
        for i in range(len(self.values)):
            copied.append(self.values[i])
        return MandelbrotGrid(self.width, self.height, copied)

    def __getitem__(self, y: Int, x: Int) -> Int32:
        return self.values[y * self.width + x]


def mandelbrot_scalar(cx: Float64, cy: Float64) -> Int32:
    var zx: Float64 = 0.0
    var zy: Float64 = 0.0
    var iter: Int32 = 0

    while iter < MAX_ITERS and (zx * zx + zy * zy) <= 4.0:
        var xtemp = zx * zx - zy * zy + cx
        zy = 2.0 * zx * zy + cy
        zx = xtemp
        iter += 1

    return iter


def compute_mandelbrot(
    width: Int,
    height: Int,
    xmin: Float64,
    ymin: Float64,
    xmax: Float64,
    ymax: Float64,
) -> MandelbrotGrid:
    """Mojo-native entry point.

    Returns a 2D grid of iteration counts shaped (height, width).
    This is used by both Mojo tests and the Python wrapper.
    """
    var values = List[Int32]()

    var dx = (xmax - xmin) / Float64(width)
    var dy = (ymax - ymin) / Float64(height)

    for y in range(height):
        var cy = ymin + dy * Float64(y)
        for x in range(width):
            var cx = xmin + dx * Float64(x)
            values.append(mandelbrot_scalar(cx, cy))

    return MandelbrotGrid(width, height, values)


def compute_mandelbrot_gpu_tensor(
    width: Int,
    height: Int,
    xmin: Float64,
    ymin: Float64,
    xmax: Float64,
    ymax: Float64,
) -> MandelbrotGrid:
    """Placeholder for a GPU-backed Mandelbrot implementation.

    This currently forwards to the CPU implementation.
    """
    return compute_mandelbrot(width, height, xmin, ymin, xmax, ymax)


@export
def PyInit_mandelbrot() -> PythonObject:
    """Initialize the Python extension module ``mandelbrot``."""
    try:
        var mb = PythonModuleBuilder("mandelbrot")
        return mb.finalize()
    except e:
        abort(String("error creating Mojo Python module 'mandelbrot': ", e))
