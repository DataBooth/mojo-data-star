mojo

from python import PythonObject
from python.bindings import PythonModuleBuilder
from tensor import Tensor, TensorShape, DType
from math import sqrt
from os import abort

# Interactive Mandelbrot kernel and host-side computation.
# NOTE: This version is written for open-source Mojo 0.25.x.
# It currently runs on the CPU but is structured so a GPU kernel
# can be slotted in following the gpu.host fundamentals docs.

comptime let MAX_ITERS: Int32 = 256

fn mandelbrot_scalar(cx: Float64, cy: Float64) -> Int32:
    var zx: Float64 = 0.0
    var zy: Float64 = 0.0
    var iter: Int32 = 0

    while iter < MAX_ITERS and (zx * zx + zy * zy) <= 4.0:
        let xtemp = zx * zx - zy * zy + cx
        zy = 2.0 * zx * zy + cy
        zx = xtemp
        iter += 1

    return iter

fn compute_mandelbrot_tensor(
    width: Int,
    height: Int,
    xmin: Float64,
    ymin: Float64,
    xmax: Float64,
    ymax: Float64,
) -> Tensor[Int32]:
    # Create a 2D tensor [height, width] of iteration counts.
    let shape = TensorShape([height, width])
    var result = Tensor[Int32](shape)

    let dx = (xmax - xmin) / Float64(width)
    let dy = (ymax - ymin) / Float64(height)

    for y in range(height):
        let cy = ymin + dy * Float64(y)
        for x in range(width):
            let cx = xmin + dx * Float64(x)
            let iters = mandelbrot_scalar(cx, cy)
            result[y, x] = iters

    return result

fn compute_mandelbrot(
    width: Int,
    height: Int,
    xmin: Float64,
    ymin: Float64,
    xmax: Float64,
    ymax: Float64,
) -> Tensor[Int32]:
    """Mojo-native entry point.

    Returns a 2D tensor of iteration counts shaped (height, width).
    This is used by both Mojo tests and the Python wrapper.
    """
    return compute_mandelbrot_tensor(width, height, xmin, ymin, xmax, ymax)

fn compute_mandelbrot_py(
    width_obj: PythonObject,
    height_obj: PythonObject,
    xmin_obj: PythonObject,
    ymin_obj: PythonObject,
    xmax_obj: PythonObject,
    ymax_obj: PythonObject,
) raises -> PythonObject:
    """PythonObject-based wrapper suitable for Python bindings.

    Accepts six Python arguments (ints/floats), converts them to Mojo
    types, calls the native compute_mandelbrot, and returns a Python
    object wrapping the resulting Tensor[Int32].
    """
    var width = Int(py=width_obj)
    var height = Int(py=height_obj)
    var xmin = Float64(py=xmin_obj)
    var ymin = Float64(py=ymin_obj)
    var xmax = Float64(py=xmax_obj)
    var ymax = Float64(py=ymax_obj)

    var iters = compute_mandelbrot(width, height, xmin, ymin, xmax, ymax)
    return PythonObject(alloc=iters)

@export
fn PyInit_mandelbrot() -> PythonObject:
    """Initialize the Python extension module ``mandelbrot``.

    This follows the pattern from "Calling Mojo from Python" in the
    Mojo manual: Python looks for ``PyInit_<module>()`` and we use
    PythonModuleBuilder to expose our wrapper function as a normal
    Python function.
    """
    try:
        var mb = PythonModuleBuilder("mandelbrot")
        mb.def_function[compute_mandelbrot_py](
            "compute_mandelbrot",
            docstring="Compute Mandelbrot escape iterations as a 2D Tensor[Int32].",
        )
        return mb.finalize()
    except e:
        abort(String("error creating Mojo Python module 'mandelbrot': ", e))
