from __future__ import annotations

import hashlib
import math
import typing_extensions
from typing import Any, Optional, TYPE_CHECKING, Union

import sympy  # noqa: TC002

import torch  # noqa: TC001
from torch.utils._ordered_set import OrderedSet
from torch.utils._pallas import has_tpu_pallas
from torch.utils._sympy.functions import ModularIndexing

from .. import config
from ..ir import (
    ComputedBuffer,
    PALLAS_EXPAND_STRIDE,
    PallasIndirectStride,
    PallasStride,
    strip_pallas_stride,
)
from ..runtime.runtime_utils import torch_dtype_to_jax
from ..utils import get_fused_kernel_name, get_kernel_metadata
from ..virtualized import V
from .block_analysis import BlockPatternMatcher
from .common import (
    BackendFeature,
    CSEVariable,
    IndentedBuffer,
    OpOverrides,
    PythonPrinter,
)
from .simd import SIMDKernel, SIMDScheduling


class PallasPrinter(PythonPrinter):
    """
    Custom sympy printer for Pallas that handles JAX-specific constructs.
    """

    def _print_Where(self, expr: sympy.Expr) -> str:
        """Convert sympy Where to jnp.where."""
        c = self.doprint(expr.args[0])
        p = self.doprint(expr.args[1])
        q = self.doprint(expr.args[2])
        return f"jnp.where({c}, {p}, {q})"

    def _print_Min(self, expr: sympy.Expr) -> str:
        """Convert sympy Min to jnp.minimum for JAX compatibility."""
        args = [self.doprint(arg) for arg in expr.args]
        result = args[0]
        for arg in args[1:]:
            result = f"jnp.minimum({result}, {arg})"
        return result

    def _print_Max(self, expr: sympy.Expr) -> str:
        """Convert sympy Max to jnp.maximum for JAX compatibility."""
        args = [self.doprint(arg) for arg in expr.args]
        result = args[0]
        for arg in args[1:]:
            result = f"jnp.maximum({result}, {arg})"
        return result

    def _print_PallasStride(self, expr: sympy.Expr) -> str:
        """Strip PallasStride marker and return just the stride value."""
        return self.doprint(expr.args[0])

    def _print_PallasIndirectStride(self, expr: sympy.Expr) -> str:
        """Strip PallasIndirectStride marker and return just the stride value."""
        return self.doprint(expr.args[0])


# Use Pallas-specific printer for expression generation
pallas_pexpr = PallasPrinter().doprint


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from ..ir import IRNode
    from ..ops_handler import ReductionType
    from ..scheduler import BaseSchedulerNode


# Main function suffix used in generated Pallas code
MAIN_SUFFIX = "main"

# Mosaic GPU warpgroup size: 4 warps × 32 threads = 128 threads per warpgroup.
# This is a hardware constant for Hopper and Blackwell GPUs.
# See: jax/_src/pallas/mosaic_gpu/lowering.py
WARPGROUP_SIZE = 128


def _align_to_warpgroup(size: int) -> int:
    """Align size to WARPGROUP_SIZE (128) for Mosaic GPU compatibility."""
    return ((size + WARPGROUP_SIZE - 1) // WARPGROUP_SIZE) * WARPGROUP_SIZE


# Logger for Pallas kernel code
kernel_code_log = torch._logging.getArtifactLogger(__name__, "kernel_code")


class PallasKernelWrapper:
    """Wrapper to provide .run() interface for Pallas kernels"""

    def __init__(
        self, kernel_fn: Callable[..., Any], kernel_path: Optional[str] = None
    ):
        self.kernel_fn = kernel_fn
        self.kernel_path = kernel_path
        kernel_code_log.info("Pallas kernel path: %s", kernel_path)

    def run(self, *args, stream=None, **kwargs):
        """
        Execute the Pallas kernel.

        Args:
            *args: Arguments to pass to the kernel function
            stream: CUDA stream to pass to the kernel function
            **kwargs: Additional keyword arguments for the kernel

        Returns:
            Result of the kernel execution
        """
        return self.kernel_fn(*args, stream=stream, **kwargs)


class Unsupported(RuntimeError):
    """Exception raised when an operation is not supported by the Pallas backend."""


class PallasKernelOverrides(OpOverrides):
    """
    Map element-wise ops to JAX/Pallas operations.

    For now, we use the default Python operators which are compatible
    with JAX numpy broadcasting semantics.
    """

    @staticmethod
    def sin(x: str) -> str:
        return f"jnp.sin({x})"

    @staticmethod
    def cos(x: str) -> str:
        return f"jnp.cos({x})"

    @staticmethod
    def tan(x: str) -> str:
        return f"jnp.tan({x})"

    @staticmethod
    def sinh(x: str) -> str:
        return f"jnp.sinh({x})"

    @staticmethod
    def cosh(x: str) -> str:
        return f"jnp.cosh({x})"

    @staticmethod
    def tanh(x: str) -> str:
        return f"jnp.tanh({x})"

    @staticmethod
    def asin(x: str) -> str:
        return f"jnp.arcsin({x})"

    @staticmethod
    def acos(x: str) -> str:
        return f"jnp.arccos({x})"

    @staticmethod
    def atan(x: str) -> str:
        return f"jnp.arctan({x})"

    @staticmethod
    def exp(x: str) -> str:
        return f"jnp.exp({x})"

    @staticmethod
    def exp2(x: str) -> str:
        return f"jnp.exp2({x})"

    @staticmethod
    def expm1(x: str) -> str:
        return f"jnp.expm1({x})"

    @staticmethod
    def log(x: str) -> str:
        return f"jnp.log({x})"

    @staticmethod
    def log10(x: str) -> str:
        return f"jnp.log10({x})"

    @staticmethod
    def log2(x: str) -> str:
        return f"jnp.log2({x})"

    @staticmethod
    def log1p(x: str) -> str:
        return f"jnp.log1p({x})"

    @staticmethod
    def sqrt(x: str) -> str:
        return f"jnp.sqrt({x})"

    @staticmethod
    def rsqrt(x: str) -> str:
        return f"(1.0 / jnp.sqrt({x}))"

    @staticmethod
    def abs(x: str) -> str:
        return f"jnp.abs({x})"

    @staticmethod
    def neg(x: str) -> str:
        return f"(-{x})"

    @staticmethod
    def floor(x: str) -> str:
        return f"jnp.floor({x})"

    @staticmethod
    def ceil(x: str) -> str:
        return f"jnp.ceil({x})"

    @staticmethod
    def trunc(x: str) -> str:
        return f"jnp.trunc({x})"

    @staticmethod
    def round(x: str) -> str:
        return f"jnp.round({x})"

    @staticmethod
    def sigmoid(x: str) -> str:
        return f"(1.0 / (1.0 + jnp.exp(-{x})))"

    @staticmethod
    def relu(x: str) -> str:
        return f"jnp.maximum({x}, 0)"

    @staticmethod
    def pow(a: str, b: str) -> str:
        return f"jnp.power({a}, {b})"

    @staticmethod
    def maximum(a: str, b: str) -> str:
        return f"jnp.maximum({a}, {b})"

    @staticmethod
    def minimum(a: str, b: str) -> str:
        return f"jnp.minimum({a}, {b})"

    @staticmethod
    def where(cond: str, a: str, b: str) -> str:
        return f"jnp.where({cond}, {a}, {b})"

    @staticmethod
    def masked(mask: str, body: Callable[[], str], other: float) -> str:
        """
        Computes body, but only uses the result where mask is true.
        Where mask is false, uses the 'other' value instead.
        """
        result = body()
        # Format the 'other' value properly for JAX
        if isinstance(other, float):
            if math.isnan(other):
                other_str = "jnp.nan"
            elif math.isinf(other):
                other_str = "jnp.inf" if other > 0 else "-jnp.inf"
            else:
                other_str = repr(other)
        else:
            other_str = repr(other)
        # Use jnp.where to select between result and other based on mask
        return f"jnp.where({mask}, {result}, {other_str})"

    @staticmethod
    def to_dtype(
        x: str,
        dtype: torch.dtype,
        src_dtype: Optional[torch.dtype] = None,
        use_compute_types: bool = True,
    ) -> str:
        jax_dtype = torch_dtype_to_jax(dtype)
        # Wrap in jnp.asarray to handle scalars from integer indexing
        return f"jnp.asarray({x}).astype({jax_dtype})"

    @staticmethod
    def to_dtype_bitcast(x: str, dtype: torch.dtype, src_dtype: torch.dtype) -> str:
        """Bitcast a value from one dtype to another with the same size."""
        jax_dtype = torch_dtype_to_jax(dtype)
        jax_src_dtype = torch_dtype_to_jax(src_dtype)
        # First ensure the value is the correct source dtype, then bitcast
        return f"jax.lax.bitcast_convert_type(jnp.asarray({x}).astype({jax_src_dtype}), {jax_dtype})"

    @staticmethod
    def index_expr(expr: sympy.Expr, dtype: torch.dtype) -> str:
        """Convert a sympy expression to a JAX array indexing expression."""
        from ..utils import get_bounds_index_expr

        # Prepare and rename indexing to register size symbols as kernel args
        prepared = V.kernel.prepare_indexing(expr)
        renamed = V.kernel.rename_indexing(prepared)
        idx_str = V.kernel.kexpr(renamed)
        var = V.kernel.cse.generate(
            V.kernel.compute, idx_str, bounds=get_bounds_index_expr(expr)
        )
        return PallasKernelOverrides.to_dtype(var, dtype)

    @staticmethod
    def constant(val, dtype: torch.dtype) -> str:
        """Convert a constant value to JAX representation."""
        jax_dtype = torch_dtype_to_jax(dtype)
        if dtype == torch.bool:
            return "True" if val else "False"
        # Handle special float values
        if isinstance(val, float):
            if math.isnan(val):
                return "jnp.nan"
            if math.isinf(val):
                return "jnp.inf" if val > 0 else "-jnp.inf"
        return f"jnp.array({val}, dtype={jax_dtype})"

    @staticmethod
    def real(x: str) -> str:
        return f"jnp.real({x})"

    @staticmethod
    def imag(x: str) -> str:
        return f"jnp.imag({x})"

    @staticmethod
    def conj(x: str) -> str:
        return f"jnp.conj({x})"

    @staticmethod
    def angle(x: str) -> str:
        return f"jnp.angle({x})"

    @staticmethod
    def view_as_real(x: str) -> str:
        """View complex tensor as real tensor with extra dimension."""
        return f"jnp.stack([jnp.real({x}), jnp.imag({x})], axis=-1)"

    @staticmethod
    def view_as_complex(x: str) -> str:
        """View real tensor as complex tensor."""
        return f"({x}[..., 0] + 1j * {x}[..., 1])"

    # Comparison operations
    @staticmethod
    def eq(a: str, b: str) -> str:
        return f"({a} == {b})"

    @staticmethod
    def ne(a: str, b: str) -> str:
        return f"({a} != {b})"

    @staticmethod
    def lt(a: str, b: str) -> str:
        return f"({a} < {b})"

    @staticmethod
    def le(a: str, b: str) -> str:
        return f"({a} <= {b})"

    @staticmethod
    def gt(a: str, b: str) -> str:
        return f"({a} > {b})"

    @staticmethod
    def isnan(x: str) -> str:
        return f"jnp.isnan({x})"

    @staticmethod
    def isinf(x: str) -> str:
        return f"jnp.isinf({x})"

    @staticmethod
    def isfinite(x: str) -> str:
        return f"jnp.isfinite({x})"

    @staticmethod
    def ge(a: str, b: str) -> str:
        return f"({a} >= {b})"

    # Logical operations
    @staticmethod
    def logical_and(a: str, b: str) -> str:
        return f"jnp.logical_and({a}, {b})"

    @staticmethod
    def logical_or(a: str, b: str) -> str:
        return f"jnp.logical_or({a}, {b})"

    @staticmethod
    def logical_not(x: str) -> str:
        return f"jnp.logical_not({x})"

    @staticmethod
    def logical_xor(a: str, b: str) -> str:
        return f"jnp.logical_xor({a}, {b})"

    # Math operations
    @staticmethod
    def atan2(a: str, b: str) -> str:
        return f"jnp.arctan2({a}, {b})"

    @staticmethod
    def hypot(a: str, b: str) -> str:
        return f"jnp.hypot({a}, {b})"

    @staticmethod
    def fmod(a: str, b: str) -> str:
        return f"jnp.fmod({a}, {b})"

    @staticmethod
    def remainder(a: str, b: str) -> str:
        return f"jnp.remainder({a}, {b})"

    @staticmethod
    def truncdiv(a: str, b: str) -> str:
        # Truncated division (rounds toward zero)
        # For integers: sign(a)*sign(b) * (abs(a) // abs(b))
        return f"(jnp.sign({a}) * jnp.sign({b}) * (jnp.abs({a}) // jnp.abs({b}))).astype({a}.dtype)"

    @staticmethod
    def floordiv(a: str, b: str) -> str:
        return f"({a} // {b})"

    @staticmethod
    def clamp(x: str, min_val: str, max_val: str) -> str:
        return f"jnp.clip({x}, {min_val}, {max_val})"

    @staticmethod
    def clip(x: str, min_val: str, max_val: str) -> str:
        return f"jnp.clip({x}, {min_val}, {max_val})"

    # Sign operations
    @staticmethod
    def sign(x: str) -> str:
        return f"jnp.sign({x})"

    @staticmethod
    def signbit(x: str) -> str:
        return f"jnp.signbit({x})"

    # Special math functions
    @staticmethod
    def erf(x: str) -> str:
        return f"jax.scipy.special.erf({x})"

    @staticmethod
    def erfc(x: str) -> str:
        return f"jax.scipy.special.erfc({x})"

    @staticmethod
    def erfinv(x: str) -> str:
        return f"jax.scipy.special.erfinv({x})"

    @staticmethod
    def lgamma(x: str) -> str:
        return f"jax.scipy.special.gammaln({x})"

    @staticmethod
    def digamma(x: str) -> str:
        return f"jax.scipy.special.digamma({x})"

    @staticmethod
    def bessel_j0(x: str) -> str:
        # bessel_jn requires float64 and has numerical issues at x=0 (returns NaN)
        # bessel_jn(x, v=n) returns array of shape (n+1, ...) with J_0 to J_n
        # Handle by: convert to float64, compute, handle x=0, convert back
        # J0(0) = 1.0
        return (
            f"jnp.where({x}.astype(jnp.float64) == 0.0, 1.0, "
            f"jax.scipy.special.bessel_jn({x}.astype(jnp.float64), v=0)[0])"
            f".astype({x}.dtype)"
        )

    @staticmethod
    def bessel_j1(x: str) -> str:
        # bessel_jn requires float64 and has numerical issues at x=0 (returns NaN)
        # bessel_jn(x, v=n) returns array of shape (n+1, ...) with J_0 to J_n
        # Handle by: convert to float64, compute, handle x=0, convert back
        # J1(0) = 0.0
        return (
            f"jnp.where({x}.astype(jnp.float64) == 0.0, 0.0, "
            f"jax.scipy.special.bessel_jn({x}.astype(jnp.float64), v=1)[1])"
            f".astype({x}.dtype)"
        )

    @staticmethod
    def modified_bessel_i0(x: str) -> str:
        # Modified Bessel function of the first kind I_0(x)
        # I_0(x) = bessel_i0e(x) * exp(|x|) where bessel_i0e is the scaled version
        return f"jax.lax.bessel_i0e({x}) * jnp.exp(jnp.abs({x}))"

    @staticmethod
    def modified_bessel_i1(x: str) -> str:
        # Modified Bessel function of the first kind I_1(x)
        # I_1(x) = bessel_i1e(x) * exp(|x|) where bessel_i1e is the scaled version
        return f"jax.lax.bessel_i1e({x}) * jnp.exp(jnp.abs({x}))"

    @staticmethod
    def spherical_bessel_j0(x: str) -> str:
        # Spherical Bessel function of the first kind j_0(x) = sin(x) / x
        # Handle x=0: j_0(0) = 1
        return f"jnp.where({x} == 0.0, 1.0, jnp.sin({x}) / {x})"

    @staticmethod
    def i0(x: str) -> str:
        # Modified Bessel function I_0 (same as modified_bessel_i0)
        return f"jax.lax.bessel_i0e({x}) * jnp.exp(jnp.abs({x}))"

    @staticmethod
    def i0e(x: str) -> str:
        # Exponentially scaled modified Bessel function I_0
        return f"jax.lax.bessel_i0e({x})"

    @staticmethod
    def i1(x: str) -> str:
        # Modified Bessel function I_1 (same as modified_bessel_i1)
        return f"jax.lax.bessel_i1e({x}) * jnp.exp(jnp.abs({x}))"

    @staticmethod
    def i1e(x: str) -> str:
        # Exponentially scaled modified Bessel function I_1
        return f"jax.lax.bessel_i1e({x})"

    @staticmethod
    def gammainc(x: str, y: str) -> str:
        # Regularized lower incomplete gamma function P(a, x)
        # Note: PyTorch uses gammainc(input, other) where input is a (shape param)
        return f"jax.scipy.special.gammainc({x}, {y})"

    @staticmethod
    def gammaincc(x: str, y: str) -> str:
        # Regularized upper incomplete gamma function Q(a, x)
        return f"jax.scipy.special.gammaincc({x}, {y})"

    @staticmethod
    def igamma(x: str, y: str) -> str:
        # Regularized lower incomplete gamma function (alias for gammainc)
        return f"jax.scipy.special.gammainc({x}, {y})"

    @staticmethod
    def igammac(x: str, y: str) -> str:
        # Regularized upper incomplete gamma function (alias for gammaincc)
        return f"jax.scipy.special.gammaincc({x}, {y})"

    @staticmethod
    def polygamma(x: str, y: str) -> str:
        # Polygamma function psi^(n)(x), x is order n, y is the value
        # Note: JAX uses polygamma(n, x) where n is integer order
        return f"jax.scipy.special.polygamma({x}.astype(jnp.int32), {y})"

    @staticmethod
    def ndtri(x: str) -> str:
        # Inverse of the standard normal CDF
        return f"jax.scipy.special.ndtri({x})"

    @staticmethod
    def zeta(x: str, y: str) -> str:
        # Hurwitz zeta function zeta(x, q) = sum_{k=0}^inf 1/(k+q)^x
        return f"jax.scipy.special.zeta({x}, {y})"

    @staticmethod
    def xlogy(x: str, y: str) -> str:
        # x * log(y), with proper handling of x=0
        return f"jax.scipy.special.xlogy({x}, {y})"

    @staticmethod
    def xlog1py(x: str, y: str) -> str:
        # x * log1p(y), with proper handling of x=0
        return f"jax.scipy.special.xlog1py({x}, {y})"

    @staticmethod
    def chebyshev_polynomial_t(x: str, n: str) -> str:
        # Chebyshev polynomial of the first kind T_n(x)
        # For |x| <= 1: T_n(x) = cos(n * arccos(x))
        # For x > 1: T_n(x) = cosh(n * arccosh(x))
        # For x < -1: T_n(x) = (-1)^n * cosh(n * arccosh(-x))
        return (
            f"jnp.where(jnp.abs({x}) <= 1, "
            f"jnp.cos({n} * jnp.arccos(jnp.clip({x}, -1, 1))), "
            f"jnp.where({x} > 1, "
            f"jnp.cosh({n} * jnp.arccosh(jnp.maximum({x}, 1.0))), "
            f"((-1.0) ** {n}) * jnp.cosh({n} * jnp.arccosh(jnp.maximum(-{x}, 1.0)))))"
        )

    @staticmethod
    def chebyshev_polynomial_u(x: str, n: str) -> str:
        # Chebyshev polynomial of the second kind U_n(x)
        # For |x| < 1: U_n(x) = sin((n+1) * arccos(x)) / sqrt(1 - x^2)
        # For x = 1: U_n(1) = n+1
        # For x = -1: U_n(-1) = (-1)^n * (n+1)
        # For x > 1: U_n(x) = sinh((n+1) * arccosh(x)) / sqrt(x^2 - 1)
        # For x < -1: U_n(x) = (-1)^n * U_n(-x) (symmetry)
        return (
            f"jnp.where(jnp.abs({x}) < 1, "
            f"jnp.sin(({n} + 1) * jnp.arccos(jnp.clip({x}, -1, 1))) / "
            f"jnp.sqrt(jnp.maximum(1 - {x}**2, 1e-10)), "
            f"jnp.where({x} >= 1, "
            f"jnp.where({x} == 1, {n} + 1.0, "
            f"jnp.sinh(({n} + 1) * jnp.arccosh(jnp.maximum({x}, 1.0))) / "
            f"jnp.sqrt(jnp.maximum({x}**2 - 1, 1e-10))), "
            f"jnp.where({x} == -1, ((-1.0) ** {n}) * ({n} + 1.0), "
            f"((-1.0) ** {n}) * jnp.sinh(({n} + 1) * jnp.arccosh(jnp.maximum(-{x}, 1.0))) / "
            f"jnp.sqrt(jnp.maximum({x}**2 - 1, 1e-10)))))"
        )

    @staticmethod
    def chebyshev_polynomial_v(x: str, n: str) -> str:
        # Chebyshev polynomial of the third kind V_n(x)
        # V_n(x) = (T_n(x) - T_{n+1}(x)) / (1 - x) for x != 1
        # V_n(1) = 1, recurrence: V_0 = 1, V_1 = 2x - 1, V_n = 2x*V_{n-1} - V_{n-2}
        # Explicit: V_0 = 1, V_1 = 2x-1, V_2 = 4x^2-2x-1, V_3 = 8x^3-4x^2-4x+1
        return (
            f"jnp.where({n} == 0, jnp.ones_like({x}), "
            f"jnp.where({n} == 1, 2*{x} - 1, "
            f"jnp.where({n} == 2, 4*{x}**2 - 2*{x} - 1, "
            f"jnp.where({n} == 3, 8*{x}**3 - 4*{x}**2 - 4*{x} + 1, "
            f"jnp.where({n} == 4, 16*{x}**4 - 8*{x}**3 - 12*{x}**2 + 4*{x} + 1, "
            f"jnp.where({n} == 5, 32*{x}**5 - 16*{x}**4 - 32*{x}**3 + 12*{x}**2 + 6*{x} - 1, "
            f"jnp.zeros_like({x})))))))"
        )

    @staticmethod
    def chebyshev_polynomial_w(x: str, n: str) -> str:
        # Chebyshev polynomial of the fourth kind W_n(x)
        # W_n(x) = (T_n(x) + T_{n+1}(x)) / (1 + x) for x != -1
        # W_n(-1) = (-1)^n, recurrence: W_0 = 1, W_1 = 2x + 1, W_n = 2x*W_{n-1} - W_{n-2}
        # Explicit: W_0 = 1, W_1 = 2x+1, W_2 = 4x^2+2x-1, W_3 = 8x^3+4x^2-4x-1
        return (
            f"jnp.where({n} == 0, jnp.ones_like({x}), "
            f"jnp.where({n} == 1, 2*{x} + 1, "
            f"jnp.where({n} == 2, 4*{x}**2 + 2*{x} - 1, "
            f"jnp.where({n} == 3, 8*{x}**3 + 4*{x}**2 - 4*{x} - 1, "
            f"jnp.where({n} == 4, 16*{x}**4 + 8*{x}**3 - 12*{x}**2 - 4*{x} + 1, "
            f"jnp.where({n} == 5, 32*{x}**5 + 16*{x}**4 - 32*{x}**3 - 12*{x}**2 + 6*{x} + 1, "
            f"jnp.zeros_like({x})))))))"
        )

    @staticmethod
    def shifted_chebyshev_polynomial_t(x: str, n: str) -> str:
        # Shifted Chebyshev polynomial of the first kind T*_n(x) = T_n(2x - 1)
        # T_n(y) where y = 2x - 1
        # Use same formula as chebyshev_polynomial_t
        y = f"(2 * {x} - 1)"
        return (
            f"jnp.where(jnp.abs({y}) <= 1, "
            f"jnp.cos({n} * jnp.arccos(jnp.clip({y}, -1, 1))), "
            f"jnp.where({y} > 1, "
            f"jnp.cosh({n} * jnp.arccosh(jnp.maximum({y}, 1.0))), "
            f"((-1.0) ** {n}) * jnp.cosh({n} * jnp.arccosh(jnp.maximum(-{y}, 1.0)))))"
        )

    @staticmethod
    def shifted_chebyshev_polynomial_u(x: str, n: str) -> str:
        # Shifted Chebyshev polynomial of the second kind U*_n(x) = U_n(2x - 1)
        # Use same formula as chebyshev_polynomial_u
        y = f"(2 * {x} - 1)"
        return (
            f"jnp.where(jnp.abs({y}) < 1, "
            f"jnp.sin(({n} + 1) * jnp.arccos(jnp.clip({y}, -1, 1))) / "
            f"jnp.sqrt(jnp.maximum(1 - ({y})**2, 1e-10)), "
            f"jnp.where({y} >= 1, "
            f"jnp.where({y} == 1, {n} + 1.0, "
            f"jnp.sinh(({n} + 1) * jnp.arccosh(jnp.maximum({y}, 1.0))) / "
            f"jnp.sqrt(jnp.maximum({y}**2 - 1, 1e-10))), "
            f"jnp.where({y} == -1, ((-1.0) ** {n}) * ({n} + 1.0), "
            f"((-1.0) ** {n}) * jnp.sinh(({n} + 1) * jnp.arccosh(jnp.maximum(-{y}, 1.0))) / "
            f"jnp.sqrt(jnp.maximum({y}**2 - 1, 1e-10)))))"
        )

    @staticmethod
    def shifted_chebyshev_polynomial_v(x: str, n: str) -> str:
        # Shifted Chebyshev polynomial of the third kind V*_n(x) = V_n(2x - 1)
        y = f"(2 * {x} - 1)"  # shifted variable
        return (
            f"jnp.where({n} == 0, jnp.ones_like({x}), "
            f"jnp.where({n} == 1, 2*{y} - 1, "
            f"jnp.where({n} == 2, 4*{y}**2 - 2*{y} - 1, "
            f"jnp.where({n} == 3, 8*{y}**3 - 4*{y}**2 - 4*{y} + 1, "
            f"jnp.where({n} == 4, 16*{y}**4 - 8*{y}**3 - 12*{y}**2 + 4*{y} + 1, "
            f"jnp.where({n} == 5, 32*{y}**5 - 16*{y}**4 - 32*{y}**3 + 12*{y}**2 + 6*{y} - 1, "
            f"jnp.zeros_like({x})))))))"
        )

    @staticmethod
    def shifted_chebyshev_polynomial_w(x: str, n: str) -> str:
        # Shifted Chebyshev polynomial of the fourth kind W*_n(x) = W_n(2x - 1)
        y = f"(2 * {x} - 1)"  # shifted variable
        return (
            f"jnp.where({n} == 0, jnp.ones_like({x}), "
            f"jnp.where({n} == 1, 2*{y} + 1, "
            f"jnp.where({n} == 2, 4*{y}**2 + 2*{y} - 1, "
            f"jnp.where({n} == 3, 8*{y}**3 + 4*{y}**2 - 4*{y} - 1, "
            f"jnp.where({n} == 4, 16*{y}**4 + 8*{y}**3 - 12*{y}**2 - 4*{y} + 1, "
            f"jnp.where({n} == 5, 32*{y}**5 + 16*{y}**4 - 32*{y}**3 - 12*{y}**2 + 6*{y} + 1, "
            f"jnp.zeros_like({x})))))))"
        )

    @staticmethod
    def hermite_polynomial_h(x: str, n: str) -> str:
        # Physicist's Hermite polynomial H_n(x)
        # H_n(x) = 2^n * x^n - n*(n-1)/2 * 2^(n-2) * x^(n-2) + ...
        # Use explicit formula: H_n(x) = n! * sum_{m=0}^{n//2} (-1)^m / (m! * (n-2m)!) * (2x)^(n-2m)
        # For simplicity, use the relation: H_n(x) = 2^(n/2) * He_n(x * sqrt(2)) where He is probabilist's
        # Actually simpler: use recurrence or closed form
        # H_0 = 1, H_1 = 2x, H_2 = 4x^2 - 2, H_3 = 8x^3 - 12x
        return (
            f"jnp.where({n} == 0, jnp.ones_like({x}), "
            f"jnp.where({n} == 1, 2 * {x}, "
            f"jnp.where({n} == 2, 4 * {x}**2 - 2, "
            f"jnp.where({n} == 3, 8 * {x}**3 - 12 * {x}, "
            f"jnp.where({n} == 4, 16 * {x}**4 - 48 * {x}**2 + 12, "
            f"jnp.where({n} == 5, 32 * {x}**5 - 160 * {x}**3 + 120 * {x}, "
            f"jnp.zeros_like({x})))))))"  # Fallback for higher n
        )

    @staticmethod
    def hermite_polynomial_he(x: str, n: str) -> str:
        # Probabilist's Hermite polynomial He_n(x)
        # He_0 = 1, He_1 = x, He_2 = x^2 - 1, He_3 = x^3 - 3x
        return (
            f"jnp.where({n} == 0, jnp.ones_like({x}), "
            f"jnp.where({n} == 1, {x}, "
            f"jnp.where({n} == 2, {x}**2 - 1, "
            f"jnp.where({n} == 3, {x}**3 - 3 * {x}, "
            f"jnp.where({n} == 4, {x}**4 - 6 * {x}**2 + 3, "
            f"jnp.where({n} == 5, {x}**5 - 10 * {x}**3 + 15 * {x}, "
            f"jnp.zeros_like({x})))))))"  # Fallback for higher n
        )

    @staticmethod
    def laguerre_polynomial_l(x: str, n: str) -> str:
        # Laguerre polynomial L_n(x)
        # L_0 = 1, L_1 = 1 - x, L_2 = (x^2 - 4x + 2)/2, L_3 = (-x^3 + 9x^2 - 18x + 6)/6
        return (
            f"jnp.where({n} == 0, jnp.ones_like({x}), "
            f"jnp.where({n} == 1, 1 - {x}, "
            f"jnp.where({n} == 2, ({x}**2 - 4*{x} + 2) / 2, "
            f"jnp.where({n} == 3, (-{x}**3 + 9*{x}**2 - 18*{x} + 6) / 6, "
            f"jnp.where({n} == 4, ({x}**4 - 16*{x}**3 + 72*{x}**2 - 96*{x} + 24) / 24, "
            f"jnp.where({n} == 5, (-{x}**5 + 25*{x}**4 - 200*{x}**3 + 600*{x}**2 - 600*{x} + 120) / 120, "
            f"jnp.zeros_like({x})))))))"  # Fallback for higher n
        )

    @staticmethod
    def legendre_polynomial_p(x: str, n: str) -> str:
        # Legendre polynomial P_n(x)
        # P_0 = 1, P_1 = x, P_2 = (3x^2 - 1)/2, P_3 = (5x^3 - 3x)/2
        return (
            f"jnp.where({n} == 0, jnp.ones_like({x}), "
            f"jnp.where({n} == 1, {x}, "
            f"jnp.where({n} == 2, (3 * {x}**2 - 1) / 2, "
            f"jnp.where({n} == 3, (5 * {x}**3 - 3 * {x}) / 2, "
            f"jnp.where({n} == 4, (35 * {x}**4 - 30 * {x}**2 + 3) / 8, "
            f"jnp.where({n} == 5, (63 * {x}**5 - 70 * {x}**3 + 15 * {x}) / 8, "
            f"jnp.zeros_like({x})))))))"  # Fallback for higher n
        )

    # Reciprocal and square
    @staticmethod
    def reciprocal(x: str) -> str:
        return f"jnp.reciprocal({x})"

    @staticmethod
    def square(x: str) -> str:
        return f"jnp.square({x})"

    # Additional operations
    @staticmethod
    def fma(a: str, b: str, c: str) -> str:
        """Fused multiply-add: a * b + c

        JAX doesn't have jnp.fma, so we use the unfused version.
        The compiler may still fuse this on supported hardware.
        """
        return f"(({a}) * ({b}) + ({c}))"

    @staticmethod
    def copysign(a: str, b: str) -> str:
        return f"jnp.copysign({a}, {b})"

    @staticmethod
    def nextafter(a: str, b: str) -> str:
        return f"jnp.nextafter({a}, {b})"

    @staticmethod
    def ldexp(a: str, b: str) -> str:
        return f"jnp.ldexp({a}, {b})"

    @staticmethod
    def frexp(x: str) -> str:
        return f"jnp.frexp({x})"

    @staticmethod
    def modf(x: str) -> str:
        return f"jnp.modf({x})"

    # Bitwise operations
    @staticmethod
    def bitwise_and(a: str, b: str) -> str:
        return f"jnp.bitwise_and({a}, {b})"

    @staticmethod
    def bitwise_or(a: str, b: str) -> str:
        return f"jnp.bitwise_or({a}, {b})"

    @staticmethod
    def bitwise_xor(a: str, b: str) -> str:
        return f"jnp.bitwise_xor({a}, {b})"

    @staticmethod
    def bitwise_not(x: str) -> str:
        return f"jnp.bitwise_not({x})"

    @staticmethod
    def left_shift(a: str, b: str) -> str:
        return f"jnp.left_shift({a}, {b})"

    @staticmethod
    def right_shift(a: str, b: str) -> str:
        return f"jnp.right_shift({a}, {b})"

    # Random number generation operations
    @staticmethod
    def load_seed(name: str, offset: str) -> str:
        """Load the random seed value from a buffer."""
        # Load the seed from the buffer and add offset for uniqueness
        seed_offset = V.kernel.args.seed_offset("load_seed_offset", offset)
        return f"({V.kernel.args.input(name)}[0] + {seed_offset})"

    @staticmethod
    def rand(seed: str, offset: str) -> str:
        """Generate uniform random numbers in [0, 1).

        Uses JAX's threefry2x32 PRNG directly for vectorized random generation.
        The seed provides the base key, offset provides per-element uniqueness.
        """
        # For vectorized random, we use jax.random.uniform with shape from offset
        # Create a base key from seed, then use fold_in with vmap for per-element keys
        # Use float32 dtype to match PyTorch's default
        return (
            f"jax.vmap(lambda o: jax.random.uniform("
            f"jax.random.fold_in(jax.random.PRNGKey(jnp.uint32({seed})), jnp.uint32(o)), (), dtype=jnp.float32))"
            f"(jnp.asarray({offset}).flatten()).reshape(jnp.asarray({offset}).shape)"
        )

    @staticmethod
    def randn(seed: str, offset: str) -> str:
        """Generate standard normal random numbers.

        Uses JAX's threefry2x32 PRNG directly for vectorized random generation.
        The seed provides the base key, offset provides per-element uniqueness.
        """
        # For vectorized random, use vmap to fold in each offset value
        # Use float32 dtype to match PyTorch's default
        return (
            f"jax.vmap(lambda o: jax.random.normal("
            f"jax.random.fold_in(jax.random.PRNGKey(jnp.uint32({seed})), jnp.uint32(o)), (), dtype=jnp.float32))"
            f"(jnp.asarray({offset}).flatten()).reshape(jnp.asarray({offset}).shape)"
        )

    @staticmethod
    def randint64(seed: str, offset: str, low: str, high: str) -> str:
        """Generate random int64 values in [low, high)."""
        # For vectorized random, use vmap to fold in each offset value
        return (
            f"jax.vmap(lambda o: jax.random.randint("
            f"jax.random.fold_in(jax.random.PRNGKey(jnp.uint32({seed})), jnp.uint32(o)), (), {low}, {high}, dtype=jnp.int64))"
            f"(jnp.asarray({offset}).flatten()).reshape(jnp.asarray({offset}).shape)"
        )


class PallasKernel(SIMDKernel):
    """
    Pallas kernel for elementwise operations with support for strided/scatter access.

    Strategy:
    - Convert index expressions to JAX-compatible array slicing
    - Load/store using indexed access: "in_ptrX[slice]" or full-array "in_ptrX[...]"
    - Compute expression with Python operators (compatible with jax.numpy broadcasting)
    - Generate Python code that defines a Pallas kernel and a host entrypoint.
    - Use async_compile.pallas path to compile and load Python code.

    For GPU (Mosaic backend):
    - Use TMA (Tensor Memory Accelerator) for automatic OOB masking
    - Falls back to legacy padding approach for reductions, broadcasting, non-contiguous tensors
    """

    overrides = PallasKernelOverrides  # type: ignore[assignment]
    kexpr: Callable[[sympy.Expr], str] = pallas_pexpr  # Use Pallas expression printer

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Determine device type once at initialization
        device = V.graph.get_current_device_or_throw()
        self.is_gpu = device.type == "cuda"
        # Use TMA (Tensor Memory Accelerator) for GPU to handle non-aligned tensor sizes
        # TMA automatically masks OOB accesses, eliminating the need for explicit
        # padding to multiples of 128. Uses lax.fori_loop with direct TMA primitives.
        self.use_emit_pipeline = self.is_gpu  # Enable TMA approach for GPU
        # Legacy: warpgroup padding (enabled when TMA approach is disabled)
        self.use_warpgroup_padding = self.is_gpu and not self.use_emit_pipeline
        # Track which output param each store uses: list of (out_ptr_name, store_line)
        self.store_with_output: list[tuple[str, str]] = []
        # Track load index expressions for argmax/argmin axis detection
        self.load_index_exprs: dict[str, sympy.Expr] = {}
        # Track outputs that need to be readable (for scatter operations)
        self.outputs_need_read: OrderedSet[str] = OrderedSet()
        # Track if any load in this kernel used transpose
        # Used to avoid double transpose (load + store)
        self.has_transposed_load = False
        # Track if any load in this kernel has expand (PALLAS_EXPAND_STRIDE)
        # Used to trigger full array store with broadcast
        self.has_expand_load = False
        # Track expand intermediate shape (after unsqueeze, before broadcast)
        # Computed from coefficient analysis in _maybe_reshape_for_expand
        self.expand_intermediate_shape: Optional[tuple[int, ...]] = None
        self.expand_input_shape: Optional[tuple[int, ...]] = None
        # Track the CSE variable name from expand load
        # Only stores of this exact variable should use expand store logic
        self.expand_load_cse_name: Optional[str] = None
        # Flag to capture CSE variable name after generation
        self._pending_expand_capture: bool = False
        # Track 2D form dimensions for pure pointwise kernels (for broadcasting)
        self._pw_2d_outer: Optional[int] = None
        self._pw_2d_inner: Optional[int] = None
        # Track pending transpose permutation for TPU strided access
        # When non-None, _build_load_expr should apply jnp.transpose with this perm
        self._pending_transpose_perm: Optional[list[int]] = None
        # Track CSE variable names that were loaded with iteration variable indexing
        # These have the iteration shape and should NOT be reshaped in strided index
        self._cse_vars_with_iter_shape: OrderedSet[str] = OrderedSet()

    def check_bounds(
        self, expr: sympy.Expr, size: sympy.Expr, lower: bool, upper: bool
    ) -> None:
        """Check array bounds for indirect indexing."""
        # For now, skip explicit bounds checking as JAX/Pallas handles this internally
        # TODO: Implement explicit bounds checking with assertions if needed

    def _get_index_str(self, index: sympy.Expr) -> str:
        """
        Convert an index expression to a string suitable for Pallas indexing.

        Pallas operates on full arrays, so we need to convert index expressions
        to JAX array slicing. For example:
        - x0 -> "..." (contiguous access, full array)
        - 2*x0 -> "::2" (strided access with stride 2)
        - 2*x0 + 1 -> "1::2" (strided access with offset 1, stride 2)

        Args:
            index: The indexing expression to convert

        Returns:
            The indexing string to use in generated code
        """
        # Prepare and simplify the index
        prepared_index = self.prepare_indexing(index)

        # Note: Block variable detection (im2col patterns) is handled in load()/store()
        # where we have access to buffer dimensions. We check the buffer size
        # against iteration variables there to detect gather patterns.

        # For simple single-symbol access (contiguous case), we can use [...]
        # which is more efficient as it operates on the entire array at once
        if isinstance(prepared_index, sympy.Symbol):
            return "..."
        elif prepared_index.is_Integer:
            # Scalar index
            return str(prepared_index)
        else:
            # Complex expression (strided/scatter access)
            # Try to extract stride and offset for common patterns
            return self._convert_to_jax_slice(prepared_index)

    def _convert_to_jax_slice(self, index: sympy.Expr) -> str:
        """
        Convert a sympy index expression to JAX slice notation.

        Handles common patterns like:
        - stride*var -> ::stride
        - stride*var + offset -> offset::stride

        For more complex patterns, falls back to explicit indexing.
        Uses BlockPatternMatcher for robust pattern matching.
        """
        # Get the iteration variables for this kernel
        if not self.range_trees:
            return "..."

        # Rename symbolic sizes to kernel parameter names upfront
        index = self.rename_indexing(index)

        # Check for ModularIndexing - this is NOT contiguous access
        # ModularIndexing is used for roll/wrap-around operations
        if index.has(ModularIndexing):
            # Generate actual index expression - iteration variables are already
            # defined as jnp.arange arrays, so we just convert to JAX code
            return self.kexpr(index)

        # Simplify the index
        index = V.graph.sizevars.simplify(index)
        # Find which iteration variable(s) are used
        used_vars = self._get_used_iter_vars(index)

        if len(used_vars) == 0:
            # No iteration variables, this is a constant index
            return str(index)
        elif len(used_vars) == 1:
            # Single iteration variable - try to extract stride and offset using BlockPatternMatcher
            var = next(iter(used_vars))

            # Get the subexpression involving this variable
            var_expr = BlockPatternMatcher.get_subexpr_involving_symbol(index, var)

            # Try to match affine pattern: stride * var
            stride = BlockPatternMatcher.match_affine_block_expr(var_expr, var)

            if stride is not None:
                offset = index - var_expr
                offset = V.graph.sizevars.simplify(offset)

                if stride < 0:
                    return self.kexpr(index)

                if offset == 0:
                    return "..."

                # Non-zero offset: check if we can use slice notation
                if stride != 1:
                    return self.kexpr(index)

                try:
                    offset_val = int(offset)
                    if offset_val < 0:
                        return self.kexpr(index)
                except (TypeError, ValueError):
                    return self.kexpr(index)

                return f"{self.kexpr(offset)}::1"
            else:
                # Couldn't match affine pattern, fall back to original logic
                offset = index - var_expr
                offset = V.graph.sizevars.simplify(offset)
                if offset == 0 and var_expr == var:
                    # Just the variable itself, unit stride
                    return "..."
        elif len(used_vars) > 1:
            # Multi-dimensional indexing
            # For contiguous multi-dim access, all terms should have unit stride
            all_unit_stride = True
            for var in used_vars:
                var_expr = BlockPatternMatcher.get_subexpr_involving_symbol(index, var)
                stride = BlockPatternMatcher.match_affine_block_expr(var_expr, var)
                if stride != 1:
                    all_unit_stride = False
                    break
            if all_unit_stride:
                # Contiguous multi-dimensional access
                return "..."
            else:
                # Strided multi-dimensional access
                # For most cases, inputs are made contiguous before passing to JAX,
                # so strided tensors become contiguous and we can use [...]
                # The buffer size check in load() handles im2col-like patterns
                return "..."

        # For complex cases, use [...] since inputs are made contiguous
        return "..."

    def _generate_strided_index(self, index: sympy.Expr) -> str:
        """
        Generate JAX code to compute an index array for strided/complex indexing patterns.

        For expressions like `2 * x3 + 32 * x2 + 256 * x1 + 1024 * x0`, we generate
        code that computes the flattened index array using broadcasting.

        The iteration variables (x0, x1, x2, x3) are already defined as jnp.arange arrays
        in the kernel. We just need to convert the sympy expression to JAX code.
        """
        # Strip PALLAS_EXPAND_STRIDE before checking free symbols
        # It's used for analysis but should be replaced with 0 for code generation
        index = self._strip_expand_stride(index)

        free_symbols = index.free_symbols
        iter_vars = self._get_iter_vars()

        # Check that all free symbols are iteration variables (no indirect vars)
        used_vars = free_symbols & iter_vars
        if used_vars != free_symbols:
            raise Unsupported(
                f"Pallas backend does not yet support mixed index pattern: {index}"
            )

        # Convert sympy expression to Python/JAX code string
        # The iteration variables are already defined as jnp.arange arrays
        index_str = self.kexpr(index)

        # Mark this as requiring flatten access
        return index_str

    def _generate_index_array(self, index: sympy.Expr) -> str:
        """
        Generate JAX code to compute an index array for complex indexing patterns.
        Delegates to _generate_strided_index.
        """
        return self._generate_strided_index(index)

    def _get_iter_vars(self) -> OrderedSet:
        """Get the set of iteration variable symbols."""
        return OrderedSet(self.range_tree_nodes.keys())

    def _get_used_iter_vars(self, index: sympy.Expr) -> OrderedSet:
        """Get iteration variables used in an index expression."""
        return index.free_symbols & self._get_iter_vars()

    def _has_iteration_vars(self, index: sympy.Expr) -> bool:
        """Check if index expression contains iteration variables."""
        return bool(self._get_used_iter_vars(index))

    def _get_indirect_vars(self, index: sympy.Expr) -> list[sympy.Symbol]:
        """Get list of indirect variable symbols (tmp*) in an index expression."""
        return [s for s in index.free_symbols if str(s).startswith("tmp")]

    def _has_indirect_vars(self, index: sympy.Expr) -> bool:
        """Check if index expression contains indirect variables."""
        return len(self._get_indirect_vars(index)) > 0

    def _strip_expand_stride(self, index: sympy.Expr) -> sympy.Expr:
        """
        Strip Pallas stride markers from the index expression.

        PALLAS_EXPAND_STRIDE and PallasStride markers are used for analysis
        (to detect expand dimensions and permutations) but should be replaced
        with their numeric values before generating actual code.
        """
        return strip_pallas_stride(index)

    def _get_strides_from_pallas_stride(self, index: sympy.Expr) -> Optional[dict]:
        """
        Extract stride info directly from PallasStride markers.

        Returns dict mapping iter_var_pos -> (stride_value, dim_index) or None.
        stride_value can be int or sympy.Expr for symbolic strides.
        """
        pallas_strides = list(index.atoms(PallasStride))
        if not pallas_strides:
            return None

        result = {}
        for ps in pallas_strides:
            if len(ps.args) >= 3:
                stride_val = ps.args[0]
                dim_idx = self._safe_int(ps.args[1])
                iter_pos = self._safe_int(ps.args[2])
                if dim_idx is not None and iter_pos is not None:
                    # Handle PALLAS_EXPAND_STRIDE
                    if stride_val == PALLAS_EXPAND_STRIDE:
                        result[iter_pos] = (0, dim_idx)  # stride=0 for expand
                    else:
                        # Try to convert to int, but keep sympy expr if symbolic
                        stride_int = self._safe_int(stride_val)
                        if stride_int is not None:
                            result[iter_pos] = (stride_int, dim_idx)
                        else:
                            # Keep symbolic stride as sympy expression
                            result[iter_pos] = (stride_val, dim_idx)
        return result if result else None

    def _get_indirect_stride_info(
        self, index: sympy.Expr
    ) -> Optional[tuple[int, int, int]]:
        """
        Extract stride info from PallasIndirectStride marker for indirect variables.

        Returns (stride_value, dim_index, iter_var_pos) or None if no indirect stride marker.
        """
        indirect_strides = list(index.atoms(PallasIndirectStride))
        if not indirect_strides:
            return None

        # Should be exactly one indirect variable in scatter patterns
        if len(indirect_strides) != 1:
            return None

        ps = indirect_strides[0]
        if len(ps.args) >= 3:
            stride_val = self._safe_int(ps.args[0])
            dim_idx = self._safe_int(ps.args[1])
            iter_pos = self._safe_int(ps.args[2])
            if stride_val is not None and dim_idx is not None and iter_pos is not None:
                return (stride_val, dim_idx, iter_pos)
        return None

    def _get_expected_output_shape(self) -> list:
        """Get the expected output shape from iteration variables.

        Iteration variables are shaped for broadcasting. For 2D outputs:
        - First var (e.g., y0) gets shape (1, N) - innermost dimension
        - Second var (e.g., x1) gets shape (M, 1) - outermost dimension
        The broadcast result is (M, N).
        """
        # Collect variable lengths
        var_items = list(self.range_tree_nodes.items())
        broadcast_vars = []
        for var_sym, entry in var_items:
            length = self._safe_int(entry.length)
            if length is not None:
                broadcast_vars.append(length)

        if len(broadcast_vars) <= 1:
            return broadcast_vars

        # For 2D case: variables are reshaped in reverse order
        # First var is innermost (last dim), second var is outermost (first dim)
        # So output shape is [second_var_length, first_var_length, ...]
        return list(reversed(broadcast_vars))

    def _get_index_expr(self, index: sympy.Expr) -> tuple[str, bool]:
        """Get the index expression string and whether it needs flattening.

        IMPORTANT: PALLAS_EXPAND_STRIDE is stripped from the index before code generation.
        The original index should be used for analysis (e.g., detecting expand dimensions).
        """
        # Strip PALLAS_EXPAND_STRIDE for code generation
        # This is used for analysis only; the expand dims contribute 0 to the index
        stripped_index = self._strip_expand_stride(index)

        has_indirect = self._has_indirect_vars(stripped_index)
        has_iter_vars = self._has_iteration_vars(stripped_index)

        if has_indirect and has_iter_vars:
            return self._handle_mixed_indexing(stripped_index), True
        elif has_indirect:
            return self.kexpr(stripped_index), False
        else:
            index_str = self._get_index_str(stripped_index)
            # Check if index contains ModularIndexing - this requires flattened access
            # ModularIndexing is used for roll/wrap-around operations
            needs_flatten = stripped_index.has(ModularIndexing) and index_str != "..."
            # If index_str is an actual expression (not "..." or a slice pattern),
            # we need flattened access because it uses block variables
            if not needs_flatten and index_str != "...":
                # Check if it's a simple slice pattern (::N or M::N)
                if not ("::" in index_str or index_str.lstrip("-").isdigit()):
                    needs_flatten = True
            return index_str, needs_flatten

    @staticmethod
    def _safe_int(val: Any) -> Optional[int]:
        """Convert value to int, returning None on failure."""
        try:
            return int(val)
        except (TypeError, ValueError):
            return None

    def _compute_prefix_numel(self, prefixes: OrderedSet) -> Optional[int]:
        """Compute total numel for given prefixes (e.g., pointwise prefixes)."""
        result = 1
        for p in prefixes:
            if p in self.numels:
                numel = self._safe_int(self.numels[p])
                if numel is None:
                    return None
                result *= numel
        return result

    def _compute_reduction_numel(self) -> Optional[int]:
        """Compute total reduction numel."""
        result = 1
        for tree in self.range_trees:
            if tree.is_reduction:
                numel = self._safe_int(tree.numel)
                if numel is None:
                    return None
                result *= numel
        return result

    def _can_use_tma_approach(self) -> bool:
        """
        Check if TMA (Tensor Memory Accelerator) approach can be used.
        TMA works for simple element-wise ops but not for:
        - Reductions (need different accumulation patterns)
          TODO: TMA supports float64 for loading but not for reductions
        - Broadcasting (inputs have different shapes or output differs)
        - Non-contiguous tensors (strided, transposed)
        """
        # Check for reductions
        reduction_numel = self._compute_reduction_numel()
        if reduction_numel is not None and reduction_numel > 1:
            return False

        # Check all input buffers for contiguity, dtype, and shape consistency
        input_shapes: list[tuple] = []
        for name in self.args.input_buffers:
            buf_obj, buf_size, buf_numel, actual_strides, is_contiguous = (
                self._get_buffer_info(name)
            )
            if not is_contiguous:
                return False

            # Check for unsupported dtypes
            # TODO: TMA supports float64 for loading but current JAX Mosaic GPU
            # implementation doesn't support it yet. Re-enable when JAX adds support.
            buf_dtype = getattr(buf_obj, "get_dtype", lambda: None)()
            if buf_dtype is not None:
                import torch

                if buf_dtype == torch.float64:
                    return False

            # Collect shape as tuple for comparison
            shape_tuple = tuple(self._safe_int(s) for s in buf_size)
            if None in shape_tuple:
                return False  # Dynamic shapes not supported
            input_shapes.append(shape_tuple)

        # Check if all input shapes are identical (no broadcasting)
        if input_shapes and len(OrderedSet(input_shapes)) > 1:
            return False

        # Check that output numel matches input numel (no broadcasting expansion)
        if input_shapes:
            input_numel = 1
            for s in input_shapes[0]:
                input_numel *= s

            # Compute output numel from pointwise range trees (non-reduction)
            output_numel = 1
            for tree in self.range_trees:
                if not tree.is_reduction:
                    numel = self._safe_int(tree.numel)
                    if numel is None:
                        return False  # Dynamic shapes not supported
                    output_numel *= numel

            if output_numel != input_numel:
                return False

        return True

    def _get_buffer_info(self, name: str) -> tuple[Any, Any, Any, list, bool]:
        """Get buffer metadata (buf_obj, buf_size, buf_numel, actual_strides, is_contiguous)."""
        buf_obj = V.graph.get_buffer(name)
        buf_size = buf_obj.get_size()
        buf_numel = 1
        for s in buf_size:
            sval = self._safe_int(s)
            buf_numel *= sval if sval is not None else s

        # Get buffer strides and check contiguity
        actual_strides: list = []
        is_contiguous = True

        layout = getattr(buf_obj, "get_layout", lambda: None)()
        buf_stride = getattr(layout, "stride", None) if layout else None

        if buf_stride is not None:
            for i in range(len(buf_size)):
                actual_stride = self._safe_int(buf_stride[i])
                actual_strides.append(actual_stride)

            # Check contiguity
            if len(buf_size) == 1:
                if actual_strides[0] is not None and actual_strides[0] != 1:
                    is_contiguous = False
            elif len(buf_size) > 1:
                expected_stride = 1
                for i in range(len(buf_size) - 1, -1, -1):
                    actual_stride = actual_strides[i]
                    if actual_stride is None or actual_stride != expected_stride:
                        is_contiguous = False
                    dim_size = self._safe_int(buf_size[i])
                    if dim_size is not None:
                        expected_stride *= dim_size

        return buf_obj, buf_size, buf_numel, actual_strides, is_contiguous

    def _compute_output_numel_from_index(
        self, index: sympy.Expr
    ) -> tuple[int, OrderedSet]:
        """Compute expected output numel and used vars from iteration variables in index."""
        used_vars = self._get_used_iter_vars(index)

        used_range_lengths = []
        for var in used_vars:
            if var in self.range_tree_nodes:
                entry = self.range_tree_nodes[var]
                length_val = self._safe_int(entry.length)
                if length_val is not None:
                    used_range_lengths.append(length_val)

        output_numel = 1
        for l in used_range_lengths:
            output_numel *= l

        return output_numel, used_vars

    def _get_dim_order_from_pallas_stride(
        self, index: sympy.Expr
    ) -> Optional[list[int]]:
        """
        Extract dimension order from PallasStride markers in index expression.

        PallasStride(stride_value, dim_index, iter_var_pos) carries metadata about
        which original dimension each buffer position came from.

        Returns:
            List where result[iter_var_pos] = dim_index, giving the permutation
            that maps buffer positions to original dimensions. Returns None if
            no PallasStride markers are found.

        Example:
            For permute(0, 2, 1) on shape (2, 4, 3):
            - Position 0 came from dim 0
            - Position 1 came from dim 2
            - Position 2 came from dim 1
            - Returns [0, 2, 1]
        """
        pallas_strides = list(index.atoms(PallasStride))
        if not pallas_strides:
            return None

        # Build mapping: iter_var_pos -> dim_index
        pos_to_dim: dict[int, int] = {}
        for ps in pallas_strides:
            if len(ps.args) >= 3:
                dim_idx = self._safe_int(ps.args[1])  # dim_index (original dimension)
                iter_pos = self._safe_int(ps.args[2])  # iter_var_pos (current position)
                if dim_idx is not None and iter_pos is not None:
                    pos_to_dim[iter_pos] = dim_idx

        if not pos_to_dim:
            return None

        # Build ordered list
        max_pos = max(pos_to_dim.keys())
        dim_order = [pos_to_dim.get(i, i) for i in range(max_pos + 1)]

        # Verify it's a valid permutation
        if sorted(dim_order) != list(range(len(dim_order))):
            return None

        return dim_order

    def _compute_permutation_for_iteration_order(
        self,
        index: sympy.Expr,
        actual_strides: list,
        used_vars: OrderedSet,
    ) -> Optional[list[int]]:
        """
        Compute permutation to reorder buffer to match iteration variable order.

        For non-contiguous buffers on TPU, we can't use flatten()[offset] because
        JAX flatten returns logical order, not physical offset. Instead, we compute
        a permutation that reorders the buffer to match iteration order.

        Uses PallasStride markers which carry dimension info directly.

        Returns:
            List of buffer dimension indices in iteration order, or None if not applicable.
        """
        dim_order = self._get_dim_order_from_pallas_stride(index)
        if dim_order is not None:
            # dim_order[i] = which original dim is at buffer position i
            # This IS the permutation we need
            if dim_order != list(range(len(dim_order))):
                return dim_order
            return None  # Identity permutation - no transpose needed
        return None  # No PallasStride markers - can't compute permutation

    def _needs_strided_indexing(
        self,
        name: str,
        index: sympy.Expr,
        index_str: str,
        needs_flatten: bool,
    ) -> tuple[str, bool]:
        """
        Check if buffer access needs strided indexing due to size mismatch or gather patterns.

        This handles cases like:
        - Pooling operations where input/output have different sizes
        - im2col-like gather patterns
        - Transposed or strided buffer access
        - Expand operations (PALLAS_EXPAND_STRIDE in index)
        """
        # Clear any pending transpose from previous loads
        self._pending_transpose_perm = None

        # Only applies when full array access is indicated
        if index_str != "..." or needs_flatten:
            return index_str, needs_flatten

        buf = V.graph.get_buffer(name)
        if buf is None:
            return index_str, needs_flatten

        # Special case: expand operations always need strided indexing
        # The buffer is smaller than the iteration space, so we need element-wise access
        # NOTE: We do NOT set up expand tracking here because the strided load
        # already handles the expansion element-wise. The store should write directly.
        if PALLAS_EXPAND_STRIDE in index.free_symbols:
            return self._generate_strided_index(index), True

        buf_obj, buf_size, buf_numel, actual_strides, is_contiguous = (
            self._get_buffer_info(name)
        )
        output_numel, used_vars = self._compute_output_numel_from_index(index)
        all_iter_vars = self._get_iter_vars()

        # Get stride info from PallasStride markers
        stride_info = self._get_strides_from_pallas_stride(index)

        # Check for symbolic strides - can't do strided indexing with symbolic strides
        has_symbolic_stride = False
        if stride_info:
            for _, (stride_val, _) in stride_info.items():
                if not isinstance(stride_val, (int, float)):
                    has_symbolic_stride = True
                    break

        # Check for gather pattern: buffer strides don't match contiguous layout
        has_non_unit_strides = False
        if stride_info and actual_strides and not has_symbolic_stride:
            # Check if any stride in the index doesn't match buffer's contiguous strides
            for iter_pos, (stride_val, dim_idx) in stride_info.items():
                if stride_val != 0 and dim_idx < len(actual_strides):
                    # Compare index stride with what contiguous layout would have
                    expected_stride = 1
                    for d in range(dim_idx + 1, len(buf_size)):
                        dim_size = self._safe_int(buf_size[d])
                        if dim_size is not None:
                            expected_stride *= dim_size
                    if stride_val != expected_stride:
                        has_non_unit_strides = True
                        break

        # Check for im2col-like pattern (more iter vars used than buffer dims)
        buf_effective_dims = sum(1 for s in buf_size if self._safe_int(s) != 1)
        not_all_vars_used = (
            len(used_vars) < len(all_iter_vars)
            and len(used_vars) > 0
            and buf_effective_dims > 1
            and len(used_vars) > len(buf_size)
        )

        # Check various conditions for skipping strided indexing
        is_tpu = torch._inductor.config._debug_cpu_to_tpu_pallas

        # For non-contiguous buffers with same numel:
        #
        # Both CPU and TPU paths make data contiguous in the wrapper:
        # - CPU: jax.dlpack.from_dlpack(tensor.detach().contiguous())
        # - TPU: jax.device_put(tensor.cpu().numpy(), ...) - JAX makes contiguous copy
        #
        # So for same-numel non-contiguous access, skip strided indexing.
        # The kernel can use full array access since wrapper handles contiguity.
        skip_for_same_numel = False
        is_known_non_contiguous = not is_contiguous and all(
            s is not None for s in actual_strides
        )
        if is_known_non_contiguous and buf_numel == output_numel:
            # Both CPU and TPU: wrapper makes buffer contiguous
            skip_for_same_numel = True
            # For TPU with permuted tensors, also set up transpose
            if is_tpu:
                perm = self._compute_permutation_for_iteration_order(
                    index, actual_strides, used_vars
                )
                if perm is not None:
                    self._pending_transpose_perm = perm

        # Check for shape mismatch or stride ordering mismatch
        # When buffer shape doesn't match iteration shape, reshape won't work
        # Example: input (32, 16) with iteration (16, 32) = transpose
        # Also check for square matrices where shapes match but strides indicate transpose
        shape_mismatch = False
        if (
            buf_numel == output_numel
            and len(buf_size) == 2
            and stride_info
            and not skip_for_same_numel
        ):
            buf_shape = [self._safe_int(s) for s in buf_size]
            if None not in buf_shape:
                # Compute iteration 2D shape the same way as _compute_pw_2d_shape
                # Find innermost variable (divisor=1, smallest length)
                inner_length = None
                inner_var = None
                outer_var = None
                pointwise_vars = [
                    (var, entry)
                    for var, entry in self.range_tree_nodes.items()
                    if not entry.is_reduction
                ]
                for var_sym, entry in pointwise_vars:
                    if entry.divisor == 1:
                        entry_len = self._safe_int(entry.length)
                        if entry_len is not None and entry_len > 1:
                            if inner_length is None or entry_len < inner_length:
                                inner_length = entry_len
                                inner_var = var_sym
                # Outer var is the one that's not inner
                for var_sym, entry in pointwise_vars:
                    if var_sym != inner_var and not entry.is_reduction:
                        outer_var = var_sym
                        break

                if inner_length is not None and inner_length > 1:
                    outer_length = buf_numel // inner_length
                    iter_shape = [outer_length, inner_length]

                    # Check 1: buffer shape vs iteration shape (non-square case)
                    if (
                        sorted(buf_shape) == sorted(iter_shape)
                        and buf_shape != iter_shape
                    ):
                        shape_mismatch = True

                    # Check 2: stride ordering (square case)
                    # Extract actual stride coefficients from index expression for each var
                    # For row-major iteration, inner var should have smaller stride than outer
                    if not shape_mismatch and inner_var and outer_var:
                        # Get coefficient of each variable in the index expression
                        stripped = strip_pallas_stride(index)
                        inner_coeff = stripped.coeff(inner_var) if hasattr(stripped, 'coeff') else None
                        outer_coeff = stripped.coeff(outer_var) if hasattr(stripped, 'coeff') else None
                        if (
                            inner_coeff is not None
                            and outer_coeff is not None
                            and self._safe_int(inner_coeff) is not None
                            and self._safe_int(outer_coeff) is not None
                        ):
                            inner_stride = self._safe_int(inner_coeff)
                            outer_stride = self._safe_int(outer_coeff)
                            # Normal: inner stride < outer stride
                            # Transposed: inner stride > outer stride
                            if inner_stride > outer_stride:
                                shape_mismatch = True

        # Determine if strided indexing is needed
        # Skip strided indexing for symbolic strides - can't generate index pattern
        if (
            output_numel > 0
            and (
                buf_numel != output_numel
                or not_all_vars_used
                or has_non_unit_strides
                or shape_mismatch
            )
            and len(used_vars) > 0
            and not skip_for_same_numel
            and not has_symbolic_stride
        ):
            # If strided indexing handles shape mismatch (transpose), mark it
            # so store doesn't apply transpose again
            if shape_mismatch:
                self.has_transposed_load = True
            return self._generate_strided_index(index), True

        return index_str, needs_flatten

    def _adjust_index_for_buffer_shape(
        self,
        name: str,
        index: sympy.Expr,
        index_str: str,
        needs_flatten: bool,
    ) -> tuple[str, bool]:
        """
        Adjust index expression based on buffer shape (0-dim scalar, multi-dim, etc.).
        """
        if needs_flatten or index_str == "...":
            return index_str, needs_flatten

        buf_obj = V.graph.get_buffer(name)
        if buf_obj is None:
            return index_str, needs_flatten

        buf_size = buf_obj.get_size()

        # 0-dimensional (scalar) buffer - use [...] to access it
        if len(buf_size) == 0:
            return "...", needs_flatten

        # Multi-dimensional buffer with constant/scalar index
        if len(buf_size) > 1:
            has_iter_vars = self._has_iteration_vars(index)
            if not has_iter_vars:
                return index_str, True  # Use flattened access
            elif "::" in index_str:
                # Strided slice patterns need flattened indexing for multi-dim
                return self._generate_strided_index(index), True

        # GPU doesn't support gather from slice patterns on 1D buffers
        if self.is_gpu and "::" in index_str:
            return self._generate_strided_index(index), True

        return index_str, needs_flatten

    def _build_load_expr(
        self,
        buf: str,
        name: str,
        index: sympy.Expr,
        index_str: str,
        needs_flatten: bool,
    ) -> str:
        """
        Build the load expression based on indexing mode.
        """
        # Check for pending transpose permutation (TPU non-contiguous access)
        # This reorders the buffer to match iteration order
        transpose_perm = self._pending_transpose_perm
        self._pending_transpose_perm = None  # Clear after use

        if needs_flatten:
            # Flatten then index for non-contiguous access (gather operation)
            has_minmax = index.has(sympy.Min) or index.has(sympy.Max)
            idx = f"({index_str}).astype(jnp.int64)" if has_minmax else index_str
            return f"{buf}[...].flatten()[{idx}]"
        else:
            # Direct indexing for contiguous access
            load_expr = f"{buf}[{index_str}]"

            # Apply transpose permutation for TPU non-contiguous access
            # This reorders the buffer to match iteration order
            if transpose_perm is not None and index_str == "...":
                perm_str = ", ".join(str(p) for p in transpose_perm)
                load_expr = f"jnp.transpose({load_expr}, ({perm_str},))"
                self.has_transposed_load = True

            return load_expr

    def _maybe_squeeze_intermediate_buffer(self, name: str, load_expr: str) -> str:
        """
        Squeeze (N,1) intermediate buffers when kernel has 1D graph inputs.

        This avoids wrong broadcasting: (N,) op (N,1) -> (N,N) instead of (N,)
        """
        if not name.startswith("buf"):
            return load_expr

        # Check if any input buffer is a 1D graph input
        has_1d_input = any(
            not buf_name.startswith("buf")
            and (buf_obj := V.graph.get_buffer(buf_name)) is not None
            and len(buf_obj.get_size()) == 1
            for buf_name in self.args.input_buffers
        )

        if has_1d_input:
            buf_obj = V.graph.get_buffer(name)
            if buf_obj is not None:
                buf_size = buf_obj.get_size()
                if len(buf_size) == 2 and buf_size[-1] == 1:
                    return f"jnp.squeeze({load_expr}, axis=-1)"

        return load_expr

    def _maybe_broadcast_1d_buffer(
        self, name: str, index: sympy.Expr, load_expr: str
    ) -> str:
        """Reshape 1D buffers (e.g., batch norm mean) for higher-dim broadcasting."""
        buf_obj = V.graph.get_buffer(name)
        if buf_obj is None or len(buf_obj.get_size()) != 1:
            return load_expr

        buf_length = self._safe_int(buf_obj.get_size()[0])
        if buf_length is None:
            return load_expr

        # Only graph inputs, not intermediate buffers or index tensors
        if name.startswith("buf"):
            return load_expr
        dtype = V.graph.get_dtype(name)
        if dtype is not None and not dtype.is_floating_point:
            return load_expr

        # Find a higher-dimensional reference buffer
        ref_buf_size = None
        for buf_name in self.args.input_buffers:
            other_buf = V.graph.get_buffer(buf_name)
            if other_buf is not None and len(other_buf.get_size()) > 1:
                ref_buf_size = [self._safe_int(s) for s in other_buf.get_size()]
                if all(s is not None for s in ref_buf_size):
                    break
                ref_buf_size = None
        if ref_buf_size is None or len(ref_buf_size) <= 1:
            return load_expr

        # Must use exactly one iteration variable
        used_vars = self._get_used_iter_vars(index)
        if len(used_vars) != 1:
            return load_expr
        used_var = next(iter(used_vars))
        if used_var not in self.range_tree_nodes:
            return load_expr

        # Verify buffer length matches variable length
        entry = self.range_tree_nodes[used_var]
        if self._safe_int(entry.length) != buf_length:
            return load_expr

        # Buffer length must uniquely match one iteration variable
        matching_vars = [
            v
            for v, e in self.range_tree_nodes.items()
            if self._safe_int(e.length) == buf_length and not e.is_reduction
        ]
        if len(matching_vars) != 1:
            return load_expr

        # Buffer length must uniquely match one ref buffer dimension
        matching_dims = [i for i, s in enumerate(ref_buf_size) if s == buf_length]
        if len(matching_dims) != 1:
            return load_expr

        axis_pos = matching_dims[0]
        if axis_pos == len(ref_buf_size) - 1:
            return load_expr  # Last dim uses default broadcasting

        reshape_dims = [1] * len(ref_buf_size)
        reshape_dims[axis_pos] = -1
        return f"{load_expr}.reshape({', '.join(map(str, reshape_dims))})"

    def _maybe_reshape_for_expand(
        self, name: str, index: sympy.Expr, load_expr: str
    ) -> str:
        """Handle expand operations using PallasStride markers."""
        # Check for expand via PallasStride
        stride_info = self._get_strides_from_pallas_stride(index)
        if not stride_info:
            return load_expr

        # Find expand dimension (stride=0)
        expand_dims = [(pos, dim) for pos, (stride, dim) in stride_info.items() if stride == 0]
        if len(expand_dims) != 1:
            return load_expr  # No expand or multiple expands - not handled

        expand_pos, expand_dim = expand_dims[0]

        # Get buffer info
        buf_obj = V.graph.get_buffer(name)
        if buf_obj is None:
            return load_expr

        buf_size = buf_obj.get_size()
        input_shape = [self._safe_int(s) for s in buf_size]
        if None in input_shape:
            return load_expr

        # Compute intermediate shape: insert 1 at expand_dim position
        intermediate_shape = list(input_shape)
        intermediate_shape.insert(expand_dim, 1)

        # Track for store
        self.expand_intermediate_shape = tuple(intermediate_shape)
        self.expand_input_shape = tuple(input_shape)
        self._pending_expand_capture = True
        self.has_expand_load = True

        return load_expr

    def _compute_pw_2d_shape(self, numel: int) -> None:
        """
        Compute 2D shape for pure pointwise kernels to enable broadcasting.

        This finds the innermost iteration variable (smallest length with divisor=1)
        and uses it to determine the inner dimension of the 2D form.
        """
        if self._pw_2d_outer is not None:
            return  # Already computed

        if not self.range_tree_nodes:
            return

        # Find pointwise variables (non-reduction)
        pointwise_vars = []
        for var_sym, entry in self.range_tree_nodes.items():
            if not entry.is_reduction:
                pointwise_vars.append((var_sym, entry))

        if not pointwise_vars:
            return

        # Find innermost variable (smallest length with divisor=1)
        inner_length = None
        for var_sym, entry in pointwise_vars:
            if entry.divisor == 1:
                entry_len = self._safe_int(entry.length)
                if entry_len is not None and entry_len > 1:
                    if inner_length is None or entry_len < inner_length:
                        inner_length = entry_len

        if inner_length is not None and inner_length > 1 and numel > inner_length:
            self._pw_2d_outer = numel // inner_length
            self._pw_2d_inner = inner_length

    def _maybe_reshape_to_nd_form(self, name: str, load_expr: str) -> str:
        """
        Reshape full buffer loads to the 2D canonical form for proper broadcasting.

        The canonical forms are:
        1. With reduction: (numel, rnumel) where numel = x*y*z
        2. Without reduction but with x and y: (y_numel, x_numel)
        3. Otherwise: 1D

        This also handles broadcasting: smaller buffers are reshaped to broadcast.

        IMPORTANT: Only reshape buffers whose total size matches expected sizes.
        Buffers with different sizes (e.g., weights, indices) should not be reshaped.
        """
        if not self.range_tree_nodes:
            return load_expr

        # Get buffer info to check size
        buf_obj = V.graph.get_buffer(name)
        if buf_obj is None:
            return load_expr

        buf_size = buf_obj.get_size()
        buf_numel = 1
        for s in buf_size:
            int_s = self._safe_int(s)
            if int_s is None:
                # Symbolic size - can't determine if reshape is safe
                return load_expr
            buf_numel *= int_s

        # Compute total pointwise numel (x*y*z)
        numel = 1
        for prefix in ["x", "y", "z"]:
            if prefix in self.numels:
                n = self._safe_int(self.numels[prefix])
                if n is None:
                    return load_expr
                numel *= n

        # Compute reduction numel
        rnumel = 1
        for prefix in ["r0_", "r1_", "r2_"]:
            if prefix in self.numels:
                r = self._safe_int(self.numels[prefix])
                if r is None:
                    return load_expr
                rnumel *= r

        # Get individual dimension sizes
        y_numel = 1
        for prefix in ["y", "z"]:
            if prefix in self.numels:
                y = self._safe_int(self.numels[prefix])
                if y is None:
                    return load_expr
                y_numel *= y
        x_numel = self._safe_int(self.numels.get("x", sympy.Integer(1)))
        if x_numel is None:
            return load_expr

        has_reduction = rnumel > 1
        has_both_xy = y_numel > 1 and x_numel > 1

        # Case 1: With reduction - use (numel, rnumel) form
        if has_reduction and numel > 1:
            total_size = numel * rnumel
            if buf_numel == total_size:
                target_shape_ints = [numel, rnumel]
            elif buf_numel == numel:
                target_shape_ints = [numel, 1]
            elif buf_numel == rnumel:
                target_shape_ints = [1, rnumel]
            else:
                return load_expr

        # Case 2: No reduction but has both x and y - use (y_numel, x_numel) form
        elif has_both_xy:
            total_size = y_numel * x_numel
            if buf_numel == total_size:
                target_shape_ints = [y_numel, x_numel]
            elif buf_numel == y_numel:
                target_shape_ints = [y_numel, 1]
            elif buf_numel == x_numel:
                target_shape_ints = [1, x_numel]
            else:
                return load_expr

        # Case 3: Only pointwise or only reduction
        elif numel > 1:
            # Compute 2D shape for broadcasting if not already done
            self._compute_pw_2d_shape(numel)

            # Check if we're using 2D form for broadcasting
            if self._pw_2d_outer is not None and self._pw_2d_inner is not None:
                # 2D form: (outer, inner)
                total_2d = self._pw_2d_outer * self._pw_2d_inner
                if buf_numel == total_2d:
                    target_shape_ints = [self._pw_2d_outer, self._pw_2d_inner]
                elif buf_numel == self._pw_2d_inner:
                    # Smaller buffer - broadcast along outer dimension
                    # Check INNER first because PyTorch aligns 1D tensors with LAST dim
                    target_shape_ints = [1, self._pw_2d_inner]
                elif buf_numel == self._pw_2d_outer:
                    # Smaller buffer - broadcast along inner dimension
                    target_shape_ints = [self._pw_2d_outer, 1]
                else:
                    return load_expr
            elif buf_numel == numel:
                target_shape_ints = [numel]
            else:
                return load_expr
        elif rnumel > 1:
            if buf_numel == rnumel:
                target_shape_ints = [rnumel]
            else:
                return load_expr
        else:
            return load_expr

        # Check if buffer already has the target shape
        if len(buf_size) == len(target_shape_ints) and all(
            self._safe_int(s) == t for s, t in zip(buf_size, target_shape_ints)
        ):
            return load_expr

        # Build target shape string
        target_shape = ", ".join(str(s) for s in target_shape_ints)
        return f"{load_expr}.reshape({target_shape})"

    def _check_im2col_pattern(
        self, index: sympy.Expr, index_str: str, needs_flatten: bool
    ) -> tuple[str, bool]:
        """
        Check for im2col-like patterns where store uses block variables but load doesn't.

        For cat/expand patterns, both load and store prepared indices share block vars.
        For im2col patterns, store compresses to block vars but load doesn't.
        """
        if index_str != "..." or needs_flatten:
            return index_str, needs_flatten

        prepared_index = self.prepare_indexing(index)
        iter_vars = self._get_iter_vars()
        store_orig_vars = self._get_used_iter_vars(index)
        store_prep_vars = (
            prepared_index.free_symbols
            if hasattr(prepared_index, "free_symbols")
            else OrderedSet()
        ) & iter_vars
        new_vars = store_prep_vars - store_orig_vars

        # Only trigger if store introduces new block vars
        if not new_vars or len(store_orig_vars) <= 1:
            return index_str, needs_flatten

        # Check if loads are compatible with broadcast or cat pattern
        has_im2col_pattern = False
        for buf_name, load_index in self.load_index_exprs.items():
            load_orig_vars = self._get_used_iter_vars(load_index)
            if not load_orig_vars:
                continue

            # Load has iteration variables
            if load_orig_vars != store_orig_vars:
                continue

            # Same vars - check if load gets compressed too
            prep_load = self.prepare_indexing(load_index)
            load_prep_vars = (
                prep_load.free_symbols
                if hasattr(prep_load, "free_symbols")
                else OrderedSet()
            ) & iter_vars

            # If store compresses but load doesn't, check for strided input vs im2col
            if load_orig_vars != load_prep_vars or store_prep_vars == store_orig_vars:
                continue

            # Check if load coefficients match buffer strides
            if not self._check_load_is_strided_input(
                buf_name, load_index, load_orig_vars
            ):
                has_im2col_pattern = True
                break

        if has_im2col_pattern:
            return self._generate_strided_index(prepared_index), True

        return index_str, needs_flatten

    def _check_store_needs_transpose(self, name: str) -> bool:
        """
        Check if output needs transpose for column-major storage.

        Transpose on store is needed when:
        - Output has column-major stride (s0 < s1)
        - But input(s) have row-major stride
        - And we haven't already transposed on load
        """
        if self.has_transposed_load:
            return False

        buf = V.graph.get_buffer(name)
        if buf is None:
            return False

        layout = getattr(buf, "get_layout", lambda: None)()
        if layout is None:
            return False

        buf_stride = getattr(layout, "stride", None)
        if buf_stride is None:
            return False

        buf_size = buf.get_size()
        if len(buf_stride) != 2 or len(buf_size) != 2:
            return False

        size0 = self._safe_int(buf_size[0])
        size1 = self._safe_int(buf_size[1])
        s0 = self._safe_int(buf_stride[0])
        s1 = self._safe_int(buf_stride[1])

        # Check if output is column-major with valid dimensions
        if not (
            s0 is not None
            and s1 is not None
            and s0 < s1
            and size0 is not None
            and size1 is not None
            and size0 > 1
            and size1 > 1
        ):
            return False

        # Check if any input is column-major (if so, no transpose needed)
        for inp_name in self.args.input_buffers:
            inp_buf = V.graph.get_buffer(inp_name)
            if inp_buf is None:
                continue
            inp_layout = getattr(inp_buf, "get_layout", lambda: None)()
            if inp_layout is None:
                continue
            inp_stride = getattr(inp_layout, "stride", None)
            if inp_stride is None or len(inp_stride) != 2:
                continue
            inp_s0 = self._safe_int(inp_stride[0])
            inp_s1 = self._safe_int(inp_stride[1])
            if inp_s0 is not None and inp_s1 is not None and inp_s0 < inp_s1:
                return False  # Input is also column-major

        return True

    def _build_expand_store_expr(
        self, out: str, value: CSEVariable, name: str
    ) -> str:
        """
        Build store expression for expand operations.

        For expand+reshape fused operations (like repeat_kv), the value has
        fewer elements than the output. We use the precomputed intermediate
        shape (from coefficient analysis) to:
        1. Reshape input to intermediate shape (with 1 at expand position)
        2. Broadcast to output shape

        This handles cases like:
        - Input: (2, 16, 2, 16) = 1024 elements
        - Intermediate: (2, 16, 2, 1, 16) - 1 inserted at expand position
        - Output: (2, 16, 2, 2, 16) = 2048 elements
        """
        if self.expand_intermediate_shape is not None:
            # Use precomputed intermediate shape from coefficient analysis
            intermediate_shape_str = ", ".join(str(s) for s in self.expand_intermediate_shape)
            return (
                f"{out}[...] = jnp.broadcast_to("
                f"jnp.asarray({value}).reshape({intermediate_shape_str}), "
                f"{out}.shape)"
            )
        else:
            # Expand without precomputed shape - this should be handled via strided indexing
            # If we reach here, the codegen should generate proper strided load/store code
            # Fall back to simple reshape which works when sizes match
            return (
                f"{out}[...] = (jnp.full({out}.shape, {value}) if jnp.asarray({value}).ndim == 0 "
                f"else (jnp.broadcast_to(jnp.asarray({value}), {out}.shape) "
                f"if jnp.asarray({value}).size != {out}.size "
                f"else jnp.asarray({value}).reshape({out}.shape)))"
            )

    def _build_full_array_store_expr(
        self, out: str, value: CSEVariable, needs_transpose: bool
    ) -> str:
        """
        Build store expression for full array assignment.

        Handles scalar broadcast, shape matching, and optional transpose.
        """
        if needs_transpose:
            return (
                f"{out}[...] = ("
                f"jnp.full({out}.shape, {value}) if jnp.asarray({value}).ndim == 0 "
                f"else jnp.transpose(jnp.asarray({value})))"
            )
        else:
            return (
                f"{out}[...] = ("
                f"jnp.full({out}.shape, {value}) if jnp.asarray({value}).ndim == 0 "
                f"else (jnp.broadcast_to(jnp.asarray({value}), {out}.shape) "
                f"if jnp.asarray({value}).size != {out}.size "
                f"else jnp.asarray({value}).reshape({out}.shape)))"
            )

    def _is_full_coverage_store(self, buf: Any, index: sympy.Expr) -> bool:
        """
        Check if a store covers all elements of the output buffer.

        For pointwise operations (no partial writes), stores always cover
        all elements. This is used to avoid the scatter pattern which
        reads from uninitialized output buffers.

        Returns True if buf_numel == iteration_numel (full coverage).
        """
        # Get buffer numel
        buf_size = buf.get_size()
        buf_numel = 1
        for s in buf_size:
            val = self._safe_int(s)
            if val is None:
                return False  # Can't determine, be conservative
            buf_numel *= val

        # Compute iteration space numel from range trees
        # Only count pointwise (non-reduction) dimensions
        iter_numel = 1
        for var_sym, entry in self.range_tree_nodes.items():
            if not entry.is_reduction:
                length = self._safe_int(entry.length)
                if length is None:
                    return False  # Can't determine, be conservative
                iter_numel *= length

        return buf_numel == iter_numel

    def _build_store_expr(
        self,
        out: str,
        name: str,
        index: sympy.Expr,
        value: CSEVariable,
        index_str: str,
        needs_flatten: bool,
        mode: Any = None,
    ) -> str:
        """
        Build the store expression based on indexing mode.
        mode can be None (set) or "atomic_add" (accumulate).
        """
        if index_str == "...":
            # Full array store with shape matching
            needs_transpose = self._check_store_needs_transpose(name)
            return self._build_full_array_store_expr(out, value, needs_transpose)

        if needs_flatten:
            # Block variable indexing - check if this is a full coverage store
            # For pointwise operations writing all elements, we can use direct assignment
            # instead of the scatter pattern (which reads from uninitialized output)
            if mode != "atomic_add":
                buf = V.graph.get_buffer(name)
                if buf is not None:
                    # Check if store covers all elements (pointwise operation)
                    # by comparing buffer size to iteration space
                    is_full_coverage = self._is_full_coverage_store(buf, index)
                    if is_full_coverage:
                        # Direct assignment - no need to read from output
                        return (
                            f"{out}[...] = jnp.asarray({value}).flatten()"
                            f".reshape({out}.shape)"
                        )

            # Partial store or atomic_add - use flattened scatter
            scatter_op = "add" if mode == "atomic_add" else "set"
            return (
                f"{out}[...] = {out}[...].flatten().at[({index_str}).flatten()].{scatter_op}("
                f"jnp.asarray({value}).flatten()).reshape({out}.shape)"
            )

        # Direct indexed assignment
        has_indirect = self._has_indirect_vars(index)
        buf = V.graph.get_buffer(name)

        if buf is not None:
            buf_size = buf.get_size()
            if len(buf_size) > 1 and not self._has_iteration_vars(index):
                # Multi-dim output with constant index - use [...] for full assignment
                return self._build_full_array_store_expr(out, value, False)

        if has_indirect:
            # Indirect indexed store (scatter): use .add() for atomic_add, .set() otherwise
            scatter_op = "add" if mode == "atomic_add" else "set"
            value_expr = (
                f"(jnp.full({index_str}.shape, {value}) "
                f"if jnp.asarray({value}).ndim == 0 else {value})"
            )
            if mode == "atomic_add":
                # For atomic_add, mark output as needing to be readable (for aliasing)
                self.outputs_need_read.add(out)
                alias_param = f"{out}_alias"
                return (
                    f"{out}[...] = {alias_param}[...].flatten().at[({index_str}).flatten()].{scatter_op}("
                    f"{value_expr}.flatten()).reshape({out}.shape)"
                )
            else:
                return f"{out}[{index_str}] = {value_expr}"

        return f"{out}[{index_str}] = {value}"

    def _build_scatter_store_expr(
        self,
        out: str,
        value: CSEVariable,
        scatter_info: dict[str, Any],
        name: str,
        mode: Any,
    ) -> str:
        """Build store expression for scatter operations (indirect indexing)."""
        is_point_scatter = scatter_info.get("is_point_scatter", False)

        # Mark this output parameter as needing to be readable (for aliasing)
        self.outputs_need_read.add(out)
        alias_param = f"{out}_alias"

        # Use .add() for atomic_add mode, .set() otherwise
        scatter_op = "add" if mode == "atomic_add" else "set"

        if is_point_scatter:
            # Single-element scatter
            indirect_var = scatter_info["indirect_var"]
            indirect_dim = scatter_info["indirect_dim"]
            output_shape = scatter_info["output_shape"]

            # Build index tuple with 0s for other dimensions
            index_parts = []
            for dim in range(len(output_shape)):
                if dim == indirect_dim:
                    index_parts.append(indirect_var)
                else:
                    index_parts.append("0")

            index_tuple = ", ".join(index_parts)
            return f"{out}[...] = {alias_param}[...].at[{index_tuple}].{scatter_op}({value})"

        # Scatter with iteration variables
        indirect_var = scatter_info["indirect_var"]
        dims_before = scatter_info["dims_before"]
        dims_after = scatter_info["dims_after"]

        # Determine if element-wise or slice-based scatter
        buf = V.graph.get_buffer(name)
        output_ndim = len(buf.get_size()) if buf is not None else 0

        num_iter_vars_in_store = len(dims_before) + len(dims_after)
        total_kernel_iter_vars = len(self.range_tree_nodes)
        remaining_dims = output_ndim - 1  # dims other than indirect

        is_element_wise = (
            num_iter_vars_in_store == remaining_dims
            and num_iter_vars_in_store == total_kernel_iter_vars
        )

        if is_element_wise:
            # Element-wise scatter: use iteration variable names
            index_parts = [var_name for var_name, size in dims_before]

            # Reshape indirect var for broadcasting if needed
            n_leading = len(dims_before)
            n_trailing = len(dims_after)
            if n_leading > 0 and n_trailing > 0:
                leading_ones = "None, " * n_leading
                trailing_nones = ", None" * n_trailing
                indirect_reshaped = f"{indirect_var}[{leading_ones}...{trailing_nones}]"
            else:
                indirect_reshaped = indirect_var
            index_parts.append(indirect_reshaped)

            index_parts.extend(var_name for var_name, size in dims_after)
        else:
            # Slice-based scatter: use : for iteration dimensions
            index_parts = [":" for _ in dims_before]
            index_parts.append(indirect_var)
            index_parts.extend(":" for _ in dims_after)

        index_tuple = ", ".join(index_parts)
        return (
            f"{out}[...] = {alias_param}[...].at[{index_tuple}].{scatter_op}({value})"
        )

    @typing_extensions.override
    def load(self, name: str, index: sympy.Expr) -> CSEVariable:
        buf = self.args.input(name)
        dtype = V.graph.get_dtype(name)

        # Track the load index expression for argmax/argmin axis detection
        self.load_index_exprs[name] = index

        # Check if this load involves expand (PALLAS_EXPAND_STRIDE in index)
        # This affects how we handle the store later
        if PALLAS_EXPAND_STRIDE in index.free_symbols:
            self.has_expand_load = True

        # Get base index expression
        index_str, needs_flatten = self._get_index_expr(index)

        # Check for buffer size mismatch requiring strided indexing
        index_str, needs_flatten = self._needs_strided_indexing(
            name, index, index_str, needs_flatten
        )

        # Adjust index for buffer shape (scalar, multi-dim, etc.)
        index_str, needs_flatten = self._adjust_index_for_buffer_shape(
            name, index, index_str, needs_flatten
        )

        # Build the load expression
        load_expr = self._build_load_expr(buf, name, index, index_str, needs_flatten)

        # Handle intermediate buffer squeezing for correct broadcasting
        if not needs_flatten and index_str == "...":
            load_expr = self._maybe_squeeze_intermediate_buffer(name, load_expr)
            # Handle 1D buffer broadcasting for higher-dimensional kernels
            load_expr = self._maybe_broadcast_1d_buffer(name, index, load_expr)
            # Handle expand operations: insert singleton dimensions for broadcast
            load_expr = self._maybe_reshape_for_expand(name, index, load_expr)
            # Reshape full buffer loads to N-dimensional canonical form for proper broadcasting
            load_expr = self._maybe_reshape_to_nd_form(name, load_expr)

        cse_var = self.cse.generate(
            self.compute,
            load_expr,
            dtype=dtype,
        )

        # Capture the CSE variable name if this was an expand load
        if self._pending_expand_capture:
            self.expand_load_cse_name = str(cse_var)
            self._pending_expand_capture = False

        # Track CSE variables that were loaded with iteration variable indexing
        # These have the iteration shape and should NOT be reshaped in strided index
        if needs_flatten and self._has_iteration_vars(index):
            self._cse_vars_with_iter_shape.add(str(cse_var))

        return cse_var

    def _handle_mixed_indexing(self, index: sympy.Expr) -> str:
        """
        Handle indexing with both indirect variables and iteration variables.

        For example, x[indices, :] generates index = i0 + stride * tmp0
        where tmp0 is loaded from indices and i0 is the iteration variable.

        We need to convert this to JAX advanced indexing with proper broadcasting.
        When there are multiple iteration variables, they need different shapes
        to form an outer product (grid) rather than broadcasting together.

        Special case: For gather operations where a single iteration variable
        and single indirect variable have the same extent, they should be
        element-wise aligned, not broadcast into an outer product.

        PyTorch advanced indexing semantics: When multiple indirect indices have
        the same shape, they are paired element-wise (not outer product), and
        the combined result dimension appears at the FRONT of the output.
        """
        used_iter_vars_set = self._get_used_iter_vars(index)

        if len(used_iter_vars_set) == 0:
            return self.kexpr(index)

        # Get stride info from PallasStride markers
        stride_info = self._get_strides_from_pallas_stride(index)

        # Build mapping from variable to stride using PallasStride info
        var_strides = {}
        if stride_info:
            var_items = list(self.range_tree_nodes.items())
            for iter_pos, (stride_val, dim_idx) in stride_info.items():
                if iter_pos < len(var_items):
                    var_sym = var_items[iter_pos][0]
                    var_strides[var_sym] = stride_val

        def get_stride(var):
            """Get stride from PallasStride info."""
            if var in var_strides:
                return var_strides[var]
            return 0  # Unknown stride

        # Sort iteration variables by their stride. Larger strides = earlier dimensions.
        used_iter_vars = sorted(used_iter_vars_set, key=get_stride, reverse=True)
        iter_coeffs = [get_stride(var) for var in used_iter_vars]

        # Rename symbolic sizes to kernel parameter names
        index_str = self.kexpr(self.rename_indexing(index))
        indirect_var_syms = self._get_indirect_vars(index)
        indirect_vars = [str(sym) for sym in indirect_var_syms]

        # Get stride for indirect vars from PallasIndirectStride
        indirect_info = self._get_indirect_stride_info(index)
        indirect_coeffs = {}
        if indirect_info:
            # Only one indirect var supported for now
            stride_val, dim_idx, _ = indirect_info
            for s in indirect_var_syms:
                indirect_coeffs[str(s)] = stride_val
        else:
            # No PallasIndirectStride - use 0 as default
            for s in indirect_var_syms:
                indirect_coeffs[str(s)] = 0

        # Special case: reduction var + single indirect var = use 2D canonical form
        # The indirect var (pointwise) gets shape (numel, 1)
        # The reduction var gets shape (1, rnumel)
        # This ensures proper broadcasting: (numel, 1) + (1, rnumel) -> (numel, rnumel)
        if len(used_iter_vars) == 1 and len(indirect_vars) == 1:
            var = used_iter_vars[0]
            var_name = str(var)
            is_reduction_var = (
                var in self.range_tree_nodes and self.range_tree_nodes[var].is_reduction
            )

            if is_reduction_var:
                # Reduction var: reshape to (1, rnumel) for 2D broadcasting
                if var in self.range_tree_nodes:
                    range_entry = self.range_tree_nodes[var]
                    range_size = range_entry.length
                    renamed_size = self.rename_indexing(range_size)
                    # 2D form: (1, length) for reduction dimension
                    arange_expr = f"jnp.arange({self.kexpr(renamed_size)})[None, :]"
                    index_str = index_str.replace(var_name, arange_expr)

                # Indirect var: flatten and reshape to (numel, 1) for 2D broadcasting
                indirect_var = indirect_vars[0]
                reshape_expr = f"jnp.asarray({indirect_var}).reshape(-1)[:, None]"
                index_str = index_str.replace(indirect_var, reshape_expr)

                return index_str
            # For pointwise vars, fall through to the complex reshape code

        # Check if multiple indirect vars should be paired element-wise.
        # In PyTorch, when multiple advanced indices have the same shape, they pair up.
        # The paired dimension goes to the FRONT of the output.
        # However, if indirect vars have different shapes (e.g., (1,4) and (4,1)),
        # they form an outer product instead.
        # We detect element-wise pairing when:
        # 1. Multiple indirect vars exist
        # 2. There's exactly ONE unused iteration variable (for the shared paired dim)
        # For outer product, there are MULTIPLE unused iter vars (one per indirect dim)
        paired_indirect = False
        if len(indirect_vars) > 1:
            # Count unused iteration variables (defined but not in index expression)
            unused_iter_vars = self._get_iter_vars() - used_iter_vars_set
            # Element-wise pairing: one unused iter var for the shared paired dimension
            # Outer product: multiple unused iter vars (one for each indirect var dimension)
            paired_indirect = len(unused_iter_vars) == 1

        if paired_indirect:
            # Multiple indirect vars with element-wise pairing
            # Output order: (paired_indirect_dim, iter_var_dims...)
            # All indirect vars get the same shape: (N, 1, 1, ...) for first dim
            # Iter vars come after: second dim onwards

            # Count total output dims: 1 (paired) + len(iter_vars) for non-newaxis
            # But some iter vars may be for newaxis dimensions (size 1)
            n_output_dims = 1 + len(used_iter_vars)

            # Reshape indirect vars to occupy the first dimension
            for indirect_var in indirect_vars:
                trailing_ones = ", 1" * len(used_iter_vars)
                reshape_expr = f"{indirect_var}.reshape(-1{trailing_ones})"
                index_str = index_str.replace(indirect_var, reshape_expr)

            # Reshape iteration variables to occupy subsequent dimensions
            # Sort by coefficient (descending) to determine order
            for i, var in enumerate(used_iter_vars):
                var_name = str(var)
                if var in self.range_tree_nodes:
                    range_entry = self.range_tree_nodes[var]
                    range_size = range_entry.length
                    # Rename to use kernel parameter names for symbolic sizes
                    renamed_size = self.rename_indexing(range_size)

                    # Shape: (1, ..., N, ..., 1) where N is at position i+1
                    # Position 0 is for paired indirect vars
                    shape_parts = ["1"] * n_output_dims
                    shape_parts[i + 1] = self.kexpr(renamed_size)
                    shape_str = ", ".join(shape_parts)
                    arange_expr = (
                        f"jnp.arange({self.kexpr(renamed_size)}).reshape({shape_str})"
                    )

                    index_str = index_str.replace(var_name, arange_expr)

            return index_str

        # Single indirect var case (or no indirect vars handled above)
        # Handle broadcasting for mixed iteration/indirect variable indexing.
        #
        # Key insight: Iteration variables are ALREADY defined at kernel setup time
        # with proper N-dimensional shapes. We should NOT replace them with new
        # jnp.arange() expressions. We only need to reshape indirect variables.

        # Count TOTAL kernel dimensions (not just vars used in this index)
        # This determines the shape of the N-D grid for broadcasting.
        total_pointwise_vars = []
        total_reduction_vars = []
        for var_sym, entry in self.range_tree_nodes.items():
            if entry.is_reduction:
                total_reduction_vars.append(var_sym)
            else:
                total_pointwise_vars.append(var_sym)

        has_indirect = len(indirect_vars) > 0
        total_num_pointwise = len(total_pointwise_vars)
        total_num_reduction = len(total_reduction_vars)
        total_num_dims = total_num_pointwise + (1 if total_num_reduction > 0 else 0)

        # Case 1: Multi-dimensional kernel (total_num_dims > 1)
        # Iteration variables are already defined with proper N-D shapes.
        # Don't replace them - just reshape indirect variables.
        if total_num_dims > 1 and total_num_pointwise > 0:
            # Indirect variables typically correspond to the "outer" dimension
            # (the dimension with the largest coefficient in the index expression).
            # Reshape them with trailing 1s for proper broadcasting.
            # EXCEPTION 1: If indirect_info is None, there's no PallasIndirectStride
            # marker, meaning the indirect vars come from prior iteration-indexed
            # computations and already have the iteration shape.
            # EXCEPTION 2: If the indirect var was directly loaded with iteration indexing.
            if indirect_info is None:
                # No PallasIndirectStride - indirect vars already have iteration shape
                return index_str
            for indirect_var in indirect_vars:
                # Skip reshape if this var already has iteration shape
                if indirect_var in self._cse_vars_with_iter_shape:
                    continue
                # Shape: (-1, 1, 1, ...) with total_num_dims-1 trailing 1s
                trailing_ones = ", 1" * (total_num_dims - 1)
                reshape_expr = f"jnp.asarray({indirect_var}).reshape(-1{trailing_ones})"
                index_str = index_str.replace(indirect_var, reshape_expr)
            return index_str

        # Case 2: 2D form with pointwise + reduction (legacy path)
        has_reduction_iter = num_reduction > 0
        use_2d = has_indirect and has_reduction_iter

        # Handle iteration variables with 2D canonical form
        for var in used_iter_vars:
            var_name = str(var)
            if var in self.range_tree_nodes:
                range_entry = self.range_tree_nodes[var]
                range_size = range_entry.length
                renamed_size = self.rename_indexing(range_size)
                arange_expr = f"jnp.arange({self.kexpr(renamed_size)})"

                if use_2d and range_entry.is_reduction:
                    # Reduction var: (1, length) shape for 2D broadcasting
                    arange_expr = f"{arange_expr}[None, :]"
                elif use_2d and not range_entry.is_reduction:
                    # Pointwise iter var: (length, 1) shape
                    arange_expr = f"{arange_expr}[:, None]"

                index_str = index_str.replace(var_name, arange_expr)

        # Reshape indirect variables for 2D canonical form
        # EXCEPTION 1: If indirect_info is None, indirect vars come from prior
        # iteration-indexed computations and already have the iteration shape.
        # EXCEPTION 2: If the indirect var was directly loaded with iteration indexing.
        if indirect_info is None and has_indirect:
            # No PallasIndirectStride - indirect vars already have iteration shape
            return index_str
        for indirect_var in indirect_vars:
            # Skip reshape if this var already has iteration shape
            if indirect_var in self._cse_vars_with_iter_shape:
                continue
            if use_2d:
                # Flatten and add trailing dim: (numel, 1)
                reshape_expr = f"jnp.asarray({indirect_var}).reshape(-1)[:, None]"
            else:
                # Just flatten
                reshape_expr = f"jnp.asarray({indirect_var}).reshape(-1)"

            index_str = index_str.replace(indirect_var, reshape_expr)

        return index_str

    @typing_extensions.override
    def store(
        self, name: str, index: sympy.Expr, value: CSEVariable, mode: Any = None
    ) -> None:
        # mode can be None (set), "atomic_add" (accumulate), etc.
        if mode is not None and mode != "atomic_add":
            raise Unsupported(f"pallas store mode '{mode}' not supported")
        out = self.args.output(name)
        self.store_buffer_names.add(name)

        # Check if this is a scalar output (reduction to scalar)
        buf = V.graph.get_buffer(name)
        is_scalar = buf is not None and len(buf.get_size()) == 0

        if is_scalar:
            # For scalar outputs, use jnp.full to handle shape mismatch
            store_expr = (
                f"{out}[...] = ("
                f"jnp.full({out}.shape, {value}) if jnp.asarray({value}).ndim == 0 "
                f"else jnp.asarray({value}).reshape({out}.shape))"
            )
        else:
            # Check for expand operations: if any load had PALLAS_EXPAND_STRIDE,
            # AND the value being stored is the EXACT expand load variable
            # (not a derived value from operations on the expand load)
            needs_expand = (
                self.has_expand_load
                and self.expand_input_shape is not None
                and self.expand_intermediate_shape is not None
                and self.expand_load_cse_name is not None
                and str(value) == self.expand_load_cse_name
            )
            if needs_expand:
                # Calculate sizes to verify this is the correct expand store
                # The intermediate shape must have the same element count as the input
                input_size = 1
                for s in self.expand_input_shape:
                    input_size *= s
                intermediate_size = 1
                for s in self.expand_intermediate_shape:
                    intermediate_size *= s
                output_size = 1
                if buf is not None:
                    for s in buf.get_size():
                        output_size *= (int(s) if hasattr(s, '__int__') else s)
                # Only use expand if:
                # 1. Output is larger than input (actual expand)
                # 2. Intermediate size matches input size (correct reshape target)
                needs_expand = (
                    output_size > input_size
                    and intermediate_size == input_size
                )

            if needs_expand:
                # Expand operation: use full array store with broadcast
                # We need to reshape the value to have the same ndim as output
                # with 1s at the expand positions for broadcast_to to work
                store_expr = self._build_expand_store_expr(out, value, name)
            else:
                # Check for scatter pattern (indirect indexing for stores)
                scatter_info = self._detect_scatter_pattern(index, name)

                if scatter_info is not None:
                    store_expr = self._build_scatter_store_expr(
                        out, value, scatter_info, name, mode
                    )
                else:
                    # Get base index expression
                    index_str, needs_flatten = self._get_index_expr(index)

                    # Check for im2col-like patterns
                    index_str, needs_flatten = self._check_im2col_pattern(
                        index, index_str, needs_flatten
                    )

                    # Build the store expression
                    store_expr = self._build_store_expr(
                        out, name, index, value, index_str, needs_flatten, mode
                    )

        self.stores.writeline(store_expr)
        # Track which output param this store uses for filtering in codegen_kernel
        self.store_with_output.append((out, store_expr))

    def _detect_scatter_pattern(
        self, index: sympy.Expr, output_name: str = ""
    ) -> Optional[dict[str, Any]]:
        """Detect scatter operation pattern using PallasIndirectStride markers.

        Returns scatter info dict or None.
        """
        indirect_syms = self._get_indirect_vars(index)
        if len(indirect_syms) != 1:
            return None

        indirect_sym = indirect_syms[0]
        indirect_var = str(indirect_sym)

        # Get indirect stride info from PallasIndirectStride marker
        indirect_info = self._get_indirect_stride_info(index)

        # Point scatter: no iteration variables, just indirect indexing
        if not self._has_iteration_vars(index):
            return self._detect_point_scatter(output_name, indirect_var, indirect_info)

        # Regular scatter: has both indirect and iteration variables
        # Get stride info from PallasStride markers for iteration vars
        stride_info = self._get_strides_from_pallas_stride(index)
        return self._detect_iter_scatter(index, indirect_var, indirect_info, stride_info)

    def _detect_point_scatter(
        self,
        output_name: str,
        indirect_var: str,
        indirect_info: Optional[tuple[int, int, int]],
    ) -> Optional[dict[str, Any]]:
        """Detect single-element scatter pattern using PallasIndirectStride."""
        if not output_name:
            return None
        if not indirect_info:
            return None

        try:
            buf = V.graph.get_buffer(output_name)
            output_shape = [int(s) for s in buf.get_size()]
        except Exception:
            return None

        if len(output_shape) < 2:
            return None

        # Get dimension directly from PallasIndirectStride marker
        indirect_stride, indirect_dim, _ = indirect_info

        return {
            "indirect_var": indirect_var,
            "indirect_dim": indirect_dim,
            "dims_before": [],
            "dims_after": [],
            "is_point_scatter": True,
            "output_shape": output_shape,
        }

    def _detect_iter_scatter(
        self,
        index: sympy.Expr,
        indirect_var: str,
        indirect_info: Optional[tuple[int, int, int]],
        stride_info: Optional[dict],
    ) -> Optional[dict[str, Any]]:
        """Detect scatter pattern with iteration variables using PallasStride markers."""
        if not indirect_info:
            return None
        if not stride_info:
            return None

        # Get indirect var info directly from PallasIndirectStride
        indirect_stride, indirect_dim, _ = indirect_info

        # Build list of (var_name, stride, dim_index, length) from PallasStride
        all_vars = []
        for iter_pos, (stride_val, dim_idx) in stride_info.items():
            if stride_val == 0:
                continue  # Skip expand dimensions
            # Find the corresponding variable
            var_items = list(self.range_tree_nodes.items())
            if iter_pos < len(var_items):
                var_sym, entry = var_items[iter_pos]
                length = self._safe_int(entry.length)
                if length is None:
                    return None
                all_vars.append((str(var_sym), stride_val, dim_idx, length))

        # Sort by stride (descending) to get dimension order
        all_vars.sort(key=lambda x: x[1], reverse=True)

        # Find position of indirect var in the sorted order based on its stride
        indirect_pos = 0
        for i, (_, stride, _, _) in enumerate(all_vars):
            if indirect_stride >= stride:
                indirect_pos = i
                break
            indirect_pos = i + 1

        return {
            "indirect_var": indirect_var,
            "indirect_dim": indirect_dim,
            "dims_before": [(n, l) for n, _, _, l in all_vars[:indirect_pos]],
            "dims_after": [(n, l) for n, _, _, l in all_vars[indirect_pos:]],
            "is_point_scatter": False,
            "output_shape": None,
        }

    def reduction(
        self,
        dtype: torch.dtype,
        src_dtype: torch.dtype,
        reduction_type: ReductionType,
        value: Union[CSEVariable, tuple[CSEVariable, ...]],
    ) -> Union[CSEVariable, tuple[CSEVariable, ...]]:  # type: ignore[override]
        """
        Generate code for reduction operations in JAX/Pallas.

        Reductions in Pallas work by:
        1. Loading the input data into the kernel
        2. Applying JAX reduction operations (jnp.sum, jnp.max, etc.)
        3. Storing the reduced result

        The reduction happens over the loaded block of data.
        """
        assert self.inside_reduction

        # Handle welford_reduce using the fallback (computes via sum reductions)
        if reduction_type == "welford_reduce":
            return self.welford_reduce_fallback(dtype, value)

        if isinstance(value, tuple):
            raise Unsupported(
                "Tuple reductions (e.g., welford_combine) not supported in Pallas backend"
            )

        # Check if this reduction is already cached
        cache_key = (src_dtype, reduction_type, value)
        if cache_key in self.cse.reduction_cache:
            return self.cse.reduction_cache[cache_key]

        # Map reduction types to JAX functions
        reduction_ops = {
            "sum": "jnp.sum",
            "prod": "jnp.prod",  # CPU only - not supported in Pallas GPU (Mosaic) backend
            "max": "jnp.max",
            "min": "jnp.min",
            "any": "jnp.any",
            "argmax": "jnp.argmax",
            "argmin": "jnp.argmin",
        }

        # Determine if this is a partial reduction (has pointwise dimensions)
        # or a full reduction to scalar
        pointwise_prefixes = OrderedSet(["x", "y", "z"])
        has_pointwise = any(p in self.numels for p in pointwise_prefixes)

        # Get the pointwise and reduction numels
        pointwise_numel: Optional[int] = self._compute_prefix_numel(pointwise_prefixes)
        reduction_numel: Optional[int] = self._compute_reduction_numel()

        # Count the number of pointwise and reduction dimensions
        n_reduction_dims = sum(
            1 for var, entry in self.range_tree_nodes.items() if entry.is_reduction
        )

        if reduction_type == "xor_sum":
            if has_pointwise and pointwise_numel and reduction_numel:
                reduction_expr = f"jnp.bitwise_xor.reduce({value}.reshape({pointwise_numel}, -1), axis=-1)"
            else:
                reduction_expr = f"jnp.bitwise_xor.reduce({value})"
        elif reduction_type in ("argmax", "argmin"):
            # For argmax/argmin, the result is indices into the reduction dimension.
            # Unlike sum/max/min, we can't just reshape because the indices depend
            # on which axis we reduce over. We need to determine the correct axis.
            reduction_op = reduction_ops[reduction_type]
            # Check if this is a true partial reduction (pointwise numel > 1)
            # When pointwise_numel == 1, it's effectively a full reduction to scalar
            is_partial_reduction = (
                has_pointwise
                and pointwise_numel
                and pointwise_numel > 1
                and reduction_numel
            )
            if is_partial_reduction and n_reduction_dims > 0:
                # Partial reduction: determine the reduction axis from PallasStride markers
                # Higher stride = outer axis (lower axis number in row-major order)
                reduction_axis = -1  # Default to last axis
                if self.load_index_exprs:
                    # Get the first load index expression
                    load_index = next(iter(self.load_index_exprs.values()))

                    # Get stride info from PallasStride markers
                    stride_info = self._get_strides_from_pallas_stride(load_index)
                    if stride_info:
                        # Build var -> stride mapping
                        var_items = list(self.range_tree_nodes.items())
                        var_to_stride = {}
                        for iter_pos, (stride_val, _) in stride_info.items():
                            if iter_pos < len(var_items):
                                var_sym = var_items[iter_pos][0]
                                var_to_stride[var_sym] = stride_val

                        # Find reduction and pointwise variable strides
                        reduction_vars = [
                            var
                            for var, entry in self.range_tree_nodes.items()
                            if entry.is_reduction
                        ]
                        pw_vars = [
                            var
                            for var, entry in self.range_tree_nodes.items()
                            if not entry.is_reduction
                        ]

                        if reduction_vars and pw_vars:
                            r_stride = var_to_stride.get(reduction_vars[0], 1)
                            pw_stride = var_to_stride.get(pw_vars[0], 1)
                            # Higher stride = earlier (outer) axis
                            reduction_axis = 0 if r_stride > pw_stride else -1
                reduction_expr = f"{reduction_op}({value}, axis={reduction_axis})"
            else:
                # Full reduction to scalar
                reduction_expr = f"{reduction_op}({value})"
        elif reduction_type in reduction_ops:
            # Check for true partial reduction (pointwise_numel > 1 means we have
            # actual pointwise dimensions, not just a scalar placeholder)
            is_partial_reduction = (
                has_pointwise
                and pointwise_numel is not None
                and pointwise_numel > 1
                and reduction_numel
            )
            # Also check for symbolic partial reduction (has both pw and reduction vars)
            is_symbolic_partial = (
                has_pointwise and n_reduction_dims > 0 and pointwise_numel is None
            )
            if is_partial_reduction:
                # For partial reductions, we need to:
                # 1. Find which axes are reduction axes (contiguous axes whose product = reduction_numel)
                # 2. Move pointwise axes to front, reduction axes to back
                # 3. Reshape to (pointwise_numel, reduction_numel) and reduce over last axis
                # 4. Reshape output with 1s in reduced dims for proper broadcasting
                reduction_op = reduction_ops[reduction_type]
                # Use a helper to find reduction axes by product matching
                reduction_expr = f"_pallas_partial_reduce({reduction_op}, {value}, {pointwise_numel}, {reduction_numel})"
            elif is_symbolic_partial:
                # Symbolic sizes: use axis-based reduction (axis=0 for outer reduction)
                reduction_expr = f"{reduction_ops[reduction_type]}({value}, axis=0)"
            else:
                # Full reduction to scalar
                reduction_expr = f"{reduction_ops[reduction_type]}({value})"
        else:
            raise Unsupported(
                f"Reduction type '{reduction_type}' not yet supported in Pallas backend. "
                f"Supported types: {list(reduction_ops.keys())}, xor_sum"
            )

        # Generate CSE variable for the reduction result
        result = self.cse.generate(
            self.compute,
            reduction_expr,
            dtype=dtype,
        )

        # Cache the result
        self.cse.reduction_cache[cache_key] = result
        return result

    @staticmethod
    def _buffer_is_contiguous(buffer_name: str) -> bool:
        buf = V.graph.get_buffer(buffer_name)
        layout = buf.get_layout()
        return layout.is_contiguous()

    def codegen_kernel(self, name: Optional[str] = None) -> str:  # type: ignore[override]
        """
        Generate the complete Pallas kernel code as a Python string.

        This includes:
        - Import statements for JAX/Pallas
        - The kernel function that operates on refs
        - The main wrapper function that handles PyTorch<->JAX conversions via DLPack

        Args:
            name: Optional kernel name (will use placeholder if not provided)

        Returns:
            str: Complete Python source code for the Pallas kernel
        """
        code = IndentedBuffer()

        # Define the Pallas kernel: accepts refs, uses broadcasted expressions
        arg_defs, call_args, _, _ = self.args.python_argdefs()
        kernel_params = [a.name for a in arg_defs]
        pure_out_params = [p for p in kernel_params if p.startswith("out_ptr")]
        output_params = [
            p for p in kernel_params if p.startswith(("out_ptr", "in_out_ptr"))
        ]
        # Identify size variable parameters (scalars like load_seed_offset)
        size_var_names = OrderedSet(self.args.sizevars.values())
        size_var_params = [p for p in kernel_params if p in size_var_names]
        if not output_params:
            raise RuntimeError("Pallas backend requires at least one output buffer")

        output_buffer_lookup = {
            inner: outer
            for outer, inner in self.args.output_buffers.items()
            if isinstance(inner, str)
        }

        kernel_name = name or "<KERNEL_NAME>"
        interpret_is_cpu = V.graph.get_current_device_or_throw().type == "cpu"
        is_tpu = torch._inductor.config._debug_cpu_to_tpu_pallas
        if is_tpu:
            if not torch._inductor.config.pallas_take_first_jax_device_only:
                raise RuntimeError(
                    "Pallas backend currently only supports using the first JAX device."
                )
            if not has_tpu_pallas():
                raise RuntimeError(
                    "PALLAS_TARGET_TPU is set, but no TPU device was found. "
                    "Please make sure that you have a TPU available and that JAX is configured correctly."
                )
        interpret_literal = "True" if interpret_is_cpu else "False"

        # For GPU (Mosaic backend), import plgpu for TMA operations
        # Import math for symbolic expressions (e.g., math.floor, math.log2)
        imports = """
import functools
import math
import torch
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from torch._inductor.runtime.runtime_utils import torch_dtype_to_jax_runtime
def _pallas_partial_reduce(reduce_fn, v, pw_numel, red_numel):
    # Helper for partial reductions: reorders axes and reduces
    # Returns result with keepdims-style shape for proper in-kernel broadcasting
    shape = tuple(v.shape)
    # Find contiguous axes whose product = red_numel (search from right)
    red_axes = None
    for i in range(len(shape) - 1, -1, -1):
        prod = 1
        for j in range(i, -1, -1):
            prod *= shape[j]
            if prod == red_numel:
                red_axes = list(range(j, i + 1))
                break
        if red_axes is not None:
            break
    if red_axes is None:
        red_axes = [len(shape) - 1]
    # Build output shape with 1s for reduced dimensions (keepdims style)
    out_shape = tuple(1 if i in red_axes else s for i, s in enumerate(shape))
    # Move pointwise axes to front, reduction axes to back
    pw_axes = [i for i in range(len(shape)) if i not in red_axes]
    reordered = jnp.moveaxis(v, pw_axes, list(range(len(pw_axes))))
    result = reduce_fn(reordered.reshape(pw_numel, red_numel), axis=-1)
    return result.reshape(out_shape)
""" + (
            "\nfrom jax.experimental.pallas import mosaic_gpu as plgpu"
            if not interpret_is_cpu
            else ""
        )
        code.splice(imports, strip=True)

        aliasable_flags: dict[str, bool] = {}
        for param in pure_out_params:
            buffer_name = output_buffer_lookup.get(param)
            is_contiguous = buffer_name is not None and self._buffer_is_contiguous(
                buffer_name
            )
            # Enable aliasing if:
            # 1. Not on CPU and buffer is contiguous (normal case), OR
            # 2. Output needs to be readable (for scatter operations)
            # outputs_need_read contains output parameter names (e.g., out_ptr0)
            needs_read = param in self.outputs_need_read
            aliasable_flags[param] = (
                (not interpret_is_cpu) and is_contiguous
            ) or needs_read
        alias_params = [
            f"{param}_alias" for param in pure_out_params if aliasable_flags[param]
        ]
        pointer_tail = [
            p for p in kernel_params if p.startswith(("in_out_ptr", "in_ptr"))
        ]
        kernel_input_params = alias_params + pointer_tail
        full_kernel_params = alias_params + kernel_params
        non_alias_out_set = OrderedSet(
            [name for name, flag in aliasable_flags.items() if not flag]
        )
        # On CPU (interpret=True), we need to copy back even aliased outputs
        # because pallas_call returns a new array (doesn't mutate in-place)
        # For outputs that need read access (scatter), we enable aliasing to read
        # current values, but still need to copy back the result
        if interpret_is_cpu:
            # Copy back all outputs on CPU
            copy_output_indices = list(range(len(output_params)))
        else:
            copy_output_indices = [
                idx
                for idx, name in enumerate(output_params)
                if name in non_alias_out_set
            ]
        self.aliasable_out_ptrs = aliasable_flags

        # Generate kernel body into a separate buffer first.
        # This allows us to discover all size variables (registered via rename_indexing)
        # before generating the kernel signature.
        kernel_body = IndentedBuffer()
        with kernel_body.indent():
            # Generate iteration variables as jnp.arange arrays using 2D canonical form.
            # All computation happens in (numel, rnumel) shape space:
            # - Pointwise vars (x*): shape (numel, 1) when rnumel > 1, else (numel,)
            # - Reduction vars (r*): shape (1, rnumel) when numel > 1, else (rnumel,)
            # This ensures all operations broadcast correctly.
            # Skip on GPU - jnp.arange is not supported by Pallas Mosaic backend
            if self.range_tree_nodes and not self.is_gpu:
                kernel_body.writeline("# Define iteration variables as JAX arrays (2D canonical form)")

                # Separate pointwise and reduction variables
                pointwise_vars = []
                reduction_vars = []
                for var_sym, entry in self.range_tree_nodes.items():
                    if entry.is_reduction:
                        reduction_vars.append((var_sym, entry))
                    else:
                        pointwise_vars.append((var_sym, entry))

                # Compute iteration space dimensions
                # Strategy:
                # 1. With reduction: use (numel, rnumel) where numel = x*y*z, rnumel = r0_*r1_*...
                # 2. Without reduction but with x and y: use (y_numel, x_numel) for broadcasting
                # 3. Otherwise: 1D form

                # Compute total pointwise numel (x*y*z)
                pointwise_prefixes = ["x", "y", "z"]
                numel_expr = sympy.Integer(1)
                for prefix in pointwise_prefixes:
                    if prefix in self.numels:
                        numel_expr = numel_expr * self.numels[prefix]
                numel_expr = V.graph.sizevars.simplify(numel_expr)

                # Compute reduction numel
                reduction_prefixes = ["r0_", "r1_", "r2_"]
                rnumel_expr = sympy.Integer(1)
                for prefix in reduction_prefixes:
                    if prefix in self.numels:
                        rnumel_expr = rnumel_expr * self.numels[prefix]
                rnumel_expr = V.graph.sizevars.simplify(rnumel_expr)

                # Get individual dimension sizes for non-reduction case
                y_numel_expr = self.numels.get("y", sympy.Integer(1))
                x_numel_expr = self.numels.get("x", sympy.Integer(1))
                y_numel_expr = V.graph.sizevars.simplify(y_numel_expr)
                x_numel_expr = V.graph.sizevars.simplify(x_numel_expr)

                numel_val = self._safe_int(numel_expr)
                rnumel_val = self._safe_int(rnumel_expr)
                y_numel_val = self._safe_int(y_numel_expr)
                x_numel_val = self._safe_int(x_numel_expr)

                # Determine what dimensions we have
                has_reduction = len(reduction_vars) > 0 and (rnumel_val is None or rnumel_val > 1)
                has_both_xy = (y_numel_val is not None and y_numel_val > 1 and
                               x_numel_val is not None and x_numel_val > 1)
                num_pointwise = len(pointwise_vars)

                # Case 1: With reduction - use (numel, rnumel) form
                if num_pointwise > 0 and has_reduction:
                    renamed_numel = self.rename_indexing(numel_expr)
                    numel_str = self.kexpr(renamed_numel)
                    renamed_rnumel = self.rename_indexing(rnumel_expr)
                    rnumel_str = self.kexpr(renamed_rnumel)

                    # Generate base pointwise index with shape (numel, 1)
                    kernel_body.writeline(f"_pw_idx = jnp.arange({numel_str})[:, None]")

                    # Generate each pointwise variable using modular arithmetic
                    for var_sym, entry in pointwise_vars:
                        var_name = str(var_sym)
                        renamed_length = self.rename_indexing(entry.length)
                        length_str = self.kexpr(renamed_length)
                        renamed_divisor = self.rename_indexing(entry.divisor)
                        divisor_str = self.kexpr(renamed_divisor)

                        if entry.divisor == 1 and entry.length == numel_expr:
                            kernel_body.writeline(f"{var_name} = _pw_idx")
                        elif entry.divisor == 1:
                            kernel_body.writeline(f"{var_name} = _pw_idx % {length_str}")
                        elif entry.length * entry.divisor == numel_expr:
                            kernel_body.writeline(f"{var_name} = _pw_idx // {divisor_str}")
                        else:
                            kernel_body.writeline(f"{var_name} = (_pw_idx // {divisor_str}) % {length_str}")

                    # Generate base reduction index with shape (1, rnumel)
                    kernel_body.writeline(f"_r_idx = jnp.arange({rnumel_str})[None, :]")

                    # Generate each reduction variable
                    for var_sym, entry in reduction_vars:
                        var_name = str(var_sym)
                        renamed_length = self.rename_indexing(entry.length)
                        length_str = self.kexpr(renamed_length)
                        renamed_divisor = self.rename_indexing(entry.divisor)
                        divisor_str = self.kexpr(renamed_divisor)

                        if entry.divisor == 1 and entry.length == rnumel_expr:
                            kernel_body.writeline(f"{var_name} = _r_idx")
                        elif entry.divisor == 1:
                            kernel_body.writeline(f"{var_name} = _r_idx % {length_str}")
                        elif entry.length * entry.divisor == rnumel_expr:
                            kernel_body.writeline(f"{var_name} = _r_idx // {divisor_str}")
                        else:
                            kernel_body.writeline(f"{var_name} = (_r_idx // {divisor_str}) % {length_str}")

                # Case 2: No reduction but has both x and y - use (y_numel, x_numel) for broadcasting
                elif num_pointwise > 0 and has_both_xy:
                    renamed_y = self.rename_indexing(y_numel_expr)
                    y_str = self.kexpr(renamed_y)
                    renamed_x = self.rename_indexing(x_numel_expr)
                    x_str = self.kexpr(renamed_x)

                    # Generate y index with shape (y_numel, 1)
                    kernel_body.writeline(f"_y_idx = jnp.arange({y_str})[:, None]")
                    # Generate x index with shape (1, x_numel)
                    kernel_body.writeline(f"_x_idx = jnp.arange({x_str})[None, :]")

                    # Generate each pointwise variable based on its prefix
                    for var_sym, entry in pointwise_vars:
                        var_name = str(var_sym)
                        prefix = var_name.rstrip('0123456789_')
                        renamed_length = self.rename_indexing(entry.length)
                        length_str = self.kexpr(renamed_length)
                        renamed_divisor = self.rename_indexing(entry.divisor)
                        divisor_str = self.kexpr(renamed_divisor)

                        if prefix == "x":
                            # x variables derive from _x_idx
                            if entry.divisor == 1 and entry.length == x_numel_expr:
                                kernel_body.writeline(f"{var_name} = _x_idx")
                            elif entry.divisor == 1:
                                kernel_body.writeline(f"{var_name} = _x_idx % {length_str}")
                            elif entry.length * entry.divisor == x_numel_expr:
                                kernel_body.writeline(f"{var_name} = _x_idx // {divisor_str}")
                            else:
                                kernel_body.writeline(f"{var_name} = (_x_idx // {divisor_str}) % {length_str}")
                        else:
                            # y, z variables derive from _y_idx
                            if entry.divisor == 1 and entry.length == y_numel_expr:
                                kernel_body.writeline(f"{var_name} = _y_idx")
                            elif entry.divisor == 1:
                                kernel_body.writeline(f"{var_name} = _y_idx % {length_str}")
                            elif entry.length * entry.divisor == y_numel_expr:
                                kernel_body.writeline(f"{var_name} = _y_idx // {divisor_str}")
                            else:
                                kernel_body.writeline(f"{var_name} = (_y_idx // {divisor_str}) % {length_str}")

                # Case 3: Only pointwise vars - use 2D form for broadcasting
                elif num_pointwise > 0:
                    # Find the innermost dimension (smallest length with divisor=1)
                    # This determines the "inner" size for 2D form
                    inner_length_expr = None
                    for var_sym, entry in pointwise_vars:
                        if entry.divisor == 1:
                            entry_len = self._safe_int(entry.length)
                            if entry_len is not None and entry_len > 1:
                                if inner_length_expr is None:
                                    inner_length_expr = entry.length
                                else:
                                    curr_len = self._safe_int(inner_length_expr)
                                    if curr_len is None or entry_len < curr_len:
                                        inner_length_expr = entry.length

                    # Use 2D form if we have a non-trivial inner dimension
                    inner_val = self._safe_int(inner_length_expr) if inner_length_expr else None
                    use_2d = (inner_val is not None and inner_val > 1 and
                              numel_val is not None and numel_val > inner_val)

                    if use_2d:
                        outer_val = numel_val // inner_val
                        outer_expr = sympy.Integer(outer_val)
                        inner_expr = inner_length_expr

                        renamed_outer = self.rename_indexing(outer_expr)
                        outer_str = self.kexpr(renamed_outer)
                        renamed_inner = self.rename_indexing(inner_expr)
                        inner_str = self.kexpr(renamed_inner)

                        # Track for _maybe_reshape_to_nd_form
                        self._pw_2d_outer = outer_val
                        self._pw_2d_inner = inner_val

                        # Generate 2D indices for broadcasting
                        kernel_body.writeline(f"_outer_idx = jnp.arange({outer_str})[:, None]")
                        kernel_body.writeline(f"_inner_idx = jnp.arange({inner_str})[None, :]")
                        kernel_body.writeline(f"_pw_idx = _outer_idx * {inner_str} + _inner_idx")

                        for var_sym, entry in pointwise_vars:
                            var_name = str(var_sym)
                            renamed_length = self.rename_indexing(entry.length)
                            length_str = self.kexpr(renamed_length)
                            renamed_divisor = self.rename_indexing(entry.divisor)
                            divisor_str = self.kexpr(renamed_divisor)

                            if entry.divisor == 1 and entry.length == numel_expr:
                                kernel_body.writeline(f"{var_name} = _pw_idx")
                            elif entry.divisor == 1:
                                kernel_body.writeline(f"{var_name} = _pw_idx % {length_str}")
                            elif entry.length * entry.divisor == numel_expr:
                                kernel_body.writeline(f"{var_name} = _pw_idx // {divisor_str}")
                            else:
                                kernel_body.writeline(f"{var_name} = (_pw_idx // {divisor_str}) % {length_str}")
                    else:
                        # Simple 1D form - no broadcasting needed
                        renamed_numel = self.rename_indexing(numel_expr)
                        numel_str = self.kexpr(renamed_numel)
                        kernel_body.writeline(f"_pw_idx = jnp.arange({numel_str})")

                        for var_sym, entry in pointwise_vars:
                            var_name = str(var_sym)
                            renamed_length = self.rename_indexing(entry.length)
                            length_str = self.kexpr(renamed_length)
                            renamed_divisor = self.rename_indexing(entry.divisor)
                            divisor_str = self.kexpr(renamed_divisor)

                            if entry.divisor == 1 and entry.length == numel_expr:
                                kernel_body.writeline(f"{var_name} = _pw_idx")
                            elif entry.divisor == 1:
                                kernel_body.writeline(f"{var_name} = _pw_idx % {length_str}")
                            elif entry.length * entry.divisor == numel_expr:
                                kernel_body.writeline(f"{var_name} = _pw_idx // {divisor_str}")
                            else:
                                kernel_body.writeline(f"{var_name} = (_pw_idx // {divisor_str}) % {length_str}")

                # Case 4: Only reduction vars
                elif has_reduction:
                    renamed_rnumel = self.rename_indexing(rnumel_expr)
                    rnumel_str = self.kexpr(renamed_rnumel)
                    kernel_body.writeline(f"_r_idx = jnp.arange({rnumel_str})")

                    for var_sym, entry in reduction_vars:
                        var_name = str(var_sym)
                        renamed_length = self.rename_indexing(entry.length)
                        length_str = self.kexpr(renamed_length)
                        renamed_divisor = self.rename_indexing(entry.divisor)
                        divisor_str = self.kexpr(renamed_divisor)

                        if entry.divisor == 1 and entry.length == rnumel_expr:
                            kernel_body.writeline(f"{var_name} = _r_idx")
                        elif entry.divisor == 1:
                            kernel_body.writeline(f"{var_name} = _r_idx % {length_str}")
                        elif entry.length * entry.divisor == rnumel_expr:
                            kernel_body.writeline(f"{var_name} = _r_idx // {divisor_str}")
                        else:
                            kernel_body.writeline(f"{var_name} = (_r_idx // {divisor_str}) % {length_str}")

                elif has_reduction:
                    # Only reduction vars, no pointwise - use 1D form
                    renamed_rnumel = self.rename_indexing(rnumel_expr)
                    rnumel_str = self.kexpr(renamed_rnumel)

                    # Generate base reduction index
                    kernel_body.writeline(f"_r_idx = jnp.arange({rnumel_str})")

                    # Generate each reduction variable
                    for var_sym, entry in reduction_vars:
                        var_name = str(var_sym)
                        renamed_length = self.rename_indexing(entry.length)
                        length_str = self.kexpr(renamed_length)
                        renamed_divisor = self.rename_indexing(entry.divisor)
                        divisor_str = self.kexpr(renamed_divisor)

                        if entry.divisor == 1 and entry.length == rnumel_expr:
                            kernel_body.writeline(f"{var_name} = _r_idx")
                        elif entry.divisor == 1:
                            kernel_body.writeline(f"{var_name} = _r_idx % {length_str}")
                        elif entry.length * entry.divisor == rnumel_expr:
                            kernel_body.writeline(f"{var_name} = _r_idx // {divisor_str}")
                        else:
                            kernel_body.writeline(f"{var_name} = (_r_idx // {divisor_str}) % {length_str}")

            # Emit compute (CSE) and store lines; they reference *_ptr[index] directly.
            for line in self.compute._lines:
                kernel_body.writeline(str(line))

        # Recompute kernel parameters after kernel body generation.
        # Size variables may have been registered during kernel body generation
        # (e.g., via rename_indexing for symbolic sizes), so we need to re-fetch
        # the arg defs to capture all parameters including newly-registered size vars.
        arg_defs, call_args, _, _ = self.args.python_argdefs()
        kernel_params = [a.name for a in arg_defs]
        size_var_names = OrderedSet(self.args.sizevars.values())
        size_var_params = [p for p in kernel_params if p in size_var_names]
        pointer_tail = [
            p for p in kernel_params if p.startswith(("in_out_ptr", "in_ptr"))
        ]
        kernel_input_params = alias_params + pointer_tail
        full_kernel_params = alias_params + kernel_params

        # Now emit the kernel function with the correct signature
        kernel_signature = f"def {kernel_name}_kernel({', '.join(full_kernel_params)}):"
        code.writeline(kernel_signature)

        with code.indent():
            for line in kernel_body._lines:
                if isinstance(line, str):
                    # Remove any existing indentation and re-add with code's indentation
                    code.writeline(line.lstrip())
                else:
                    code._lines.append(line)

            # Add store lines (using recomputed full_kernel_params)
            # Filter stores to only emit those for outputs that are in kernel params.
            # This handles cases where an intermediate value was stored but the buffer
            # was later optimized away (not passed to the kernel).
            for out_ptr, store_line in self.store_with_output:
                if out_ptr in full_kernel_params:
                    code.writeline(store_line)

        jit_wrapper_name = f"{kernel_name}_jit_wrapper"
        donate_indices = []
        # Offset by 2 for (out_shapes, out_dtypes), plus size_var_params count
        base_offset = 2 + len(size_var_params)
        for idx, name in enumerate(kernel_input_params):
            if (name in alias_params) or name.startswith("in_out_ptr"):
                donate_indices.append(idx + base_offset)
        if donate_indices:
            donate_literal = "(" + ", ".join(str(x) for x in donate_indices) + ",)"
        else:
            donate_literal = "()"
        # Size variables are static args (after out_shapes and out_dtypes)
        static_argnums = list(range(2 + len(size_var_params)))
        static_argnums_literal = "(" + ", ".join(str(x) for x in static_argnums) + ",)"
        # Always set backend='cpu' when using interpret mode (CPU execution)
        # This ensures JAX doesn't try to use TPU even when PALLAS_TARGET_TPU=1
        # (that flag affects codegen patterns, not actual execution device)
        if interpret_is_cpu:
            jit_decorator = (
                f"@functools.partial(jax.jit, static_argnums={static_argnums_literal}, "
                f"donate_argnums={donate_literal}, backend='cpu')"
            )
        else:
            jit_decorator = (
                f"@functools.partial(jax.jit, static_argnums={static_argnums_literal}, "
                f"donate_argnums={donate_literal})"
            )
        code.writeline(jit_decorator)
        # Include size_var_params in wrapper signature
        wrapper_params = (
            ["out_shapes", "out_dtypes"] + size_var_params + kernel_input_params
        )
        code.writeline(f"def {jit_wrapper_name}({', '.join(wrapper_params)}):")
        with code.indent():
            code.writeline("out_specs = tuple(")
            code.writeline("    jax.ShapeDtypeStruct(shape, dtype)")
            code.writeline("    for shape, dtype in zip(out_shapes, out_dtypes)")
            code.writeline(")")

            alias_pairs: list[tuple[int, int]] = []
            for out_idx, name in enumerate(output_params):
                if name.startswith("out_ptr"):
                    if aliasable_flags.get(name, False):
                        alias_name = f"{name}_alias"
                        input_idx = kernel_input_params.index(alias_name)
                        alias_pairs.append((input_idx, out_idx))
                else:
                    input_idx = kernel_input_params.index(name)
                    alias_pairs.append((input_idx, out_idx))
            alias_map_literal = ", ".join(f"{i}: {o}" for (i, o) in alias_pairs)

            # Wrap kernel with functools.partial to pass scalar arguments (size variables)
            partial_args = []
            for sv_param in size_var_params:
                partial_args.append(f"{sv_param}={sv_param}")

            if partial_args:
                kernel_arg = f"functools.partial({kernel_name}_kernel, {', '.join(partial_args)}),"
            else:
                kernel_arg = f"{kernel_name}_kernel,"

            # Use plgpu.kernel for GPU (Mosaic), pl.pallas_call for CPU/TPU
            # TMA approach requires: no reductions, all inputs contiguous, same sizes
            use_tma = (
                self.is_gpu and self.use_emit_pipeline and self._can_use_tma_approach()
            )
            if use_tma:
                # Use lax.fori_loop with direct TMA for automatic OOB masking
                # TMA (Tensor Memory Accelerator) automatically handles out-of-bounds
                # accesses, eliminating the need for explicit padding to multiples of 128
                code.writeline("# Use lax.fori_loop with TMA for automatic OOB masking")
                code.writeline("from jax import lax")
                code.writeline("_tile_size = 128  # Warpgroup size")
                code.writeline("_orig_out_shapes = out_shapes")

                # Calculate max numel across all inputs/outputs for grid calculation
                code.writeline("_max_numel = 0")
                for param in kernel_input_params:
                    code.writeline(f"_max_numel = max(_max_numel, {param}.size)")
                code.writeline("for shape in out_shapes:")
                code.writeline("    _numel = 1")
                code.writeline("    for s in shape:")
                code.writeline("        _numel *= s")
                code.writeline("    _max_numel = max(_max_numel, _numel)")

                code.writeline(
                    "_num_tiles = (_max_numel + _tile_size - 1) // _tile_size"
                )

                # Build param names for the kernel
                gmem_input_params = [f"{p}_gmem" for p in kernel_input_params]
                gmem_output_params = [f"{p}_gmem" for p in output_params]
                smem_input_params = [f"{p}_smem" for p in kernel_input_params]
                smem_output_params = [f"{p}_smem" for p in output_params]

                # Generate the TMA kernel with fori_loop
                code.writeline("")
                code.writeline("# Wrapper kernel using lax.fori_loop with direct TMA")

                # Kernel receives: *input_gmem_refs, *output_gmem_refs (from plgpu.kernel)
                # Plus scratch SMEM buffers for inputs and outputs, and barriers for TMA
                wrapper_kernel_params = gmem_input_params + gmem_output_params
                all_smem_params = smem_input_params + smem_output_params
                # Barrier params for TMA operations
                barrier_params = [
                    f"_barrier_{i}" for i in range(len(kernel_input_params))
                ]
                scratch_params = ", ".join(all_smem_params + barrier_params)

                code.writeline(
                    f"def _tma_kernel({', '.join(wrapper_kernel_params)}, *, {scratch_params}):"
                )
                with code.indent():
                    # Define the loop body function
                    code.writeline("")
                    code.writeline("def _tile_body(_tile_idx, _):")
                    with code.indent():
                        code.writeline("_tile_start = _tile_idx * _tile_size")
                        code.writeline("")

                        # TMA load inputs from GMEM to SMEM
                        code.writeline(
                            "# TMA load inputs from GMEM to SMEM (OOB auto-masked)"
                        )
                        for i, (gmem_in, smem_in) in enumerate(
                            zip(gmem_input_params, smem_input_params)
                        ):
                            code.writeline(
                                f"plgpu.copy_gmem_to_smem({gmem_in}.at[pl.ds(_tile_start, _tile_size)], {smem_in}, _barrier_{i})"
                            )

                        # Wait for all input loads
                        code.writeline("")
                        code.writeline("# Wait for TMA loads to complete")
                        for i, _ in enumerate(gmem_input_params):
                            code.writeline(f"plgpu.barrier_wait(_barrier_{i})")

                        # Call the original kernel function with SMEM refs
                        code.writeline("")
                        code.writeline("# Compute on SMEM tiles")
                        kernel_call_args = smem_input_params + smem_output_params
                        kernel_fn = kernel_arg.rstrip(",").strip()
                        code.writeline(f"{kernel_fn}({', '.join(kernel_call_args)})")

                        # TMA store outputs from SMEM to GMEM
                        code.writeline("")
                        code.writeline(
                            "# TMA store outputs from SMEM to GMEM (OOB auto-masked)"
                        )
                        code.writeline("plgpu.commit_smem()")
                        for gmem_out, smem_out in zip(
                            gmem_output_params, smem_output_params
                        ):
                            code.writeline(
                                f"plgpu.copy_smem_to_gmem({smem_out}, {gmem_out}.at[pl.ds(_tile_start, _tile_size)])"
                            )
                        code.writeline("plgpu.wait_smem_to_gmem(0)")
                        code.writeline("")
                        code.writeline("return None")

                    # Run the loop over all tiles
                    code.writeline("")
                    code.writeline("# Iterate over all tiles")
                    code.writeline("lax.fori_loop(0, _num_tiles, _tile_body, None)")

                # Build scratch_shapes dict for SMEM buffers and TMA barriers
                code.writeline("")
                code.writeline(
                    "# Build SMEM scratch shapes for inputs, outputs, and TMA barriers"
                )
                code.writeline("_scratch_shapes = {}")
                for i, smem_param in enumerate(smem_input_params):
                    # Get dtype from input param
                    orig_param = kernel_input_params[i]
                    code.writeline(
                        f"_scratch_shapes['{smem_param}'] = plgpu.SMEM((_tile_size,), {orig_param}.dtype)"
                    )
                for i, smem_param in enumerate(smem_output_params):
                    code.writeline(
                        f"_scratch_shapes['{smem_param}'] = plgpu.SMEM((_tile_size,), out_dtypes[{i}])"
                    )
                # Add barriers for TMA GMEM->SMEM operations
                for barrier_param in barrier_params:
                    code.writeline(
                        f"_scratch_shapes['{barrier_param}'] = plgpu.Barrier(num_arrivals=1)"
                    )

                # Create flattened and aligned output specs for TMA
                code.writeline("")
                code.writeline("# Create flattened output specs aligned to tile size")
                code.writeline("_flat_out_specs = []")
                code.writeline("for shape, dtype in zip(out_shapes, out_dtypes):")
                code.writeline("    _numel = 1")
                code.writeline("    for s in shape:")
                code.writeline("        _numel *= s")
                code.writeline(
                    "    _aligned_numel = ((_numel + _tile_size - 1) // _tile_size) * _tile_size"
                )
                code.writeline(
                    "    _flat_out_specs.append(jax.ShapeDtypeStruct((_aligned_numel,), dtype))"
                )
                code.writeline("_flat_out_specs = tuple(_flat_out_specs)")

                # Call plgpu.kernel with the TMA kernel
                code.writeline("")
                code.writeline("# Call plgpu.kernel with TMA kernel")
                code.writeline("_result = plgpu.kernel(")
                with code.indent():
                    code.writeline("_tma_kernel,")
                    code.writeline("out_shape=_flat_out_specs,")
                    code.writeline("scratch_shapes=_scratch_shapes,")
                code.writeline(")(")
                # Pass flattened inputs for 1D tiled processing
                for param in kernel_input_params:
                    code.writeline(f"    {param}.flatten(),")
                code.writeline(")")

                # Reshape outputs to original shapes
                code.writeline("")
                code.writeline("# Reshape results to original shapes")
                code.writeline("if not isinstance(_result, tuple):")
                code.writeline("    _result = (_result,)")
                code.writeline("_final_results = []")
                code.writeline("for _res, _shape in zip(_result, _orig_out_shapes):")
                code.writeline("    _orig_numel = 1")
                code.writeline("    for _s in _shape:")
                code.writeline("        _orig_numel *= _s")
                code.writeline(
                    "    _final_results.append(_res[:_orig_numel].reshape(_shape))"
                )
                code.writeline(
                    "return _final_results[0] if len(_final_results) == 1 else tuple(_final_results)"
                )
            elif self.is_gpu:
                # Legacy GPU path with explicit padding (use_emit_pipeline=False)
                # For GPU, pad inputs to align to WARPGROUP_SIZE (128)
                # Mosaic GPU requires tensor sizes to be multiples of 128
                # BUT: only apply padding when all tensors have the same size
                # (no broadcasting). If inputs have different sizes, we need
                # to preserve shapes for proper broadcasting semantics.

                # First, check if all inputs and outputs have the same numel
                code.writeline(
                    "# Check if all tensors have same size (no broadcasting)"
                )
                code.writeline("_all_sizes = []")
                for i, param in enumerate(kernel_input_params):
                    code.writeline(f"_all_sizes.append({param}.size)")
                code.writeline("for shape in out_shapes:")
                code.writeline("    _numel = 1")
                code.writeline("    for s in shape:")
                code.writeline("        _numel *= s")
                code.writeline("    _all_sizes.append(_numel)")
                code.writeline("_unique_sizes = set(_all_sizes)")
                code.writeline(
                    "_can_pad = len(_unique_sizes) == 1 and all(s > 1 for s in _unique_sizes)"
                )

                code.writeline("")
                code.writeline("if _can_pad:")
                code.writeline("    # All tensors same size - safe to flatten and pad")
                code.writeline("    _orig_out_shapes = out_shapes")
                code.writeline("    _padded_inputs = []")
                for i, param in enumerate(kernel_input_params):
                    code.writeline(f"    _orig_size_{i} = {param}.size")
                    code.writeline(
                        f"    _aligned_size_{i} = ((_orig_size_{i} + 127) // 128) * 128"
                    )
                    code.writeline(f"    if _orig_size_{i} != _aligned_size_{i}:")
                    code.writeline(f"        _flat_{i} = {param}.flatten()")
                    code.writeline(
                        f"        _padded_{i} = jnp.pad(_flat_{i}, (0, _aligned_size_{i} - _orig_size_{i}))"
                    )
                    code.writeline(f"        _padded_inputs.append(_padded_{i})")
                    code.writeline("    else:")
                    code.writeline(f"        _padded_inputs.append({param}.flatten())")

                code.writeline("    # Align output shapes to warpgroup size (128)")
                code.writeline("    _aligned_out_specs = []")
                code.writeline("    _is_scalar_output = []")
                code.writeline("    for shape, dtype in zip(out_shapes, out_dtypes):")
                code.writeline("        _numel = 1")
                code.writeline("        for s in shape:")
                code.writeline("            _numel *= s")
                code.writeline("        if _numel <= 1:")
                code.writeline(
                    "            _aligned_out_specs.append(jax.ShapeDtypeStruct(shape, dtype))"
                )
                code.writeline("            _is_scalar_output.append(True)")
                code.writeline("        else:")
                code.writeline(
                    "            _aligned_numel = ((_numel + 127) // 128) * 128"
                )
                code.writeline(
                    "            _aligned_out_specs.append(jax.ShapeDtypeStruct((_aligned_numel,), dtype))"
                )
                code.writeline("            _is_scalar_output.append(False)")
                code.writeline("    _aligned_out_specs = tuple(_aligned_out_specs)")

                code.writeline("    _result = plgpu.kernel(")
                code.writeline("        " + kernel_arg)
                code.writeline("        out_shape=_aligned_out_specs,")
                code.writeline("    )(*_padded_inputs)")

                code.writeline("    # Remove padding from results")
                code.writeline("    if not isinstance(_result, tuple):")
                code.writeline("        _result = (_result,)")
                code.writeline("    _unpadded_results = []")
                code.writeline(
                    "    for _res, _shape, _is_scalar in zip(_result, _orig_out_shapes, _is_scalar_output):"
                )
                code.writeline("        if _is_scalar:")
                code.writeline("            _unpadded_results.append(_res)")
                code.writeline("        else:")
                code.writeline("            _orig_numel = 1")
                code.writeline("            for _s in _shape:")
                code.writeline("                _orig_numel *= _s")
                code.writeline(
                    "            _unpadded = _res[:_orig_numel].reshape(_shape)"
                )
                code.writeline("            _unpadded_results.append(_unpadded)")
                code.writeline(
                    "    return _unpadded_results[0] if len(_unpadded_results) == 1 else tuple(_unpadded_results)"
                )

                code.writeline("else:")
                code.writeline(
                    "    # Different sizes - check if it's a reduction (scalar output)"
                )
                code.writeline("    _out_numel = 1")
                code.writeline("    for s in out_shapes[0]:")
                code.writeline("        _out_numel *= s")
                code.writeline("    ")
                code.writeline("    if _out_numel <= 1:")
                code.writeline(
                    "        # Scalar output (reduction) - pad inputs but keep scalar output"
                )
                code.writeline("        _orig_out_shapes = out_shapes")
                code.writeline("        _padded_inputs = []")
                for i, param in enumerate(kernel_input_params):
                    code.writeline(f"        _orig_size_{i} = {param}.size")
                    code.writeline(
                        f"        _aligned_size_{i} = ((_orig_size_{i} + 127) // 128) * 128"
                    )
                    code.writeline(f"        if _orig_size_{i} != _aligned_size_{i}:")
                    code.writeline(f"            _flat_{i} = {param}.flatten()")
                    code.writeline(
                        f"            _padded_{i} = jnp.pad(_flat_{i}, (0, _aligned_size_{i} - _orig_size_{i}))"
                    )
                    code.writeline(f"            _padded_inputs.append(_padded_{i})")
                    code.writeline("        else:")
                    code.writeline(
                        f"            _padded_inputs.append({param}.flatten())"
                    )
                code.writeline("        ")
                code.writeline("        # Scalar output - don't pad")
                code.writeline("        _aligned_out_specs = tuple(")
                code.writeline("            jax.ShapeDtypeStruct(shape, dtype)")
                code.writeline(
                    "            for shape, dtype in zip(out_shapes, out_dtypes)"
                )
                code.writeline("        )")
                code.writeline("        ")
                code.writeline("        _result = plgpu.kernel(")
                code.writeline("            " + kernel_arg)
                code.writeline("            out_shape=_aligned_out_specs,")
                code.writeline("        )(*_padded_inputs)")
                code.writeline("        return _result")
                code.writeline("    else:")
                code.writeline(
                    "        # Non-scalar output with broadcasting - broadcast inputs to output shape"
                )
                code.writeline("        _target_shape = out_shapes[0]")
                code.writeline("        _target_numel = _out_numel")
                code.writeline("        _orig_out_shapes = out_shapes")
                code.writeline("        ")
                code.writeline(
                    "        # Broadcast and flatten all inputs to target shape"
                )
                code.writeline("        _padded_inputs = []")
                for i, param in enumerate(kernel_input_params):
                    code.writeline(
                        f"        _broadcasted_{i} = jnp.broadcast_to({param}, _target_shape).flatten()"
                    )
                    code.writeline(
                        f"        _aligned_size_{i} = ((_target_numel + 127) // 128) * 128"
                    )
                    code.writeline(f"        if _target_numel != _aligned_size_{i}:")
                    code.writeline(
                        f"            _padded_{i} = jnp.pad(_broadcasted_{i}, (0, _aligned_size_{i} - _target_numel))"
                    )
                    code.writeline(f"            _padded_inputs.append(_padded_{i})")
                    code.writeline("        else:")
                    code.writeline(
                        f"            _padded_inputs.append(_broadcasted_{i})"
                    )
                code.writeline("        ")
                code.writeline("        # Align output shapes to warpgroup size (128)")
                code.writeline("        _aligned_out_specs = []")
                code.writeline(
                    "        for shape, dtype in zip(out_shapes, out_dtypes):"
                )
                code.writeline("            _numel = 1")
                code.writeline("            for s in shape:")
                code.writeline("                _numel *= s")
                code.writeline(
                    "            _aligned_numel = ((_numel + 127) // 128) * 128"
                )
                code.writeline(
                    "            _aligned_out_specs.append(jax.ShapeDtypeStruct((_aligned_numel,), dtype))"
                )
                code.writeline("        _aligned_out_specs = tuple(_aligned_out_specs)")
                code.writeline("        ")
                code.writeline("        _result = plgpu.kernel(")
                code.writeline("            " + kernel_arg)
                code.writeline("            out_shape=_aligned_out_specs,")
                code.writeline("        )(*_padded_inputs)")
                code.writeline("        ")
                code.writeline("        # Remove padding from results")
                code.writeline("        if not isinstance(_result, tuple):")
                code.writeline("            _result = (_result,)")
                code.writeline("        _unpadded_results = []")
                code.writeline(
                    "        for _res, _shape in zip(_result, _orig_out_shapes):"
                )
                code.writeline("            _orig_numel = 1")
                code.writeline("            for _s in _shape:")
                code.writeline("                _orig_numel *= _s")
                code.writeline(
                    "            _unpadded = _res[:_orig_numel].reshape(_shape)"
                )
                code.writeline("            _unpadded_results.append(_unpadded)")
                code.writeline(
                    "        return _unpadded_results[0] if len(_unpadded_results) == 1 else tuple(_unpadded_results)"
                )
            else:
                code.writeline("return pl.pallas_call(")
                code.writeline("    " + kernel_arg)
                code.writeline("    out_shape=out_specs,")
                code.writeline(f"    interpret={interpret_literal},")
                code.writeline("    grid=(1,),")
                code.writeline(
                    f"    input_output_aliases={{ {alias_map_literal} }},"
                    if alias_pairs
                    else "    input_output_aliases={},"
                )
                code.writeline(")(")
                if kernel_input_params:
                    code.writeline(f"    {', '.join(kernel_input_params)},")
                code.writeline(")")

        main_name = f"{kernel_name}_main"
        code.writeline(
            f"def {main_name}({', '.join(full_kernel_params)}, stream=None):"
        )
        with code.indent():
            code.writeline("# Enable JAX x64 mode for float64/int64 support")
            code.writeline("jax.config.update('jax_enable_x64', True)")
            # Clear JAX caches to avoid Mosaic GPU backend state issues
            code.writeline("jax.clear_caches()")

            if alias_params:
                code.writeline("# Convert Torch -> JAX for donated outputs")
                for alias_name in alias_params:
                    # TODO: The `jax.device_put` path is a temporary workaround for a Mosaic compiler bug
                    # that occurs with DLPack. Once TorchTPU provides a direct method for placing a
                    # `torch.Tensor` on a TPU device, this should be reverted to use the
                    #  `jax.dlpack.from_dlpack` path.
                    if is_tpu:
                        code.writeline(
                            f"{alias_name}_jax = jax.device_put({alias_name}.cpu().numpy(), device=jax.devices('tpu')[0])"
                        )
                    else:
                        code.writeline(
                            f"{alias_name}_jax = jax.dlpack.from_dlpack({alias_name}.detach())"
                        )
            code.writeline("# Convert Torch -> JAX for in-place tensors")
            for ptr in pointer_tail:
                if ptr.startswith("in_out_ptr"):
                    if is_tpu:
                        code.writeline(
                            f"{ptr}_jax = jax.device_put({ptr}.cpu().numpy(), device=jax.devices('tpu')[0])"
                        )
                    else:
                        code.writeline(
                            f"{ptr}_jax = jax.dlpack.from_dlpack({ptr}.detach())"
                        )
            code.writeline("# Convert Torch -> JAX for inputs")
            for ptr in pointer_tail:
                if ptr.startswith("in_ptr"):
                    if is_tpu:
                        code.writeline(
                            f"{ptr}_jax = jax.device_put({ptr}.cpu().numpy(), device=jax.devices('tpu')[0])"
                        )
                    else:
                        code.writeline(
                            f"{ptr}_jax = jax.dlpack.from_dlpack({ptr}.detach().contiguous())"
                        )

            code.writeline("# Prepare output metadata from PyTorch tensor")
            code.writeline(
                "out_shapes = ("
                + ", ".join([f"tuple({name}.shape)" for name in output_params])
                + ",)"
            )
            code.writeline(
                "out_dtypes = ("
                + ", ".join(
                    [
                        f"torch_dtype_to_jax_runtime({name}.dtype)"
                        for name in output_params
                    ]
                )
                + ",)"
            )
            arg_name_map: dict[str, str] = {}
            for alias_name in alias_params:
                arg_name_map[alias_name] = f"{alias_name}_jax"
            for ptr in pointer_tail:
                arg_name_map[ptr] = f"{ptr}_jax"

            # Build the jit_wrapper call with size vars and tensor args
            wrapper_call_args = ["out_shapes", "out_dtypes"]
            # Add size variable params (they're already available as locals in main)
            wrapper_call_args.extend(size_var_params)
            # Add tensor args (with _jax suffix)
            wrapper_call_args.extend(arg_name_map[name] for name in kernel_input_params)
            code.writeline(f"res = {jit_wrapper_name}({', '.join(wrapper_call_args)})")
            if copy_output_indices:
                code.writeline(
                    "result_values = res if isinstance(res, tuple) else (res,)"
                )
                for idx in copy_output_indices:
                    name = output_params[idx]
                    if is_tpu:
                        code.writeline(
                            f"res_cpu = jax.device_get(result_values[{idx}])"
                        )
                        code.writeline(f"{name}.copy_(torch.from_dlpack(res_cpu))")
                    else:
                        code.writeline(
                            f"{name}.copy_(torch.from_dlpack(result_values[{idx}]))"
                        )

        return code.getvalue()

    def call_kernel(self, name: str, node: Optional[IRNode] = None) -> None:  # type: ignore[override]
        """Generate the Python code that calls this Pallas kernel."""
        wrapper = V.graph.wrapper_code
        arg_defs, call_args, _, _ = self.args.python_argdefs()
        kernel_param_names = [a.name for a in arg_defs]
        pure_out_params = [p for p in kernel_param_names if p.startswith("out_ptr")]
        call_arg_strs = list(map(str, call_args))
        aliasable = getattr(self, "aliasable_out_ptrs", {})
        alias_call_args = [
            call_arg_strs[kernel_param_names.index(p)]
            for p in pure_out_params
            if aliasable.get(p, False)
        ]

        # Generate kernel call: kernel_name.run(arg1, arg2, ...)
        # Note: async_compile.pallas loads {name}_main function and wraps it in PallasKernelWrapper
        # which exposes a run() method
        kernel_call = f"{name}.run({', '.join(alias_call_args + call_arg_strs)})"
        wrapper.writeline(kernel_call)


class PallasScheduling(SIMDScheduling):
    kernel_type = PallasKernel  # type: ignore[assignment]

    @classmethod
    def get_backend_features(cls, device: torch.device) -> OrderedSet[BackendFeature]:
        # Pallas/JAX can handle reductions to single elements efficiently
        # without requiring split reductions
        return OrderedSet([BackendFeature.REDUCE_TO_SINGLE_ELEMENT])

    def define_kernel(
        self,
        src_code: str,
        node_schedule: Sequence[BaseSchedulerNode],
        kernel: PallasKernel,
    ) -> str:  # type: ignore[override]
        wrapper = V.graph.wrapper_code
        if src_code in wrapper.src_to_kernel:
            return wrapper.src_to_kernel[src_code]

        fused_name = (
            get_fused_kernel_name(node_schedule, config.triton.descriptive_names)
            if config.triton.descriptive_names
            else ""
        )
        kernel_hash = hashlib.sha256(src_code.encode("utf-8")).hexdigest()[:8]
        if fused_name == "fused":
            kernel_name = f"pallas_{kernel_hash}"
        else:
            kernel_name = f"pallas_{fused_name}_{kernel_hash}"
        wrapper.src_to_kernel[src_code] = kernel_name

        # Replace placeholder if any
        src_code = src_code.replace("<KERNEL_NAME>", kernel_name)

        compile_wrapper = IndentedBuffer()
        compile_wrapper.writeline(f"async_compile.pallas({kernel_name!r}, r'''")
        compile_wrapper.splice(src_code, strip=True)
        compile_wrapper.writeline("''')")

        origins, detailed_origins = get_kernel_metadata(node_schedule, wrapper)
        metadata_comment = f"{origins}\n{detailed_origins}"
        wrapper.define_kernel(kernel_name, compile_wrapper.getvalue(), metadata_comment)

        return kernel_name
