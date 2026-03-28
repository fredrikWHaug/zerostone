//! Feature-gated floating-point type and math wrappers.
//!
//! By default, Zerostone uses `f64` for all computation. When the `f32`
//! feature is enabled (for embedded targets like Cortex-M4F where only
//! f32 has hardware FPU support), all computation uses `f32` instead.
//!
//! # Usage
//!
//! Library code should use [`Float`] instead of `f64`, and call math
//! functions through this module (e.g., `float::sqrt`) rather than
//! calling `libm` directly.
//!
//! ```
//! use zerostone::float::{self, Float};
//!
//! let x: Float = 4.0;
//! let y = float::sqrt(x);
//! assert!((y - 2.0).abs() < 1e-6);
//! ```

/// The floating-point type used throughout the library.
///
/// - Default: `f64` (desktop, Python bindings, offline analysis)
/// - With `f32` feature: `f32` (embedded, Cortex-M4F, NPU inference)
#[cfg(feature = "f32")]
pub type Float = f32;

/// The floating-point type used throughout the library.
#[cfg(not(feature = "f32"))]
pub type Float = f64;

// ---- Constants ----

/// Pi.
#[cfg(feature = "f32")]
pub const PI: Float = core::f32::consts::PI;
#[cfg(not(feature = "f32"))]
pub const PI: Float = core::f64::consts::PI;

/// Euler's number.
#[cfg(feature = "f32")]
pub const E: Float = core::f32::consts::E;
#[cfg(not(feature = "f32"))]
pub const E: Float = core::f64::consts::E;

/// 2 * Pi.
pub const TAU: Float = PI * 2.0;

/// ln(2).
#[cfg(feature = "f32")]
pub const LN_2: Float = core::f32::consts::LN_2;
#[cfg(not(feature = "f32"))]
pub const LN_2: Float = core::f64::consts::LN_2;

/// Maximum finite value.
#[cfg(feature = "f32")]
pub const MAX: Float = f32::MAX;
#[cfg(not(feature = "f32"))]
pub const MAX: Float = f64::MAX;

/// Smallest positive normal value.
#[cfg(feature = "f32")]
pub const MIN_POSITIVE: Float = f32::MIN_POSITIVE;
#[cfg(not(feature = "f32"))]
pub const MIN_POSITIVE: Float = f64::MIN_POSITIVE;

/// Positive infinity.
#[cfg(feature = "f32")]
pub const INFINITY: Float = f32::INFINITY;
#[cfg(not(feature = "f32"))]
pub const INFINITY: Float = f64::INFINITY;

// ---- Math wrappers ----
// Each function dispatches to the f32 or f64 variant of libm.

#[inline(always)]
pub fn sqrt(x: Float) -> Float {
    #[cfg(feature = "f32")]
    {
        libm::sqrtf(x)
    }
    #[cfg(not(feature = "f32"))]
    {
        libm::sqrt(x)
    }
}

#[inline(always)]
pub fn abs(x: Float) -> Float {
    #[cfg(feature = "f32")]
    {
        libm::fabsf(x)
    }
    #[cfg(not(feature = "f32"))]
    {
        libm::fabs(x)
    }
}

#[inline(always)]
pub fn exp(x: Float) -> Float {
    #[cfg(feature = "f32")]
    {
        libm::expf(x)
    }
    #[cfg(not(feature = "f32"))]
    {
        libm::exp(x)
    }
}

#[inline(always)]
pub fn log(x: Float) -> Float {
    #[cfg(feature = "f32")]
    {
        libm::logf(x)
    }
    #[cfg(not(feature = "f32"))]
    {
        libm::log(x)
    }
}

#[inline(always)]
pub fn sin(x: Float) -> Float {
    #[cfg(feature = "f32")]
    {
        libm::sinf(x)
    }
    #[cfg(not(feature = "f32"))]
    {
        libm::sin(x)
    }
}

#[inline(always)]
pub fn cos(x: Float) -> Float {
    #[cfg(feature = "f32")]
    {
        libm::cosf(x)
    }
    #[cfg(not(feature = "f32"))]
    {
        libm::cos(x)
    }
}

#[inline(always)]
pub fn tan(x: Float) -> Float {
    #[cfg(feature = "f32")]
    {
        libm::tanf(x)
    }
    #[cfg(not(feature = "f32"))]
    {
        libm::tan(x)
    }
}

#[inline(always)]
pub fn atan(x: Float) -> Float {
    #[cfg(feature = "f32")]
    {
        libm::atanf(x)
    }
    #[cfg(not(feature = "f32"))]
    {
        libm::atan(x)
    }
}

#[inline(always)]
pub fn atan2(y: Float, x: Float) -> Float {
    #[cfg(feature = "f32")]
    {
        libm::atan2f(y, x)
    }
    #[cfg(not(feature = "f32"))]
    {
        libm::atan2(y, x)
    }
}

#[inline(always)]
pub fn tanh(x: Float) -> Float {
    #[cfg(feature = "f32")]
    {
        libm::tanhf(x)
    }
    #[cfg(not(feature = "f32"))]
    {
        libm::tanh(x)
    }
}

#[inline(always)]
pub fn sinh(x: Float) -> Float {
    #[cfg(feature = "f32")]
    {
        libm::sinhf(x)
    }
    #[cfg(not(feature = "f32"))]
    {
        libm::sinh(x)
    }
}

#[inline(always)]
pub fn pow(x: Float, y: Float) -> Float {
    #[cfg(feature = "f32")]
    {
        libm::powf(x, y)
    }
    #[cfg(not(feature = "f32"))]
    {
        libm::pow(x, y)
    }
}

#[inline(always)]
pub fn floor(x: Float) -> Float {
    #[cfg(feature = "f32")]
    {
        libm::floorf(x)
    }
    #[cfg(not(feature = "f32"))]
    {
        libm::floor(x)
    }
}

#[inline(always)]
pub fn ceil(x: Float) -> Float {
    #[cfg(feature = "f32")]
    {
        libm::ceilf(x)
    }
    #[cfg(not(feature = "f32"))]
    {
        libm::ceil(x)
    }
}

#[inline(always)]
pub fn round(x: Float) -> Float {
    #[cfg(feature = "f32")]
    {
        libm::roundf(x)
    }
    #[cfg(not(feature = "f32"))]
    {
        libm::round(x)
    }
}

/// Log-gamma function. Returns `(lgamma, sign)`.
#[inline(always)]
pub fn lgamma(x: Float) -> (Float, i32) {
    #[cfg(feature = "f32")]
    {
        libm::lgammaf_r(x)
    }
    #[cfg(not(feature = "f32"))]
    {
        libm::lgamma_r(x)
    }
}

#[inline(always)]
pub fn log2(x: Float) -> Float {
    #[cfg(feature = "f32")]
    {
        libm::log2f(x)
    }
    #[cfg(not(feature = "f32"))]
    {
        libm::log2(x)
    }
}

#[inline(always)]
pub fn log10(x: Float) -> Float {
    #[cfg(feature = "f32")]
    {
        libm::log10f(x)
    }
    #[cfg(not(feature = "f32"))]
    {
        libm::log10(x)
    }
}

#[inline(always)]
pub fn copysign(x: Float, y: Float) -> Float {
    #[cfg(feature = "f32")]
    {
        libm::copysignf(x, y)
    }
    #[cfg(not(feature = "f32"))]
    {
        libm::copysign(x, y)
    }
}

/// Returns true if x is finite (not NaN or infinite).
#[inline(always)]
pub fn is_finite(x: Float) -> bool {
    #[cfg(feature = "f32")]
    {
        f32::is_finite(x)
    }
    #[cfg(not(feature = "f32"))]
    {
        f64::is_finite(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sqrt() {
        assert!((sqrt(4.0) - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_trig() {
        assert!(sin(0.0).abs() < 1e-6);
        assert!((cos(0.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    #[allow(clippy::approx_constant)]
    fn test_constants() {
        assert!((PI - 3.14159).abs() < 0.001);
        assert!((E - 2.71828).abs() < 0.001);
    }
}
