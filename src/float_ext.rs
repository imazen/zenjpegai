//! Standard float operations for targets without a standard library.
//! Kept private and enabled only without std; std builds retain their methods.

pub(crate) trait FloatExt: Sized {
    fn floor(self) -> Self;
    fn sqrt(self) -> Self;
    fn round_ties_even(self) -> Self;
    fn mul_add(self, b: Self, c: Self) -> Self;
}
macro_rules! impl_float {
    ($t:ty, $floor:path, $sqrt:path, $round:path, $fma:path) => {
        impl FloatExt for $t {
            #[inline]
            fn floor(self) -> Self {
                $floor(self)
            }
            #[inline]
            fn sqrt(self) -> Self {
                $sqrt(self)
            }
            #[inline]
            fn round_ties_even(self) -> Self {
                $round(self)
            }
            #[inline]
            fn mul_add(self, b: Self, c: Self) -> Self {
                $fma(self, b, c)
            }
        }
    };
}
impl_float!(f32, libm::floorf, libm::sqrtf, libm::roundevenf, libm::fmaf);
impl_float!(f64, libm::floor, libm::sqrt, libm::roundeven, libm::fma);
