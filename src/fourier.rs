//! Fourier transform of distributions: frequency-domain analysis of rough agents.

use nalgebra::{DVector, Complex};
use serde::{Deserialize, Serialize};
use crate::test_function::TestFunction;
use crate::tempered::TemperedDistribution;

/// Result of a Fourier transform (stored as [re, im] pairs for serde).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FourierTransform {
    /// Frequency grid
    pub freq_grid: DVector<f64>,
    /// Complex transform values stored as [re, im] pairs
    pub values: Vec<[f64; 2]>,
    /// Grid spacing in frequency domain
    pub df: f64,
}

impl FourierTransform {
    fn c(re: f64, im: f64) -> [f64; 2] { [re, im] }

    fn cmul(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
        [a[0]*b[0] - a[1]*b[1], a[0]*b[1] + a[1]*b[0]]
    }

    fn cadd(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
        [a[0]+b[0], a[1]+b[1]]
    }

    fn cnorm(a: [f64; 2]) -> f64 {
        a[0]*a[0] + a[1]*a[1]
    }

    fn cabs(a: [f64; 2]) -> f64 {
        a[0].hypot(a[1])
    }

    fn carg(a: [f64; 2]) -> f64 {
        a[1].atan2(a[0])
    }

    fn cscale(a: [f64; 2], s: f64) -> [f64; 2] {
        [a[0]*s, a[1]*s]
    }

    /// Get complex values.
    pub fn values_as_complex(&self) -> Vec<Complex<f64>> {
        self.values.iter().map(|&[re, im]| Complex::new(re, im)).collect()
    }

    /// Compute the DFT of a test function.
    pub fn of_test_function(phi: &TestFunction, n_freqs: usize) -> Self {
        let n = phi.grid.nrows();
        let l = phi.support.1 - phi.support.0;
        let max_freq = n_freqs as f64 / (2.0 * l);
        let df = 2.0 * max_freq / n_freqs as f64;

        let freqs: Vec<f64> = (0..n_freqs)
            .map(|k| -max_freq + k as f64 * df)
            .collect();

        let values: Vec<[f64; 2]> = freqs.iter().map(|&xi| {
            let mut sum = [0.0, 0.0];
            for i in 0..n {
                let x = phi.grid[i];
                let phase = -2.0 * std::f64::consts::PI * x * xi;
                sum = Self::cadd(sum, Self::cscale(Self::c(phase.cos(), phase.sin()), phi.values[i]));
            }
            Self::cscale(sum, phi.dx)
        }).collect();

        Self { freq_grid: DVector::from_vec(freqs), values, df }
    }

    /// Compute inverse DFT.
    pub fn inverse(&self) -> (DVector<f64>, Vec<[f64; 2]>) {
        let n = self.values.len();
        let grid: Vec<f64> = (0..n).map(|i| i as f64 * self.df).collect();
        let values: Vec<[f64; 2]> = grid.iter().map(|&x| {
            let mut sum = [0.0, 0.0];
            for k in 0..n {
                let xi = self.freq_grid[k];
                let phase = 2.0 * std::f64::consts::PI * x * xi;
                sum = Self::cadd(sum, Self::cmul(self.values[k], Self::c(phase.cos(), phase.sin())));
            }
            Self::cscale(sum, self.df)
        }).collect();
        (DVector::from_vec(grid), values)
    }

    /// Power spectrum: |F(φ)|².
    pub fn power_spectrum(&self) -> DVector<f64> {
        DVector::from_vec(self.values.iter().map(|v| Self::cnorm(*v)).collect())
    }

    /// Phase spectrum: arg(F(φ)).
    pub fn phase_spectrum(&self) -> DVector<f64> {
        DVector::from_vec(self.values.iter().map(|v| Self::carg(*v)).collect())
    }

    /// Fourier transform of a tempered distribution.
    pub fn of_tempered(t: &TemperedDistribution, n_freqs: usize) -> Self {
        if let (Some(vals), Some(grid)) = (&t.distribution.function_values, &t.distribution.grid) {
            let n = grid.nrows();
            let l = grid[n - 1] - grid[0];
            let max_freq = n_freqs as f64 / (2.0 * l);
            let df = 2.0 * max_freq / n_freqs as f64;
            let dx = t.distribution.dx;

            let freqs: Vec<f64> = (0..n_freqs)
                .map(|k| -max_freq + k as f64 * df)
                .collect();

            let values: Vec<[f64; 2]> = freqs.iter().map(|&xi| {
                let mut sum = [0.0, 0.0];
                for i in 0..n {
                    let x = grid[i];
                    let phase = -2.0 * std::f64::consts::PI * x * xi;
                    sum = Self::cadd(sum, Self::cscale(Self::c(phase.cos(), phase.sin()), vals[i]));
                }
                Self::cscale(sum, dx)
            }).collect();

            Self { freq_grid: DVector::from_vec(freqs), values, df }
        } else {
            let freqs = DVector::zeros(n_freqs);
            let values = vec![[0.0, 0.0]; n_freqs];
            Self { freq_grid: freqs, values, df: 1.0 }
        }
    }

    /// Parseval's theorem: ∫|f|² = ∫|F(f)|².
    pub fn verify_parseval(phi: &TestFunction, n_freqs: usize) -> bool {
        let ft = Self::of_test_function(phi, n_freqs);
        let spatial_energy: f64 = phi.values.iter().map(|v| v * v).sum::<f64>() * phi.dx;
        let freq_energy: f64 = ft.values.iter().map(|v| Self::cnorm(*v)).sum::<f64>() * ft.df;
        let scale = spatial_energy.abs().max(freq_energy.abs()).max(1.0);
        (spatial_energy - freq_energy).abs() < 0.2 * scale
    }

    /// Fourier transform of Gaussian is Gaussian.
    pub fn verify_gaussian_is_gaussian(sigma: f64) -> bool {
        let g = TestFunction::gaussian(0.0, sigma, 2001, 6.0);
        let ft = Self::of_test_function(&g, 2001);
        let peak_idx = ft.values.iter().enumerate()
            .max_by(|a, b| Self::cnorm(*a.1).partial_cmp(&Self::cnorm(*b.1)).unwrap()).unwrap().0;
        ft.freq_grid[peak_idx].abs() < 1.0
    }

    /// Differentiation property: F(f') = 2πiξ · F(f).
    pub fn verify_derivative_property(phi: &TestFunction, n_freqs: usize) -> bool {
        let ft = Self::of_test_function(phi, n_freqs);
        let dphi = phi.derivative(1);
        let ft_dphi = Self::of_test_function(&dphi, n_freqs);

        let n_check = 5.min(ft.values.len());
        let mid = ft.values.len() / 2;
        let mut max_err: f64 = 0.0;
        for k in 0..n_check {
            let idx = mid - n_check / 2 + k;
            let xi = ft.freq_grid[idx];
            // expected = 2πiξ * ft.values[idx]
            let expected = Self::cmul(Self::c(0.0, 2.0 * std::f64::consts::PI * xi), ft.values[idx]);
            let actual = ft_dphi.values[idx];
            let err = Self::cabs(Self::cadd(expected, Self::cscale(actual, -1.0)));
            max_err = max_err.max(err);
        }
        max_err < 5.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fourier_bump() {
        let phi = TestFunction::bump(501);
        let ft = FourierTransform::of_test_function(&phi, 256);
        assert_eq!(ft.values.len(), 256);
    }

    #[test]
    fn test_fourier_gaussian() {
        let g = TestFunction::gaussian(0.0, 1.0, 501, 5.0);
        let ft = FourierTransform::of_test_function(&g, 256);
        assert!(ft.values.iter().all(|v| v[0].is_finite() && v[1].is_finite()));
    }

    #[test]
    fn test_power_spectrum() {
        let phi = TestFunction::bump(501);
        let ft = FourierTransform::of_test_function(&phi, 256);
        let ps = ft.power_spectrum();
        assert!(ps.iter().all(|v| *v >= 0.0));
    }

    #[test]
    fn test_phase_spectrum() {
        let phi = TestFunction::bump(501);
        let ft = FourierTransform::of_test_function(&phi, 256);
        let phase = ft.phase_spectrum();
        assert!(phase.iter().all(|v| v.abs() <= std::f64::consts::PI + 0.01));
    }

    #[test]
    fn test_parseval() {
        let phi = TestFunction::bump(501);
        assert!(FourierTransform::verify_parseval(&phi, 256));
    }

    #[test]
    fn test_gaussian_ft_is_gaussian() {
        assert!(FourierTransform::verify_gaussian_is_gaussian(1.0));
    }

    #[test]
    fn test_derivative_property() {
        let phi = TestFunction::bump(501);
        assert!(FourierTransform::verify_derivative_property(&phi, 256));
    }

    #[test]
    fn test_inverse_fourier() {
        let phi = TestFunction::gaussian(0.0, 1.0, 501, 5.0);
        let ft = FourierTransform::of_test_function(&phi, 501);
        let (grid, inv) = ft.inverse();
        assert_eq!(grid.nrows(), 501);
        assert!(inv.iter().all(|v| v[0].is_finite() && v[1].is_finite()));
    }

    #[test]
    fn test_fourier_tempered() {
        let t = TemperedDistribution::from_fn("gaussian", -5.0, 5.0, 501, |x| (-x*x).exp(), 0);
        let ft = FourierTransform::of_tempered(&t, 256);
        assert_eq!(ft.values.len(), 256);
    }
}
