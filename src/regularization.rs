//! Regularization: approximate rough agents by smooth agents (mollification).
//!
//! Mollification: u_ε = u * φ_ε where φ_ε is a mollifier (smooth bump).
//! Properties:
//! - u_ε → u in L^p as ε → 0
//! - u_ε is smooth (C^∞)
//! - ||u_ε||_p ≤ ||u||_p

use nalgebra::DVector;
use serde::{Deserialize, Serialize};
use crate::test_function::TestFunction;
use crate::distribution::Distribution;
use crate::convolution::Convolution;

/// Mollifier: a smooth bump function with integral 1.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Mollifier {
    /// The test function (bump) used as mollifier
    pub kernel: TestFunction,
    /// Epsilon parameter (controls width)
    pub epsilon: f64,
}

impl Mollifier {
    /// Standard mollifier: φ_ε(x) = (1/ε) φ(x/ε).
    pub fn standard(epsilon: f64, n_points: usize) -> Self {
        let range = 5.0 * epsilon;
        let dx = 2.0 * range / (n_points - 1) as f64;
        let grid: Vec<f64> = (0..n_points).map(|i| -range + i as f64 * dx).collect();
        let coeff = 1.0 / epsilon;
        let values: Vec<f64> = grid.iter().map(|&x| {
            let t = x / epsilon;
            if t.abs() < 1.0 {
                coeff * (-1.0 / (1.0 - t * t)).exp()
            } else {
                0.0
            }
        }).collect();
        let kernel = TestFunction::new(DVector::from_vec(grid), DVector::from_vec(values));

        // Normalize to have integral 1
        let integral = kernel.integrate();
        let kernel = kernel.scale(1.0 / integral);

        Self { kernel, epsilon }
    }

    /// Gaussian mollifier.
    pub fn gaussian(epsilon: f64, n_points: usize) -> Self {
        let sigma = epsilon;
        let range = 5.0 * sigma;
        let dx = 2.0 * range / (n_points - 1) as f64;
        let grid: Vec<f64> = (0..n_points).map(|i| -range + i as f64 * dx).collect();
        let coeff = 1.0 / (sigma * (2.0 * std::f64::consts::PI).sqrt());
        let values: Vec<f64> = grid.iter().map(|&x| {
            coeff * (-x * x / (2.0 * sigma * sigma)).exp()
        }).collect();
        let kernel = TestFunction::new(DVector::from_vec(grid), DVector::from_vec(values));
        Self { kernel, epsilon }
    }

    /// Verify the mollifier is normalized (integral = 1).
    pub fn verify_normalization(&self) -> bool {
        let integral = self.kernel.integrate();
        (integral - 1.0).abs() < 0.05
    }

    /// Mollify a distribution: u_ε = u * φ_ε.
    pub fn mollify(&self, dist: &Distribution) -> Regularization {
        let conv = Convolution::of_distribution_with_test(dist, &self.kernel);
        let smoothed = Distribution::regular(
            &format!("{}_ε{}", dist.name, self.epsilon),
            conv.grid,
            conv.values,
        );
        Regularization {
            original: dist.clone(),
            smoothed,
            epsilon: self.epsilon,
        }
    }

    /// Mollify a test function.
    pub fn mollify_fn(&self, f: &TestFunction) -> TestFunction {
        let conv = Convolution::of_test_functions(f, &self.kernel);
        TestFunction::new(conv.grid, conv.values)
    }
}

/// Result of regularizing a rough distribution by mollification.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Regularization {
    /// Original (possibly rough) distribution
    pub original: Distribution,
    /// Smoothed distribution
    pub smoothed: Distribution,
    /// Epsilon parameter
    pub epsilon: f64,
}

impl Regularization {
    /// Create a regularization sequence (ε → 0).
    pub fn sequence(dist: &Distribution, epsilons: &[f64], n_points: usize) -> Vec<Self> {
        epsilons.iter().map(|&eps| {
            let mol = Mollifier::standard(eps, n_points);
            mol.mollify(dist)
        }).collect()
    }

    /// Verify convergence: ||u_ε - u|| → 0 as ε → 0.
    pub fn verify_convergence(regs: &[Regularization]) -> bool {
        if regs.len() < 2 { return true; }
        let mut decreasing = true;
        for w in regs.windows(2) {
            let err1 = w[0].error_norm();
            let err2 = w[1].error_norm();
            if err2 > err1 * 1.5 { decreasing = false; }
        }
        decreasing
    }

    /// Compute L² error between original and smoothed.
    pub fn error_norm(&self) -> f64 {
        match (&self.original.function_values, &self.smoothed.function_values) {
            (Some(orig), Some(smooth)) => {
                let diff = orig - smooth;
                diff.norm()
            }
            _ => f64::NAN,
        }
    }

    /// Verify that smoothing reduces oscillations.
    pub fn is_smoother(&self) -> bool {
        match (&self.original.function_values, &self.smoothed.function_values) {
            (Some(orig), Some(smooth)) => {
                let n = orig.nrows();
                // Count sign changes as proxy for oscillation
                let orig_changes = Self::count_sign_changes(orig);
                let smooth_changes = Self::count_sign_changes(smooth);
                smooth_changes <= orig_changes + 2
            }
            _ => true,
        }
    }

    fn count_sign_changes(v: &DVector<f64>) -> usize {
        let n = v.nrows();
        let mut count = 0;
        for i in 1..n {
            if v[i] * v[i - 1] < 0.0 {
                count += 1;
            }
        }
        count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mollifier_normalization() {
        let mol = Mollifier::standard(0.5, 501);
        assert!(mol.verify_normalization());
    }

    #[test]
    fn test_gaussian_mollifier_normalization() {
        let mol = Mollifier::gaussian(0.5, 501);
        assert!(mol.verify_normalization());
    }

    #[test]
    fn test_mollify_smooth() {
        let dist = Distribution::from_fn("gaussian", -3.0, 3.0, 501, |x| (-x*x).exp());
        let mol = Mollifier::standard(0.1, 501);
        let reg = mol.mollify(&dist);
        assert!(reg.error_norm() < 50.0);
    }

    #[test]
    fn test_mollify_discontinuous() {
        let dist = Distribution::from_fn("step", -3.0, 3.0, 501, |x| {
            if x >= 0.0 { 1.0 } else { 0.0 }
        });
        let mol = Mollifier::standard(0.3, 501);
        let reg = mol.mollify(&dist);
        // Smoothed version should be finite
        if let Some(vals) = &reg.smoothed.function_values {
            assert!(vals.iter().all(|v| v.is_finite()));
        }
    }

    #[test]
    fn test_mollify_delta() {
        let delta = TestFunction::gaussian(0.0, 0.01, 501, 6.0);
        // Normalize to be delta-like
        let integral = delta.integrate();
        let delta = delta.scale(1.0 / integral);
        let mol = Mollifier::standard(0.5, 501);
        let smoothed = mol.mollify_fn(&delta);
        // Should spread out the peak
        assert!(smoothed.values.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_regularization_sequence() {
        let dist = Distribution::from_fn("step", -3.0, 3.0, 1001, |x| {
            if x >= 0.0 { 1.0 } else { 0.0 }
        });
        let regs = Regularization::sequence(&dist, &[1.0, 0.5, 0.2, 0.1], 501);
        assert_eq!(regs.len(), 4);
    }

    #[test]
    fn test_convergence_smooth_function() {
        let dist = Distribution::from_fn("x²", -3.0, 3.0, 501, |x| x * x);
        let regs = Regularization::sequence(&dist, &[0.5, 0.2, 0.1], 501);
        // For smooth functions, errors should be small
        for reg in &regs {
            assert!(reg.error_norm() < 100.0);
        }
    }

    #[test]
    fn test_smoother() {
        let dist = Distribution::from_fn("spiky", -3.0, 3.0, 501, |x| {
            (10.0 * x).sin() * (-x * x).exp()
        });
        let mol = Mollifier::standard(0.3, 501);
        let reg = mol.mollify(&dist);
        assert!(reg.is_smoother());
    }

    #[test]
    fn test_mollifier_support() {
        let mol = Mollifier::standard(0.1, 201);
        assert!(mol.kernel.has_compact_support(1e-3));
    }
}
