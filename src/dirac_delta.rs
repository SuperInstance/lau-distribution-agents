//! Dirac delta: δ(x-a) = "point observation" at agent state a.
//!
//! The Dirac delta is the fundamental singular distribution.
//! δ_a(φ) = φ(a) — it evaluates the test function at a point.
//! For agents, this represents an instantaneous observation or impulse
//! at a specific state.

use nalgebra::DVector;
use serde::{Deserialize, Serialize};
use crate::test_function::TestFunction;
use crate::distribution::Distribution;

/// Dirac delta distribution δ(x - a).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DiracDelta {
    /// Center point a
    pub center: f64,
    /// Label
    pub label: String,
}

impl DiracDelta {
    /// Create δ(x - a) centered at `a`.
    pub fn at(center: f64) -> Self {
        Self {
            center,
            label: format!("δ(x-{})", center),
        }
    }

    /// Create δ(x) = δ(x - 0).
    pub fn at_origin() -> Self {
        Self::at(0.0)
    }

    /// Apply δ_a to a test function: δ_a(φ) = φ(a).
    pub fn apply(&self, phi: &TestFunction) -> f64 {
        phi.eval(self.center)
    }

    /// Finite difference approximation: δ_ε(x) = (1/(ε√π)) exp(-x²/ε²).
    pub fn approximate(&self, epsilon: f64, n_points: usize) -> TestFunction {
        let range = 10.0 * epsilon;
        let a = self.center - range;
        let b = self.center + range;
        let dx = (b - a) / (n_points - 1) as f64;
        let grid: Vec<f64> = (0..n_points).map(|i| a + i as f64 * dx).collect();
        let coeff = 1.0 / (epsilon * std::f64::consts::PI.sqrt());
        let values: Vec<f64> = grid.iter().map(|&x| {
            coeff * (-(x - self.center).powi(2) / (epsilon * epsilon)).exp()
        }).collect();
        TestFunction::new(DVector::from_vec(grid), DVector::from_vec(values))
    }

    /// Convert to a Distribution.
    pub fn as_distribution(&self) -> Distribution {
        Distribution::singular(&self.label.clone(), "dirac_0")
    }

    /// Sifting property: ∫ δ(x-a) f(x) dx = f(a).
    pub fn sift(&self, f: impl Fn(f64) -> f64) -> f64 {
        f(self.center)
    }

    /// Scaling property: δ(ax) = (1/|a|) δ(x).
    pub fn scaled(&self, a: f64) -> DiracDelta {
        let scaled_center = self.center / a;
        let mut d = DiracDelta::at(scaled_center);
        d.label = format!("(1/|{}|)·δ(x-{})", a, scaled_center);
        d
    }

    /// Shifted delta: δ(x - (a + shift)).
    pub fn shifted(&self, shift: f64) -> DiracDelta {
        DiracDelta::at(self.center + shift)
    }

    /// Approximate integral: ∫ δ_ε(x-a) dx ≈ 1.
    pub fn verify_normalization(&self, epsilon: f64, n_points: usize) -> f64 {
        let approx = self.approximate(epsilon, n_points);
        approx.integrate()
    }
}

/// A linear combination of Dirac deltas: Σ cᵢ δ(x - aᵢ).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DiracComb {
    pub centers: Vec<f64>,
    pub weights: Vec<f64>,
}

impl DiracComb {
    /// Create a Dirac comb.
    pub fn new(centers: Vec<f64>, weights: Vec<f64>) -> Self {
        assert_eq!(centers.len(), weights.len());
        Self { centers, weights }
    }

    /// Shah/comb function: Σ δ(x - nT) for integer n.
    pub fn shah(period: f64, n_terms: i32) -> Self {
        let centers: Vec<f64> = (-n_terms..=n_terms).map(|n| n as f64 * period).collect();
        let weights = vec![1.0; centers.len()];
        Self::new(centers, weights)
    }

    /// Apply the comb to a test function.
    pub fn apply(&self, phi: &TestFunction) -> f64 {
        self.centers.iter()
            .zip(self.weights.iter())
            .map(|(&c, &w)| w * phi.eval(c))
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dirac_at_origin() {
        let delta = DiracDelta::at_origin();
        assert!((delta.center).abs() < 1e-10);
    }

    #[test]
    fn test_dirac_sifting() {
        let delta = DiracDelta::at(2.0);
        let val = delta.sift(|x| x * x);
        assert!((val - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_dirac_apply_bump() {
        let delta = DiracDelta::at_origin();
        let phi = TestFunction::bump(1001);
        let val = delta.apply(&phi);
        assert!((val - (-1.0_f64).exp()).abs() < 0.01);
    }

    #[test]
    fn test_dirac_approximate_normalization() {
        let delta = DiracDelta::at_origin();
        let integral = delta.verify_normalization(0.1, 5001);
        assert!((integral - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_dirac_approximate_peak() {
        let delta = DiracDelta::at(1.0);
        let approx = delta.approximate(0.05, 1001);
        let peak_idx = approx.values.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
        assert!((approx.grid[peak_idx] - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_dirac_shifted() {
        let delta = DiracDelta::at(1.0);
        let shifted = delta.shifted(2.0);
        assert!((shifted.center - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_dirac_scaled() {
        let delta = DiracDelta::at(2.0);
        let scaled = delta.scaled(2.0);
        assert!((scaled.center - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_dirac_comb() {
        let comb = DiracComb::new(vec![0.0, 1.0], vec![1.0, 2.0]);
        let phi = TestFunction::bump(1001);
        let result = comb.apply(&phi);
        assert!(result > 0.0);
    }

    #[test]
    fn test_shah_comb() {
        let shah = DiracComb::shah(1.0, 3);
        assert_eq!(shah.centers.len(), 7);
    }

    #[test]
    fn test_dirac_as_distribution() {
        let delta = DiracDelta::at_origin();
        let dist = delta.as_distribution();
        assert!(!dist.is_regular());
    }

    #[test]
    fn test_dirac_convergence() {
        let delta = DiracDelta::at(0.5);
        let phi = TestFunction::bump(2001);
        let exact = delta.apply(&phi);
        // Approximation should converge as ε→0
        for eps in [1.0, 0.5, 0.1] {
            let approx = delta.approximate(eps, 2001);
            let approx_val = approx.inner_product(&phi);
            if eps < 0.2 {
                assert!((approx_val - exact).abs() < 0.5,
                    "eps={}: approx={} exact={}", eps, approx_val, exact);
            }
        }
    }
}
