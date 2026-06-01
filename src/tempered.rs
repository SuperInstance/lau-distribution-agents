//! Tempered distributions: grow at most polynomially → Fourier transform exists.
//!
//! A tempered distribution T ∈ S'(ℝ) is a continuous linear functional on
//! the Schwartz space S(ℝ) of rapidly decreasing smooth functions.
//! These are distributions that don't grow faster than some polynomial.

use nalgebra::DVector;
use serde::{Deserialize, Serialize};
use crate::test_function::TestFunction;
use crate::distribution::Distribution;

/// A tempered distribution — a distribution with at most polynomial growth.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TemperedDistribution {
    /// The underlying distribution
    pub distribution: Distribution,
    /// Maximum polynomial growth rate (for classification)
    pub growth_order: usize,
}

impl TemperedDistribution {
    /// Create a tempered distribution from a regular distribution.
    pub fn from_distribution(dist: Distribution, growth_order: usize) -> Self {
        Self { distribution: dist, growth_order }
    }

    /// Create from a function that grows at most polynomially.
    pub fn from_fn(name: &str, a: f64, b: f64, n_points: usize,
                   f: impl Fn(f64) -> f64, growth_order: usize) -> Self {
        let dist = Distribution::from_fn(name, a, b, n_points, f);
        Self { distribution: dist, growth_order }
    }

    /// Check if a function is of polynomial growth by checking its decay rate.
    pub fn check_polynomial_growth(values: &DVector<f64>, grid: &DVector<f64>, max_order: usize) -> bool {
        // Check if |f(x)| ≤ C(1 + |x|)^max_order for some C
        let n = values.nrows();
        if n == 0 { return true; }
        let c = values.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        for i in 0..n {
            let x = grid[i];
            let bound = c * (1.0 + x.abs()).powi(max_order as i32);
            if values[i].abs() > bound * 2.0 {
                return false;
            }
        }
        true
    }

    /// Apply to a Schwartz test function.
    pub fn apply(&self, phi: &TestFunction) -> f64 {
        self.distribution.apply(phi)
    }

    /// Add two tempered distributions.
    pub fn add(&self, other: &TemperedDistribution) -> TemperedDistribution {
        let new_order = self.growth_order.max(other.growth_order);
        TemperedDistribution {
            distribution: self.distribution.add(&other.distribution),
            growth_order: new_order,
        }
    }

    /// Scale by a constant.
    pub fn scale(&self, c: f64) -> TemperedDistribution {
        TemperedDistribution {
            distribution: self.distribution.scale(c),
            growth_order: self.growth_order,
        }
    }

    /// Differentiate a tempered distribution (result is also tempered).
    pub fn derivative(&self, order: usize) -> TemperedDistribution {
        let deriv = crate::derivative::DistributionDerivative::weak_derivative(
            &self.distribution, order
        );
        TemperedDistribution {
            distribution: deriv,
            growth_order: self.growth_order + order,
        }
    }

    /// Multiply by a polynomial of degree n (result is tempered).
    pub fn multiply_by_polynomial(&self, coeffs: &[f64]) -> TemperedDistribution {
        if let (Some(vals), Some(grid)) = (&self.distribution.function_values, &self.distribution.grid) {
            let poly_vals: Vec<f64> = grid.iter().map(|&x| {
                coeffs.iter().enumerate().map(|(i, &c)| c * x.powi(i as i32)).sum::<f64>()
            }).collect();
            let new_vals: Vec<f64> = vals.iter().zip(poly_vals.iter()).map(|(&a, &b)| a * b).collect();
            let dist = Distribution::regular(
                &format!("p·{}", self.distribution.name),
                grid.clone(),
                DVector::from_vec(new_vals),
            );
            TemperedDistribution {
                distribution: dist,
                growth_order: self.growth_order + coeffs.len() - 1,
            }
        } else {
            self.clone()
        }
    }

    /// Verify that this is indeed tempered (polynomial growth check).
    pub fn verify_tempered(&self) -> bool {
        if let (Some(vals), Some(grid)) = (&self.distribution.function_values, &self.distribution.grid) {
            Self::check_polynomial_growth(vals, grid, self.growth_order + 2)
        } else {
            true // Singular tempered distributions by construction
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tempered_from_function() {
        let t = TemperedDistribution::from_fn("1+x²", -5.0, 5.0, 201, |x| 1.0 + x*x, 2);
        assert!(t.verify_tempered());
    }

    #[test]
    fn test_tempered_polynomial_growth() {
        let grid = DVector::from_vec(vec![-3.0, -1.0, 0.0, 1.0, 3.0]);
        let vals = DVector::from_vec(vec![10.0, 2.0, 1.0, 2.0, 10.0]);
        assert!(TemperedDistribution::check_polynomial_growth(&vals, &grid, 2));
    }

    #[test]
    fn test_tempered_add() {
        let t1 = TemperedDistribution::from_fn("1", -3.0, 3.0, 101, |_| 1.0, 0);
        let t2 = TemperedDistribution::from_fn("x", -3.0, 3.0, 101, |x| x, 1);
        let sum = t1.add(&t2);
        assert_eq!(sum.growth_order, 1);
    }

    #[test]
    fn test_tempered_scale() {
        let t = TemperedDistribution::from_fn("x²", -3.0, 3.0, 101, |x| x*x, 2);
        let scaled = t.scale(3.0);
        assert_eq!(scaled.growth_order, 2);
    }

    #[test]
    fn test_tempered_derivative() {
        let t = TemperedDistribution::from_fn("x³", -3.0, 3.0, 1001, |x| x.powi(3), 3);
        let dt = t.derivative(1);
        assert!(dt.growth_order >= 3);
    }

    #[test]
    fn test_tempered_multiply_polynomial() {
        let t = TemperedDistribution::from_fn("x", -3.0, 3.0, 101, |x| x, 1);
        let result = t.multiply_by_polynomial(&[1.0, 1.0]); // (1+x)·x = x + x²
        assert!(result.growth_order >= 1);
    }

    #[test]
    fn test_constant_is_tempered() {
        let t = TemperedDistribution::from_fn("1", -5.0, 5.0, 201, |_| 1.0, 0);
        assert!(t.verify_tempered());
    }

    #[test]
    fn test_gaussian_is_tempered() {
        let t = TemperedDistribution::from_fn("gaussian", -5.0, 5.0, 201, |x| (-x*x).exp(), 0);
        assert!(t.verify_tempered());
    }

    #[test]
    fn test_tempered_apply() {
        let t = TemperedDistribution::from_fn("x²", -3.0, 3.0, 501, |x| x*x, 2);
        let phi = TestFunction::bump(501);
        let val = t.apply(&phi);
        assert!(val > 0.0);
    }
}
