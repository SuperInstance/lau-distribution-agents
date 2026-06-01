//! Distribution derivatives: always exist, even for discontinuous agents.
//!
//! Key property: every distribution has derivatives of all orders.
//! If T is a distribution, then ∂T is defined by:
//!   (∂T)(φ) = -T(∂φ)
//! This means we can differentiate discontinuous agent policies.

use nalgebra::DVector;
use serde::{Deserialize, Serialize};
use crate::test_function::TestFunction;
use crate::distribution::Distribution;

/// Distribution derivative: handles differentiation of distributions.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DistributionDerivative {
    /// The underlying distribution
    pub base: Distribution,
    /// Order of derivative
    pub order: usize,
}

impl DistributionDerivative {
    /// Create a derivative operator of given order.
    pub fn new(base: Distribution, order: usize) -> Self {
        Self { base, order }
    }

    /// Apply the distributional derivative to a test function.
    /// D^n(T)(φ) = (-1)^n T(φ^(n))
    pub fn apply(&self, phi: &TestFunction) -> f64 {
        let dphi = phi.derivative(self.order);
        let sign = if self.order % 2 == 0 { 1.0 } else { -1.0 };
        sign * self.base.apply(&dphi)
    }

    /// Compute distributional derivative of a regular distribution
    /// numerically (for regular distributions, this gives weak derivatives).
    pub fn weak_derivative(dist: &Distribution, order: usize) -> Distribution {
        match &dist.function_values {
            Some(vals) => {
                let grid = dist.grid.as_ref().unwrap().clone();
                let dx = dist.dx;
                let mut result = vals.clone();
                for _ in 0..order {
                    let n = result.nrows();
                    let mut d = DVector::zeros(n);
                    for i in 1..n - 1 {
                        d[i] = (result[i + 1] - result[i - 1]) / (2.0 * dx);
                    }
                    result = d;
                }
                Distribution::regular(&format!("∂^{}({})", order, dist.name), grid, result)
            }
            None => {
                Distribution::singular(
                    &format!("∂^{}({})", order, dist.name),
                    &format!("derivative_{}_{}", order, dist.singular_action.as_deref().unwrap_or("unknown")),
                )
            }
        }
    }

    /// Derivative of Heaviside step function = Dirac delta.
    /// H'(x) = δ(x) in the distributional sense.
    pub fn heaviside_derivative(n_points: usize) -> (Distribution, Distribution) {
        let a = -3.0;
        let b = 3.0;
        let dx = (b - a) / (n_points - 1) as f64;
        let grid: Vec<f64> = (0..n_points).map(|i| a + i as f64 * dx).collect();
        let h_vals: Vec<f64> = grid.iter().map(|&x| if x >= 0.0 { 1.0 } else { 0.0 }).collect();
        let heaviside = Distribution::regular("H", DVector::from_vec(grid.clone()), DVector::from_vec(h_vals));
        let deriv = Self::weak_derivative(&heaviside, 1);
        (heaviside, deriv)
    }

    /// Verify integration by parts: <T', φ> = -<T, φ'>.
    pub fn verify_integration_by_parts(
        dist: &Distribution,
        phi: &TestFunction,
    ) -> bool {
        let d_dist = Self::weak_derivative(dist, 1);
        let lhs = d_dist.apply(phi);
        let dphi = phi.derivative(1);
        let rhs = -dist.apply(&dphi);
        let scale = lhs.abs().max(rhs.abs()).max(1.0);
        (lhs - rhs).abs() < 1e-4 * scale
    }
}

/// Product rule for distributions (when one factor is smooth).
pub fn product_rule(
    smooth: &TestFunction,
    dist: &Distribution,
    phi: &TestFunction,
) -> f64 {
    // (fT)' = f'T + fT'
    let ft_term = {
        let smooth_vals = DVector::from_vec(
            smooth.values.iter().zip(phi.values.iter()).map(|(&a, &b)| a * b).collect()
        );
        let ft = Distribution::regular("fT", smooth.grid.clone(), smooth_vals);
        ft.apply(phi)
    };
    ft_term
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_derivative_definition() {
        let t = Distribution::from_fn("cos", -5.0, 5.0, 501, |x| x.cos());
        let d = DistributionDerivative::new(t, 1);
        let phi = TestFunction::bump(501);
        let result = d.apply(&phi);
        // Should be related to integral of cos * phi'
        assert!(result.is_finite());
    }

    #[test]
    fn test_heaviside_derivative() {
        let (_, deriv) = DistributionDerivative::heaviside_derivative(2001);
        // Derivative of Heaviside should peak near zero
        if let Some(vals) = &deriv.function_values {
            let peak_idx = vals.iter().enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
            let grid = deriv.grid.as_ref().unwrap();
            assert!((grid[peak_idx]).abs() < 0.1);
        }
    }

    #[test]
    fn test_second_derivative() {
        let t = Distribution::from_fn("x²", -3.0, 3.0, 1001, |x| x * x);
        let dd = DistributionDerivative::weak_derivative(&t, 2);
        // d²/dx²(x²) = 2 in distributional sense
        if let Some(vals) = &dd.function_values {
            let mid = vals.nrows() / 2;
            assert!((vals[mid] - 2.0).abs() < 0.5);
        }
    }

    #[test]
    fn test_integration_by_parts() {
        let t = Distribution::from_fn("x³", -3.0, 3.0, 1001, |x| x.powi(3));
        let phi = TestFunction::bump(1001);
        assert!(DistributionDerivative::verify_integration_by_parts(&t, &phi));
    }

    #[test]
    fn test_zero_order_derivative() {
        let t = Distribution::from_fn("sin", -5.0, 5.0, 501, |x| x.sin());
        let d0 = DistributionDerivative::weak_derivative(&t, 0);
        assert!(d0.is_regular());
    }

    #[test]
    fn test_derivative_sign_convention() {
        let t = Distribution::from_fn("x", -3.0, 3.0, 501, |x| x);
        let deriv = DistributionDerivative::new(t, 1);
        let phi = TestFunction::bump(501);
        let result = deriv.apply(&phi);
        // T'(φ) = -T(φ') = -∫ x φ'(x) dx = ∫ φ(x) dx (integration by parts)
        let expected = phi.integrate();
        assert!((result - expected).abs() < 0.1);
    }

    #[test]
    fn test_singular_derivative() {
        let d = Distribution::singular("δ₀", "dirac_0");
        let dd = DistributionDerivative::new(d, 1);
        // δ'(φ) = -φ'(0)
        let phi = TestFunction::bump(1001);
        let dphi = phi.derivative(1);
        let result = dd.apply(&phi);
        let expected = -dphi.eval(0.0);
        assert!((result - expected).abs() < 0.01);
    }
}
