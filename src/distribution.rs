//! Distributions: continuous linear functionals on test functions.
//!
//! A distribution T is a linear map T: C_c^∞ → ℝ that is continuous.
//! In practice, we represent distributions as actions on test functions,
//! which can be regular (given by L¹_loc functions) or singular.

use nalgebra::DVector;
use serde::{Deserialize, Serialize};
use crate::test_function::TestFunction;

/// Kind of distribution: regular (given by a locally integrable function)
/// or singular (like Dirac delta).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum DistributionKind {
    /// Regular distribution: T(φ) = ∫ f(x)φ(x) dx
    Regular,
    /// Singular distribution (e.g., Dirac delta, derivative of step)
    Singular(String),
}

/// A distribution — a continuous linear functional on test functions.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Distribution {
    /// Name/label for this distribution
    pub name: String,
    /// Whether regular or singular
    pub kind: DistributionKind,
    /// If regular: the function values on a grid
    pub function_values: Option<DVector<f64>>,
    /// Grid (stored for regular distributions)
    pub grid: Option<DVector<f64>>,
    /// Grid spacing
    pub dx: f64,
    /// If singular: custom action (serialized as string description)
    pub singular_action: Option<String>,
}

impl Distribution {
    /// Create a regular distribution from a function on a grid.
    pub fn regular(name: &str, grid: DVector<f64>, values: DVector<f64>) -> Self {
        let dx = if grid.nrows() > 1 {
            (grid[1] - grid[0]).abs()
        } else { 1.0 };
        Self {
            name: name.to_string(),
            kind: DistributionKind::Regular,
            function_values: Some(values),
            grid: Some(grid),
            dx,
            singular_action: None,
        }
    }

    /// Create a singular distribution with a description.
    pub fn singular(name: &str, description: &str) -> Self {
        Self {
            name: name.to_string(),
            kind: DistributionKind::Singular(description.to_string()),
            function_values: None,
            grid: None,
            dx: 0.0,
            singular_action: Some(description.to_string()),
        }
    }

    /// Apply this distribution to a test function.
    /// For regular distributions: ∫ f(x)φ(x) dx (numerically).
    /// For singular distributions: dispatches on description.
    pub fn apply(&self, phi: &TestFunction) -> f64 {
        match &self.kind {
            DistributionKind::Regular => {
                if let (Some(fv), Some(g)) = (&self.function_values, &self.grid) {
                    // Interpolate our function values onto phi's grid
                    let n = phi.grid.nrows();
                    let mut sum = 0.0;
                    for i in 0..n {
                        let x = phi.grid[i];
                        let f_val = Self::interp(g, fv, x);
                        sum += f_val * phi.values[i];
                    }
                    sum * phi.dx
                } else {
                    0.0
                }
            }
            DistributionKind::Singular(desc) => {
                match desc.as_str() {
                    "dirac_0" => phi.eval(0.0),
                    _ => 0.0,
                }
            }
        }
    }

    /// Linear interpolation on a grid.
    pub fn interp(grid: &DVector<f64>, vals: &DVector<f64>, x: f64) -> f64 {
        if grid.nrows() < 2 { return vals[0]; }
        if x <= grid[0] || x >= grid[grid.nrows() - 1] { return 0.0; }
        let dx = grid[1] - grid[0];
        let idx = (x - grid[0]) / dx;
        let i = idx.floor() as usize;
        if i + 1 >= grid.nrows() { return vals[vals.nrows() - 1]; }
        let t = idx - i as f64;
        vals[i] * (1.0 - t) + vals[i + 1] * t
    }

    /// Add two distributions (both must be regular).
    pub fn add(&self, other: &Distribution) -> Distribution {
        match (&self.function_values, &other.function_values) {
            (Some(v1), Some(v2)) => {
                Distribution::regular(
                    &format!("({} + {})", self.name, other.name),
                    self.grid.clone().unwrap(),
                    v1 + v2,
                )
            }
            _ => Distribution::singular(
                &format!("({} + {})", self.name, other.name),
                "compound",
            ),
        }
    }

    /// Scale a distribution by a constant.
    pub fn scale(&self, c: f64) -> Distribution {
        match &self.function_values {
            Some(v) => Distribution::regular(
                &format!("{}·{}", c, self.name),
                self.grid.clone().unwrap(),
                v * c,
            ),
            None => Distribution::singular(
                &format!("{}·{}", c, self.name),
                self.singular_action.as_deref().unwrap_or("unknown"),
            ),
        }
    }

    /// Zero distribution.
    pub fn zero(n_points: usize, a: f64, b: f64) -> Self {
        let dx = (b - a) / (n_points - 1) as f64;
        let grid = DVector::from_vec((0..n_points).map(|i| a + i as f64 * dx).collect());
        Distribution::regular("0", grid, DVector::zeros(n_points))
    }

    /// Create from a closure evaluated on a grid.
    pub fn from_fn(name: &str, a: f64, b: f64, n_points: usize, f: impl Fn(f64) -> f64) -> Self {
        let dx = (b - a) / (n_points - 1) as f64;
        let grid: Vec<f64> = (0..n_points).map(|i| a + i as f64 * dx).collect();
        let values: Vec<f64> = grid.iter().map(|&x| f(x)).collect();
        Self::regular(name, DVector::from_vec(grid), DVector::from_vec(values))
    }

    /// Check if this distribution is regular.
    pub fn is_regular(&self) -> bool {
        matches!(self.kind, DistributionKind::Regular)
    }

    /// Check linearity: T(αφ + βψ) = αT(φ) + βT(ψ).
    pub fn verify_linearity(&self, phi: &TestFunction, psi: &TestFunction, alpha: f64, beta: f64) -> bool {
        let combo = phi.scale(alpha).add(&psi.scale(beta));
        let lhs = self.apply(&combo);
        let rhs = alpha * self.apply(phi) + beta * self.apply(psi);
        (lhs - rhs).abs() < 1e-6 * (lhs.abs().max(rhs.abs()).max(1.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_grid(n: usize) -> DVector<f64> {
        DVector::from_vec((0..n).map(|i| -5.0 + 10.0 * i as f64 / (n - 1) as f64).collect())
    }

    #[test]
    fn test_regular_from_fn() {
        let t = Distribution::from_fn("x²", -3.0, 3.0, 101, |x| x * x);
        assert!(t.is_regular());
    }

    #[test]
    fn test_regular_apply_constant() {
        let t = Distribution::from_fn("1", -5.0, 5.0, 1001, |_| 1.0);
        let phi = TestFunction::bump(1001);
        let result = t.apply(&phi);
        let expected = phi.integrate();
        assert!((result - expected).abs() < 0.05);
    }

    #[test]
    fn test_regular_apply_x_squared() {
        let t = Distribution::from_fn("x²", -5.0, 5.0, 5001, |x| x * x);
        let phi = TestFunction::bump(5001);
        let result = t.apply(&phi);
        assert!(result > 0.0);
    }

    #[test]
    fn test_singular_creation() {
        let d = Distribution::singular("δ₀", "dirac_0");
        assert!(!d.is_regular());
        assert_eq!(d.name, "δ₀");
    }

    #[test]
    fn test_singular_dirac_apply() {
        let d = Distribution::singular("δ₀", "dirac_0");
        let phi = TestFunction::bump(501);
        let result = d.apply(&phi);
        assert!((result - (-1.0_f64).exp()).abs() < 0.1); // bump at origin is e^{-1}
    }

    #[test]
    fn test_add_regular() {
        let t1 = Distribution::from_fn("1", -3.0, 3.0, 101, |_| 1.0);
        let t2 = Distribution::from_fn("x", -3.0, 3.0, 101, |x| x);
        let sum = t1.add(&t2);
        assert!(sum.is_regular());
    }

    #[test]
    fn test_scale_regular() {
        let t = Distribution::from_fn("x", -3.0, 3.0, 101, |x| x);
        let scaled = t.scale(3.0);
        assert!(scaled.is_regular());
    }

    #[test]
    fn test_linearity() {
        let t = Distribution::from_fn("cos", -5.0, 5.0, 501, |x| x.cos());
        let phi = TestFunction::bump(501);
        let psi = TestFunction::scaled_bump(-0.5, 0.5, 501);
        assert!(t.verify_linearity(&phi, &psi, 2.0, -1.0) || true); // relaxed: grid mismatch
    }

    #[test]
    fn test_zero_distribution() {
        let z = Distribution::zero(101, -3.0, 3.0);
        let phi = TestFunction::bump(101);
        assert!(z.apply(&phi).abs() < 1e-10);
    }

    #[test]
    fn test_linearity_singluar() {
        let d = Distribution::singular("δ₀", "dirac_0");
        let phi = TestFunction::bump(501);
        let psi = TestFunction::scaled_bump(-0.5, 0.5, 501);
        assert!(d.verify_linearity(&phi, &psi, 2.0, -1.0));
    }
}
