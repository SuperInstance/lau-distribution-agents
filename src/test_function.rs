//! Test functions: smooth compactly-supported functions (the "probes").
//!
//! In Schwartz distribution theory, test functions are infinitely differentiable
//! functions with compact support, denoted C_c^∞(ℝ). They serve as "probes" that
//! distributions act upon.

use nalgebra::DVector;
use serde::{Deserialize, Serialize};

/// A smooth (infinitely differentiable) compactly-supported test function.
///
/// Represented by evaluating the function at discrete grid points.
/// The canonical bump function φ(x) = exp(-1/(1-x²)) for |x|<1, 0 otherwise
/// is the prototype.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TestFunction {
    /// Grid points where the function is evaluated
    pub grid: DVector<f64>,
    /// Function values at grid points
    pub values: DVector<f64>,
    /// Support interval [a, b]
    pub support: (f64, f64),
    /// Grid spacing
    pub dx: f64,
}

impl TestFunction {
    /// Create a new test function from grid and values.
    pub fn new(grid: DVector<f64>, values: DVector<f64>) -> Self {
        assert_eq!(grid.nrows(), values.nrows(), "Grid and values must have same length");
        let dx = if grid.nrows() > 1 {
            (grid[1] - grid[0]).abs()
        } else {
            1.0
        };
        let a = grid[0];
        let b = grid[grid.nrows() - 1];
        Self { grid, values, support: (a, b), dx }
    }

    /// Canonical bump function: φ(x) = exp(-1/(1-x²)) for |x|<1, 0 otherwise.
    pub fn bump(n_points: usize) -> Self {
        let a = -1.0;
        let b = 1.0;
        let dx = (b - a) / (n_points - 1) as f64;
        let grid: Vec<f64> = (0..n_points).map(|i| a + i as f64 * dx).collect();
        let values: Vec<f64> = grid.iter().map(|&x| {
            if x.abs() < 1.0 {
                let inner = 1.0 - x * x;
                (-1.0 / inner).exp()
            } else {
                0.0
            }
        }).collect();
        Self::new(DVector::from_vec(grid), DVector::from_vec(values))
    }

    /// Scaled bump function with support [a, b].
    pub fn scaled_bump(a: f64, b: f64, n_points: usize) -> Self {
        let dx = (b - a) / (n_points - 1) as f64;
        let center = (a + b) / 2.0;
        let half_width = (b - a) / 2.0;
        let grid: Vec<f64> = (0..n_points).map(|i| a + i as f64 * dx).collect();
        let values: Vec<f64> = grid.iter().map(|&x| {
            let t = (x - center) / half_width;
            if t.abs() < 1.0 {
                let inner = 1.0 - t * t;
                (-1.0 / inner).exp()
            } else {
                0.0
            }
        }).collect();
        Self::new(DVector::from_vec(grid), DVector::from_vec(values))
    }

    /// Gaussian test function (not compactly supported but exponentially decaying).
    pub fn gaussian(center: f64, sigma: f64, n_points: usize, range: f64) -> Self {
        let a = center - range * sigma;
        let b = center + range * sigma;
        let dx = (b - a) / (n_points - 1) as f64;
        let grid: Vec<f64> = (0..n_points).map(|i| a + i as f64 * dx).collect();
        let values: Vec<f64> = grid.iter().map(|&x| {
            (-(x - center).powi(2) / (2.0 * sigma * sigma)).exp()
        }).collect();
        Self::new(DVector::from_vec(grid), DVector::from_vec(values))
    }

    /// Compute the L² inner product with another test function.
    pub fn inner_product(&self, other: &TestFunction) -> f64 {
        assert_eq!(self.grid.nrows(), other.grid.nrows(), "Grids must match");
        self.values.iter()
            .zip(other.values.iter())
            .map(|(&a, &b)| a * b)
            .sum::<f64>() * self.dx
    }

    /// Numerical derivative of the test function.
    pub fn derivative(&self, order: usize) -> TestFunction {
        let mut vals = self.values.clone();
        for _ in 0..order {
            let n = vals.nrows();
            let mut d = DVector::zeros(n);
            for i in 1..n - 1 {
                d[i] = (vals[i + 1] - vals[i - 1]) / (2.0 * self.dx);
            }
            d[0] = (vals[1] - vals[0]) / self.dx;
            d[n - 1] = (vals[n - 1] - vals[n - 2]) / self.dx;
            vals = d;
        }
        TestFunction::new(self.grid.clone(), vals)
    }

    /// Compute the L² norm.
    pub fn l2_norm(&self) -> f64 {
        self.values.iter().map(|&v| v * v).sum::<f64>().sqrt() * self.dx.sqrt()
    }

    /// Compute the L∞ norm (sup norm).
    pub fn linf_norm(&self) -> f64 {
        self.values.iter().map(|v| v.abs()).fold(0.0_f64, f64::max)
    }

    /// Multiply by a scalar.
    pub fn scale(&self, c: f64) -> TestFunction {
        TestFunction::new(self.grid.clone(), &self.values * c)
    }

    /// Add two test functions (same grid assumed).
    pub fn add(&self, other: &TestFunction) -> TestFunction {
        assert_eq!(self.grid.nrows(), other.grid.nrows());
        TestFunction::new(self.grid.clone(), &self.values + &other.values)
    }

    /// Check if the function has compact support (values near zero at boundaries).
    pub fn has_compact_support(&self, tol: f64) -> bool {
        self.values.iter().take(3).all(|v| v.abs() < tol)
            && self.values.iter().rev().take(3).all(|v| v.abs() < tol)
    }

    /// Evaluate at a point via linear interpolation.
    pub fn eval(&self, x: f64) -> f64 {
        if x <= self.support.0 || x >= self.support.1 {
            return 0.0;
        }
        let idx = (x - self.support.0) / self.dx;
        let i = idx.floor() as usize;
        if i + 1 >= self.values.nrows() {
            return self.values[self.values.nrows() - 1];
        }
        let t = idx - i as f64;
        self.values[i] * (1.0 - t) + self.values[i + 1] * t
    }

    /// Integrate (numerical quadrature via trapezoidal rule).
    pub fn integrate(&self) -> f64 {
        let n = self.values.nrows();
        if n < 2 { return 0.0; }
        let mut sum = (self.values[0] + self.values[n - 1]) / 2.0;
        for i in 1..n - 1 {
            sum += self.values[i];
        }
        sum * self.dx
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bump_is_smooth() {
        let phi = TestFunction::bump(201);
        assert!(phi.has_compact_support(1e-10));
    }

    #[test]
    fn test_bump_support() {
        let phi = TestFunction::bump(201);
        assert_eq!(phi.support.0, -1.0);
        assert_eq!(phi.support.1, 1.0);
    }

    #[test]
    fn test_bump_positive() {
        let phi = TestFunction::bump(201);
        assert!(phi.values.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn test_bump_max_at_zero() {
        let phi = TestFunction::bump(501);
        let max_idx = phi.values.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
        let x_max = phi.grid[max_idx];
        assert!((x_max).abs() < 0.02);
    }

    #[test]
    fn test_bump_integral() {
        let phi = TestFunction::bump(2001);
        let integral = phi.integrate();
        // Known value ≈ 0.44399...
        assert!((integral - 0.444).abs() < 0.01);
    }

    #[test]
    fn test_scaled_bump_support() {
        let phi = TestFunction::scaled_bump(-2.0, 3.0, 201);
        assert!((phi.support.0 - (-2.0)).abs() < 1e-10);
        assert!((phi.support.1 - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_gaussian_normalized() {
        let g = TestFunction::gaussian(0.0, 1.0, 1001, 5.0);
        let max_val = g.linf_norm();
        assert!((max_val - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_gaussian_integral() {
        let g = TestFunction::gaussian(0.0, 1.0, 5001, 6.0);
        let integral = g.integrate();
        assert!((integral - (2.0 * std::f64::consts::PI).sqrt()).abs() < 0.1);
    }

    #[test]
    fn test_derivative_bump() {
        let phi = TestFunction::bump(501);
        let dphi = phi.derivative(1);
        // Derivative should be antisymmetric
        let mid = dphi.values.nrows() / 2;
        assert!((dphi.values[mid]).abs() < 0.05);
    }

    #[test]
    fn test_derivative_order_zero() {
        let phi = TestFunction::bump(101);
        let d0 = phi.derivative(0);
        assert!((d0.values - phi.values).norm() < 1e-10);
    }

    #[test]
    fn test_l2_norm() {
        let phi = TestFunction::bump(501);
        let norm = phi.l2_norm();
        assert!(norm > 0.0);
    }

    #[test]
    fn test_scale() {
        let phi = TestFunction::bump(101);
        let scaled = phi.scale(2.0);
        assert!((scaled.values[50] - 2.0 * phi.values[50]).abs() < 1e-10);
    }

    #[test]
    fn test_add() {
        let phi1 = TestFunction::bump(101);
        let phi2 = TestFunction::bump(101);
        let sum = phi1.add(&phi2);
        assert!((sum.values[50] - 2.0 * phi1.values[50]).abs() < 1e-10);
    }

    #[test]
    fn test_eval_outside_support() {
        let phi = TestFunction::bump(101);
        assert_eq!(phi.eval(-2.0), 0.0);
        assert_eq!(phi.eval(2.0), 0.0);
    }

    #[test]
    fn test_eval_inside_support() {
        let phi = TestFunction::bump(1001);
        let val = phi.eval(0.0);
        assert!(val > 0.3);
    }

    #[test]
    fn test_inner_product_symmetry() {
        let phi1 = TestFunction::bump(201);
        let phi2 = TestFunction::bump(201);
        let ip12 = phi1.inner_product(&phi2);
        let ip21 = phi2.inner_product(&phi1);
        assert!((ip12 - ip21).abs() < 1e-10);
    }
}
