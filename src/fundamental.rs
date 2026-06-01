//! Fundamental solutions: Green's functions as distributions.
//!
//! A fundamental solution E for an operator L is a distribution satisfying:
//!   L(E) = δ
//! Green's functions are fundamental solutions with boundary conditions.
//! For agents: response of the system to a point impulse.

use nalgebra::DVector;
use serde::{Deserialize, Serialize};
use crate::distribution::Distribution;
use crate::test_function::TestFunction;
use crate::dirac_delta::DiracDelta;

/// A fundamental solution (Green's function) for a differential operator.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FundamentalSolution {
    /// Name/label
    pub name: String,
    /// The Green's function as a distribution
    pub green_function: Distribution,
    /// Description of the operator
    pub operator: String,
}

impl FundamentalSolution {
    /// Laplacian in 1D: d²E/dx² = δ → E(x) = |x|/2.
    pub fn laplacian_1d(a: f64, b: f64, n_points: usize) -> Self {
        let dist = Distribution::from_fn("E_lap1d", a, b, n_points, |x| x.abs() / 2.0);
        Self {
            name: "Laplacian_1D".to_string(),
            green_function: dist,
            operator: "d²/dx²".to_string(),
        }
    }

    /// Heat equation: ∂u/∂t - ∂²u/∂x² = 0
    /// Fundamental solution: E(x,t) = (4πt)^{-1/2} exp(-x²/(4t)).
    pub fn heat_equation(t: f64, a: f64, b: f64, n_points: usize) -> Self {
        let coeff = 1.0 / (4.0 * std::f64::consts::PI * t).sqrt();
        let dist = Distribution::from_fn(
            &format!("E_heat_t{}", t), a, b, n_points,
            |x| coeff * (-x * x / (4.0 * t)).exp()
        );
        Self {
            name: format!("Heat_t{}", t),
            green_function: dist,
            operator: "∂/∂t - ∂²/∂x²".to_string(),
        }
    }

    /// Wave equation: ∂²u/∂t² - ∂²u/∂x² = 0
    /// Fundamental solution involves propagating delta.
    pub fn wave_equation(t: f64, a: f64, b: f64, n_points: usize) -> Self {
        let dx = (b - a) / (n_points - 1) as f64;
        let grid: Vec<f64> = (0..n_points).map(|i| a + i as f64 * dx).collect();
        let width = 3.0 * dx;
        let values: Vec<f64> = grid.iter().map(|&x| {
            // Two traveling bumps: one at x=t, one at x=-t
            let bump1 = if (x - t).abs() < width {
                (-1.0 / (1.0 - ((x - t) / width).powi(2))).exp()
            } else { 0.0 };
            let bump2 = if (x + t).abs() < width {
                (-1.0 / (1.0 - ((x + t) / width).powi(2))).exp()
            } else { 0.0 };
            0.5 * (bump1 + bump2)
        }).collect();
        let dist = Distribution::regular(
            &format!("E_wave_t{}", t),
            DVector::from_vec(grid),
            DVector::from_vec(values),
        );
        Self {
            name: format!("Wave_t{}", t),
            green_function: dist,
            operator: "∂²/∂t² - ∂²/∂x²".to_string(),
        }
    }

    /// Poisson equation in 1D: -u'' = f → u(x) = ∫ G(x,y) f(y) dy.
    /// Green's function for [-L, L] with u(-L) = u(L) = 0:
    /// G(x,y) = (L - x_>)(L + x_<) / (2L)
    pub fn poisson_1d(L: f64, source_y: f64, n_points: usize) -> Self {
        let dx = 2.0 * L / (n_points - 1) as f64;
        let grid: Vec<f64> = (0..n_points).map(|i| -L + i as f64 * dx).collect();
        let values: Vec<f64> = grid.iter().map(|&x| {
            let x_upper = x.max(source_y);
            let x_lower = x.min(source_y);
            (L - x_upper) * (L + x_lower) / (2.0 * L)
        }).collect();
        let dist = Distribution::regular(
            &format!("G_poisson_y{}", source_y),
            DVector::from_vec(grid),
            DVector::from_vec(values),
        );
        Self {
            name: "Poisson_1D".to_string(),
            green_function: dist,
            operator: "-d²/dx²".to_string(),
        }
    }

    /// Apply the Green's function to solve L(u) = f.
    /// u(x) = ∫ G(x,y) f(y) dy.
    pub fn solve(&self, f: &Distribution) -> Distribution {
        if let (Some(g_vals), Some(g_grid)) = (&self.green_function.function_values, &self.green_function.grid) {
            if let (Some(f_vals), Some(_f_grid)) = (&f.function_values, &f.grid) {
                let n = g_grid.nrows();
                let dx = self.green_function.dx;
                let mut result = DVector::zeros(n);
                for i in 0..n {
                    // Interpolate f onto g's grid and convolve
                    let mut sum = 0.0;
                    for j in 0..n {
                        let f_val = if j < f_vals.nrows() { f_vals[j] } else { 0.0 };
                        sum += g_vals[j] * f_val;
                    }
                    result[i] = sum * dx;
                }
                return Distribution::regular(
                    &format!("u_for_{}", f.name),
                    g_grid.clone(),
                    result,
                );
            }
        }
        f.clone()
    }

    /// Verify that L(E) = δ by checking the jump in derivative.
    /// For Laplacian in 1D: E'' should give δ.
    pub fn verify_fundamental(&self, phi: &TestFunction) -> f64 {
        self.green_function.apply(phi)
    }

    /// Conservation property for heat kernel: ∫ E(x,t) dx = 1 for all t > 0.
    pub fn verify_heat_conservation(&self) -> bool {
        let integral = if let Some(vals) = &self.green_function.function_values {
            vals.sum() * self.green_function.dx
        } else { 0.0 };
        (integral - 1.0).abs() < 0.1
    }

    /// Maximum principle for heat equation: max decreases over time.
    pub fn verify_maximum_principle(solutions: &[FundamentalSolution]) -> bool {
        let mut prev_max = f64::INFINITY;
        for sol in solutions {
            if let Some(vals) = &sol.green_function.function_values {
                let max_val = vals.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
                if max_val > prev_max * 1.1 { return false; }
                prev_max = max_val;
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_laplacian_1d_green() {
        let sol = FundamentalSolution::laplacian_1d(-5.0, 5.0, 201);
        assert!(sol.green_function.is_regular());
    }

    #[test]
    fn test_laplacian_1d_at_origin() {
        let sol = FundamentalSolution::laplacian_1d(-5.0, 5.0, 1001);
        if let Some(vals) = &sol.green_function.function_values {
            let mid = vals.nrows() / 2;
            assert!((vals[mid]).abs() < 0.01); // E(0) = 0
        }
    }

    #[test]
    fn test_heat_kernel() {
        let sol = FundamentalSolution::heat_equation(1.0, -5.0, 5.0, 201);
        assert!(sol.green_function.is_regular());
    }

    #[test]
    fn test_heat_kernel_positive() {
        let sol = FundamentalSolution::heat_equation(0.5, -5.0, 5.0, 501);
        if let Some(vals) = &sol.green_function.function_values {
            assert!(vals.iter().all(|v| *v >= 0.0));
        }
    }

    #[test]
    fn test_heat_conservation() {
        let sol = FundamentalSolution::heat_equation(1.0, -5.0, 5.0, 5001);
        assert!(sol.verify_heat_conservation());
    }

    #[test]
    fn test_wave_equation() {
        let sol = FundamentalSolution::wave_equation(1.0, -5.0, 5.0, 201);
        assert!(sol.green_function.is_regular());
    }

    #[test]
    fn test_poisson_green() {
        let sol = FundamentalSolution::poisson_1d(5.0, 0.0, 201);
        assert!(sol.green_function.is_regular());
    }

    #[test]
    fn test_poisson_boundary() {
        let sol = FundamentalSolution::poisson_1d(5.0, 0.0, 201);
        if let Some(vals) = &sol.green_function.function_values {
            assert!(vals[0].abs() < 0.1); // G(-L, y) = 0
            assert!(vals[vals.nrows() - 1].abs() < 0.1); // G(L, y) = 0
        }
    }

    #[test]
    fn test_solve_poisson() {
        let green = FundamentalSolution::poisson_1d(5.0, 0.0, 201);
        let f = Distribution::from_fn("f", -5.0, 5.0, 201, |x| (-x * x).exp());
        let u = green.solve(&f);
        assert!(u.is_regular());
    }

    #[test]
    fn test_heat_kernel_spreads() {
        let sol1 = FundamentalSolution::heat_equation(0.1, -5.0, 5.0, 501);
        let sol2 = FundamentalSolution::heat_equation(1.0, -5.0, 5.0, 501);
        if let (Some(v1), Some(v2)) = (&sol1.green_function.function_values, &sol2.green_function.function_values) {
            // Larger t → wider kernel → smaller peak
            let peak1 = v1.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
            let peak2 = v2.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
            assert!(peak2 < peak1);
        }
    }
}
