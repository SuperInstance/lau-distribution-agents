//! Application: analyze discontinuous agent policies (e.g., bang-bang control).
//!
//! Agent policies in control theory are often discontinuous:
//! - Bang-bang control: u(t) = +1 or -1, switching instantaneously
//! - Threshold policies: step-function decisions
//! - Sliding mode: high-frequency switching near a surface
//!
//! Distribution theory lets us differentiate, convolve, and analyze these
//! in a rigorous mathematical framework.

use nalgebra::DVector;
use serde::{Deserialize, Serialize};
use crate::distribution::Distribution;
use crate::test_function::TestFunction;
use crate::derivative::DistributionDerivative;
use crate::regularization::{Mollifier, Regularization};
use crate::convolution::Convolution;
use crate::dirac_delta::DiracDelta;

/// An agent policy — possibly discontinuous, analyzed as a distribution.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AgentPolicy {
    /// The policy as a distribution
    pub distribution: Distribution,
    /// Time grid
    pub time_grid: DVector<f64>,
    /// Description
    pub description: String,
}

impl AgentPolicy {
    /// Create a bang-bang control policy: u(t) = +u_max for t < t_switch, -u_max otherwise.
    pub fn bang_bang(t_switch: f64, u_max: f64, a: f64, b: f64, n_points: usize) -> Self {
        let dt = (b - a) / (n_points - 1) as f64;
        let grid: Vec<f64> = (0..n_points).map(|i| a + i as f64 * dt).collect();
        let values: Vec<f64> = grid.iter().map(|&t| {
            if t < t_switch { u_max } else { -u_max }
        }).collect();
        let dist = Distribution::regular(
            &format!("bang_bang(t={})", t_switch),
            DVector::from_vec(grid.clone()),
            DVector::from_vec(values),
        );
        Self {
            distribution: dist,
            time_grid: DVector::from_vec(grid),
            description: format!("Bang-bang: +{} for t < {}, -{} otherwise", u_max, t_switch, u_max),
        }
    }

    /// Create a proportional control policy: u(t) = K * error(t).
    pub fn proportional_control(k: f64, setpoint: f64, a: f64, b: f64, n_points: usize,
                                 state_fn: impl Fn(f64) -> f64) -> Self {
        let dt = (b - a) / (n_points - 1) as f64;
        let grid: Vec<f64> = (0..n_points).map(|i| a + i as f64 * dt).collect();
        let values: Vec<f64> = grid.iter().map(|&t| {
            k * (setpoint - state_fn(t))
        }).collect();
        let dist = Distribution::regular(
            "proportional",
            DVector::from_vec(grid.clone()),
            DVector::from_vec(values),
        );
        Self {
            distribution: dist,
            time_grid: DVector::from_vec(grid),
            description: format!("P control: K={}, setpoint={}", k, setpoint),
        }
    }

    /// Threshold policy: u(t) = u_high if x(t) > threshold, u_low otherwise.
    pub fn threshold_policy(threshold: f64, u_low: f64, u_high: f64,
                            a: f64, b: f64, n_points: usize,
                            state_fn: impl Fn(f64) -> f64) -> Self {
        let dt = (b - a) / (n_points - 1) as f64;
        let grid: Vec<f64> = (0..n_points).map(|i| a + i as f64 * dt).collect();
        let values: Vec<f64> = grid.iter().map(|&t| {
            if state_fn(t) > threshold { u_high } else { u_low }
        }).collect();
        let dist = Distribution::regular(
            "threshold",
            DVector::from_vec(grid.clone()),
            DVector::from_vec(values),
        );
        Self {
            distribution: dist,
            time_grid: DVector::from_vec(grid),
            description: format!("Threshold: thresh={}, low={}, high={}", threshold, u_low, u_high),
        }
    }

    /// Sliding mode: high-frequency switching near a surface.
    pub fn sliding_mode(surface: f64, gain: f64, freq: f64,
                        a: f64, b: f64, n_points: usize) -> Self {
        let dt = (b - a) / (n_points - 1) as f64;
        let grid: Vec<f64> = (0..n_points).map(|i| a + i as f64 * dt).collect();
        let values: Vec<f64> = grid.iter().map(|&t| {
            let s = t - surface; // simplified: surface at x=surface
            if s.abs() < 0.2 {
                gain * (2.0 * std::f64::consts::PI * freq * t).signum()
            } else {
                gain * s.signum()
            }
        }).collect();
        let dist = Distribution::regular(
            "sliding_mode",
            DVector::from_vec(grid.clone()),
            DVector::from_vec(values),
        );
        Self {
            distribution: dist,
            time_grid: DVector::from_vec(grid),
            description: format!("Sliding mode: surface={}, gain={}, freq={}", surface, gain, freq),
        }
    }

    /// Compute the distributional derivative of the policy.
    /// For bang-bang: derivative has Dirac deltas at switching times.
    pub fn derivative(&self, order: usize) -> Distribution {
        DistributionDerivative::weak_derivative(&self.distribution, order)
    }

    /// Regularize (smooth) the policy via mollification.
    pub fn regularize(&self, epsilon: f64, n_mollifier_points: usize) -> Regularization {
        let mol = Mollifier::standard(epsilon, n_mollifier_points);
        mol.mollify(&self.distribution)
    }

    /// Compute the total variation of the policy.
    pub fn total_variation(&self) -> f64 {
        if let Some(vals) = &self.distribution.function_values {
            let mut tv = 0.0;
            for i in 1..vals.nrows() {
                tv += (vals[i] - vals[i - 1]).abs();
            }
            tv
        } else {
            f64::INFINITY
        }
    }

    /// Count discontinuities (jumps above threshold).
    pub fn count_discontinuities(&self, threshold: f64) -> usize {
        if let Some(vals) = &self.distribution.function_values {
            let dx = self.distribution.dx;
            (0..vals.nrows()-1).filter(|&i| (vals[i+1] - vals[i]).abs() > threshold).count()
        } else {
            0
        }
    }

    /// Compute energy: ∫ u(t)² dt.
    pub fn energy(&self) -> f64 {
        if let Some(vals) = &self.distribution.function_values {
            vals.iter().map(|v| v * v).sum::<f64>() * self.distribution.dx
        } else {
            0.0
        }
    }

    /// Apply the policy to a test function (distributional pairing).
    pub fn apply(&self, phi: &TestFunction) -> f64 {
        self.distribution.apply(phi)
    }

    /// Compute the cost functional J(u) = ∫ (u² + α u'²) dt.
    pub fn cost_functional(&self, alpha: f64) -> f64 {
        let u_energy = self.energy();
        let du = self.derivative(1);
        let du_energy = if let Some(vals) = &du.function_values {
            vals.iter().map(|v| v * v).sum::<f64>() * du.dx
        } else {
            0.0
        };
        u_energy + alpha * du_energy
    }

    /// Check if policy is bang-bang (only takes values ±u_max).
    pub fn is_bang_bang(&self, tol: f64) -> bool {
        if let Some(vals) = &self.distribution.function_values {
            let max = vals.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
            if max < 1e-10 { return false; }
            vals.iter().all(|v| (v.abs() - max).abs() < tol * max)
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bang_bang_creation() {
        let policy = AgentPolicy::bang_bang(1.0, 1.0, 0.0, 2.0, 201);
        assert!(policy.distribution.is_regular());
    }

    #[test]
    fn test_bang_bang_values() {
        let policy = AgentPolicy::bang_bang(1.0, 1.0, 0.0, 2.0, 201);
        if let Some(vals) = &policy.distribution.function_values {
            assert_eq!(vals[0], 1.0);    // t < 1
            assert_eq!(vals[200], -1.0);  // t > 1
        }
    }

    #[test]
    fn test_bang_bang_discontinuity() {
        let policy = AgentPolicy::bang_bang(1.0, 1.0, 0.0, 2.0, 201);
        assert!(policy.count_discontinuities(0.5) >= 1);
    }

    #[test]
    fn test_bang_bang_derivative() {
        let policy = AgentPolicy::bang_bang(1.0, 1.0, 0.0, 2.0, 2001);
        let deriv = policy.derivative(1);
        // Derivative should spike at the switching point
        if let Some(vals) = &deriv.function_values {
            let peak = vals.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
            assert!(peak > 5.0); // Large spike at discontinuity
        }
    }

    #[test]
    fn test_bang_bang_total_variation() {
        let policy = AgentPolicy::bang_bang(1.0, 1.0, 0.0, 2.0, 2001);
        let tv = policy.total_variation();
        // Should be approximately 2.0 (one jump from +1 to -1)
        assert!((tv - 2.0).abs() < 1.0);
    }

    #[test]
    fn test_bang_bang_energy() {
        let policy = AgentPolicy::bang_bang(1.0, 1.0, 0.0, 2.0, 2001);
        let energy = policy.energy();
        assert!((energy - 2.0).abs() < 0.1); // ∫0^2 1 dt = 2
    }

    #[test]
    fn test_bang_bang_is_bang_bang() {
        let policy = AgentPolicy::bang_bang(1.0, 1.0, 0.0, 2.0, 201);
        assert!(policy.is_bang_bang(0.1));
    }

    #[test]
    fn test_proportional_control() {
        let policy = AgentPolicy::proportional_control(
            2.0, 1.0, 0.0, 5.0, 201, |t| 0.5 * t
        );
        assert!(policy.distribution.is_regular());
    }

    #[test]
    fn test_threshold_policy() {
        let policy = AgentPolicy::threshold_policy(
            0.5, 0.0, 1.0, 0.0, 2.0, 201, |t| (t * std::f64::consts::PI).sin()
        );
        assert!(policy.count_discontinuities(0.5) >= 1);
    }

    #[test]
    fn test_sliding_mode() {
        let policy = AgentPolicy::sliding_mode(1.0, 1.0, 100.0, 0.0, 2.0, 5001);
        assert!(policy.total_variation() > 1.0); // High-frequency switching
    }

    #[test]
    fn test_regularization_smooths_bang_bang() {
        let policy = AgentPolicy::bang_bang(1.0, 1.0, 0.0, 2.0, 501);
        let reg = policy.regularize(0.1, 501);
        // Smoothed version should have fewer discontinuities
        let smooth_policy = AgentPolicy {
            distribution: reg.smoothed,
            time_grid: policy.time_grid.clone(),
            description: "smoothed".to_string(),
        };
        assert!(smooth_policy.count_discontinuities(0.5) == 0);
    }

    #[test]
    fn test_cost_functional() {
        let policy = AgentPolicy::bang_bang(1.0, 1.0, 0.0, 2.0, 2001);
        let cost = policy.cost_functional(0.1);
        assert!(cost > 0.0);
        assert!(cost.is_finite());
    }

    #[test]
    fn test_policy_apply() {
        let policy = AgentPolicy::bang_bang(1.0, 1.0, -1.0, 2.0, 501);
        let phi = TestFunction::bump(501);
        let val = policy.apply(&phi);
        assert!(val.is_finite());
    }

    #[test]
    fn test_multiple_switches() {
        // Bang-bang with 2 switches
        let dt = 2.0 / 200.0;
        let grid: Vec<f64> = (0..201).map(|i| -1.0 + i as f64 * dt).collect();
        let values: Vec<f64> = grid.iter().map(|&t| {
            if t < -0.3 { 1.0 } else if t < 0.3 { -1.0 } else { 1.0 }
        }).collect();
        let dist = Distribution::regular("multi_switch", DVector::from_vec(grid.clone()), DVector::from_vec(values));
        let policy = AgentPolicy {
            distribution: dist,
            time_grid: DVector::from_vec(grid),
            description: "2 switches".to_string(),
        };
        assert!(policy.count_discontinuities(0.5) >= 2);
        let tv = policy.total_variation();
        assert!(tv > 3.0); // Two jumps of size 2
    }
}
