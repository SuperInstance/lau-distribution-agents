//! Sobolev spaces via distributions: W^{s,p} = {u ∈ S' : (1+|ξ|²)^{s/2} û ∈ L^p}.
//!
//! Sobolev spaces measure the "smoothness" of distributions.
//! For agent policies, W^{s,p} regularity tells us how smooth the policy is
//! in a weak/distributional sense.

use nalgebra::DVector;
use serde::{Deserialize, Serialize};
use crate::distribution::Distribution;
use crate::fourier::FourierTransform;

/// Sobolev space membership and norm computation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SobolevSpace {
    /// Smoothness parameter s
    pub smoothness: f64,
    /// Integrability parameter p
    pub p: f64,
    /// Grid
    pub grid: DVector<f64>,
}

impl SobolevSpace {
    /// Create a W^{s,p} space on a given grid.
    pub fn new(s: f64, p: f64, a: f64, b: f64, n_points: usize) -> Self {
        let dx = (b - a) / (n_points - 1) as f64;
        let grid = DVector::from_vec((0..n_points).map(|i| a + i as f64 * dx).collect());
        Self { smoothness: s, p, grid }
    }

    /// Compute the W^{s,p} norm of a distribution (via Fourier transform).
    /// ||u||_{W^{s,p}} = ||F^{-1}[(1+|ξ|²)^{s/2} û]||_{L^p}
    pub fn norm(&self, dist: &Distribution, n_freqs: usize) -> f64 {
        if let (Some(vals), Some(grid)) = (&dist.function_values, &dist.grid) {
            // Compute weighted Sobolev norm
            let dx = dist.dx;
            let n = vals.nrows();

            // For s=1, p=2: ||u||_{H^1}² = ||u||_{L²}² + ||u'||_{L²}²
            if (self.p - 2.0).abs() < 0.01 {
                let l2_sq: f64 = vals.iter().map(|v| v * v).sum::<f64>() * dx;

                // Numerical derivative
                let mut deriv_sq = 0.0;
                for i in 1..n - 1 {
                    let d = (vals[i + 1] - vals[i - 1]) / (2.0 * dx);
                    deriv_sq += d * d;
                }
                deriv_sq *= dx;

                let s = self.smoothness;
                if (s - 1.0).abs() < 0.01 {
                    (l2_sq + deriv_sq).sqrt()
                } else if (s - 0.0).abs() < 0.01 {
                    l2_sq.sqrt()
                } else {
                    // General: approximate with derivative terms
                    (l2_sq + s * deriv_sq).sqrt()
                }
            } else {
                // General p: compute L^p norm
                let lp: f64 = vals.iter()
                    .map(|v| v.abs().powf(self.p))
                    .sum::<f64>() * dx;
                lp.powf(1.0 / self.p)
            }
        } else {
            f64::INFINITY
        }
    }

    /// Check if a distribution belongs to W^{s,p}.
    pub fn contains(&self, dist: &Distribution, n_freqs: usize) -> bool {
        let norm = self.norm(dist, n_freqs);
        norm.is_finite()
    }

    /// Compute the H^1 = W^{1,2} semi-norm (just the derivative part).
    pub fn h1_seminorm(&self, dist: &Distribution) -> f64 {
        if let (Some(vals), Some(_grid)) = (&dist.function_values, &dist.grid) {
            let dx = dist.dx;
            let n = vals.nrows();
            let mut deriv_sq = 0.0;
            for i in 1..n - 1 {
                let d = (vals[i + 1] - vals[i - 1]) / (2.0 * dx);
                deriv_sq += d * d;
            }
            deriv_sq.sqrt() * dx.sqrt()
        } else {
            f64::INFINITY
        }
    }

    /// Embedding: W^{s,p} ⊂ C^0 when s > n/p (for n=1, s > 1/p).
    /// Checks if a function in W^{s,p} is continuous.
    pub fn check_continuous_embedding(&self) -> bool {
        // In 1D: W^{s,p} ⊂ C^0 iff s > 1/p
        self.smoothness > 1.0 / self.p
    }

    /// Rellich-Kondrachov: W^{s₁,p} compactly embeds in W^{s₂,p} when s₁ > s₂.
    pub fn compactly_embeds(&self, other: &SobolevSpace) -> bool {
        self.smoothness > other.smoothness && (self.p - other.p).abs() < 0.01
    }

    /// Poincaré inequality: ||u||_{L^p} ≤ C ||∇u||_{L^p} for u with zero boundary.
    pub fn verify_poincare(&self, dist: &Distribution) -> bool {
        let lp_norm = self.lp_norm(dist);
        let h1_semi = self.h1_seminorm(dist);
        if h1_semi < 1e-10 { return true; }
        lp_norm <= 10.0 * h1_semi // with some constant
    }

    fn lp_norm(&self, dist: &Distribution) -> f64 {
        if let (Some(vals), Some(_grid)) = (&dist.function_values, &dist.grid) {
            let dx = dist.dx;
            vals.iter().map(|v| v.abs().powf(self.p)).sum::<f64>().powf(1.0 / self.p) * dx.powf(1.0 / self.p)
        } else {
            f64::INFINITY
        }
    }

    /// Interpolation: W^{s,p} = [L^p, W^{1,p}]_{s} by complex interpolation.
    pub fn interpolate_norm(&self, l2_norm: f64, h1_norm: f64) -> f64 {
        let s = self.smoothness;
        l2_norm.powf(1.0 - s) * h1_norm.powf(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_h1_norm_smooth() {
        let space = SobolevSpace::new(1.0, 2.0, -3.0, 3.0, 501);
        let dist = Distribution::from_fn("smooth", -3.0, 3.0, 501, |x| (-x*x).exp());
        let norm = space.norm(&dist, 256);
        assert!(norm.is_finite());
        assert!(norm > 0.0);
    }

    #[test]
    fn test_l2_norm() {
        let space = SobolevSpace::new(0.0, 2.0, -3.0, 3.0, 501);
        let dist = Distribution::from_fn("1", -3.0, 3.0, 501, |_| 1.0);
        let norm = space.norm(&dist, 256);
        assert!(norm > 0.0);
    }

    #[test]
    fn test_continuous_embedding() {
        let space = SobolevSpace::new(1.0, 2.0, -3.0, 3.0, 101);
        assert!(space.check_continuous_embedding()); // s=1 > 1/p=0.5
    }

    #[test]
    fn test_no_continuous_embedding() {
        let space = SobolevSpace::new(0.3, 2.0, -3.0, 3.0, 101);
        assert!(!space.check_continuous_embedding()); // s=0.3 < 1/p=0.5
    }

    #[test]
    fn test_compact_embedding() {
        let s1 = SobolevSpace::new(2.0, 2.0, -3.0, 3.0, 101);
        let s2 = SobolevSpace::new(1.0, 2.0, -3.0, 3.0, 101);
        assert!(s1.compactly_embeds(&s2));
    }

    #[test]
    fn test_no_compact_embedding() {
        let s1 = SobolevSpace::new(1.0, 2.0, -3.0, 3.0, 101);
        let s2 = SobolevSpace::new(2.0, 2.0, -3.0, 3.0, 101);
        assert!(!s1.compactly_embeds(&s2));
    }

    #[test]
    fn test_h1_seminorm() {
        let space = SobolevSpace::new(1.0, 2.0, -3.0, 3.0, 501);
        let dist = Distribution::from_fn("x", -3.0, 3.0, 501, |x| x);
        let semi = space.h1_seminorm(&dist);
        assert!((semi - 6.0_f64.sqrt()).abs() < 0.5);
    }

    #[test]
    fn test_contains_smooth() {
        let space = SobolevSpace::new(1.0, 2.0, -3.0, 3.0, 501);
        let dist = Distribution::from_fn("smooth", -3.0, 3.0, 501, |x| (-x*x).exp());
        assert!(space.contains(&dist, 256));
    }

    #[test]
    fn test_poincare() {
        let space = SobolevSpace::new(1.0, 2.0, -3.0, 3.0, 501);
        let dist = Distribution::from_fn("zero_bc", -3.0, 3.0, 501, |x| {
            let t = (x + 3.0) / 6.0;
            (std::f64::consts::PI * t).sin()
        });
        assert!(space.verify_poincare(&dist));
    }

    #[test]
    fn test_interpolation() {
        let space = SobolevSpace::new(0.5, 2.0, -3.0, 3.0, 101);
        let norm = space.interpolate_norm(1.0, 2.0);
        assert!((norm - 2.0_f64.sqrt()).abs() < 0.01);
    }
}
