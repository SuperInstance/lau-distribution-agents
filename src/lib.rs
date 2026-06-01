//! # lau-distribution-agents
//!
//! Schwartz distribution theory for agents — when agent behavior is too rough
//! for classical analysis, distributions extend the analysis.
//!
//! Agent policies may have discontinuities, impulses, or singularities.
//! Distribution theory provides the framework to handle these.

pub mod test_function;
pub mod distribution;
pub mod dirac_delta;
pub mod derivative;
pub mod convolution;
pub mod tempered;
pub mod fourier;
pub mod sobolev;
pub mod regularization;
pub mod fundamental;
pub mod agent_policy;

pub use test_function::TestFunction;
pub use distribution::Distribution;
pub use dirac_delta::DiracDelta;
pub use derivative::DistributionDerivative;
pub use convolution::Convolution;
pub use tempered::TemperedDistribution;
pub use fourier::FourierTransform;
pub use sobolev::SobolevSpace;
pub use regularization::Regularization;
pub use fundamental::FundamentalSolution;
pub use agent_policy::AgentPolicy;
