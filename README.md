# lau-distribution-agents

**Schwartz distribution theory for agents — when behavior is too rough for classical analysis.**

[![Tests](https://img.shields.io/badge/tests-112-passing-brightgreen)]()
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)]()

---

## What This Does

Agent policies in the real world are often **discontinuous**: bang-bang control flips between ±1 instantaneously, threshold policies jump at boundaries, sliding mode controllers chatter at high frequency. Classical calculus breaks down at these discontinuities.

This crate provides the complete mathematical toolkit of **Schwartz distribution theory** — test functions, distributions, the Dirac delta, distributional derivatives, convolution, tempered distributions, Fourier transforms, Sobolev spaces, regularization (mollification), and fundamental solutions (Green's functions) — all applied to analyzing rough agent policies.

You can:
- **Differentiate anything.** Every distribution has derivatives of all orders. Even a Heaviside step function has a derivative (the Dirac delta).
- **Convolve rough policies with smooth kernels.** Mollification turns bang-bang control into a smooth approximation that converges to the original.
- **Take Fourier transforms** of tempered distributions for frequency-domain analysis.
- **Measure smoothness** via Sobolev norms and verify embedding theorems.
- **Solve PDEs** using Green's functions (heat equation, wave equation, Poisson equation).
- **Compute cost functionals** for discontinuous control policies.

---

## Key Idea

The central abstraction:

| Classical Analysis | Distribution Theory |
|---|---|
| Functions f(x) | Distributions T[φ] = ⟨T, φ⟩ |
| Derivative ∂f | Distributional derivative: ⟨∂T, φ⟩ = −⟨T, ∂φ⟩ |
| Point evaluation | Pairing with test functions |
| Discontinuities are problematic | Every distribution is infinitely differentiable |
| Dirac delta is "infinite" | δ₀[φ] = φ(0) — perfectly rigorous |
| Convolution smooths | T ∗ φ ∈ C^∞ for any distribution T |

A **distribution** is a continuous linear functional on the space of test functions (smooth, compactly supported). This seemingly abstract definition lets us handle impulses, discontinuities, and singularities in a mathematically rigorous way — and it's exactly what we need for rough agent policies.

---

## Install

Add to your `Cargo.toml`:

```toml
[dependencies]
lau-distribution-agents = { git = "https://github.com/SuperInstance/lau-distribution-agents" }
```

**Dependencies:** `serde` (with `derive`), `nalgebra` (with `serde-serialize`).

---

## Quick Start

### Bang-Bang Control Analysis

```rust
use lau_distribution-agents::AgentPolicy;

// Create a bang-bang controller: +1 before t=1, -1 after
let policy = AgentPolicy::bang_bang(1.0, 1.0, 0.0, 2.0, 2001);

// It's discontinuous
assert!(policy.count_discontinuities(0.5) >= 1);

// But we can differentiate it! The derivative is a spike at the switch
let deriv = policy.derivative(1);

// Compute total variation and energy
let tv = policy.total_variation();  // ≈ 2.0 (one jump of size 2)
let energy = policy.energy();       // ≈ 2.0 (integral of u²)

// Smooth it via mollification
let reg = policy.regularize(0.1, 501);
let smooth_policy = AgentPolicy {
    distribution: reg.smoothed,
    time_grid: policy.time_grid.clone(),
    description: "smoothed".to_string(),
};
assert_eq!(smooth_policy.count_discontinuities(0.5), 0); // no more jumps
```

### The Dirac Delta

```rust
use lau_distribution_agents::{DiracDelta, DiracComb, TestFunction};

// Point evaluation: δ₀[φ] = φ(0)
let delta = DiracDelta::at_origin();
let phi = TestFunction::bump(1001);
let val = delta.apply(&phi);  // = φ(0) ≈ e^{-1}

// Sifting property: ∫ δ(x-a) f(x) dx = f(a)
assert_eq!(delta.sift(|x| x * x), 0.0);
let delta_at_2 = DiracDelta::at(2.0);
assert_eq!(delta_at_2.sift(|x| x * x), 4.0);

// Dirac comb (Shah function)
let comb = DiracComb::shah(1.0, 3);  // Σ δ(x - n) for n = -3,...,3
```

### Distributional Derivatives

```rust
use lau_distribution_agents::{Distribution, TestFunction, DistributionDerivative};

// Derivative of Heaviside step function = Dirac delta
let (heaviside, deriv) = DistributionDerivative::heaviside_derivative(2001);
// deriv peaks at x=0, which is the delta

// Integration by parts: ⟨T', φ⟩ = -⟨T, φ'⟩
let t = Distribution::from_fn("x³", -3.0, 3.0, 1001, |x| x.powi(3));
let phi = TestFunction::bump(1001);
assert!(DistributionDerivative::verify_integration_by_parts(&t, &phi));
```

### Green's Functions

```rust
use lau_distribution_agents::FundamentalSolution;

// Heat kernel: E(x,t) = (4πt)^{-1/2} exp(-x²/(4t))
let heat = FundamentalSolution::heat_equation(1.0, -5.0, 5.0, 501);
assert!(heat.verify_heat_conservation());  // ∫ E dx = 1

// Poisson equation: -u'' = f on [-L, L] with zero boundary
let green = FundamentalSolution::poisson_1d(5.0, 0.0, 201);
let f = Distribution::from_fn("source", -5.0, 5.0, 201, |x| (-x*x).exp());
let u = green.solve(&f);  // Solution via Green's function convolution
```

---

## API Reference

### Core Types

| Type | Module | Description |
|---|---|---|
| `TestFunction` | `test_function` | Smooth compactly-supported function (bump, Gaussian, scaled) |
| `Distribution` | `distribution` | Continuous linear functional on test functions (regular or singular) |
| `DiracDelta` | `dirac_delta` | Point evaluation distribution δ(x − a) |
| `DiracComb` | `dirac_delta` | Linear combination of Dirac deltas / Shah function |
| `DistributionDerivative` | `derivative` | Distributional derivative: ⟨∂T, φ⟩ = −⟨T, ∂φ⟩ |
| `Convolution` | `convolution` | Smoothing via convolution: T ∗ φ is always smooth |
| `TemperedDistribution` | `tempered` | Distribution with polynomial growth (Fourier transformable) |
| `FourierTransform` | `fourier` | DFT of test functions and tempered distributions |
| `SobolevSpace` | `sobolev` | W^{s,p} spaces: smoothness measurement via weak derivatives |
| `Mollifier` | `regularization` | Smooth bump kernel for approximating rough distributions |
| `Regularization` | `regularization` | Result of mollifying: original + smoothed + convergence check |
| `FundamentalSolution` | `fundamental` | Green's functions for PDE operators |
| `AgentPolicy` | `agent_policy` | High-level API for discontinuous control policies |

### `TestFunction` — The Probes

```rust
TestFunction::bump(201)                        // Canonical bump: φ(x) = exp(-1/(1-x²))
TestFunction::scaled_bump(-2.0, 3.0, 201)     // Bump on [a, b]
TestFunction::gaussian(0.0, 1.0, 501, 5.0)    // Gaussian with center, σ, points, range

phi.derivative(2)     // 2nd numerical derivative
phi.integrate()       // Trapezoidal rule
phi.l2_norm()         // L² norm
phi.linf_norm()       // Sup norm
phi.inner_product(&psi) // ⟨φ, ψ⟩
phi.eval(0.5)         // Linear interpolation
phi.has_compact_support(1e-10) // Check boundary vanishing
```

### `Distribution` — The Central Concept

```rust
// Regular: from a function on a grid
let t = Distribution::from_fn("cos", -5.0, 5.0, 501, |x| x.cos());

// Singular: Dirac delta
let d = Distribution::singular("δ₀", "dirac_0");

// Operations
t.apply(&phi)           // ⟨T, φ⟩ — the pairing
t.add(&other)           // T₁ + T₂
t.scale(3.0)            // 3T
t.is_regular()          // Regular or singular?
t.verify_linearity(&phi, &psi, α, β) // Check T(αφ + βψ) = αT(φ) + βT(ψ)
```

### `AgentPolicy` — High-Level API

```rust
// Pre-built policies
AgentPolicy::bang_bang(t_switch, u_max, a, b, n)
AgentPolicy::proportional_control(K, setpoint, a, b, n, |t| state(t))
AgentPolicy::threshold_policy(thresh, u_low, u_high, a, b, n, |t| state(t))
AgentPolicy::sliding_mode(surface, gain, freq, a, b, n)

// Analysis
policy.derivative(1)                  // Distributional derivative
policy.regularize(0.1, 501)          // Mollify with ε=0.1
policy.total_variation()              // TV norm
policy.count_discontinuities(0.5)    // Jump detection
policy.energy()                       // ∫ u² dt
policy.cost_functional(α)             // ∫ (u² + α u'²) dt
policy.is_bang_bang(0.1)             // Check if ±u_max only
```

### `SobolevSpace` — Regularity Measurement

```rust
let h1 = SobolevSpace::new(1.0, 2.0, -3.0, 3.0, 501); // H¹ = W^{1,2}
h1.norm(&dist, 256)                    // W^{s,p} norm
h1.h1_seminorm(&dist)                  // |u|_{H¹} = ||∇u||_{L²}
h1.check_continuous_embedding()        // W^{s,p} ⊂ C⁰? (s > 1/p)
h1.compactly_embeds(&h0)              // Rellich-Kondrachov
h1.verify_poincare(&dist)             // Poincaré inequality
```

### `FourierTransform` — Frequency Analysis

```rust
let ft = FourierTransform::of_test_function(&phi, 256);
ft.power_spectrum()                                    // |F̂(ξ)|²
ft.phase_spectrum()                                    // arg(F̂(ξ))
ft.inverse()                                           // Inverse DFT

FourierTransform::of_tempered(&t, 256)                // DFT of tempered dist
FourierTransform::verify_parseval(&phi, 256)           // Parseval's theorem
FourierTransform::verify_gaussian_is_gaussian(1.0)    // F(Gauss) = Gauss
FourierTransform::verify_derivative_property(&phi, 256) // F(f') = 2πiξ F(f)
```

---

## How It Works

The crate is organized in layers, each building on the previous:

### Layer 1: Test Functions (`test_function`)

Everything in distribution theory starts with **test functions** — smooth (C^∞) functions with compact support. The prototype is the canonical bump function:

$$\varphi(x) = \exp\!\left(\frac{-1}{1 - x^2}\right) \text{ for } |x| < 1, \quad 0 \text{ otherwise}$$

`TestFunction` stores a function evaluated on a discrete grid, supporting numerical derivatives (central differences), integration (trapezoidal rule), L²/L^∞ norms, inner products, and linear interpolation for evaluation.

### Layer 2: Distributions (`distribution`)

A `Distribution` is a continuous linear functional on test functions. There are two kinds:
- **Regular**: given by a locally integrable function f, acting as T[φ] = ∫ f(x)φ(x) dx
- **Singular**: defined by a custom action (like the Dirac delta: δ₀[φ] = φ(0))

The crate implements linearity verification (T(αφ + βψ) = αT(φ) + βT(ψ)), addition, and scalar multiplication.

### Layer 3: Fundamental Singular Distributions (`dirac_delta`)

The **Dirac delta** δ(x − a) is the quintessential singular distribution. It evaluates the test function at a point: δ_a[φ] = φ(a). The crate supports:
- Point evaluation (sifting property)
- Gaussian approximation δ_ε → δ as ε → 0
- Scaling and shifting
- **Dirac combs** (Shah functions): Σ cᵢ δ(x − aᵢ)

### Layer 4: Distributional Derivatives (`derivative`)

The key insight: **every distribution has derivatives of all orders**, defined by:

$$\langle \partial T, \varphi \rangle = -\langle T, \partial\varphi \rangle$$

This means we can differentiate discontinuities. The derivative of the Heaviside step function H(x) is δ(x). The derivative of x² is 2 (as a weak derivative). Integration by parts ⟨T', φ⟩ = −⟨T, φ'⟩ is verified numerically.

### Layer 5: Convolution (`convolution`)

Convolution of a distribution with a smooth function always yields a smooth function: (T ∗ φ)(x) = T[φ(x − ·)]. This is the engine behind mollification. The crate verifies:
- **Commutativity**: f ∗ g = g ∗ f
- **Associativity**: (f ∗ g) ∗ h = f ∗ (g ∗ h)
- **Young's inequality**: ‖f ∗ g‖_r ≤ ‖f‖_p · ‖g‖_q

### Layer 6: Tempered Distributions (`tempered`)

A **tempered distribution** grows at most polynomially. This is the largest class for which the Fourier transform is defined. The crate checks polynomial growth bounds, supports addition, scaling, differentiation (which preserves temperedness), and multiplication by polynomials.

### Layer 7: Fourier Analysis (`fourier`)

The DFT is computed for test functions and tempered distributions. Properties verified:
- **Parseval's theorem**: ∫|f|² = ∫|F̂|²
- **Gaussian preservation**: F(gaussian) = gaussian
- **Derivative property**: F(f') = 2πiξ · F(f)
- Power spectrum and phase spectrum computation
- Inverse DFT

### Layer 8: Sobolev Spaces (`sobolev`)

Sobolev spaces W^{s,p} measure distributional smoothness. The H¹ norm combines L² and derivative L²: ‖u‖²_{H¹} = ‖u‖²_{L²} + ‖∇u‖²_{L²}. The crate implements:
- Sobolev norm computation
- **Sobolev embedding**: W^{s,p} ⊂ C⁰ when s > 1/p
- **Rellich-Kondrachov compactness**: W^{s₁,p} ↪↪ W^{s₂,p} for s₁ > s₂
- **Poincaré inequality**: ‖u‖ ≤ C‖∇u‖
- Complex interpolation of norms

### Layer 9: Regularization (`regularization`)

**Mollification** is the process of smoothing a rough distribution by convolving with a mollifier (smooth bump with unit integral). The standard mollifier is φ_ε(x) = (1/ε)φ(x/ε). Properties:
- u_ε → u in L^p as ε → 0
- u_ε is smooth (C^∞)
- ‖u_ε‖_p ≤ ‖u‖_p
- The crate generates regularization sequences and verifies convergence

### Layer 10: Fundamental Solutions (`fundamental`)

**Green's functions** are fundamental solutions of differential operators: L(E) = δ. The crate provides:
- **Laplacian in 1D**: E(x) = |x|/2
- **Heat equation**: E(x,t) = (4πt)^{-1/2} exp(−x²/(4t))
- **Wave equation**: Propagating delta functions
- **Poisson equation**: G(x,y) = (L − x_>)(L + x_<)/(2L) with zero boundary conditions

Each comes with verification: heat conservation, maximum principle, boundary conditions.

### Layer 11: Agent Policies (`agent_policy`)

The top-level API ties everything together for discontinuous control policies:
- **Bang-bang control**: u(t) = +u_max or −u_max
- **Proportional control**: u(t) = K(setpoint − state(t))
- **Threshold policy**: step function triggered by state
- **Sliding mode**: high-frequency switching near a surface

Each can be differentiated, regularized, and analyzed for total variation, energy, discontinuity count, and cost functionals.

---

## The Math

### Schwartz Distribution Theory

A **distribution** T ∈ D'(ℝ) is a continuous linear functional on the space of test functions D(ℝ) = C_c^∞(ℝ). The key operations:

- **Derivative**: ⟨∂^n T, φ⟩ = (−1)^n ⟨T, ∂^n φ⟩ — always exists
- **Convolution**: (T ∗ φ)(x) = ⟨T, φ(x − ·)⟩ — always smooth
- **Fourier transform** (for tempered T): ⟨F(T), ψ⟩ = ⟨T, F(ψ)⟩

The **Dirac delta** δ₀ is defined by δ₀[φ] = φ(0). It's not a function — it's a singular distribution. Approximations converge: δ_ε → δ as ε → 0 in D'.

### Sobolev Embedding Theorems

In 1D:
- W^{1,p}(ℝ) ⊂ C⁰(ℝ) for all p > 1 (Sobolev embedding)
- H¹(ℝ) = W^{1,2}(ℝ) ⊂ C⁰(ℝ) with ‖u‖_∞ ≤ C‖u‖_{H¹}
- W^{s₁,p} compactly embeds in W^{s₂,p} for s₁ > s₂ (Rellich-Kondrachov)

### Mollification

For any u ∈ L^p_loc, the mollification u_ε = u ∗ φ_ε satisfies:
- u_ε ∈ C^∞
- u_ε → u in L^p as ε → 0
- ‖u_ε‖_p ≤ ‖u‖_p

### Green's Functions

For the operator L with fundamental solution E (L(E) = δ), the solution to L(u) = f is:
$$u(x) = \int G(x, y) f(y) \, dy$$
where G is the Green's function incorporating boundary conditions.

---

## Test Suite

112 tests across 11 modules:

| Module | Tests | Coverage |
|---|---|---|
| `test_function` | 16 | Bump, Gaussian, derivatives, norms, support, integration |
| `distribution` | 10 | Regular, singular, Dirac, linearity, addition, scaling |
| `dirac_delta` | 11 | Sifting, approximation, scaling, shifting, comb, convergence |
| `derivative` | 7 | Weak derivative, Heaviside, integration by parts, sign convention |
| `convolution` | 7 | Commutativity, smoothing, delta convolution, Young's inequality |
| `tempered` | 9 | Growth check, polynomial growth, derivative, multiplication |
| `fourier` | 9 | DFT, Parseval, Gaussian, derivative property, inverse, tempered |
| `sobolev` | 10 | H¹ norm, embeddings, compactness, Poincaré, interpolation |
| `regularization` | 9 | Mollifier, smoothing, convergence, discontinuous functions |
| `fundamental` | 10 | Laplacian, heat, wave, Poisson, conservation, boundaries |
| `agent_policy` | 14 | Bang-bang, P-control, threshold, sliding mode, cost |

Run all tests:

```bash
cargo test
```

---

## License

MIT
