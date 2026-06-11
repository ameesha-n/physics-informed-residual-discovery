# Hybrid Physics-Informed Residual Discovery Framework

## 1. Introduction

Many Physics-Informed Neural Networks (PINNs) assume that the governing partial differential equation (PDE) is completely known. However, real-world systems often contain unknown forcing terms, missing physics, modeling errors, or unresolved dynamics.

Current approaches typically model these unknown components using neural networks, resulting in black-box residual representations that are difficult to interpret physically.

This work proposes a hybrid framework that combines:

* Known physics constraints
* Symbolic residual discovery using SINDy
* Learnable residual weighting coefficients
* Adaptive loss balancing

The objective is to learn PDE dynamics while simultaneously identifying unknown physical mechanisms in an interpretable manner.

---

## 2. Research Question

Can partially known PDEs be solved more accurately and interpretably by combining:

1. Physics-informed learning,
2. Symbolic residual discovery,
3. Learnable residual weighting,
4. Adaptive loss balancing,

instead of relying solely on black-box neural residual models?

---

## 3. Known Physics Backbone

Assume that part of the governing PDE is known.

General formulation:

u_t = N(u)

where:

* u represents the system state
* N(u) represents known physical dynamics

Example: Burgers Equation

u_t = -u u_x + ν u_xx

where:

* u u_x is the nonlinear advection term
* ν u_xx is the diffusion term
* ν is the viscosity coefficient

The known PDE backbone acts as the physics prior of the framework.

---

## 4. Residual Discovery Module

Assume the true PDE is only partially known:

u_t = N(u) + R(u)

where:

* N(u) = known physics
* R(u) = unknown residual dynamics

Examples:

R(u) = 0.1u⁴

R(u) = sin(3u)

R(u) = unknown forcing term

Instead of representing R(u) using a neural network, the framework uses Sparse Identification of Nonlinear Dynamics (SINDy).

Residual computation:

R̂ = u_t − N(u)

A candidate library Θ(u) is constructed:

Θ(u) = [1, u, u², u³, u⁴, u_x, u_xx, u u_x, ...]

Sparse regression is then performed:

R̂ ≈ Θ(u)Ξ

where:

* Θ(u) is the symbolic library
* Ξ contains sparse coefficients

This produces an interpretable symbolic expression for the unknown residual dynamics.

Advantages:

* Interpretable residuals
* Physical insight
* Reduced black-box behavior
* Potential scientific discovery of missing physics

---

## 5. Learnable Alpha Module

Traditional PINN formulations typically use a fixed residual weighting parameter:

u_t = N(u) + αR(u)

where α is chosen manually.

This work proposes learning α directly from data.

Modified formulation:

u_t = N(u) + αR(u)

where:

α ∈ ℝ

is treated as a trainable parameter.

During optimization:

α ← α - η ∂L/∂α

where:

* η is the learning rate
* L is the total loss

Advantages:

* Eliminates manual hyperparameter tuning
* Automatically estimates residual importance
* Adapts to different physical regimes
* Provides interpretable coefficient estimates

Initial implementation:

Global learnable α

Future extension:

Spatial-temporal coefficient:

α(x,t)

---

## 6. Adaptive Loss Balancing

Physics-informed learning often involves multiple competing objectives.

Total loss:

L = λ_d L_data + λ_p L_physics + λ_r L_residual

where:

L_data:
Data fitting loss

L_physics:
Physics residual loss

L_residual:
Residual discovery loss

Traditional approaches require manual tuning of:

λ_d
λ_p
λ_r

This work proposes treating these weights as trainable variables.

Advantages:

* Automatic balancing of competing objectives
* Reduced sensitivity to hyperparameter choices
* Improved training stability
* Better scalability across PDE systems

Possible extensions include:

* GradNorm
* Curriculum learning
* Uncertainty-based weighting

---

## 7. Training Procedure

Step 1:

Collect trajectory data from the PDE solver.

Step 2:

Train the solution network to approximate:

u(x,t)

Step 3:

Compute known physics contribution:

N(u)

Step 4:

Compute residual:

R̂ = u_t − N(u)

Step 5:

Apply SINDy to discover symbolic residual terms.

Step 6:

Update learnable α.

Step 7:

Update adaptive loss weights.

Step 8:

Optimize the full framework jointly.

Repeat until convergence.

---

## 8. Experimental Validation

The framework will be evaluated on the Burgers equation.

Experiments:

1. Recovery of known PDE dynamics

2. Recovery of unknown coefficient α

3. Recovery of hidden residual terms

Examples:

R(u)=0.1u⁴

R(u)=sin(3u)

Evaluation metrics:

* PDE coefficient error
* Residual recovery error
* Relative L2 solution error
* Training stability

---

## 9. Expected Contributions

1. A hybrid physics-informed and symbolic-discovery framework.

2. Symbolic residual discovery using SINDy instead of purely neural residual models.

3. Learnable residual weighting coefficient α.

4. Adaptive loss balancing without manual tuning.

5. Improved interpretability for partially known PDE systems.

---

## 10. Future Work

1. Koopman Operator-based residual discovery

2. Dynamic Mode Decomposition (DMD)

3. Spatially varying alpha fields α(x,t)

4. Neural-symbolic hybrid residual models

5. Extension to higher-dimensional PDEs

6. Real-world scientific datasets
