# Mathematical Derivations

## 1. Linear Gaussian State-Space Model

### Generative Model

The standard Linear SSM is defined by:

```
State dynamics:     x_t = A * x_{t-1} + w_t,    w_t ~ N(0, Q)
Observation model:  y_t = C * x_t + v_t,         v_t ~ N(0, R)
Initial state:      x_0 ~ N(μ_0, P_0)
```

### Kalman Filter (Forward Pass)

**Predict step:**
```
μ_{t|t-1} = A * μ_{t-1|t-1}
P_{t|t-1} = A * P_{t-1|t-1} * A^T + Q
```

**Update step:**
```
Innovation:      ε_t = y_t - C * μ_{t|t-1}
Innovation cov:  S_t = C * P_{t|t-1} * C^T + R
Kalman gain:     K_t = P_{t|t-1} * C^T * S_t^{-1}
Filtered mean:   μ_{t|t} = μ_{t|t-1} + K_t * ε_t
Filtered cov:    P_{t|t} = (I - K_t * C) * P_{t|t-1}
```

**Log-likelihood contribution:**
```
log p(y_t | y_{1:t-1}) = log N(y_t; C*μ_{t|t-1}, S_t)
                        = -½ [d*log(2π) + log|S_t| + ε_t^T * S_t^{-1} * ε_t]
```

### RTS Smoother (Backward Pass)

After the forward pass, the Rauch-Tung-Striebel smoother refines estimates:

```
Smoother gain:    G_t = P_{t|t} * A^T * P_{t+1|t}^{-1}
Smoothed mean:    μ_{t|T} = μ_{t|t} + G_t * (μ_{t+1|T} - A*μ_{t|t})
Smoothed cov:     P_{t|T} = P_{t|t} + G_t * (P_{t+1|T} - P_{t+1|t}) * G_t^T
```

---

## 2. Switching State-Space Model

### Generative Model

```
Regime transition:   s_t ~ Categorical(π_{s_{t-1}})  — Markov chain
State dynamics:      x_t = A_{s_t} * x_{t-1} + w_t^{(s_t)}
Observation:         y_t = C_{s_t} * x_t + v_t^{(s_t)}
```

Each regime `k` has its own parameters `{A_k, C_k, Q_k, R_k}`.

### Exact Inference (Intractable)

Exact inference requires tracking all possible regime histories:
```
p(x_t, s_t | y_{1:t}) — exponential in t due to mixture growth
```

At time T, we would have K^T mixture components — intractable.

### GPB2 Approximation

The Generalized Pseudo-Bayesian order 2 approximation:

1. Maintain K parallel Kalman filters (one per regime)
2. Compute regime probabilities via Bayes' rule
3. **Collapse** the K^2 mixtures back to K at each step

**Predicted regime probabilities (Chapman-Kolmogorov):**
```
P(s_t = j | y_{1:t-1}) = Σ_i π_{ij} * P(s_{t-1} = i | y_{1:t-1})
```

**Update via Bayes:**
```
P(s_t = k | y_{1:t}) ∝ p(y_t | s_t=k, y_{1:t-1}) * P(s_t=k | y_{1:t-1})
```

Where the likelihood for regime k uses its Kalman filter:
```
p(y_t | s_t=k, y_{1:t-1}) = N(y_t; C_k * μ_{t|t-1}^{(k)}, S_t^{(k)})
```

**Mixture collapse (GPB1 for next step):**
```
μ_collapsed = Σ_k P(s_t=k) * μ_{t|t}^{(k)}
P_collapsed = Σ_k P(s_t=k) * [P_{t|t}^{(k)} + (μ^{(k)} - μ_c)(μ^{(k)} - μ_c)^T]
```

---

## 3. Extended Kalman Filter (Nonlinear)

For nonlinear models:
```
x_t = f(x_{t-1}) + w_t
y_t = h(x_t) + v_t
```

Linearize around current estimate:
```
F_t = ∂f/∂x |_{x = μ_{t-1|t-1}}    (Jacobian of f)
H_t = ∂h/∂x |_{x = μ_{t|t-1}}      (Jacobian of h)
```

Then apply standard Kalman equations with F_t, H_t replacing A, C.

---

## 4. Unscented Kalman Filter

Instead of Jacobians, the UKF propagates **sigma points** through the
nonlinear function.

**Sigma points (2n+1 points):**
```
χ_0 = μ
χ_i = μ + (√((n+λ)P))_i      for i = 1,...,n
χ_{n+i} = μ - (√((n+λ)P))_i  for i = 1,...,n
```

**Weights:**
```
W_0^m = λ/(n+λ)
W_i^m = 1/(2(n+λ))   for i = 1,...,2n

W_0^c = λ/(n+λ) + (1-α²+β)
W_i^c = 1/(2(n+λ))   for i = 1,...,2n
```

**Unscented Transform through f:**
```
μ_pred = Σ_i W_i^m * f(χ_i)
P_pred = Σ_i W_i^c * (f(χ_i) - μ_pred)(f(χ_i) - μ_pred)^T + Q
```

UKF is more accurate than EKF for strongly nonlinear systems and does not
require computing Jacobians analytically.

---

## 5. Viterbi Algorithm for Regime Decoding

Finds the most likely sequence of regimes:
```
s_{1:T}* = argmax_{s_{1:T}} p(s_{1:T} | y_{1:T})
```

**Log-domain recursion:**
```
δ_t(k) = max_{s_{1:t-1}} log p(y_{1:t}, s_t=k)
        = log p(y_t | s_t=k) + max_j [log π_{jk} + δ_{t-1}(j)]
```

**Backtracking:**
```
ψ_t(k) = argmax_j [log π_{jk} + δ_{t-1}(j)]
s_T* = argmax_k δ_T(k)
s_t* = ψ_{t+1}(s_{t+1}*)  for t = T-1,...,1
```
