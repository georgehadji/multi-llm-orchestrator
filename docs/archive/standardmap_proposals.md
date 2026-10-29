# Chirikov Standard Map - Scientific Projects Proposals

The **Chirikov Standard Map** (or Chirikov-Taylor map) is a fundamental, area-preserving chaotic map defined on a torus by the recurrence relations:

$$p_{n+1} = p_n + K \sin(\theta_n) \pmod{2\pi}$$
$$\theta_{n+1} = \theta_n + p_{n+1} \pmod{2\pi}$$

It serves as the paradigm for the transition from integrability to chaos in Hamiltonian systems, modeling physical setups like the kicked rotor, charged particles in magnetic fields, and plasma containment.

---

### 1. Greene’s Residue Method and the Breakdown of the Last KAM Tori
* **Scientific Sub-discipline:** Mathematical Physics / Nonlinear Dynamics
* **The Core Concept:** According to Kolmogorov-Arnold-Moser (KAM) theory, as the perturbation parameter $K$ increases, quasiperiodic orbits (invariant tori) with irrational winding numbers are deformed but survive. The "last" torus to break down is the one associated with the most irrational number, the Golden Ratio $\omega_c = \frac{\sqrt{5}-1}{2}$. 
* **Key Computational Tasks:**
  1. Approximate the Golden Ratio using Fibonacci fractions $q_n / q_{n+1}$ representing periodic orbits.
  2. Implement **Greene's Residue Method** to calculate the stability (residue $R$) of these periodic orbits.
  3. Determine the critical parameter $K_c$ where the residue of these high-order orbits transitions from $< 0.25$ (stable/regular) to $> 0.25$ (unstable/chaotic). The literature places this at $K_c \approx 0.9716354$.
  4. Measure the fractal scaling properties of the phase-space coordinates near the critical curve.
* **Visual Deliverables:** High-resolution phase portraits zooming into the critical Golden Torus fractal boundary, and a plot showing orbit residues converging to the critical value $R \approx 0.25008$.

---

### 2. Anomalous Momentum Diffusion and Accelerator Modes
* **Scientific Sub-discipline:** Statistical Mechanics / Plasma Physics
* **The Core Concept:** In the chaotic sea ($K > K_c$), a collection of particles will undergo diffusion in momentum space. For most values of $K$, this is normal diffusion: the mean squared momentum grows linearly with time, $\langle p^2 \rangle \sim D(K) \cdot t$. However, for specific narrow intervals of $K$ (e.g., near $K \approx 2\pi$), **accelerator modes** appear. These are small stable islands that transport particles linearly in momentum, leading to *superdiffusion* where $\langle p^2 \rangle \sim t^\gamma$ with $\gamma > 1$.
* **Key Computational Tasks:**
  1. Initialize an ensemble of $10^5$ particles in a small chaotic patch.
  2. Simulate their trajectories for $10^4$ steps under varying $K$ (from $1.0$ to $10.0$).
  3. Calculate the diffusion coefficient $D(K) = \lim_{t \to \infty} \frac{\langle p^2 \rangle}{t}$ and plot it. Compare it with the quasilinear analytical approximation $D_{QL} = \frac{K^2}{2}$.
  4. Identify the superdiffusive regions ($\gamma > 1$) and locate the fractal boundaries of the accelerator mode islands.
* **Visual Deliverables:** Log-log plots of $\langle p^2 \rangle$ vs. $t$ showing the transition from normal diffusion to ballistic transport ($\gamma \to 2$), and a density plot of particle positions in momentum space showing "Lévy flights."

---

### 3. Dynamical Localization in the Quantum Standard Map
* **Scientific Sub-discipline:** Quantum Chaos / Condensed Matter Physics
* **The Core Concept:** The quantum version of the standard map represents a quantum kicked rotor. Classically, when $K > K_c$, the energy (momentum squared) grows linearly forever due to chaotic diffusion. Quantum mechanically, quantum interference suppresses this classical diffusion after a characteristic "break time" $\tau^*$. The wave function in momentum space becomes exponentially localized—a phenomenon called **Dynamical Localization**, which is mathematically mapped directly to 1D Anderson Localization in disordered solids.
* **Key Computational Tasks:**
  1. Formulate the quantum evolution operator (Floquet operator) $U = e^{-i \hat{T}} e^{-i \hat{V}}$, where $\hat{T}$ is the kinetic energy operator and $\hat{V}$ is the kicking potential.
  2. Implement the split-operator method using Fast Fourier Transforms (FFTs) to alternate wave-function propagation between position and momentum representations.
  3. Simulate the time evolution of an initial ground-state wave packet.
  4. Track the growth of kinetic energy $\langle p^2 \rangle$ over time in both classical and quantum regimes, demonstrating the saturation of quantum growth.
* **Visual Deliverables:** A comparison plot of classical vs. quantum energy growth showing the quantum localization cutoff, and a semilog plot of the quantum wave-packet momentum envelope showing exponential decay ($|\psi(p)|^2 \sim e^{-|p|/\xi}$).

---

### 4. Active Control of Chaotic Transport via Directed Symmetry Breaking
* **Scientific Sub-discipline:** Nonlinear Control Theory / Nanotechnology
* **The Core Concept:** In a symmetrical standard map, particles diffuse symmetrically in both directions ($\langle p \rangle = 0$). By breaking the spatial or temporal symmetry of the kicking potential (e.g., adding a second harmonic $\sin(2\theta)$ or introducing a phase shift), one can induce a **chaotic ratchet** effect. This generates a net directed transport (current) without any net external force.
* **Key Computational Tasks:**
  1. Modify the map to: $p_{n+1} = p_n + K [\sin(\theta_n) + a \sin(2\theta_n + \phi)]$.
  2. Simulate the collective center-of-mass velocity of a chaotic ensemble of particles as a function of the symmetry-breaking parameters $a$ and $\phi$.
  3. Map out regions of parameter space that yield the maximum positive or negative particle current.
  4. Analyze how stable islands coordinate with the chaotic sea to "rectify" chaotic motion into targeted transport.
* **Visual Deliverables:** Heatmaps of net drift velocity in the $(a, \phi)$ parameter plane, and trajectories showing chaotic particles being guided into specific directional paths.

---

### 5. Symplectic Neural Networks and AI Trajectory Classification
* **Scientific Sub-discipline:** Machine Learning / Computational Physics
* **The Core Concept:** Standard maps are area-preserving, meaning they preserve the symplectic 2-form of Hamiltonian mechanics. Standard neural network simulators do not preserve this structure, causing long-term energy drift. This project trains a **Symplectic Neural Network (SympNet)** to learn the dynamics of the standard map directly from data, and uses a classification network to map out the boundary of chaos.
* **Key Computational Tasks:**
  1. Generate a dataset of trajectories of the standard map with mixed regular and chaotic states.
  2. Train a classification network (such as an LSTM or a simple Feedforward network evaluating Lyapunov exponents) to identify whether a given short sequence of points belongs to a stable island, a KAM torus, or the chaotic sea.
  3. Construct and train a Symplectic Neural Network (using volume-preserving linear and activation layers) to predict $x_{n+1}$ from $x_n$.
  4. Verify that the SympNet preserves area (Jacobian determinant = 1) over $10^5$ steps, unlike a standard multi-layer perceptron (MLP).
* **Visual Deliverables:** A neural-network-reconstructed phase portrait of the standard map, a classification map showing AI-predicted chaotic boundaries, and a comparison plot of area-preservation errors over long time steps between MLP and SympNet.
