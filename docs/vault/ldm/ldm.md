# High-Resolution Image Synthesis with Latent Diffusion Models — Vault Note

## 0) TL;DR CARD (10 lines max)

* **Citation:** Rombach, Blattmann, Lorenz, Esser, Ommer. *High-Resolution Image Synthesis with Latent Diffusion Models.* arXiv:2112.10752v2. 
* **Problem (1–2 lines):** Pixel-space diffusion is expensive because UNet compute scales with image resolution; training/sampling at high-res is slow and costly. 
* **Core idea (2–4 lines):** Learn a *perceptual* autoencoder $E,D$ that mildly compresses images to latents $z$, then train a diffusion model *in latent space* (LDM). Conditioning is added via concatenation or cross-attention.
* **Key contributions (≤3 bullets):**

  * Train diffusion in a learned latent $z$ for large speed/compute gains without big quality loss. 
  * Cross-attention conditioning for text/layout/class etc.
  * Convolutional sampling beyond training resolution for dense conditioning tasks.
* **Main results (numbers + dataset/metric):**

  * CelebA-HQ: best LDM reports FID $\approx 4.02$ (200 DDIM steps) in Table 1. 
  * ImageNet class-cond: LDM-4-G FID 3.60 (250 DDIM steps, classifier-free guidance).
  * MS-COCO text-cond eval: LDM-KL-8-G FID 12.63, IS 30.29 (250 DDIM steps).
* **What’s actually new vs prior work:** Two-stage *perceptual* compression + latent-space diffusion scaling, plus a general cross-attention conditioning interface.
* **Assumptions / scope:** Autoencoder preserves perceptual equivalence; diffusion operates on 2D latent grid; schedules typically linear with $T=1000$.
* **When it fails / limitations:** Too much compression (large $f$) harms fidelity; latent variance affects SNR and large-res “convolutional sampling” stability.
* **If you remember only 3 things:** (1) Train DM on $z=E(x)$ not pixels. (2) Train with $\ell_2$ noise-prediction loss (Eq. 2/15→1). (3) Condition via cross-attention with $Q$ from UNet, $K,V$ from conditioner.

**Implementation translation**

* Tensors: images $x\in\mathbb{R}^{B\times3\times H\times W}$, latents $z\in\mathbb{R}^{B\times C\times h\times w}$, noise $\epsilon$ same shape as $z$, timestep $t\in{1..T}$.
* Precompute: schedule arrays $(\alpha_t,\sigma_t)$ and optionally SNR$(t)$.
* Numerical issues: latent scaling (Sec. G) can change effective SNR; guidance involves gradients through decoder (can explode).

---

## 1) GLOSSARY & NOTATION (NO EXCEPTIONS)

> **Mini-explainer (what “latent” means):** A latent $z$ is a compressed representation of $x$. Example: if $x$ is $256\times256$, a latent might be $32\times32$ with more channels, keeping semantics but dropping imperceptible detail.

| Symbol                           | Meaning                                   | Shape/type                                                | Where defined (sec/eq/page)                            | Notes                                                |                                           |                 |
| -------------------------------- | ----------------------------------------- | --------------------------------------------------------- | ------------------------------------------------------ | ---------------------------------------------------- | ----------------------------------------- | --------------- |
| $x$                              | RGB image                                 | $\mathbb{R}^{H\times W\times 3}$ (or $3\times H\times W$) | Sec. 3.1, PDF p.4                                      | Data sample                                          |                                           |                 |
| $p(x)$                           | data distribution / model likelihood      | distribution                                              | Sec. 3.2 (concept), PDF p.4–5                          | Goal: learn                                          |                                           |                 |
| $E$                              | encoder                                   | NN                                                        | Sec. 3.1, PDF p.4                                      | $z=E(x)$                                             |                                           |                 |
| $D$                              | decoder                                   | NN                                                        | Sec. 3.1, PDF p.4                                      | $\tilde x=D(z)$                                      |                                           |                 |
| $\tilde x$                       | reconstruction                            | same as $x$                                               | Sec. 3.1, PDF p.4                                      | $\tilde x=D(E(x))$                                   |                                           |                 |
| $z$                              | latent representation                     | $\mathbb{R}^{h\times w\times c}$                          | Sec. 3.1, PDF p.4                                      | Often use BCHW in code                               |                                           |                 |
| $f$                              | downsampling factor                       | integer                                                   | Sec. 3.1, PDF p.4                                      | $f=H/h=W/w$                                          |                                           |                 |
| KL-reg                           | VAE-like latent regularization            | method                                                    | Sec. 3.1, PDF p.4; Sec. G, PDF p.29                    | tiny KL weight $\sim10^{-6}$                         |                                           |                 |
| VQ-reg                           | vector quantization latent reg            | method                                                    | Sec. 3.1, PDF p.4; Sec. G, PDF p.29                    | codebook size $                                      | Z                                         | $               |
| $                                | Z                                         | $                                                         | VQ codebook size                                       | integer                                              | Tabs 12–15, PDF p.24–25; Sec. G, PDF p.29 | Hyperparam      |
| $D_\psi$                         | discriminator for autoencoder training    | NN                                                        | Sec. G, Eq. (25), PDF p.29                             | patch-based                                          |                                           |                 |
| $\mathcal{L}_\text{Autoencoder}$ | AE objective                              | scalar loss                                               | Sec. G, Eq. (25), PDF p.29                             | min-max                                              |                                           |                 |
| $\mathcal{L}_\text{rec}$         | reconstruction loss                       | scalar                                                    | Eq. (25), PDF p.29                                     | paper doesn’t expand fully                           |                                           |                 |
| $\mathcal{L}_\text{adv}$         | adversarial term                          | scalar                                                    | Eq. (25), PDF p.29                                     | patch realism                                        |                                           |                 |
| $\mathcal{L}_\text{reg}$         | latent regularizer                        | scalar                                                    | Eq. (25), PDF p.29                                     | KL or VQ                                             |                                           |                 |
| $T$                              | diffusion steps                           | integer (often 1000)                                      | Tabs 12–15, PDF p.24–25                                | Fixed                                                |                                           |                 |
| $t$                              | timestep index                            | integer in ${1,\dots,T}$                                  | Eq. (1)/(2)/(3), Sec. 3.2–3.3, PDF p.5                 | Uniformly sampled in training                        |                                           |                 |
| $x_0$                            | clean data sample in diffusion derivation | tensor                                                    | Appx B, Eq. (4)/(9), PDF p.16–17                       | For pixels                                           |                                           |                 |
| $x_t$                            | noisy version of $x_0$                    | tensor                                                    | Appx B, Eq. (4), PDF p.16                              | Forward diffusion                                    |                                           |                 |
| $q(\cdot)$                       | forward (noising) process                 | distribution                                              | Appx B, Eq. (4)–(7), PDF p.16                          | Fixed                                                |                                           |                 |
| $p(\cdot)$                       | reverse (generative) process              | distribution                                              | Appx B, Eq. (8), PDF p.16                              | Learned via parameters                               |                                           |                 |
| $\alpha_t,\sigma_t$              | signal/noise coefficients                 | scalars (per t)                                           | Appx B, Eq. (4), PDF p.16                              | Often $\alpha_t^2+\sigma_t^2=1$ in DDPM-like setups  |                                           |                 |
| $\text{SNR}(t)$                  | signal-to-noise ratio                     | scalar                                                    | Appx B, before Eq. (4), PDF p.16                       | $\text{SNR}(t)=\alpha_t^2/\sigma_t^2$                |                                           |                 |
| $\alpha_{t                       | s}$                                       | conditional signal coeff                                  | scalar                                                 | Appx B, Eq. (6), PDF p.16                            | $\alpha_t/\alpha_s$                       |                 |
| $\sigma^2_{t                     | s}$                                       | conditional noise variance                                | scalar                                                 | Appx B, Eq. (7), PDF p.16                            | $\sigma_t^2-\alpha_{t                     | s}^2\sigma_s^2$ |
| $x_\theta(x_t,t)$                | predicted clean sample                    | tensor                                                    | Appx B, Eq. (10)–(13), PDF p.16–17                     | “denoised” estimate                                  |                                           |                 |
| $\epsilon$                       | standard Gaussian noise                   | tensor                                                    | Eq. (1)/(2)/(15), PDF p.5 and p.17                     | $\epsilon\sim\mathcal{N}(0,I)$                       |                                           |                 |
| $\epsilon_\theta(\cdot)$         | NN noise predictor                        | tensor                                                    | Eq. (1)/(2)/(15), PDF p.5 and p.17                     | Implemented by UNet backbone                         |                                           |                 |
| $\mu_\theta(x_t,t)$              | reverse mean                              | tensor                                                    | Appx B, Eq. (12), PDF p.16–17                          | used for sampling                                    |                                           |                 |
| $y$                              | conditioning input                        | varies                                                    | Sec. 3.3, PDF p.5                                      | text, layout, class, etc.                            |                                           |                 |
| $\tau_\theta$                    | conditioning encoder                      | NN                                                        | Sec. 3.3, PDF p.5; Appx E.2.1, Eq. (18)–(24), PDF p.25 | transformer etc.                                     |                                           |                 |
| $\phi_i(z_t)$                    | UNet intermediate features                | tensor                                                    | Sec. 3.3, PDF p.5                                      | flattened to $N\times d^i_\epsilon$                  |                                           |                 |
| $Q,K,V$                          | attention query/key/value                 | tensors                                                   | Sec. 3.3, PDF p.5                                      | $Q$ from UNet, $K,V$ from $\tau_\theta(y)$           |                                           |                 |
| $W_Q^{(i)},W_K^{(i)},W_V^{(i)}$  | attention projections                     | matrices                                                  | Sec. 3.3, PDF p.5                                      | learnable                                            |                                           |                 |
| $\zeta$                          | token embeddings from $\tau_\theta$       | $\mathbb{R}^{M\times d_\tau}$                             | Appx E.2.1, Eq. (18), PDF p.25                         | used as $K,V$                                        |                                           |                 |
| $\hat\mu$                        | guided mean (post-hoc)                    | tensor                                                    | Appx C, Eq. (16) (concept), PDF p.18–19                | via guidance gradients                               |                                           |                 |
| $p_\Phi(y                        | z_t)$                                     | guider distribution                                       | distribution                                           | Appx C, Eq. (16)–(17), PDF p.18–19                   | classifier or regression guider           |                 |
| $s$                              | guidance scale                            | scalar                                                    | Tables 2/3/10/fig captions, PDF p.7 + p.36–38          | used for classifier-free guidance (paper cites [32]) |                                           |                 |

**Implementation translation**

* Shapes (PyTorch BCHW):

  * $x$: `[B,3,H,W]`, $z$: `[B,C,h,w]` where `h=H/f`, `w=W/f`.
  * $\epsilon$ same as $z$ (latent diffusion) or $x$ (pixel diffusion).
  * $\tau_\theta(y)=\zeta$: `[B,M,d_tau]`; UNet feature flatten: `phi_i`: `[B, N=h_i*w_i, d_i]`.
* Precompute: attention projections per block; schedule buffers for $\alpha_t,\sigma_t$, and $\alpha_{t|s},\sigma_{t|s}$ if you implement exact Eq. (12).
* Numerical: attention softmax can overflow (use fp32 softmax); latent rescaling affects training stability.

---

## 2) PROBLEM SETUP

### What is the data distribution and modeling goal?

* Data: images $x\sim p_\text{data}(x)$ in RGB space. Goal: learn a generative model producing realistic samples from (approximately) $p_\text{data}$.
* Core bottleneck: diffusion in pixel space requires UNet evaluations over all pixels at all steps; high resolution $\Rightarrow$ expensive. 

### Generative story (random variables + dependencies)

Two-stage story:

1. **Perceptual autoencoder (fixed after training)**

* $z = E(x)$, $\tilde x = D(z)$, with $z$ lower-dimensional but “perceptually equivalent.” Sec. 3.1, PDF p.4 

2. **Latent diffusion prior over $z$**

* Define a diffusion model over latents (or over pixels in the derivation appendix):

  * Forward: $q(z_t|z_0)$ Gaussian noising (latent analogue of Appx B Eq. (4)).
  * Reverse: $p_\theta(z_{t-1}|z_t)$ parameterized by UNet.
* Sample: draw $z_T\sim \mathcal{N}(0,I)$, remember $p(z_0)=\int p(z_T)\prod_t p(z_{t-1}|z_t),dz_{1:T}$ (Appx B Eq. (8) for $x$, same structure for $z$). 
* Decode once: $x=D(z_0)$. 

### What is assumed known / fixed vs learned?

* Fixed: forward schedule $(\alpha_t,\sigma_t)$ and $T$; noising process $q$.
* Learned: autoencoder params (during stage 1); diffusion UNet $\epsilon_\theta(\cdot)$ (stage 2); optionally conditioner $\tau_\theta$ (jointly trained for conditional LDMs via Eq. (3)).

**Mini-explainer (posterior):** In Bayes, a posterior is “what you believe about an earlier variable given a later one.” Example: $q(x_{t-1}|x_t,x_0)$ is the true reverse distribution if you knew the clean $x_0$.

**Implementation translation**

* Stage 1 trainables: $E,D$ (+ discriminator $D_\psi$ if adversarial).
* Stage 2 trainables: UNet $\epsilon_\theta$ (+ conditioner $\tau_\theta$ for conditional tasks).
* Fixed artifacts to store: trained $E,D$ checkpoints; diffusion schedule buffers; latent scaling stats $\hat\sigma$ (Sec. G). 

---

## 3) THE METHOD (PLAIN ENGLISH FIRST)

1. Train an autoencoder $E,D$ that reconstructs images well *perceptually* (not just pixel-wise), using perceptual + patch adversarial losses and a very small latent regularizer (KL or VQ).
2. Freeze that autoencoder and map each training image $x$ to a latent $z=E(x)$ of spatial size $(H/f)\times(W/f)$. 
3. Define a Gaussian diffusion process in latent space: for a random timestep $t$, corrupt $z$ to $z_t$ by mixing it with Gaussian noise. (Same math as Appx B Eq. (4), but applied to $z$.) 
4. Train a time-conditional UNet $\epsilon_\theta$ to predict the noise you added, using a simple MSE loss. This is Eq. (2) for latent diffusion and comes from the ELBO derivation in Appx B Eq. (15) + reweighting.
5. For conditional tasks, feed a conditioning representation $\tau_\theta(y)$ into the UNet either by (a) concatenating spatial maps to the UNet input, or (b) using cross-attention where UNet features query the conditioning tokens (Sec. 3.3).
6. At sampling time, start from pure noise $z_T\sim\mathcal{N}(0,I)$ and iteratively denoise using the learned reverse transitions. 
7. After reaching $z_0$, decode a single time with $D$ to get an image. 
8. For large images in dense-conditioning tasks, apply the same UNet convolutionally on larger latent grids (“convolutional sampling”), sometimes stabilized by latent rescaling or guidance.
9. The “one trick”: move diffusion from pixel space to a learned *mildly compressed* latent where computation scales with $(H/f)\times(W/f)$, yet reconstructions stay high-fidelity.

**Mini-explainer (ELBO):** The ELBO is a lower bound on log-likelihood used when exact likelihood is hard. Example: VAEs maximize ELBO instead of $\log p(x)$ directly; DDPMs derive a training objective from an ELBO decomposition (Appx B Eq. (9)). 

**Implementation translation**

* Training forward pass: `z = E(x)` then `z_t = alpha[t]*z + sigma[t]*eps` then `eps_hat = UNet(z_t, t, cond)` then `loss = mse(eps_hat, eps)`.
* Sampling: loop `t=T..1` to update `z_{t-1}`; decode once at the end.
* Numerical: sampling needs careful schedule + variance choice; attention conditioning needs stable LayerNorm + fp32 softmax.

---

## 4) MAIN EQUATIONS (THE CANONICAL SET)

Below is the **minimal sufficient set** to re-derive + implement LDMs.

### Eq. (25) — Autoencoder training objective (stage 1)

From Sec. G, PDF p.29: 
$$
\mathcal{L}*\text{Autoencoder}=\min*{E,D}\max_{\psi}\Big(
\mathcal{L}*\text{rec}(x,D(E(x))) - \mathcal{L}*\text{adv}(D(E(x))) + \log D_\psi(x) + \mathcal{L}_\text{reg}(x;E,D)
\Big).
\tag{25}
$$

* **Terms:** reconstruction $\mathcal{L}*\text{rec}$, adversarial term $\mathcal{L}*\text{adv}$, discriminator log-likelihood $\log D_\psi(x)$, latent regularizer $\mathcal{L}_\text{reg}$.
* **Why it matters:** defines the perceptual latent space where diffusion will operate; small $\mathcal{L}_\text{reg}$ prevents exploding latent variance.
* **Used next:** yields fixed $E,D$ and latent scaling procedure (Sec. G) for diffusion training.

> **paper jump:** The paper does not expand exact formulas for $\mathcal{L}*\text{rec}$ and $\mathcal{L}*\text{adv}$ here; it references prior work [23]. In practice (common VQGAN-style), $\mathcal{L}*\text{rec}$ is a weighted mix of $L_1$ + perceptual (LPIPS), and $\mathcal{L}*\text{adv}$ is a hinge GAN loss. Treat exact weights as hyperparams.

---

### Eq. (2) — Latent diffusion training objective (stage 2, unconditional)

Sec. 3.2, PDF p.5: 
$$
\mathcal{L}*\text{LDM}:=\mathbb{E}*{E(x),,\epsilon\sim\mathcal{N}(0,1),,t}\Big[|\epsilon-\epsilon_\theta(z_t,t)|_2^2\Big].
\tag{2}
$$

* **Terms:** $z_0=E(x)$; sample $t$ uniformly; create $z_t$ by forward diffusion; predict noise $\epsilon_\theta$.
* **Why it matters:** this is the actual loss you implement in PyTorch.
* **Used next:** combined with conditioning (Eq. (3)) and sampling formulas (Appx B).

---

### Eq. (3) — Conditional latent diffusion objective

Sec. 3.3, PDF p.5: 
$$
\mathcal{L}*\text{LDM}:=\mathbb{E}*{E(x),,y,,\epsilon\sim\mathcal{N}(0,1),,t}\Big[|\epsilon-\epsilon_\theta(z_t,t,\tau_\theta(y))|_2^2\Big].
\tag{3}
$$

* **Terms:** conditioning input $y$, encoded into tokens/features $\tau_\theta(y)$.
* **Why it matters:** same training, but with control signals.
* **Used next:** cross-attention mechanism (below) + conditional sampling.

---

### Appx B Eq. (4)–(7) — Forward diffusion / schedule definition

Appx B, PDF p.16: 
$$
q(x_t|x_0)=\mathcal{N}(x_t,|,\alpha_t x_0,\sigma_t^2 I).
\tag{4}
$$
For $s<t$:
$$
q(x_t|x_s)=\mathcal{N}(x_t,|,\alpha_{t|s}x_s,\sigma_{t|s}^2 I),
\tag{5}
$$
$$
\alpha_{t|s}=\frac{\alpha_t}{\alpha_s},
\tag{6}
$$
$$
\sigma_{t|s}^2=\sigma_t^2-\alpha_{t|s}^2\sigma_s^2.
\tag{7}
$$

* **Terms:** $(\alpha_t,\sigma_t)$ are schedule parameters; SNR$(t)=\alpha_t^2/\sigma_t^2$.
* **Why it matters:** lets you compute $x_t$ (or $z_t$) in one shot during training.

> **Mini-explainer (schedule):** A schedule tells you how much signal/noise at each timestep. Example: if $\alpha_t=0.7,\sigma_t=0.714$, then SNR$(t)=0.49/0.51\approx0.96$ (about equal signal and noise).

---

### Appx B Eq. (8)–(12) — Reverse model parameterization

Appx B, PDF p.16–17:
Generative chain:
$$
p(x_0)=\int p(x_T)\prod_{t=1}^T p(x_{t-1}|x_t),dx_{1:T}.
\tag{8}
$$
ELBO:
$$
-\log p(x_0)\le \mathrm{KL}(q(x_T|x_0)|p(x_T))+\sum_{t=1}^T \mathbb{E}*{q(x_t|x_0)}\mathrm{KL}\big(q(x*{t-1}|x_t,x_0)|p(x_{t-1}|x_t)\big).
\tag{9}
$$
Parameterize reverse step by plugging in estimate $x_\theta$:
$$
p(x_{t-1}|x_t):=q(x_{t-1}|x_t,x_\theta(x_t,t)).
\tag{10}
$$
Then (Gaussian):
$$
p(x_{t-1}|x_t)=\mathcal{N}\Big(x_{t-1},\big|,\mu_\theta(x_t,t),;\frac{\sigma_{t|t-1}^2\sigma_{t-1}^2}{\sigma_t^2}I\Big),
\tag{11}
$$
with mean
$$
\mu_\theta(x_t,t)=\frac{\alpha_{t|t-1}\sigma_{t-1}^2}{\sigma_t^2}x_t+\frac{\alpha_{t-1}\sigma_{t|t-1}^2}{\sigma_t^2}x_\theta(x_t,t).
\tag{12}
$$

* **Why it matters:** Eq. (12) gives the sampling update if you implement exact DDPM-style sampling with your $(\alpha,\sigma)$ parametrization.

---

### Appx B Eq. (13)–(15) — From ELBO term to denoising MSE

Appx B, PDF p.17: 
$$
\sum_{t=1}^T\mathbb{E}*{q(x_t|x_0)}\mathrm{KL}(\cdot)
=\sum*{t=1}^T\mathbb{E}*{\epsilon\sim\mathcal{N}(0,I)}\frac{1}{2}\big(\mathrm{SNR}(t!-!1)-\mathrm{SNR}(t)\big),|x_0-x*\theta(\alpha_t x_0+\sigma_t\epsilon,t)|*2^2.
\tag{13}
$$
Reparameterization:
$$
\epsilon*\theta(x_t,t)=\frac{x_t-\alpha_t x_\theta(x_t,t)}{\sigma_t}.
\tag{14}
$$
Then:
$$
|x_0-x_\theta(\alpha_t x_0+\sigma_t\epsilon,t)|*2^2
=\frac{\sigma_t^2}{\alpha_t^2}|\epsilon-\epsilon*\theta(\alpha_t x_0+\sigma_t\epsilon,t)|_2^2.
\tag{15}
$$

* **Why it matters:** Eq. (15) is the algebraic bridge that justifies training $\epsilon_\theta$ with MSE.

---

### Sec. 3.3 — Cross-attention conditioning equations

Sec. 3.3, PDF p.5:
Attention:
$$
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\Big(\frac{QK^\top}{\sqrt{d}}\Big)V,
$$
with
$$
Q=W_Q^{(i)}\phi_i(z_t),\qquad K=W_K^{(i)}\tau_\theta(y),\qquad V=W_V^{(i)}\tau_\theta(y).
$$

* **Terms:** $\phi_i(z_t)$ is UNet feature map at layer $i$ (flattened); $\tau_\theta(y)\in\mathbb{R}^{M\times d_\tau}$ are condition tokens.
* **Why it matters:** this is the *general conditioning interface* used for text/layout/class.

> **Mini-explainer (cross-attention):** Queries ask “what do I need?”, keys/values provide “what’s available.” Example: image feature at a spatial location queries text tokens, pulling relevant token values into that location’s feature.

---

### Appx C Eq. (16)–(17) — Post-hoc guidance (score correction)

Appx C, PDF p.18–19: 
$$
\hat\epsilon \leftarrow \epsilon_\theta(z_t,t)+\sqrt{1-\alpha_t^2},\nabla_{z_t}\log p_\Phi(y|z_t).
\tag{16}
$$
Example Gaussian guider:
$$
\log p_\Phi(y|z_t)=-\frac{1}{2}|y-T(D(z_0(z_t)))|_2^2.
\tag{17}
$$

* **Why it matters:** lets you steer unconditional models at test-time (e.g., for upsampling via downsample consistency).

---

### Equation dependency map (minimal)

* Eq. (25) $\rightarrow$ train $E,D$ $\rightarrow$ define latents $z=E(x)$ (Sec. 3.1)
* Appx B Eq. (4)–(7) $\rightarrow$ construct $z_t$ from $z_0$ during training
* Appx B Eq. (15) $\rightarrow$ simple noise-prediction MSE $\rightarrow$ Eq. (1)/(2)/(3)
* Eq. (3) + cross-attention $\rightarrow$ conditional training
* Appx B Eq. (10)–(12) $\rightarrow$ sampling update (mean/variance)
* Appx C Eq. (16)–(17) $\rightarrow$ optional guidance during sampling

**Implementation translation**

* You can implement training using only Eq. (2)/(3) plus the “one-shot” forward diffusion $z_t=\alpha_t z_0+\sigma_t\epsilon$ (latent analogue of Eq. (4)).
* Exact sampling uses Eq. (12) plus $x_\theta$ derived from $\epsilon_\theta$ (derivation in Sec. 5 below).
* Guidance uses gradients through $D$ and transformation $T$; use gradient scaling/clipping.

---

## 5) DERIVATION MAP (NO BIG JUMPS)

### Goal

Show: assumptions $\rightarrow$ ELBO $\rightarrow$ weighted reconstruction MSE $\rightarrow$ noise-prediction loss $\rightarrow$ Eq. (1)/(2)/(3).

> **Mini-explainer (why denoising = learning):** If you can predict the noise added at each step, you can reverse highlighting what the clean sample must have been. Example: if $x_t=x_0+0.1\epsilon$ and you predict $\epsilon$, you can recover $x_0\approx x_t-0.1\epsilon$.

---

### Step chain

**Step 1 (Appx B Eq. (4), PDF p.16): define forward noising** 
Start:
$$
q(x_t|x_0)=\mathcal{N}(x_t|\alpha_t x_0,\sigma_t^2 I).
$$

* Identity used: Gaussian affine transform.

**Step 2 (Appx B Eq. (8)–(9), PDF p.16–17): ELBO decomposition** 
Start:
$$
-\log p(x_0)\le \mathrm{KL}(q(x_T|x_0)|p(x_T))+\sum_{t=1}^T \mathbb{E}*{q(x_t|x_0)}\mathrm{KL}(q(x*{t-1}|x_t,x_0)|p(x_{t-1}|x_t)).
$$

* Identity used: standard variational bound for latent-variable models (ELBO).

**Step 3 (Appx B Eq. (10), PDF p.16–17): choose reverse parameterization** 
Starting expression: $p(x_{t-1}|x_t)$ is unknown.
Identity used: approximate true posterior by plugging in estimate of $x_0$.
Result:
$$
p(x_{t-1}|x_t):=q(x_{t-1}|x_t,x_\theta(x_t,t)).
$$
What changed/why: replace dependence on unknown $x_0$ with NN prediction $x_\theta$.

**Step 4 (Appx B Eq. (11)–(12), PDF p.16–17): get Gaussian form** 
Starting expression: $q(x_{t-1}|x_t,x_0)$ is Gaussian under the forward process.
Identity used: conditional of joint Gaussian.
Result:
$$
p(x_{t-1}|x_t)=\mathcal{N}\Big(x_{t-1}\big|\mu_\theta(x_t,t),\frac{\sigma_{t|t-1}^2\sigma_{t-1}^2}{\sigma_t^2}I\Big)
$$
with $\mu_\theta$ as Eq. (12).
What changed/why: turns reverse step into mean+variance computation.

**Step 5 (Appx B Eq. (13), PDF p.17): KL terms become weighted squared error** 
Starting expression:
$$
\mathbb{E}*{q(x_t|x_0)}\mathrm{KL}(q(x*{t-1}|x_t,x_0)|p(x_{t-1}|x_t)).
$$
Identity used: KL between Gaussians with same covariance reduces to squared difference of means (plus constants).
Result (paper’s simplification):
$$
\sum_{t=1}^T\mathbb{E}*{\epsilon}\frac{1}{2}(\mathrm{SNR}(t-1)-\mathrm{SNR}(t))|x_0-x*\theta(\alpha_t x_0+\sigma_t\epsilon,t)|_2^2.
\tag{13}
$$
What changed/why: objective is now a weighted denoising reconstruction loss.

> **paper jump:** The paper states Eq. (13) directly. If you want the missing algebra:
>
> * Write both Gaussians $q(x_{t-1}|x_t,x_0)$ and $q(x_{t-1}|x_t,x_\theta)$ with identical covariance (from Eq. (11)).
> * KL$(\mathcal{N}(\mu_1,\Sigma)|\mathcal{N}(\mu_2,\Sigma))=\frac{1}{2}(\mu_2-\mu_1)^\top\Sigma^{-1}(\mu_2-\mu_1)$.
> * Express $\mu_1-\mu_2$ as a scalar multiple of $(x_0-x_\theta)$; the scalar becomes $(\mathrm{SNR}(t-1)-\mathrm{SNR}(t))$.
>   This is standard DDPM algebra but not expanded here. (Call it “DDPM posterior-KL trick.”) 

**Step 6 (Appx B Eq. (14), PDF p.17): reparameterize the model to predict noise** 
Starting expression: predicted clean sample $x_\theta(x_t,t)$.
Identity used: solve Eq. (14) for $x_\theta$ or express error in $\epsilon$-space.
Define:
$$
\epsilon_\theta(x_t,t)=\frac{x_t-\alpha_t x_\theta(x_t,t)}{\sigma_t}.
\tag{14}
$$
What changed/why: predicting $\epsilon$ is often numerically easier and matches score-matching interpretation.

**Step 7 (Appx B Eq. (15), PDF p.17): convert reconstruction MSE to noise MSE** 
We now fill in the algebra carefully.

* **Starting expression:** with $x_t=\alpha_t x_0+\sigma_t\epsilon$,
  $$
  |x_0-x_\theta(x_t,t)|_2^2.
  $$

* **Identity used:** from Eq. (14),
  $$
  x_t-\alpha_t x_\theta(x_t,t)=\sigma_t \epsilon_\theta(x_t,t)
  \Rightarrow
  x_\theta(x_t,t)=\frac{x_t-\sigma_t\epsilon_\theta(x_t,t)}{\alpha_t}.
  $$

* **Substitute into starting expression:**
  $$
  x_0-x_\theta(x_t,t)=x_0-\frac{x_t-\sigma_t\epsilon_\theta(x_t,t)}{\alpha_t}
  =\frac{\alpha_t x_0-x_t+\sigma_t\epsilon_\theta(x_t,t)}{\alpha_t}.
  $$

* **Replace $x_t=\alpha_t x_0+\sigma_t\epsilon$:**
  $$
  \alpha_t x_0-x_t+\sigma_t\epsilon_\theta
  =\alpha_t x_0-(\alpha_t x_0+\sigma_t\epsilon)+\sigma_t\epsilon_\theta
  =\sigma_t(\epsilon_\theta-\epsilon).
  $$

* **Resulting expression:**
  $$
  x_0-x_\theta(x_t,t)=\frac{\sigma_t}{\alpha_t}(\epsilon_\theta(x_t,t)-\epsilon)
  \Rightarrow
  |x_0-x_\theta(x_t,t)|*2^2=\frac{\sigma_t^2}{\alpha_t^2}|\epsilon-\epsilon*\theta(x_t,t)|_2^2,
  $$
  which is Eq. (15).

* **What changed/why:** the loss is now directly on predicted noise.

**Step 8 (Appx B text after Eq. (15) → Eq. (1), PDF p.17 and Sec. 3.2 PDF p.5): reweight to equal weights**
Starting expression: remember Eq. (13) has weight $\frac{1}{2}(\mathrm{SNR}(t-1)-\mathrm{SNR}(t))\cdot\frac{\sigma_t^2}{\alpha_t^2}$ multiplying $|\epsilon-\epsilon_\theta|^2$.
Identity used: choose a “reweighted variant” where each timestep contributes equally (paper cites [30]).
Result:
$$
\mathcal{L}*\text{DM}=\mathbb{E}*{x,\epsilon,t}|\epsilon-\epsilon_\theta(x_t,t)|_2^2,
\tag{1}
$$
and in latent space (Eq. (2)).
What changed/why: optimization focuses on perceptually relevant bits and stabilizes training; weights are absorbed/ignored.

**Implementation translation**

* Core derivation implies you only need to implement Eq. (2)/(3) but sampling needs converting $\epsilon_\theta$ back to $x_\theta$ (next section).
* Precompute: $\alpha_t,\sigma_t$ arrays; store as buffers for speed.
* Numerical: division by $\alpha_t$ (near 0 at late steps) can be unstable; clamp or use fp32 for those steps. 

---

## 6) OBJECTIVE / LOSS (FINAL FORM + INTERPRETATION)

### Final training loss actually used

Unconditional latent diffusion (Sec. 3.2, Eq. (2), PDF p.5): 
$$
\mathcal{L}*\text{LDM}=\mathbb{E}*{z_0=E(x),,\epsilon\sim\mathcal{N}(0,I),,t}|\epsilon-\epsilon_\theta(z_t,t)|_2^2.
\tag{2}
$$

Conditional LDM (Sec. 3.3, Eq. (3), PDF p.5): 
$$
\mathcal{L}*\text{LDM}=\mathbb{E}*{z_0=E(x),,y,,\epsilon,,t}|\epsilon-\epsilon_\theta(z_t,t,\tau_\theta(y))|_2^2.
\tag{3}
$$

### Interpretation of the loss

* $\epsilon$ is what you *actually added* in the forward process. The network learns to undo the corruption.
* Minimizing MSE on $\epsilon$ implies (via Eq. (15)) minimizing a weighted reconstruction error of the clean sample, which corresponds to minimizing ELBO terms (Appx B Eq. (13)–(15)). 

**Mini-explainer (score):** The score is $\nabla_x \log p(x)$, pointing toward higher-probability regions. Diffusion models can be trained to approximate score-like quantities; noise prediction is closely related.

### Weighted vs unweighted (“reweighted bound”)

* The strict ELBO-derived objective weights timesteps by SNR differences (Eq. (13)). 
* The paper uses an equally weighted variant (after Eq. (15)) to obtain Eq. (1)/(2).
* **Tradeoff intuition:**

  * Weighted can emphasize certain timesteps (often early/late depending on schedule).
  * Unweighted is simpler, common, and empirically strong.

**Implementation translation**

* You implement unweighted MSE; optionally add timestep weighting if experimenting.
* Needed tensors: `eps` sampled, `eps_hat` predicted.
* Numerical: MSE in fp16 can underflow for small noise; keep loss accumulation in fp32.

---

## 7) ALGORITHMS (TRAINING + SAMPLING)

### Pseudocode — Training loop (unconditional LDM)

**Inputs:** dataset of images; pretrained $E,D$; schedule $(\alpha_t,\sigma_t)$; steps $T$
**Outputs:** trained $\epsilon_\theta$

```
for each minibatch of images x:
    z0 = E(x)                          # latent encode
    if KL-reg and using rescaling:
        z0 = z0 / sigma_hat            # Sec. G (latent std)  (paper)
    t ~ Uniform({1,...,T})
    eps ~ N(0, I)
    zt = alpha[t] * z0 + sigma[t] * eps
    eps_hat = UNet_eps(z_t=zt, t=t, cond=None)
    loss = ||eps - eps_hat||^2
    update theta to minimize loss
```

Anchors: Eq. (2) (Sec. 3.2, PDF p.5) and forward diffusion (Appx B Eq. (4), PDF p.16).

### Pseudocode — Training loop (conditional LDM)

Same, but:

* compute $\zeta=\tau_\theta(y)$
* feed cond via concat or cross-attn
* optimize both $\theta$ (UNet) and conditioner params (Eq. (3)).

---

### Sampling / inference procedure (DDPM-style using Eq. (12))

You need a way to compute $x_\theta(x_t,t)$ from $\epsilon_\theta(x_t,t)$.

**Derivation (not explicitly written as a numbered equation; derived from Appx B Eq. (14), PDF p.17)** 

* Starting expression (Eq. (14)):
  $$
  \epsilon_\theta(x_t,t)=\frac{x_t-\alpha_t x_\theta(x_t,t)}{\sigma_t}.
  $$
* Identity used: solve for $x_\theta$.
* Result:
  $$
  x_\theta(x_t,t)=\frac{x_t-\sigma_t\epsilon_\theta(x_t,t)}{\alpha_t}.
  \quad\text{(derived from Eq. (14))}
  $$
* What changed/why: converts noise prediction into clean-sample prediction required by Eq. (12).

Now the sampling step (Appx B Eq. (11)–(12), PDF p.16–17): 

* Given $x_t$ (or $z_t$), compute $\mu_\theta(x_t,t)$ using Eq. (12).
* Sample:
  $$
  x_{t-1}\sim\mathcal{N}\Big(\mu_\theta(x_t,t),\frac{\sigma_{t|t-1}^2\sigma_{t-1}^2}{\sigma_t^2}I\Big).
  $$

**Pseudocode (latent space)**

```
zT ~ N(0, I)
z = zT
for t = T ... 1:
    eps_hat = UNet_eps(z, t, cond)
    z0_hat = (z - sigma[t]*eps_hat)/alpha[t]     # from Eq. (14)
    mu = (alpha[t|t-1]*sigma[t-1]^2/sigma[t]^2)*z + (alpha[t-1]*sigma[t|t-1]^2/sigma[t]^2)*z0_hat   # Eq. (12)
    var = (sigma[t|t-1]^2 * sigma[t-1]^2 / sigma[t]^2)
    z = mu + sqrt(var) * N(0, I)
x = D(z)
```

> **paper jump:** The main paper reports using DDIM sampling steps in evaluations (Tables mention “DDIM steps”), but does not provide DDIM update equations; it cites [84] for DDIM.
> If you want DDIM: use the standard deterministic update with an $\eta$-controlled noise term; treat it as an external sampler module.

---

### Minimal PyTorch-like skeleton (structure only)

```python
class Autoencoder(nn.Module):
    def __init__(self): ...
    def encode(self, x): return z   # [B,C,h,w]
    def decode(self, z): return x   # [B,3,H,W]

class Conditioner(nn.Module):
    def forward(self, y): return zeta  # [B,M,d_tau]

class UNetEps(nn.Module):
    def forward(self, zt, t, cond=None): 
        # if cond: apply cross-attn blocks or concat input channels
        return eps_hat  # same shape as zt

def train_step(x, y=None):
    z0 = ae.encode(x)
    if use_rescale:
        z0 = z0 / sigma_hat
    t = randint(1, T+1, (B,))
    eps = torch.randn_like(z0)
    zt = alpha[t]*z0 + sigma[t]*eps
    cond = conditioner(y) if y is not None else None
    eps_hat = unet(zt, t, cond)
    loss = ((eps - eps_hat)**2).mean()
    loss.backward()
    opt.step(); opt.zero_grad()
```

**Implementation translation**

* Randomness enters via $\epsilon\sim\mathcal{N}(0,I)$ and timestep sampling $t\sim\text{Uniform}({1..T})$ (Eq. (2)/(3)).
* Hyperparams explicitly listed include: $T=1000$, “linear” noise schedule, UNet channels/depth/attention resolutions, batch size, iterations, learning rate (Tabs 12–15).
* Numerical: index-based buffers for alpha/sigma must broadcast to `[B,1,1,1]`.

---

## 8) DESIGN CHOICES & ABLATIONS

| Choice                    | Options tried                                   | What changed                               | Effect on results                                                     | My takeaway                                      |
| ------------------------- | ----------------------------------------------- | ------------------------------------------ | --------------------------------------------------------------------- | ------------------------------------------------ |
| Compression factor $f$    | $f\in{1,2,4,8,16,32}$                           | latent spatial size $h=H/f$                | Too large $f$ hurts quality; mid ($4$–$8$) strong.                    | Pick mild compression; don’t over-compress       |
| Latent regularization     | KL-reg vs VQ-reg                                | continuous vs codebook                     | VQ sometimes improves sample quality even if recon slightly worse.    | VQ can help diffusion prior learning             |
| Latent rescaling          | w/ vs w/o rescaling                             | $z\leftarrow z/\hat\sigma$                 | stabilizes “convolutional sampling”; reduces induced SNR.             | Always check latent variance; store stats        |
| Conditioning injection    | concat vs cross-attn                            | spatial concat vs token attention          | Cross-attn enables general token conditioning (text/layout/class).    | Use concat for dense maps, cross-attn for tokens |
| Conditioner $\tau_\theta$ | transformer depth/width                         | $\zeta=\tau_\theta(y)$ tokens              | text-to-image uses larger $d_\tau$; see Tabs 15–17.                   | Conditioner capacity matters for text            |
| Sampling steps            | DDIM steps: 50–500                              | fewer denoise steps                        | faster sampling with some quality tradeoff; reported with DDIM steps. | Choose steps based on target latency             |
| Guidance                  | classifier guidance vs classifier-free guidance | gradient correction / unconditional mixing | classifier-free guidance improves FID/IS strongly in reported tables. | Guidance is often “quality knob”                 |
| Post-hoc image guidance   | Eq. (16)–(17)                                   | add $\nabla\log p_\Phi$ term               | helps large-res coherence in convolutional sampling (Fig. 14).        | Useful when unconditional samples drift          |

**Implementation translation**

* Choices map to code toggles: latent `f`, latent reg type, `rescale_latent`, `cond_mode in {concat, crossattn}`, sampler steps, guidance scale.
* Ablations imply logging FID vs steps; store exact eval pipeline (Sec. E.3). 
* Numerical: guidance and rescaling can shift effective distribution; treat as part of model config.

---

## 9) IMPLEMENTATION & REPRODUCTION NOTES

### Datasets + preprocessing (explicit)

* Text-to-image model trained on LAION-400M; evaluated on MS-COCO val protocol.
* Inpainting uses Places dataset with synthetic masks from [88], train crops $256^2$, test crops $512^2$. 
* Super-resolution uses ImageNet; mentions BSR degradation pipeline for “real-world” SR variant. 
* Unconditional synthesis evaluated on CelebA-HQ, FFHQ, LSUN Churches/Bedrooms.

### Architecture (explicit tables)

* UNet hyperparams for many models listed in Tabs 12–15 (channels, depth, channel multipliers, attention resolutions, heads, batch size, iterations, LR).
* Conditioner $\tau_\theta$ for text/layout: unmasked transformer, E.2.1 Eq. (18)–(24), plus sequence length, depth, dim in Tab 17.
* Cross-attention integrated into UNet intermediate layers (Sec. 3.3). 

> **paper jump (optimizer details):** The supplement provides many hyperparameters, but I did not find explicit optimizer type (Adam/AdamW), betas, EMA decay, weight decay, grad clipping in the excerpts above. Treat these as missing; you’ll need to follow common diffusion defaults or their released code.

### Optimization (explicit)

* Learning rates, batch sizes, iterations for each model are given in Tabs 12–15.
* Compute hardware: most models trained on single NVIDIA A100; inpainting model trained on 8× V100.

### Evaluation details (explicit)

* FID/Precision/Recall computed using 50k samples and the full training set stats; FID via `torch-fidelity`, also compared with Dhariwal & Nichol script. Sec. E.3.1, PDF p.27 
* Text-to-image FID/IS computed against 30k MS-COCO val samples. Sec. E.3.2, PDF p.27 

### Gotchas & stability notes

1. **Latent variance / SNR coupling:** if Var$(z)$ is large, effective SNR changes and can harm large-res convolutional sampling; rescaling helps (Sec. D.1 + Sec. G).
2. **$\alpha_t$ near 0:** converting $\epsilon_\theta\to z_0$ divides by $\alpha_t$; use fp32 and maybe clamp. (Derived from Eq. (14)). 
3. **Guidance gradients:** Eq. (16) requires $\nabla_{z_t}\log p_\Phi$; backprop through $D$ and transforms can explode; use gradient scaling/clipping.
4. **Evaluation pipeline mismatch:** FID can differ depending on preprocessing scripts; they observed small differences and warn about unified procedures. 

### Hyperparameters that matter most (ranked)

1. Compression factor $f$ (controls compute–quality frontier).
2. Latent scaling / regularization (controls induced SNR).
3. Sampling steps (DDIM step count) and guidance scale $s$ (quality knob).
4. UNet capacity (channels, attention resolutions). 

**Implementation translation**

* Repro checklist: store exact (a) autoencoder checkpoint, (b) latent scaling stats, (c) diffusion schedule, (d) UNet/cond hyperparams from the correct table.
* Missing defaults to decide (label as inference if you pick): optimizer, EMA, normalization ranges.
* Numerical: always evaluate with the same FID pipeline to compare fairly. 

---

## 10) RESULTS & EVALUATION

### Metrics used (direction)

* FID ↓ (lower better), Inception Score ↑, Precision ↑, Recall ↑, PSNR ↑, SSIM ↑ (for SR).

### Main tables (summary)

**Unconditional image synthesis (Table 1):**

* CelebA-HQ: LDM-8 (200 DDIM steps) reports FID 4.02, Prec 0.64, Recall 0.52. 
* LSUN Bedrooms: LDM-4 (200 DDIM) FID 2.95, Prec 0.66, Recall 0.48. 

**Text-conditional (Table 2):**

* LDM-KL-8-G: FID 12.63, IS 30.29 (250 DDIM, classifier-free guidance). 

**ImageNet class-conditional (Table 3 / Table 10):**

* LDM-4-G: FID 3.60 with classifier-free guidance (scale 1.5), 250 DDIM steps.

**Compute comparison (Table 18):**

* LDM-4 has much higher throughput (samples/sec) and lower “V100-days” vs ADM/StyleGAN2 baselines (table provides detailed numbers). 

**Super-resolution (Table 11):**

* LDM-4 SR gets strong FID and IS, and guidance variants are compared; includes a pixel-space baseline with matched compute. 

### Baselines and fairness

* They compare against ADM, BigGAN, StyleGAN2, etc; compute parity discussed by converting A100-days to V100-days (assumed 2.2× speedup). 
* Evaluation protocol is mostly standard but they note pipeline differences can change FID slightly. 

### Failure modes / weaknesses (from paper + inferred)

* Over-compression (large $f$) reduces sample quality. 
* Convolutional sampling at larger-than-trained resolutions can yield incoherent global structure without guidance (Fig. 14). 
* **Inference:** adversarially-trained autoencoder + diffusion prior may inherit dataset biases; ethical concerns noted. 

**Implementation translation**

* Log exact: sampling steps (DDIM steps), $\eta$ in captions where given, guidance scale $s$ (tables/captions).
* For likelihood reporting: this paper focuses on sample quality; ELBO derivation is used for training justification, not NLL reporting. 

---

## 11) INTUITION & CONNECTIONS

### Mechanistic intuition (not vibes)

* Diffusion models learn to invert a gradual corruption process. The UNet sees a noisy latent grid $z_t$ and learns the noise pattern consistent with the data manifold at that noise level.
* LDM’s key insight: pixel-space contains many “imperceptible” degrees of freedom. Even if the loss underweights them, the network still pays compute to process every pixel. Compress first into a perceptual latent so compute focuses on semantic content.
* Cross-attention makes conditioning modular: keep the same UNet “denoiser,” swap in different $\tau_\theta$ encoders for different modalities.

### Connections (3–7)

* **DDPM (Ho et al. 2020)**: provides the ELBO-to-denoising-loss derivation; LDM reuses this and shifts to latent space (Appx B).
* **ADM (Dhariwal & Nichol 2021)**: strong pixel diffusion baseline; LDM compares compute/sample quality and uses similar UNet backbone ideas.
* **VQGAN / Taming Transformers (Esser et al. 2020)**: perceptual + adversarial autoencoder training; LDM uses similar first-stage recipe (Sec. 3.1, Sec. G).
* **Classifier guidance**: gradient-based guidance (Appx C Eq. (16)) generalizes to image-to-image steering via $p_\Phi(y|z_t)$. 
* **Classifier-free guidance (Ho & Salimans)**: quality boost via conditional/unconditional mixing; paper uses it but doesn’t re-derive formula here (Tables).

### “What this paper secretly is”

* A **compute-scaling trick**: diffusion stays the same mathematically; the main shift is *representation learning + factorizing perceptual vs semantic modeling*.

**Implementation translation**

* You can swap samplers (DDPM/DDIM) without changing training.
* Conditioner modules are pluggable: implement `tau_theta(y)` and cross-attn blocks once.
* Numerical: “secret sauce” stability often comes from latent scaling and guidance tuning, not the MSE loss itself.

---

## 12) LIMITATIONS, ASSUMPTIONS, AND OPEN QUESTIONS

### Explicit limitations / concerns (from paper)

* Ethical concerns: easier image manipulation, potential privacy leakage, biases in training data. 
* Reliance on large compute for some models (e.g., 1.45B text model), even if more accessible than some baselines. 

### Implicit limitations (inferred from method + appendices)

* If autoencoder discards information relevant for generation, diffusion cannot recover it. Compression is a lossy bottleneck.
* Convolutional sampling beyond training resolution is not guaranteed stable; depends on induced SNR and guidance (Sec. D.1, Fig. 14).
* Missing optimizer/EMA details in the text can hinder exact reproduction without code.

### Open questions / things I’d test next

1. Replace fixed schedule with learned schedule tuned for latent-space statistics.
2. Systematically compare weighting schemes (Eq. (13) weights vs unweighted Eq. (2)) in latent space.
3. Quantify how much perceptual loss choice in AE impacts diffusion sample quality.
4. Explore timestep-dependent conditioning in $\tau_\theta$ (they mention but avoid for speed).
5. Measure memorization risk vs pixel diffusion baselines (privacy). 
6. Large-res convolutional sampling: when does it break? how does latent scaling threshold relate to SNR?

### What breaks if assumptions are violated?

* If $E,D$ are not perceptually faithful, LDM samples decode to artifacts or lack detail.
* If latent variance is large and not rescaled, the effective noise schedule is “wrong” relative to training assumptions, destabilizing sampling.

**Implementation translation**

* When adapting to a new domain: first validate recon quality of AE at chosen $f$; compute latent variance and decide rescaling.
* For high-res generation beyond training: integrate guidance or multi-scale constraints early. 

---

## 13) “STEAL THIS” SECTION (PORTABLE IDEAS)

1. **Two-stage factorization: perceptual AE + generative prior**

   * Where: Sec. 3.1–3.2.
   * Why: reduces compute by shrinking spatial grid; preserves semantics.
   * How elsewhere: any expensive generative model over images/videos → learn perceptual latent first.

2. **Noise-prediction MSE from ELBO**

   * Where: Appx B Eq. (13)–(15) → Eq. (1)/(2).
   * Why: simple stable objective.
   * How: implement one-shot noising $x_t=\alpha_t x_0+\sigma_t\epsilon$ + MSE.

3. **Cross-attention as universal conditioning interface**

   * Where: Sec. 3.3 (Attention formulas) + Eq. (3).
   * Why: supports arbitrary token-based modalities (text, layout, class).
   * How: treat UNet features as queries, conditioning tokens as keys/values.

4. **Latent rescaling to control effective SNR**

   * Where: Sec. G and D.1.
   * Why: stabilizes sampling; matches schedule assumptions.
   * How: estimate per-component std $\hat\sigma$ and divide latents.

5. **Post-hoc guidance as score correction**

   * Where: Appx C Eq. (16)–(17). 
   * Why: condition an unconditional model at test-time with differentiable constraints.
   * How: add gradient of log-likelihood of desired constraint into $\epsilon$ prediction.

**Implementation translation**

* These are modular: (1) AE training module, (2) diffusion training module, (3) conditioning module, (4) scaling module, (5) guidance module.
* Biggest numerical risk: guidance gradients + divide-by-$\alpha_t$ near late steps.

---

## 14) SELF-TEST (FOR LEARNING)

### 10 short questions

1. What is $f$ and how does it affect latent shape?
2. Write Eq. (2) and identify each random variable.
3. What is SNR$(t)$?
4. What do $Q,K,V$ come from in cross-attention?
5. What distribution is $q(x_t|x_0)$?
6. How do you compute $z_t$ from $z_0$ and $\epsilon$?
7. What does “reweighted bound” mean in this context?
8. What does latent rescaling do?
9. What are DDIM steps used for?
10. What is the purpose of $\tau_\theta$?

### 5 medium questions

1. Starting from Eq. (14), derive $x_\theta(x_t,t)$ in terms of $\epsilon_\theta(x_t,t)$.
2. Explain why Eq. (15) implies that noise MSE corresponds to reconstruction MSE.
3. Compare concat vs cross-attention conditioning: when would you use each?
4. Why can convolutional sampling beyond training resolution fail, and what fix is suggested?
5. How is FID computed in their evaluation (sample count + tool)?

### 2 hard questions

1. Suppose your latent variance doubles but you keep the same schedule $(\alpha_t,\sigma_t)$. Predict how sampling changes and propose a fix grounded in Sec. D.1/G.
2. Design a new guider $p_\Phi(y|z_t)$ for “make the image match a target style embedding,” using Eq. (16)–(17) as template.

#### Answers (collapsible)

<details><summary>Short answers</summary>

1. $f=H/h=W/w$; latent shape scales as $(H/f)\times(W/f)$. Sec. 3.1, PDF p.4. 
2. Eq. (2): $\mathbb{E}*{E(x),\epsilon,t}|\epsilon-\epsilon*\theta(z_t,t)|^2$. Sec. 3.2, PDF p.5. 
3. SNR$(t)=\alpha_t^2/\sigma_t^2$. Appx B, PDF p.16. 
4. $Q=W_Q\phi_i(z_t)$, $K=W_K\tau_\theta(y)$, $V=W_V\tau_\theta(y)$. Sec. 3.3, PDF p.5. 
5. Gaussian: $\mathcal{N}(x_t|\alpha_t x_0,\sigma_t^2 I)$. Eq. (4), PDF p.16. 
6. $z_t=\alpha_t z_0+\sigma_t\epsilon$ (latent analogue of Eq. (4)).
7. Replace weighted ELBO terms with equal weights across $t$ to get simple MSE (after Eq. (15) → Eq. (1)).
8. Divide $z$ by estimated std $\hat\sigma$ so latents have unit std (Sec. G). 
9. Faster sampling with fewer steps; paper reports “N DDIM steps” in tables.
10. Encodes conditioning $y$ into intermediate representation $\tau_\theta(y)$, often tokens. Sec. 3.3 + Appx E.2.1.

</details>

<details><summary>Medium answers</summary>

1. From Eq. (14): $\epsilon_\theta=(x_t-\alpha_t x_\theta)/\sigma_t \Rightarrow x_\theta=(x_t-\sigma_t\epsilon_\theta)/\alpha_t$. 
2. Plug $x_t=\alpha_t x_0+\sigma_t\epsilon$ and the solved $x_\theta$ into $|x_0-x_\theta|^2$ to get Eq. (15). 
3. Concat for spatially aligned maps (SR/inpainting/semantic maps); cross-attn for tokens (text/layout/class). Sec. 3.3 + Table 15 conditioning column.
4. Large-res convolutional sampling can become incoherent if induced SNR is high; rescaling latents and/or guidance helps (Sec. D.1, Fig. 14; Sec. G).
5. FID/Prec/Recall: 50k samples vs training set; torch-fidelity; check with Dhariwal & Nichol script (Sec. E.3.1). 

</details>

<details><summary>Hard answers (sketch)</summary>

1. If Var$(z)$ doubles, effective SNR$(t)=\alpha_t^2\text{Var}(z)/\sigma_t^2$ increases, causing early steps to commit too much structure and harming large-res sampling; fix by rescaling $z\leftarrow z/\hat\sigma$ (Sec. G) or adjust schedule.
2. Define guider score via a differentiable style loss $L_\text{style}(D(z_0(z_t)))$; set $\log p_\Phi\propto -L_\text{style}$ and plug $\nabla_{z_t}\log p_\Phi$ into Eq. (16). 

</details>

**Implementation translation**

* Self-test derivations correspond directly to code snippets: (a) `z0_hat` formula, (b) attention source tensors, (c) latent rescaling.
* Numerical: ensure gradients for guidance are computed w.r.t. $z_t$ (not $z_0$), matching Eq. (16). 

---

## 15) FINAL CHECKLIST (MUST INCLUDE)

* [x] I listed the canonical equations and explained each
* [x] I made a derivation map with no big jumps (and marked “paper jump” where needed)
* [x] I gave sampling + training pseudocode
* [x] I extracted all experimental/repro details available in the paper (and flagged missing optimizer defaults)
* [x] I summarized results + ablations + limitations
* [x] I wrote a complete notation glossary with shapes
* [x] I included “gotchas” and “what to test next”
