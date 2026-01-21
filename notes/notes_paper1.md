# DDPM (Denoising Diffusion Probabilistic Models)

### Abstract

Diffusion Probabilistic Models - A class of latent variable models inspired by considerations from nonequilibrium thermodynamics.

The obtained it's best results when they trained on a weighted variational bound based on a novel connection between diffusion probabilistic models and denoising score matching with Langevin dynamics.

>The models that they built naturally admit a progressive lossy decompression scheme that can be interpreted as a generalization of autoregressive decoding.

### Introduction

Deep generative models like - GANs, Autoregressive models, flows, and VAEs have synthesized striking image and audio samples. There have been remarkable advances in **energy-based modeling and score matching** that have produced images comparable to those of GANs.

A diffusion probabilistic model (diffusion model) is a parameterized markov chain trained using variational inference to produce samples matching the data after finite time.

So, the transition of this chain are learned to reverse a diffusion process, which is a markov chain that gradually adds noise to the data in the opposite direction of sampling until signal is destroyed. 

When the diffusion consists of small amounts of gaussian noise, it is sufficient to set the sampling chain transitions to conditional gaussians too, allowing for a particularly simple neural network parameterization.

### Background

Diffusion models are latent variable models of the form $p_\theta(x_0) := \int p_\theta(x_{0:T}) dx_{1:T}$, where $x_1, ...., x_T$ are latents of the same dimensionality as the data $x_0 \sim q(x_0)$.

The joint distribution $p_\theta(x_{0:T})$ is called the reverse process, and it is defined as a markov chain with learned gaussian starting at $p(x_T) = \mathcal{N}(x_T;0,\textrm{I})$:

