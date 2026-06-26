# EEG-to-Video Reconstruction via CARD Transformer

An end-to-end **Brain-Computer Interface (BCI)** framework that reconstructs videos directly from EEG signals by integrating a **CARD Transformer** with **Video Latent Diffusion Models (VideoLDM)**. The project bridges neural signal processing and generative AI to translate human brain activity into semantically meaningful video content.

---

## Overview

This project proposes a deep learning pipeline that learns rich representations from EEG recordings and conditions a latent diffusion model to generate temporally coherent video sequences.
The framework combines transformer-based temporal modeling, semantic alignment, and diffusion-based generation to improve both structural quality and semantic consistency.

---

## Motivation

Reconstructing visual information directly from brain activity is a challenging problem in Brain-Computer Interfaces.
This project investigates how transformer-based representation learning and latent diffusion models can be combined to generate semantically meaningful videos from EEG recordings.

---

## Key Features

- Developed a **CARD Transformer** for learning spatial-temporal representations from 128-channel EEG signals.
- Integrated **CLIP semantic alignment** to bridge EEG embeddings with visual representations.
- Implemented **Video Latent Diffusion Models (VideoLDM)** for high-quality video reconstruction.
- Introduced **EMA-based temporal smoothing** and **causal temporal attention** to improve frame-to-frame consistency.
- Built an end-to-end training and evaluation pipeline with checkpointing, ablation studies, and automated performance evaluation.

---

## Model Pipeline

EEG Signals

↓

Preprocessing & Windowing

↓

CARD Transformer Encoder

↓

CLIP-Aligned Latent Embedding

↓

Video Latent Diffusion Model

↓

Generated Video Sequence

---

## Technologies Used

- Python
- PyTorch
- CARD Transformer
- Video Latent Diffusion Models (VideoLDM)
- CLIP
- OpenCV
- NumPy
- Diffusers
- TorchMetrics

---

## Results

| Metric | Score |
|--------|------:|
| SSIM | **0.321** |
| CLIP Similarity | **0.736** |

The proposed framework outperformed the EEG2Video baseline in structural similarity while maintaining strong semantic consistency on the **SEED-DV benchmark**.

---

## Project Highlights

- End-to-end EEG-to-video generation framework.
- Long-range temporal modeling using Transformer architecture.
- Diffusion-based video synthesis with semantic guidance.
- Modular architecture supporting training, evaluation, checkpointing, and ablation experiments.

---

## Course Information

This project was developed as part of the **Deep Learning** course at **Indian Institute of Technology (IIT) Mandi**.
The objective was to explore state-of-the-art deep learning techniques for **Brain-Computer Interfaces (BCI)** by reconstructing videos directly from EEG signals using transformer-based architectures and latent diffusion models.

