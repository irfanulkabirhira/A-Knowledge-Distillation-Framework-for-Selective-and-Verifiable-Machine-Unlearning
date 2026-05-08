<div align="center">

# 🧠 Forgetting-Aware Knowledge Distillation for Selective and Verifiable Machine Unlearning

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat-square&logo=pytorch)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-Preprint-orange?style=flat-square)](.)
[![Stars](https://img.shields.io/github/stars/irfanulkabirhira/A-Knowledge-Distillation-Framework-for-Selective-and-Verifiable-Machine-Unlearning?style=flat-square)](.)

**Umme Sara · Md Irfanul Kabir Hira**

*Machine Learning & Privacy Research*

---

> **"Can a neural network be taught to forget — selectively, verifiably, and without retraining from scratch?"**
>
> This repository answers that question with a rigorous empirical framework built on knowledge distillation.

</div>

---

## 📖 Table of Contents

- [What is Machine Unlearning?](#-what-is-machine-unlearning)
- [Why Our Approach?](#-why-our-approach-is-different--better)
- [Framework Overview](#-framework-overview)
- [Methodology](#-methodology)
- [Datasets](#-datasets)
- [Evaluation Metrics](#-evaluation-metrics)
- [Key Results](#-key-results)
- [Repository Structure](#-repository-structure)
- [Installation & Usage](#-installation--usage)
- [Reproducibility](#-reproducibility)
- [Limitations & Future Work](#-limitations--future-work)
- [Citation](#-citation)

---

## 🔍 What is Machine Unlearning?

When a deep learning model is trained on a dataset, it implicitly **memorizes** patterns from every training sample — including sensitive, private, or legally protected data. After deployment, regulations such as the **EU General Data Protection Regulation (GDPR)** grant individuals the **right to erasure**: the right to demand that their data — and its influence on any model — be completely removed.

The naive solution is to **retrain the model from scratch** on the remaining data. This is:
- ❌ **Computationally prohibitive** for large models
- ❌ **Impractical** for frequent deletion requests
- ❌ **Unverifiable** — how do you *prove* a model has forgotten?

**Machine unlearning** is the field that asks: can we efficiently remove the influence of specific data from an already-trained model, and can we *verify* that the removal was successful?

---

## 🏆 Why Our Approach Is Different & Better

Most existing machine unlearning methods fall into one of two camps — and both have serious shortcomings:

| Approach | How It Works | Problem |
|---|---|---|
| **Full Retraining** | Train from scratch on retained data | Prohibitively expensive; impractical at scale |
| **Gradient Ascent** | Maximize loss on forgotten samples | Requires access to forget set during unlearning; can destabilize the model |
| **SISA Training** | Partition data into shards; retrain affected shard | Requires special data partitioning *before* training; not applicable post-hoc |
| **Our Approach** | Train student on retained data via KD | ✅ No access to forget set needed · ✅ No model destabilization · ✅ Multi-metric verifiable |

### What makes our framework stand out:

**1. 🎯 Verification-First Design**

We do not just claim that forgetting happened — we *prove* it using five independent metrics spanning output behavior, internal representations, prediction uncertainty, and privacy leakage. No single metric can be gamed or misinterpreted.

**2. 🔒 No Forget-Set Required at Unlearning Time**

Unlike gradient-ascent methods, our student model never sees forgotten class samples during training. This means the framework is applicable even when the original forget-set data is no longer accessible — a realistic real-world constraint.

**3. 🧪 Empirically Grounded, Not Just Theoretically Claimed**

We validate across **four diverse benchmark datasets** (Caltech-101, Caltech-256, CIFAR-100, Stanford Dogs) and **three forgetting scopes** (5, 10, 15 classes) — totalling 12 independent experimental conditions per metric. Our conclusions are backed by comprehensive cross-dataset evidence.

**4. 🔐 Privacy-Aware by Design**

We embed **Membership Inference Attack (MIA) transfer analysis** as a core evaluation step — not an afterthought. If an attacker trained on the teacher model cannot successfully attack the student model, the membership signal has been effectively erased.

**5. ⚡ Efficient Relative to Full Retraining**

The teacher is frozen after initial training. Only the lighter student network is optimized, and only on the retained subset `D_r ⊂ D`. This provides a proportional reduction in computational cost relative to full retraining.

**6. 📐 Honest About Its Limits**

Unlike many papers that overstate guarantees, we explicitly distinguish between *empirical behavioral forgetting* (what we demonstrate) and *certified deletion* (what we do not claim). This intellectual honesty makes the framework trustworthy for real deployment assessment.

---

## 🗺️ Framework Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        PHASE 1: TEACHER TRAINING                    │
│                                                                     │
│   Full Dataset D ──► ResNet-18 Teacher f_θ ──► Frozen Reference    │
│   (All classes)        (trained on D)           (never updated)    │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    PHASE 2: FORGETTING SETUP                        │
│                                                                     │
│   D ──► Forget Set D_f (5/10/15 classes)                           │
│     └── Retain Set D_r (remaining classes)                         │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   PHASE 3: STUDENT TRAINING                         │
│                                                                     │
│   D_r + Teacher Soft Labels ──► Student g_φ                        │
│                                                                     │
│   L_total = λ · L_KD + (1-λ) · L_CE                               │
│                                                                     │
│   • L_KD: Temperature-scaled KL divergence (T=4)                   │
│   • L_CE: Supervised cross-entropy on retained classes             │
│   • λ = 0.5 (balanced via grid search)                             │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  PHASE 4: MULTI-METRIC VERIFICATION                 │
│                                                                     │
│   Behavioral    │ Representational │ Uncertainty │ Privacy          │
│   ─────────     │ ──────────────── │ ─────────── │ ───────          │
│   Logit MSE     │ Cosine Sim.      │ Prediction  │ MIA Transfer     │
│   Accuracy Gap  │ Feature Drift    │ Entropy     │ AUC / Accuracy   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🔬 Methodology

### Core Formulation

Let `D` be the full training dataset and `C_f ⊂ C` the set of classes to be forgotten.

**Retain set:** `D_r = D \ D_f`

**Forget set:** `D_f = {(x,y) ∈ D | y ∈ C_f}`

The **teacher** `f_θ` is trained on all of `D`. The **student** `g_φ` is trained only on `D_r` with no exposure to `D_f`.

### Distillation Objective

```
L_KD = T² · KL( σ(z_t / T) ‖ σ(z_s / T) )
```

where `z_t`, `z_s` are teacher and student logits over retained classes, and `T=4` softens the teacher distribution to carry inter-class relational structure.

### Final Loss

```
L_total = λ · L_KD + (1 - λ) · L_CE
```

This balances knowledge transfer from the teacher with direct supervised learning on retained labels.

### Why Temperature Scaling Matters

Setting `T=4` produces softer probability distributions from the teacher. This means the student learns not just *which* class is most likely, but *how similar* retained classes are to each other — a richer supervisory signal that improves generalization on the retained task while naturally diverging on forgotten classes.

---

## 📊 Datasets

| Dataset | Classes | Resolution | Granularity | Link |
|---|---|---|---|---|
| **Caltech-101** | 101 | Variable | Coarse object categories | [Download](https://www.kaggle.com/datasets/imbikramsaha/caltech-101) |
| **Caltech-256** | 257 | Variable | Broad object recognition | [Download](https://www.kaggle.com/datasets/jessicali9530/caltech256) |
| **CIFAR-100** | 100 | 32×32 | Natural scene categories | [Download](https://www.kaggle.com/datasets/melikechan/cifar100) |
| **Stanford Dogs** | 120 | Variable | Fine-grained dog breeds | [Download](https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset) |

> ⚠️ Datasets are **not included** in this repository. Download from the links above and place in the corresponding dataset folder.

**Why these four datasets?** They span a spectrum from coarse-grained (Caltech-101, CIFAR-100) to fine-grained (Stanford Dogs), and from small (CIFAR-100 resolution) to large-scale (Caltech-256). This diversity tests whether our verification signals hold across visually distinct recognition settings — not just a single favorable benchmark.

---

## 📐 Evaluation Metrics

We evaluate unlearning quality across four dimensions. Every claim in this framework is backed by evidence from multiple independent signals.

### 1. Behavioral Divergence — *Did the student forget?*

**Logit MSE Distance**
```
D_logit = (1/N_f) · Σ ‖z_t(i) - z_s(i)‖²
```
Higher values → stronger behavioral divergence on forgotten samples.

**Teacher–Student Accuracy Gap**
```
Δ_acc = Acc_teacher - Acc_student
```
A persistent gap confirms the student does not replicate the teacher's knowledge of forgotten classes.

### 2. Representational Drift — *How deep does forgetting go?*

**Feature Cosine Similarity**
```
CosSim(f_t, f_s) = (f_t · f_s) / (‖f_t‖ · ‖f_s‖)
```
Computed at the penultimate layer. Lower similarity = stronger representational drift on forgotten samples.

### 3. Uncertainty Evidence — *Is the student confused about forgotten classes?*

**Prediction Entropy**
```
H(p) = -Σ p_c · log(p_c)
```
Higher entropy on forgotten samples indicates the student has lost discriminative knowledge about those classes.

### 4. Privacy Leakage — *Can an attacker still exploit the student?*

**Membership Inference Attack (MIA) Transfer**

An attack classifier trained on the teacher's outputs is transferred to the student. If AUC ≈ 0.50 (random guess), the membership signal has been erased. This is the strongest privacy-level verification available without formal certification.

---

## 📈 Key Results

### Logit MSE — Caltech-256

| Forget Config | Mean MSE | Std |
|---|---|---|
| Forget 5 Classes | 74.59 | 39.22 |
| Forget 10 Classes | 89.51 | 53.82 |
| Forget 15 Classes | 89.87 | 47.59 |

↑ MSE increases and stabilizes — consistent suppression of forgotten knowledge.

### MIA Transfer Results (AUC)

| Dataset | Forget Config | Teacher AUC | Student AUC |
|---|---|---|---|
| Caltech-256 | 5 classes | 0.8154 | 0.5586 |
| Caltech-256 | 15 classes | 0.8099 | **0.4890** |
| Stanford Dogs | 15 classes | 0.8692 | **0.4942** |
| CIFAR-100 | 15 classes | 0.6932 | **0.4949** |
| Caltech-101 | 15 classes | 0.8611 | **0.4986** |

✅ Student AUC consistently drops toward 0.50 (random) — the membership signal is erased.

### Teacher vs Student Accuracy

| Dataset | Teacher Acc | Student Acc (15 classes) | Gap |
|---|---|---|---|
| Caltech-101 | 93.53% | 50.72% | 42.81% |
| Caltech-256 | 71.24% | 44.81% | 26.43% |
| CIFAR-100 | 77.87% | 61.82% | 16.05% |
| Stanford Dogs | 61.06% | 26.42% | 34.64% |

The growing gap confirms selective forgetting — the student preserves the retained task while losing full knowledge.

---

## 📁 Repository Structure

```
.
├── Caltech101/
│   ├── teacher_training.py          # Teacher model training on full dataset
│   ├── student_training.py          # Student KD training on retained data
│   ├── evaluate_forgetting.py       # Multi-metric forgetting verification
│   └── results/                     # Saved metrics, plots, logs
│
├── Caltech256/
│   ├── teacher_training.py
│   ├── student_training.py
│   ├── evaluate_forgetting.py
│   └── results/
│
├── CIFAR100/
│   ├── teacher_training.py
│   ├── student_training.py
│   ├── evaluate_forgetting.py
│   └── results/
│
├── StanfordDogs/
│   ├── teacher_training.py
│   ├── student_training.py
│   ├── evaluate_forgetting.py
│   └── results/
│
├── notebooks/
│   ├── 01_teacher_training.ipynb
│   ├── 02_student_distillation.ipynb
│   ├── 03_forgetting_verification.ipynb
│   └── 04_mia_evaluation.ipynb
│
├── utils/
│   ├── metrics.py                   # Logit MSE, cosine similarity, entropy
│   ├── mia.py                       # Membership inference attack
│   ├── dataset_utils.py             # Retain/forget set construction
│   └── visualization.py            # Histogram and accuracy plots
│
├── configs/
│   └── hyperparams.yaml             # All experiment hyperparameters
│
├── results/                         # Cross-dataset summary results
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation & Usage

### Requirements

```bash
git clone https://github.com/irfanulkabirhira/A-Knowledge-Distillation-Framework-for-Selective-and-Verifiable-Machine-Unlearning.git
cd A-Knowledge-Distillation-Framework-for-Selective-and-Verifiable-Machine-Unlearning
pip install -r requirements.txt
```

**requirements.txt**
```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
scikit-learn>=1.2.0
matplotlib>=3.7.0
tqdm>=4.65.0
pyyaml>=6.0
```

### Step 1: Train the Teacher

```python
# Train ResNet-18 on full dataset (e.g., Caltech-101)
python Caltech101/teacher_training.py \
    --dataset caltech101 \
    --epochs 50 \
    --lr 0.01 \
    --batch_size 64
```

### Step 2: Define the Forget Set & Train the Student

```python
# Train student via knowledge distillation on retained classes only
python Caltech101/student_training.py \
    --forget_classes 5 \       # Number of classes to forget (5, 10, or 15)
    --temperature 4 \          # KD temperature
    --lambda_kd 0.5 \          # Balance between KD and CE loss
    --epochs 50 \
    --lr 0.01
```

### Step 3: Run Forgetting Verification

```python
# Compute all five verification metrics
python Caltech101/evaluate_forgetting.py \
    --teacher_checkpoint checkpoints/teacher.pth \
    --student_checkpoint checkpoints/student.pth \
    --forget_classes 5
```

This outputs:
- Logit MSE distance (mean, std, min, max)
- Feature cosine similarity
- Prediction entropy distribution
- Teacher vs student accuracy comparison
- MIA transfer accuracy and AUC

### Hyperparameter Reference

| Parameter | Value | Rationale |
|---|---|---|
| Optimizer | SGD with Momentum | Better-calibrated confidence values for MIA evaluation |
| Learning Rate | 0.01 | Standard for ResNet fine-tuning |
| LR Schedule | Cosine Annealing | Fast early convergence, stable final epochs |
| Batch Size | 64 | Efficient gradient estimation |
| Momentum | 0.9 | Stability in optimization |
| Weight Decay | 5×10⁻⁴ | Prevents overfitting to retained classes |
| KD Temperature (T) | 4 | Softens teacher distribution; preserves inter-class structure |
| λ (KD/CE balance) | 0.5 | Selected via grid search over {0.3, 0.5, 0.7} |
| Epochs | 50 | With early stopping (patience=10) |
| Validation Split | 10–15% | Per-class stratified split |

---

## 🔁 Reproducibility

All results in the paper can be reproduced with the following guarantees:

- ✅ **Same hyperparameters** used across all datasets (see `configs/hyperparams.yaml`)
- ✅ **Class-balanced splits** — equal samples per class to eliminate selection bias
- ✅ **Fixed preprocessing pipeline** — all images resized to 224×224, normalized with ImageNet mean/std
- ✅ **Identical teacher architecture** — ResNet-18 with ImageNet pre-trained weights across all datasets
- ✅ **Three forgetting scopes tested** — 5, 10, 15 forgotten classes per dataset
- ✅ **Complete code** — from data loading to metric computation and visualization

---

## ⚠️ Honest Limitations

We believe transparency about limitations is what makes a research framework trustworthy.

| Limitation | Details |
|---|---|
| **No formal deletion guarantees** | Results are empirical behavioral evidence, not certified removal proofs |
| **Class-level only** | Instance-level or subject-level unlearning is not addressed |
| **Image classification scope** | Generalization to language models or multimodal models is not validated |
| **No wall-clock efficiency measurement** | Timing comparisons against retraining are not formally benchmarked |
| **Architecture-specific** | Results are based on CNN (ResNet-18); transformer architectures may behave differently |

---

## 🔭 Limitations & Future Work

- **Instance-level unlearning** — extend to sample-level or attribute-level deletion
- **Certified removal bounds** — integrate differential privacy or influence function analysis
- **Transformer architectures** — validate framework on ViT, DINO, CLIP-based models
- **Explicit forgetting signals** — augment with entropy maximization or adversarial forgetting loss to strengthen separation
- **Medical imaging applications** — apply to high-stakes domains requiring regulatory compliance
- **Advanced representation analysis** — incorporate CKA, influence functions, and latent drift metrics

---

## 📚 Citation

If you find this work useful for your research, please cite:

```bibtex
@article{sara2025forgetting,
  title     = {Forgetting-Aware Knowledge Distillation for Selective and Verifiable Machine Unlearning},
  author    = {Sara, Umme},
  journal   = {Preprint},
  year      = {2025},
  note      = {Available at: https://github.com/irfanulkabirhira/A-Knowledge-Distillation-Framework-for-Selective-and-Verifiable-Machine-Unlearning}
}
```

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

The datasets used are publicly available benchmarks. Please refer to their respective licenses before using them in your own work.

---

## 🙏 Acknowledgements

This work uses no external funding. Language editing assistance was provided by AI tools (ChatGPT, Gemini). All experimental design, implementation, and analysis were conducted independently by the authors.

---

<div align="center">

**If this repository helped your research, please ⭐ star it and share it with others working on machine unlearning, privacy-preserving ML, or knowledge distillation.**

*Questions? Open an [Issue](../../issues) or reach out directly.*

</div>
