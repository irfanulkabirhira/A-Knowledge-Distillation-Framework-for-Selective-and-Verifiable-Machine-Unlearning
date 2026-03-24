# 🧠 A Knowledge Distillation Framework for Selective and Verifiable Machine Unlearning

## 📌 Overview

This repository implements a **knowledge distillation-based selective machine unlearning framework** for image classification tasks.

The goal is to **remove specific class-level knowledge (forget set)** while preserving performance on the remaining data (**retain set**) — without full retraining.

---

## 🚀 Key Idea

* Train a **Teacher Model** on full dataset
* Remove selected classes (Forget Set)
* Train a **Student Model** using:

  * Retained data only
  * Knowledge Distillation (KD)
* Evaluate:

  * Forgetting effectiveness
  * Retained performance
  * Privacy leakage

---

## 🏗️ Methodology Pipeline

1. Full dataset training (Teacher)
2. Class removal (Forget Set)
3. Student training on retained data
4. Knowledge Distillation (KD Loss + CE Loss)
5. Multi-metric evaluation

---

## 📊 Datasets Used

| Dataset       | Link                                                                |
| ------------- | ------------------------------------------------------------------- |
| Caltech-101   | https://www.kaggle.com/datasets/imbikramsaha/caltech-101            |
| Caltech-256   | https://www.kaggle.com/datasets/jessicali9530/caltech256            |
| Stanford Dogs | https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset |
| CIFAR-100     | https://www.kaggle.com/datasets/melikechan/cifar100                 |

> ⚠️ Note: Datasets are not included in this repo. Please download from the above links.

---

## 🧪 Experiments

We evaluate selective unlearning under different forgetting settings:

* Forget 5 Classes
* Forget 10 Classes
* Forget 15 Classes

---

## 📈 Evaluation Metrics

### 🔹 Utility Metrics

* Retained Accuracy
* Teacher vs Student Accuracy Gap

### 🔹 Forgetting Metrics

* Logit MSE Distance
* Feature Cosine Similarity
* Prediction Entropy

### 🔹 Privacy Metrics

* Membership Inference Attack (MIA)

  * Accuracy
  * AUC

---

## 📊 Key Results

* Strong divergence on forgotten classes
* Stable retained-task performance
* Significant reduction in MIA transferability
* Efficient alternative to full retraining

---

## ⚙️ Implementation Details

* Framework: PyTorch
* Model: ResNet-18
* Optimizer: SGD with Momentum
* KD Temperature: 4
* Batch Size: 64
* LR Scheduler: Cosine Annealing

---

## 📁 Repository Structure


---
├── Caltech101/
├── Caltech256/
├── CIFAR100/
├── StanfordDogs/
├── notebooks/
├── results/
└── README.md
```

```

## 🔬 Reproducibility

* Same pipeline used across all datasets
* Class-balanced splits
* Multiple runs averaged for stability

---

## 🔐 Privacy Perspective

The framework reduces **membership inference attack success**, indicating suppression of memorized sensitive data.

---

## 📌 Future Work

* Instance-level unlearning
* Certified forgetting guarantees
* Transformer-based architectures
* Multimodal datasets

---

## 👨‍💻 Author

**Md Irfanul Kabir Hira**
Machine Learning & Data Science Researcher

---

## ⭐ If you find this useful

Please ⭐ star the repository and share!

---
