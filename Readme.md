# Data Removal-enabled Random Forest (Unlearning with Tree-based Models)

This project implements a **Data Removal-enabled Random Forest** (DaRE), a machine unlearning method designed for fast and accurate removal of training data from decision trees and random forest models without retraining from scratch. The implementation is inspired by recent research in machine unlearning that focuses on providing efficient, verifiable forgetting mechanisms for non-parametric models.

---

## 🧠 What is Machine Unlearning?

**Machine Unlearning** refers to the process of removing the influence of specific data points from a trained model. This is essential for ensuring compliance with privacy regulations such as the **Right to be Forgotten** under GDPR.

This project provides an unlearning mechanism for decision trees and random forests that:
- Efficiently removes the impact of specified samples.
- Avoids retraining the entire model.
- Maintains high accuracy on the retained dataset.

The method is based on the idea that tree-based models can be updated by **pruning or adjusting decision paths** affected by the forgotten data points.

---

## 📦 Features

- Unlearning without full retraining.
- Support for both single decision trees and ensemble random forests.
- Metrics to evaluate:
  - Relative Retain Accuracy (Ar)
  - Relative Forget Accuracy (Af)
- Reproducible experiments with controlled data removal.

---
