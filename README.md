# CloudCons: A Comprehensive End-to-End Benchmark for Cloud Resource Consolidation

## 📖 Overview

**CloudCons** is designed to bridge the gap between time series forecasting accuracy and downstream decision utility in cloud resource consolidation. By integrating diverse workloads from [Huawei Cloud](https://github.com/sir-lab/data-release/blob/main/README_data_release_2025.md), [Microsoft Azure](https://github.com/Azure/AzurePublicDataset/blob/master/AzurePublicDatasetV2.md), and [Google Borg](https://github.com/google/cluster-data/blob/master/ClusterData2019.md) , it provides a standardized framework to evaluate how the predictive performance of forecasting models (statistical, deep learning, and foundation models) impacts resource efficiency and service reliability.



## 🌟 Impact

CloudCons aims to significantly advance the field of AIOps and Cloud Resource Management by addressing critical gaps in existing research.

### 1. Evaluated on Diverse, Real-World Workloads

CloudCons incorporates heterogeneous real-world workload traces to ensure robustness against concept drifts and varying cloud environments:

- **Huawei2025:** Characterized by structural heterogeneity and strong temporal dependencies.
- **Azure2019:** Exhibits distinctive "pulse-like" morphology with high spikiness.
- **Borg2019-d:** Dominated by low resource utilization accompanied by frequent, chaotic high-frequency jitter.
- **Borg2019-e:** Demonstrates exceptional periodic regularity with highly synchronized 24-hour diurnal rhythms.

### 2. Bridging the Gap between Forecasting and Decision-making

By introducing an end-to-end evaluation suite, we enable researchers to optimize models for actual system performance (Resource Efficiency & Reliability) rather than just curve fitting.

### 3. Actionable Insights for Industry

We provide a rigorous analysis of **Predictive Quantile Selection**, offering a strategic lever to balance the trade-off between resource efficiency and service reliability. Our findings guide practitioners on when to use median forecasts (cost-oriented) versus high-quantile forecasts (reliability-oriented) .



## 🚀 Accessibility 

We are committed to open science and ensuring that our benchmark is easily accessible to the research community without barriers.

### 1. Open Data Access

- **No Authentication Required:** All processed datasets (Huawei2025, Azure2019, Borg2019-d, Borg2019-e) are hosted on Hugging Face. You can access and download them directly via the following link: [https://huggingface.co/datasets/kdd2026-cloudcons/CloudCons-ds](https://huggingface.co/datasets/kdd2026-cloudcons/CloudCons-ds). 
- **Standardized Format:** The datasets have been processed through a standardized pipeline and are provided in universal formats to ensure compatibility with standard data science tools.

### 2. Open Source Code

- **`forecasting_bench/`**: **Time Series Forecasting Evaluation**

  - It includes scripts to generate standard rolling-window predictions and compute error-based metrics, such as **MASE** and **CRPS**, to benchmark forecasting accuracy .

- **`predictor/`**: **Statistical & Foundation Models**

  - This directory contains the inference wrappers for models that do not require dataset-specific training.

- **`DeepModel/`**: **Deep Learning Model Training & Evaluation**

  - This folder manages the full lifecycle (training, hyperparameter tuning, evaluation) of deep learning baselines.

- **`simulation/`**: **End-to-End Simulation & Optimization**

  - This is the core engine for the "Forecast-then-Optimize" workflow, built on the **SimPy** discrete-event simulation framework.
  - **Environment:** Simulates cloud data center operations, including VM allocation and resource usage tracking.
  - **Optimization Algorithms (Scheduler):** Implements various packing strategies, ranging from heuristic algorithms (**FFD**, **BFD**) and meta-heuristics (**ACO**) to the exact solver (**Gurobi**).
  - **Evaluation Metrics:** Calculates downstream decision utility metrics, including Util, RAR, VR, VS, PICP, MPIW, Winker Score.

  

## ⚖️ Ethics and Fairness

We strictly adhere to ethical guidelines regarding data privacy, bias mitigation, and responsible AI development.

### 1. Data Provenance and Privacy 

- **Source:** The datasets included in CloudCons are derived from publicly released traces by **Huawei Cloud**, **Microsoft Azure**, and **Google Borg**.
- **Anonymization:** These datasets contain strictly technical metric logs (e.g., CPU usage, memory usage) . No Personally Identifiable Information (PII) or sensitive user content is involved.

### 2. Bias Mitigation

Real-world workloads can be highly skewed. By curating multi-cloud datasets that cover varying behaviors, we mitigate the risk of overfitting to specific provider architectures or workload types.

### 3. Environmental Responsibility

The core objective of CloudCons is to improve resource Consolidation. While the direct measure is reducing the number of active physical servers, the ultimate goal is to enhance resource utilization efficiency, thereby minimizing energy consumption.

