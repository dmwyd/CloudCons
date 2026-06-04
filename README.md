# Cloud Consolidation-Cloudcons

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

We provide a rigorous analysis of predictive quantile selection, offering a strategic lever to balance the trade-off between resource efficiency and service reliability. Our findings guide practitioners on when to use median forecasts (cost-oriented) versus high-quantile forecasts (reliability-oriented) .



## 🚀 How to use? 

### 1. Data Access

All processed datasets (Huawei2025, Azure2019, Borg2019-d, Borg2019-e) are hosted on Hugging Face. You can access and download them directly via the following link: [https://huggingface.co/datasets/kdd2026-cloudcons/CloudCons-ds](https://huggingface.co/datasets/kdd2026-cloudcons/CloudCons-ds). 

### 2. Source Code

- **`forecasting_bench/`**: **Time Series Forecasting Evaluation**
- **`predictor/`**: **Statistical & Foundation Models**
- **`DeepModel/`**: **Deep Learning Model Training & Evaluation**
- **`simulation/`**: **End-to-End Simulation & Optimization**
  

## ⚖️ Data Provenance and Privacy 

- **Source:** The datasets included in CloudCons are derived from publicly released traces by Huawei Cloud (via `sir-lab`), Microsoft Azure (Azure Public Dataset V2), and Google Borg (ClusterData 2019). As a curated benchmark, CloudCons represents a derivative work—involving filtering, normalization, and format conversion—and is distributed in compliance with the original data owners' open-source license (CC BY 4.0).
- **Anonymization:** These datasets contain strictly technical metric logs (e.g., CPU usage, memory usage) . No Personally Identifiable Information (PII) or sensitive user content is involved.
