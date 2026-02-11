# CloudCons Datasets

## Overview

The **[CloudCons](https://huggingface.co/datasets/kdd2026-cloudcons/CloudCons-ds)** datasets comprise heterogeneous workload traces collected from real-world cloud production environments. Unlike traditional time series benchmarks that focus solely on forecasting accuracy, the datasets are designed for an **End-to-End Evaluation** framework.

**Core Tasks:**

1.  **Time Series Forecasting:** Evaluating the accuracy of models in predicting future resource demands.
2.  **Resource Consolidation:** Assessing the downstream decision utility of forecasts by simulating virtual machine consolidation.

**Problem Solved:**
Existing benchmarks often fail to bridge the gap between forecasting accuracy (e.g., MSE, MAE) and downstream decision utility (e.g., resource efficiency, service reliability). CloudCons addresses this misalignment by providing a standardized benchmark to verify the practical utility of statistical, deep learning, and foundation models in dynamic cloud resource consolidation scenarios.

## Data Collection

The datasets are derived from three major cloud providers: **Huawei Cloud**, **Microsoft Azure**, and **Google Borg**. The raw data sources are publicly available and have been curated to ensure high quality and diversity.

### Data Sources

* **Huawei2025**
    * **Source:** [Huawei Cloud Traces](https://github.com/sir-lab/data-release/blob/main/README_data_release_2025.md)
    * **Description:** Captures 31 days of VM and serverless workload traces from a production environment, featuring diverse patterns including periodicity and noise.
* **Azure2019**
    * **Source:** [Azure Public Dataset V2](https://github.com/Azure/AzurePublicDataset/blob/master/AzurePublicDatasetV2.md)
    * **Description:** Provides detailed VM workload characteristics, including lifecycle and resource consumption, collected from Azure’s production clusters. It is characterized by "pulse-like" bursty workloads.
* **Borg2019 (d & e)**
    * **Source:** [Google ClusterData 2019 Traces](https://github.com/google/cluster-data/blob/master/ClusterData2019.md)
    * **Description:** Records machine events and instance usage within Google’s compute cells.
        * **Borg2019-d:** Characterized by low utilization and high-frequency jitter.
        * **Borg2019-e:** Exhibits strong 24-hour diurnal rhythms (overnight trough and afternoon peak).

## Metadata and Statistics

Data are organized into two categories: datasets for general time series forecasting tasks and datasets specifically processed for resource consolidation tasks. All data files are stored in **Apache Arrow** format for efficient loading.

### Time Series Forecasting Datasets

| Dataset        | Number of Series | Total Points | Attributes                                                   |
| :------------- | :--------------- | :----------- | :----------------------------------------------------------- |
| **Huawei2025** | 174              | 3.3 M        | `avg_cpu_usage`, `avg_memory_usage` , `requests`, `start`,`frequency` |
| **Azure2019**  | 10,800           | 104.5 M      | `avg_cpu_usage`, `min_cpu_usage`, `max_cpu_usage`, `start_time`,`frequency` |
| **Borg2019-d** | 414              | 3.9 M        | `avg_cpu_usage`, `avg_memory_usage`, `assigned_memory`, `page_cache_memory`, `start_time`,`frequency` |
| **Borg2019-e** | 618              | 5.9 M        | `avg_cpu_usage`, `avg_memory_usage`, `assigned_memory`, `page_cache_memory`, `start_time`,`frequency` |

### Resource Consolidation Datasets

| Dataset        | # VMs | Time Span     | Key Attributes                                     |
| :------------- | :---- | :------------ | :------------------------------------------------- |
| **Huawei2025** | 575   | $\ge$ 14 days | `cpu_usage` (normalized), `start_time`,`frequency` |
| **Azure2019**  | 150   | $\ge$ 14 days | `cpu_usage` (normalized), `start_time`,`frequency` |
| **Borg2019-d** | 127   | $\ge$ 14 days | `cpu_usage` (normalized), `start_time`,`frequency` |
| **Borg2019-e** | 68    | $\ge$ 14 days | `cpu_usage` (normalized), `start_time`,`frequency` |

## Curation & Preprocessing

The detailed data processing pipeline—including cleaning, imputation, aggregation, and feature extraction—is described in the paper.

**Balanced Sampling (BLAST):** To facilitate easier access and improve evaluation efficiency, we employed the "BLAST" balanced sampling method. This approach is based on the KDD '25 paper: *["BLAST: Balanced Sampling Time Series Corpus for Universal Forecasting Models"](https://dl.acm.org/doi/10.1145/3711896.3736860)*.

We provide a specific subset of the data that reflects the diversity of the original datasets. Researchers aiming to evaluate **forecasting tasks** can utilize this subset to enhance efficiency.

## Accessibility & Usage

Due to file size limitations, the datasets are hosted on Hugging Face. You can access them here:   🤗**[Hugging Face Dataset: kdd2026-cloudcons/CloudCons-ds](https://huggingface.co/datasets/kdd2026-cloudcons/CloudCons-ds)**
