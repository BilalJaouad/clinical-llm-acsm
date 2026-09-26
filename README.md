# Adapting Open-Source LLMs for Safe Cardiac Exercise Prescription

This repository contains the methodology, data processing pipelines, and deployment scripts for adapting open-source Large Language Models (LLMs) to safely prescribe cardiac exercise intensity based on the American College of Sports Medicine (ACSM) guidelines.

## Clinical Context
Cardiovascular diseases require precise, tailored rehabilitation. Medical professionals rely on rigorous ACSM guidelines to prescribe exercise safely. However, generalist open-source LLMs lack the specialized deterministic reasoning required to act as reliable clinical assistants, posing significant safety risks (such as over-prescribing vigorous exercise to high-risk patients).

This project addresses this reliability gap by evaluating and fine-tuning medical foundation models specifically for cardiac rehabilitation.

## Methodology

### 1. Benchmark and Model Selection
We developed a synthetic ground-truth benchmark of 50 patient vignettes to evaluate models' abilities to classify clinical exercise intensity safely (Classes A, B, C, D).
* **Models Evaluated:** Llama-3-8B-Instruct, MEDITRON-7B, and BioMistral-7B.
* **Results:** BioMistral-7B emerged as the safest and most accurate foundation model, achieving 88.0% accuracy using a Few-Shot English prompting strategy and eliminating severe over-prescription errors entirely.

### 2. Corpus Preparation
A specialized cardiac corpus was ingested and cleaned to adapt the base model:
* **Patient Education Workbooks & Protocols:** Cleaned programmatically to remove noise and isolate relevant languages.
* **Complex Documents:** Utilized the `Camelot` library to parse multi-column clinical tables (e.g., CPG STEMI 2019) and `Tesseract` OCR for complex vector image flowcharts.

### 3. QLoRA Fine-Tuning
The model was fine-tuned using Parameter-Efficient Fine-Tuning (PEFT):
* **Base Model:** BioMistral-7B-BnB.4 (4-bit quantization).
* **Hardware:** Google Colab T4 GPU (16 GB VRAM).
* **LoRA Parameters:** Rank (r=16), Alpha=32, targeting `q_proj`, `k_proj`, `v_proj`, and `o_proj`.

## Installation and Deployment

The deployment stack relies on `vLLM` for high-throughput batching and inference, paired with a `Streamlit` application for the user interface.

### Prerequisites
* Python 3.9+
* CUDA-compatible GPU

### Setup Instructions

1. Clone the repository:
   ```bash
   git clone [https://github.com/BilalJaouad/clinical-llm-acsm.git](https://github.com/BilalJaouad/clinical-llm-acsm.git)
   cd clinical-llm-acsm
