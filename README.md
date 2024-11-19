# LLM-based Query System for Traffic Analysis Using CCTV Footage

This repository contains the code and resources for developing a **LLM-based Query System** for analyzing CCTV footage to extract traffic statistics and detect accidents. The pipeline integrates state-of-the-art models for video analysis, including YOLO for object detection and LLAVA Onevision for multimodal inference. 

---

## Table of Contents
- [Introduction](#introduction)
- [Pipeline Overview](#pipeline-overview)
- [Features](#features)
- [Requirements](#requirements)
- [Setup](#setup)
- [Usage](#usage)
- [Dataset Details](#dataset-details)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Model Deployment](#model-deployment)
- [Future Work](#future-work)
- [References](#references)

---

## Introduction

This project aims to develop a system that processes natural language queries on CCTV footage to:
- Identify specific vehicles based on attributes (e.g., type, color).
- Count and classify vehicles.
- Detect traffic accidents.
- Generate statistical reports for traffic management.

The system combines transformer-based large language models (LLMs) with computer vision pipelines for robust and efficient traffic analysis.

---

## Pipeline Overview

The main pipeline consists of the following steps:

1. **Frame Extraction**:
   - Extract frames from video files at regular intervals.

2. **Vehicle Detection and Annotation**:
   - Use YOLOv5 to detect and classify vehicles into categories such as cars, trucks, bikes, and bicycles.
   - Generate annotations in JSON format for training and validation.

3. **Inference**:
   - Use the `LlavaOnevisionForConditionalGeneration` model to process annotated frames and generate responses.

4. **Finetuning**:
   - Employ QLoRA for finetuning on custom datasets.
   - Optimize recall and precision for accident detection and vehicle classification tasks.

5. **Evaluation**:
   - Evaluate using metrics like Accuracy, F1 Score, Precision, Recall, and Mean Absolute Error (MAE).

---

## Features

- **Accident Detection**: Identify traffic accidents in video footage.
- **Vehicle Counting**: Count vehicles by type (car, truck, bike, bicycle).
- **Interactive Video QA**: Query video data using natural language.
- **Efficient Finetuning**: Use QLoRA to adapt large LLMs efficiently.
- **Dataset Annotation**: Automatically generate and preprocess dataset annotations.

---

## Requirements

- Python 3.9+
- CUDA-enabled GPU for faster processing.
- Libraries:
  - PyTorch
  - Hugging Face Transformers
  - OpenCV
  - YOLOv5
  - WandB
  - Jupyter Notebook

Install the required libraries using:
```bash
# LLM-Based Query System for Traffic Analysis Using CCTV Footage

This repository contains the implementation of an LLM-based query system designed to analyze traffic scenarios using CCTV footage. The system uses multimodal large language models (LLMs) to process and query traffic data, enabling insights into vehicle counts, accidents, and traffic patterns.

---

## Setup

### Clone the Repository
bash
git clone https://github.com/Shubham23094/LLM-based-Query-System-for-Traffic-Analysis-Using-CCTV-Footage.git
cd LLM-based-Query-System-for-Traffic-Analysis-Using-CCTV-Footage


### Install Dependencies
bash
pip install -r requirements.txt


### Download the Datasets
- *DoTA*: Download and place the dataset in the datasets/dota directory.
- *HEVI*: Download and place the dataset in the datasets/hevi directory.

### Configure Paths in the Notebook
- Open /llava_finetune_lora_hf.ipynb and update the paths for the datasets and outputs.

---

## Usage

### Running the Pipeline
1. *Open the Jupyter Notebook:*
   bash
   jupyter notebook
   
2. **Navigate to /llava_finetune_lora_hf.ipynb** and run the cells sequentially to:
   - Preprocess datasets.
   - Train and finetune the model using QLoRA.
   - Generate inference results.

### Command Line Usage
To run the finetuning script directly:
bash
python llava_finetune.py --data_dir ./datasets --output_dir ./llava_finetuned


---

## Dataset Details

### DoTA Dataset
- *Description*: Accident detection dataset with 1,450 annotated videos.
- *Classes*: Cars, Trucks, Bikes, Bicycles, Accident/Non-Accident.

### HEVI Dataset
- *Description*: High-resolution videos focusing on traffic scenarios (250 videos).
- *Annotations*: Includes labels for accident and non-accident cases.

### Dataset Split
- *Training Set*: 100 accident and 100 non-accident videos.
- *Test Set*: 50 accident and 50 non-accident videos.

---

## Evaluation Metrics

The following metrics are used to evaluate the system:
- *Accuracy*
- *F1 Score*
- *Precision and Recall*
- *Mean Absolute Error (MAE)*

---

## Results

### Baseline vs Finetuned Model
| Metric         | Baseline Model | Finetuned Model |
|----------------|----------------|-----------------|
| *Accuracy*   | 0.65           | 0.80            |
| *MAE (Car)*  | 4.30           | 2.68            |
| *MAE (Truck)*| 0.90           | 0.72            |
| *MAE (Bike)* | 0.50           | 0.70            |
| *MAE (Bicycle)*| 0.65         | 0.55            |

---

## Model Deployment

The finetuned model is deployed as a chatbot for interactive querying. Users can query:
- Vehicle counts.
- Traffic accidents.
- Time intervals with specific traffic patterns.

---

## Future Work

- Deploy the model for real-time analysis.
- Expand the dataset to improve generalization across diverse scenarios.
- Optimize the pipeline for lower latency and scalability.

---

## References

1. *Jangam et al.*, "Leveraging LLMs for Video Querying," DOI.
2. *Tami et al.*, "Using Multimodal LLMs for Automated Detection of Traffic Safety-Critical Events," DOI.
3. *Kong et al.*, "WTS: A Pedestrian-Centric Traffic Video Dataset," DOI.
4. *Patel et al.*, "Multimodal Video Analysis Using LLMs for Traffic Safety and Efficiency," DOI.

---

## Acknowledgements

This project was developed as part of the coursework at IIIT Delhi under the guidance of *Prof. Rajiv Ratn Shah*.

