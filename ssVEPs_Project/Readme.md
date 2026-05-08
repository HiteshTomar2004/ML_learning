# EEG-Based Machine Learning Analysis for ssVEP Task Classification and Neural Feature Prediction
# Project Overview
This repository contains an end-to-end data science and machine learning pipeline for analyzing continuous Brain-Computer Interface (BCI) data. The project leverages Steady-State Visual Evoked Potentials (ssVEPs) to decode cognitive intent and evaluate neural pathway health.

Unlike standard tabular datasets, this project tackles the complexities of raw, noisy time-series physiological data, requiring rigorous digital signal processing, feature extraction, and leakage-proof cross-validation strategies.

## Key Objectives:
BCI Intent Recognition (Classification): Decode the subject's current cognitive task (e.g., Spatial Image Search vs. Motor Imagery) based solely on 1-second epochs of brainwave activity.

Cognitive Agility & Neural Health (Regression): Investigate if demographic factors (Age, Gender) biologically impact visual pathway speed (using an occipital P100 latency proxy) or focus acquisition time.

# Dataset
The project utilizes the BCI-SSVEP Database collected at the Autonomous University of Queretaro, Mexico.

Subjects: 30 individuals with varied demographics (Age, Gender, Vision parameters).

Hardware: Commercial Emotiv EEG headset (14 channels: AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4).

Sampling Rate: 128 Hz.

### Tasks: Visual tracking (Five Box), complex spatial scanning (Image Search), and Motor Imagery (Hand Shake).

# Methodology & Pipeline Architecture
## 1. Data Fusion & Ingestion
Raw .csv files contain only anonymous voltage readings. The pipeline programmatically maps static clinical metadata (from Signal Database.xlsx) to thousands of raw EEG files, broadcasting Subject ID, Age, Gender, and Target Task to every 7.8ms sample.

## 2. Digital Signal Processing & Feature Engineering
Raw voltage is biologically noisy. The pipeline extracts machine-learning-ready features:

Filtering: A 1-30 Hz Butterworth bandpass filter is applied to the continuous session to prevent edge artifacts.

Epoching: Data is sliced into 1-second windows (128 samples/epoch).

Frequency Domain (Focus): Welch's Method (Power Spectral Density) extracts Alpha (8-12 Hz), Beta (12-30 Hz), and Gamma (30-60 Hz) power across all 14 channels.

Time Domain (Wiring Speed): An algorithm scans the 80-120 ms window of the O1/O2 (occipital) electrodes to extract a positive-peak latency, acting as a P100 Proxy for visual pathway speed.

## 3. Exploratory Data Analysis (EDA)
To prevent statistical inflation from repeated measures (epochs), data is strictly aggregated to the Session_ID level before hypothesis testing.

T-Tests & Mann-Whitney U Tests: Evaluated baseline differences in P100 latency across genders.

Correlation Heatmaps: Proved high spatial multicollinearity among posterior electrodes, justifying regularized regression models.

## 4. Predictive Modeling
Intent Decoder (Classification): Uses Random Forest and XGBoost to classify the cognitive task.

Leakage Prevention: Implements StratifiedGroupKFold cross-validation grouped by Subject_ID to ensure the model learns generalized task signatures rather than memorizing individual skull/brain topographies.

Interpretability: Permutation importance and SHAP are used to evaluate the biological relevance of Age and Gender in predicting focus speed.

## Key Findings
Subject-Generalizable Decoding: The Random Forest classifier successfully beat the Dummy Majority Baseline (27.86% vs 22.64%) using strict grouped cross-validation, proving evidence of generalizable neurological task signatures across different human brains.

Biological Task Validation: The model excelled at isolating complex visual processing (IMAGE SEARCH F1-score: 0.38) but struggled to classify Motor Imagery using primarily visual/occipital electrodes, perfectly aligning with expected neuroanatomy.

Demographic Impact on Neural Speed: While non-parametric tests showed a "statistically significant" difference in visual pathway speed (P100 proxy) between genders, the absolute difference was <1 ms. Because the hardware resolution is 7.8 ms (128 Hz), we concluded this difference is a mathematical artifact and biologically negligible.

# How to Run the Project
Prerequisites
Make sure you have Python 3.8+ installed along with the following libraries:
pip install pandas numpy scipy scikit-learn xgboost matplotlib seaborn openpyxl

### Execution
Clone the repository:
git clone https://github.com/HiteshTomar2004/SSVEP-ML-Pipeline.git

Ensure the dataset files (Signal Database.xlsx and the .csv EEG recordings) are located in the root directory.
dataset link : https://drive.google.com/drive/folders/1VlWuuyYTGiPJBRfiOQXJJxfPCcc9hVu6?usp=sharing

Open and run the Jupyter Notebook:
jupyter notebook SSVEPsML_Project.ipynb
Note: The notebook handles the full pipeline sequentially (Data Fusion -> Feature Engineering -> EDA -> Modeling).

## Repository Structure
SSVEPsML_Project.ipynb - Main Jupyter Notebook containing the full ML pipeline

Project_Report_ssVEPs.pdf - Detailed academic report of methodologies and findings

README.md - Project documentation

✍️ Author
Hitesh Tomar
