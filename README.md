# Academic Paper Discipline Classification System

A sophisticated academic research paper classification system that automatically categorizes research papers by discipline using deep learning embeddings and unsupervised clustering techniques.

## Overview

This system analyzes PDF research papers and automatically classifies them into academic disciplines (e.g., Computer Science, Social Science, Natural Science, Mathematics, Engineering, Medicine/Health) using a combination of:

- **SciBERT embeddings** for semantic text representation
- **K-means clustering** for unsupervised grouping
- **Keyword-based discipline prediction** for automatic labeling
- **Silhouette analysis** for optimal cluster determination

## Features

- **PDF Text Extraction**: Automatically extracts and preprocesses text from academic PDFs
- **Semantic Clustering**: Groups papers based on content similarity using SciBERT embeddings
- **Discipline Classification**: Predicts academic disciplines using keyword-based scoring
- **Visualization**: 2D PCA visualization of paper clusters
- **Performance Metrics**: Silhouette analysis and cluster quality assessment

## Required Libraries

```python
import os
import PyPDF2
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, silhouette_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
```

## Installation

```bash
pip install transformers torch PyPDF2 scikit-learn matplotlib pandas numpy
```

## Directory Structure

```
project/
│
├── papers/                 # Directory containing PDF files
│   ├── paper1.pdf
│   ├── paper2.pdf
│   └── ...
│
└── Methology_Research.ipynb  # Main notebook
```

## Usage

1. **Place PDF files** in a `papers/` directory
2. **Run the notebook** which will:
   - Extract text from all PDFs
   - Generate SciBERT embeddings
   - Perform clustering analysis
   - Predict academic disciplines
   - Generate visualizations

## Key Components

### 1. Text Extraction
- Uses PyPDF2 to extract text from PDF files
- Processes first 5 pages for better classification
- Handles extraction errors gracefully

### 2. SciBERT Embeddings
- Utilizes `allenai/scibert_scivocab_uncased` model
- Generates 768-dimensional embeddings
- Truncates text to 512 tokens (model limit)

### 3. Clustering Analysis
- K-means clustering with automatic k selection
- Silhouette analysis for optimal cluster number
- Quality metrics including cluster cohesion

### 4. Discipline Classification
- Keyword-based scoring system
- Confidence scores for predictions
- Supports 6 main disciplines:
  - Computer Science
  - Social Science  
  - Natural Science
  - Mathematics
  - Engineering
  - Medicine/Health

### 5. Visualization
- 2D PCA projection of high-dimensional embeddings
- Color-coded clusters with paper labels
- Silhouette score analysis plots

## Output

The system provides:
- **Individual paper classifications** with confidence scores
- **Cluster analysis** showing paper groupings
- **Discipline distribution** within each cluster
- **Final classification results** mapping papers to disciplines
- **Visualization plots** for cluster analysis


This system provides a robust foundation for automated academic paper classification and can be extended for specific research domains or institutional requirements.
