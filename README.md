![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
# End-to-end Machine Learning Pipeline
This repository contains an end-to-end machine learning pipeline for predicting O-level scores of students.

| Stage  | Tools |
| ------------- | ------------- |
| Hyperparameter Tuning  | MLFlow, Hyperopt  |
| Deployment  | FastAPI, Docker  |

## Objective
The main objective of this project is to predict O-level scores for students. By building a comprehensive machine learning pipeline, we aim to achieve accurate predictions that can assist educators and students in understanding and improving academic performance.

## Key Features
- **Machine Learning Workflow**: The pipeline encompasses the entire machine learning workflow, from data preprocessing and model selection to training, evaluation, and deployment.

- **MLFlow Integration**: We use MLFlow to manage experiments, track model versions, and provide model reproducibility, making it easier to collaborate and iterate.

- **Hyperparameter Optimization**: Hyperopt is employed to automate hyperparameter tuning, ensuring that our models are optimized for predictive accuracy.

- **FastAPI API**: We create a FastAPI web service that exposes endpoints for making predictions with the trained models. This allows easy integration with other applications.

- **Dockerization**: The entire project is containerized using Docker, ensuring consistent environments across development and deployment.

## Getting Started

To get started with this project, follow these steps:

**1. Clone the repository:**
   ```bash
   git clone https://github.com/Joanna-Khek/o-level-score-regression-model
   cd your-repo
   ```
**2. Build and run docker image**
   ```bash
   docker build -t regression-model:latest .
   docker run -p 8001:8001 regression-model:latest
   ```
**3. Access FastAPI API**
   ```bash
   http://localhost:8001/
   ```
