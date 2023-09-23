![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
# End-to-end Machine Learning Pipeline

This repository contains an end-to-end machine learning pipeline for predicting O-level scores of students. It serves as a machine learning practice problem for me to practice various tools.

| Concept  | Tools |
| ------------- | ------------- |
| Hyperparameter Tuning  | MLflow, Hyperopt  |
| Deployment  | FastAPI, Docker  |
| Software Engineering | tox, Unit testing |

<img src="https://github.com/Joanna-Khek/o-level-score-regression-model/assets/53141849/187dbe05-b77d-4faf-a886-1b001de55d7c" width="150" height="60">

<img src="https://github.com/Joanna-Khek/o-level-score-regression-model/assets/53141849/3c087b89-08d0-430d-b3b3-bcbd14025da7" width="100" height="100">

<img src="https://github.com/Joanna-Khek/o-level-score-regression-model/assets/53141849/f3814583-6a69-4e58-bf7e-248a21d3d4aa" width="300" height="100">

<img src="https://github.com/Joanna-Khek/o-level-score-regression-model/assets/53141849/3811f491-b9a5-420e-b26f-971b0650c5ed" width="180" height="60">

<img src="https://github.com/Joanna-Khek/o-level-score-regression-model/assets/53141849/744dfaaf-4954-441e-af33-01adab369800" width="100" height="50">

<img src="https://github.com/Joanna-Khek/o-level-score-regression-model/assets/53141849/7012fe97-c6a8-483f-bcd0-94e4d3451d2e" width="100" height="80">





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
