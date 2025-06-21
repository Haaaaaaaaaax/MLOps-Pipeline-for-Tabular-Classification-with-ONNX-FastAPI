# Titanic Classification with PyTorch, ONNX, FastAPI, and Docker

This project predicts Titanic passenger survival using a neural network built with PyTorch, exported to ONNX for efficient inference, and deployed as a FastAPI web service. The app is containerized with Docker and designed for automated deployment (CI/CD) via GitHub Actions to Docker Hub.

## Features

- **Model Training**: PyTorch-based Artificial Neural Network (ANN) trained on Titanic dataset.
- **Model Export**: Trained model exported to ONNX format for fast, portable inference.
- **API Service**: FastAPI app for serving predictions via a RESTful endpoint.
- **Containerization**: Dockerfile for easy deployment anywhere.
- **CI/CD Ready**: Intended for automated Docker builds and pushes using GitHub Actions.
- **Notebook Workflow**: Jupyter notebooks for data preprocessing and model training.

## Project Structure

```
.
├── Dockerfile
├── infer.py           # ONNX inference logic
├── main.py            # FastAPI app
├── Models/
│   ├── ANN.onnx       # Exported ONNX model
│   └── ANN.pth        # PyTorch model weights
├── Notebooks/
│   ├── Preprocess.ipynb
│   └── train.ipynb
└── README.md
```

## How it Works

1. **Training**:  
   - Data is preprocessed and the ANN is trained in `Notebooks/train.ipynb`.
   - Model is saved as both PyTorch (`ANN.pth`) and ONNX (`ANN.onnx`).

2. **Inference**:  
   - `infer.py` loads the ONNX model and provides a `predict` function.
   - `main.py` exposes a `/predict` endpoint using FastAPI, accepting passenger features and returning survival prediction.

3. **Deployment**:  
   - The Dockerfile sets up the environment and runs the FastAPI app with Uvicorn.
   - GitHub Actions can be configured to build and push the Docker image to Docker Hub automatically.

## Quickstart

### 1. Build and Run with Docker

```bash
docker build -t titanic-fastapi-app .
docker run -p 8000:8000 titanic-fastapi-app
```

### 2. API Usage

Send a POST request to `http://localhost:8000/predict` with JSON body:
```json
{
  "features": [3, 1, 22.0, 1, 0, 7.25, 2]
}
```
Response:
```json
{
  "Prediction": 0
}
```

### 3. Training the Model

- Use the Jupyter notebooks in `Notebooks/` to preprocess data and train the model.
- Export the trained model to ONNX as shown in the notebook.

## Requirements

- Python 3.11+
- PyTorch, ONNX, ONNX Runtime, FastAPI, Uvicorn, NumPy, Pandas, scikit-learn

## CI/CD (GitHub Actions)

You can automate Docker builds and pushes to Docker Hub by adding a workflow YAML file in `.github/workflows/`.  
*(Sample workflow can be provided if needed.)*

---
