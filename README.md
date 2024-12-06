[![ML Pipeline](https://github.com/SunilKV123/ERAv3-Assignment5/actions/workflows/ml-pipeline.yml/badge.svg)](https://github.com/SunilKV123/ERAv3-Assignment5/actions/workflows/ml-pipeline.yml)
# MNIST CNN Model with CI/CD Pipeline

This project implements a CNN model for MNIST digit classification with a complete CI/CD pipeline.

    mnist_pipeline/
    │
    ├── .github/
    │ └── workflows/
    │ └── ml_pipeline.yml # GitHub Actions workflow configuration
    │
    ├── src/
    │ ├── init.py # Makes src a Python package
    │ ├── model.py # CNN model architecture
    │ ├── train.py # Training script
    │ └── utils.py # Utility functions
    │
    ├── tests/
    │ ├── init.py # Makes tests a Python package
    │ ├── test_model.py # Model architecture tests
    │ └── test_training.py # Training pipeline tests
    │
    ├── models/ # Directory for saved models
    │ └── .gitkeep # Keeps empty models directory in git
    │
    ├── .gitignore # Specifies which files Git should ignore
    ├── requirements.txt # Project dependencies for GPU development
    ├── requirements-cpu.txt # Project dependencies for CPU-only execution
    ├── setup.py # Package installation configuration
    └── README.md # Project documentation

## Requirements
- Python 3.8+
- PyTorch (CPU or GPU version)
- torchvision
- pytest

## Installation

### For GPU Development
```bash
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```
### For CPU-Only Development or CI/CD
```bash
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate
pip install -r requirements-cpu.txt
pip install -e .
```
## Local Testing
```bash
python -m unittest discover tests/
python src/train.py
```

## CI/CD Pipeline

The GitHub Actions workflow automatically:
1. Sets up a Python environment
2. Installs CPU-only dependencies
3. Runs all tests
4. Trains the model
5. Saves the trained model as an artifact

## Model Details
- Architecture: Convolutional Neural Network (CNN)
- Parameters: <25,000
- Input shape: 28x28 grayscale images
- Output: 10 classes (digits 0-9)
- Target accuracy: >95% training accuracy in 1 epoch

## Model Artifacts
Models are automatically saved with the following naming convention:
`mnist_model_<accuracy>acc_<timestamp>.pth`


## GitHub Actions
The pipeline runs automatically on every push to the repository:
- Uses CPU-only PyTorch version
- Executes all tests
- Trains model
- Stores model artifacts
- Results viewable in Actions tab

## Development Notes
- Local development supports both CPU and GPU
- CI/CD pipeline runs on CPU only
- All tests must pass for successful deployment
- Model artifacts are retained for 90 days


## Notes
- The model is trained on CPU in GitHub Actions
- Trained models are saved as artifacts in the workflow
- All tests must pass for successful deployment


## To use this project:
1. Clone the repository
2. Create a new branch for your changes
3. Push your changes to the new branch
4. Create a pull request to merge your changes into the main branch
5. The workflow will automatically run and deploy your changes if all tests pass
