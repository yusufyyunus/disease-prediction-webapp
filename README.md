# Health Assessment Web Application 🩺💻

## 📂 Project Structure

### Folders Overview
- **`colab/`**: Data science and model training resources
  - `datasets/`: Contains raw and processed datasets
  - `disease_prediction_training.ipynb`: Jupyter notebook for model training

- **`model/`**: Pre-trained machine learning models
  - Contains Random Forest models for:
    * Diabetes prediction
    * Stroke prediction
    * Hypertension prediction

- **`models/`**: Base models for the main application
  - Core machine learning and prediction models

- **`utils/`**: Utility functions
  - Helper scripts and common utility functions

## 🚀 Setup and Installation

### Prerequisites
- Python 3.8+
- pip (Python package manager)

### Installation Steps

1. **Download the code**
   Extract the zip file to a folder

2. **Create Virtual Environment** (Optional but Recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Development Setup

#### Backend Server
1. Run the FastAPI backend:
   ```bash
   uvicorn main:app --reload
   ```

#### Frontend
1. **VS Code Live Server**
   - Install Live Server Extension on VS Code
   - Open `index.html`
   - Click "Go Live" at the bottom right of VS Code

#### Database Access
- **SQLite Database**: `health_assessment.db`
- Recommended: Install SQLite extension in VS Code

### MailHog Installation (Mac)

#### Homebrew
```bash
brew install mailhog
brew services start mailhog
```

Hopefullt this tutorial is helpful: https://www.youtube.com/watch?v=IWJKRmFLn-g

## 🌐 Accessing the Application

- **Web App**: `http://localhost:5500`
- **Backend API**: `http://localhost:8000`
- **MailHog SMTP**: `localhost:1025`
- **MailHog Web Interface**: `http://localhost:8025`