# CloudBurst Prediction using Machine Learning

A Machine Learning project that builds a binary classification system to predict whether a **cloudburst will occur the next day**, using current meteorological readings as input features.

---

## Project Structure

```
Final/
├── CloudBurst_ML_Project.ipynb        # Main Jupyter Notebook (all 4 phases)
├── cloudpredictionsystemproject.csv   # Dataset (145,460 rows x 23 columns)
├── app.py                             # Streamlit web application
├── cloudburst_rf_model.pkl            # Trained Random Forest model
├── preprocessor.pkl                   # Fitted data preprocessor
├── scaler.pkl                         # Fitted feature scaler
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

---

## Dataset

- **Source:** [Kaggle — CloudBurst Dataset](https://www.kaggle.com/datasets/akshat234/cloudburst)
- **Size:** 145,460 rows x 23 columns
- **Location:** Multiple Australian cities (Albury, Sydney, Melbourne, Brisbane, etc.)
- **Target Variable:** `CloudBurstTomorrow` — Will a cloudburst occur tomorrow? (Yes/No)

The dataset file `cloudpredictionsystemproject.csv` is included in this repository.

---

## Requirements

### Python Version

Python **3.8+** is recommended.

### Install Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter streamlit joblib
```

---

## How to Run

### Option 1 — Jupyter Notebook (Recommended for Training & Analysis)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/prithvirajarora/cloudburst-ml-project.git
   cd cloudburst-ml-project
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```

4. **Open** `CloudBurst_ML_Project.ipynb` in the browser tab that opens.

5. **Run all cells:**
   Go to **Kernel > Restart & Run All** to execute the full pipeline end-to-end.

### Option 2 — VS Code with Jupyter Extension

1. Open the project folder in VS Code.
2. Install the [Jupyter extension](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter).
3. Open `CloudBurst_ML_Project.ipynb`.
4. Click **Run All** at the top of the notebook.

### Option 3 — Streamlit Web Application

To launch the interactive prediction interface:

```bash
streamlit run app.py
```

Ensure that `cloudburst_rf_model.pkl`, `preprocessor.pkl`, and `scaler.pkl` are present in the same directory as `app.py` before running.

---

## Project Phases

The notebook is organized into **4 phases** corresponding to the project grading rubric:

| Phase | Description |
|-------|-------------|
| **Phase 1** | Data Understanding and Preprocessing — Exploratory data analysis, missing value handling, and feature encoding |
| **Phase 2** | Supervised Learning — Logistic Regression baseline, performance metrics, and model evaluation |
| **Phase 3** | Optimization and Unsupervised Learning — Regularization techniques, PCA dimensionality reduction, and K-Means Clustering |
| **Phase 4** | Advanced Models — Decision Tree, Random Forest, Hyperparameter Tuning via GridSearchCV, and final model comparison |

---

## Key Results

| Model | Accuracy | F1 Score | AUC-ROC |
|-------|----------|----------|---------|
| Logistic Regression (Baseline) | ~0.76 | ~0.60 | ~0.81 |
| Logistic Regression (Optimized) | ~0.77 | ~0.61 | ~0.82 |
| Decision Tree (Tuned) | ~0.78 | ~0.62 | ~0.83 |
| **Random Forest (Tuned)** | **~0.82** | **~0.67** | **~0.88** |

**Final Model:** Tuned Random Forest — achieves the highest F1 Score and AUC-ROC across all evaluated models.

---

## Notes

- The notebook expects the dataset file `cloudpredictionsystemproject.csv` to be located in the **same directory** as the notebook file.
- Model training may take **several minutes** on the full dataset (145,460 rows), particularly during GridSearchCV for the Random Forest model.
- All plots are generated inline within the notebook.
- The Streamlit application requires all three model artifact files (`cloudburst_rf_model.pkl`, `preprocessor.pkl`, `scaler.pkl`) to be present at runtime.

---

## Report

Refer to the full written project report submitted alongside this repository for detailed methodology, analysis, and conclusions.

---

## Course Information

- **Student:** Prithviraj Arora
- **Enrollment No.:** 2023BTCSE025L
- **Course:** Machine Learning
- **Institution:** Jagran Lakecity University (JLU)
- **Semester:** 6
