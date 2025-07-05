# Machine Learning & Deep Learning Portfolio

Welcome to my portfolio of machine learning projects! This repository contains a collection of notebooks where I explore various datasets, apply different machine learning algorithms, and practice the end-to-end data science workflow.

My goal is to continuously add new projects, tackling different challenges and exploring a wide range of techniques, from foundational models to advanced deep learning architectures.

---

## About Me

I am a passionate and aspiring Machine Learning Engineer with a strong foundation in Python and a keen interest in turning data into actionable insights. I enjoy the process of diving deep into data, uncovering patterns, and building models that solve real-world problems.

*   **LinkedIn:** [linkedin.com/in/amazingpaddy](https://www.linkedin.com/in/amazingpaddy/)
*   **Email:** padmanabhan5789@gmail.com

---

## Skills & Technologies

Here are some of the key skills and technologies I've demonstrated in my projects or am actively developing.

*   **Languages:** Python
*   **Core Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
*   **Machine Learning:** Supervised Learning (Classification, Regression), Exploratory Data Analysis (EDA), Feature Engineering & Selection, Hyperparameter Tuning (Cross-Validation, GridSearchCV), Model Evaluation (Accuracy, Confusion Matrix, Precision-Recall Trade-off)
*   **Future Learning & Development:**
    *   **Deep Learning Frameworks:** TensorFlow, Keras, PyTorch
    *   **Advanced Topics:** Neural Networks, Computer Vision (CNNs), Natural Language Processing (NLP)
*   **Tools:** Jupyter Notebook, Git & GitHub

---

## Projects

Below is a summary of the projects included in this repository. Each project includes a detailed notebook with code, visualizations, and explanations.

### 1. Breast Cancer Classification: A Study in Model Responsibility

*   **Notebook:** [`breast-cancer-classification-project/breast_cancer_prediction.ipynb`](./breast-cancer-classification-project/breast_cancer_prediction.ipynb)
*   **Objective:** To build a model that can accurately and, more importantly, safely classify breast tumors as malignant or benign.
*   **Models Used:** K-Nearest Neighbors (KNN), Logistic Regression
*   **Key Steps & Findings:**
    *   **Problem Framing:** Identified that for this medical diagnosis task, minimizing **False Negatives** (failing to detect a malignant tumor) is far more critical than overall accuracy.
    *   **Baseline Modeling:** A tuned KNN model achieved 95.6% accuracy but produced 3 dangerous False Negatives.
    *   **Iterative Improvement:** A baseline Logistic Regression model immediately improved results, reducing False Negatives to 1.
    *   **Goal-Oriented Tuning:** Used `GridSearchCV` to optimize the Logistic Regression model, but with a crucial change: the scoring metric was set to **'recall'** instead of 'accuracy'. This explicitly instructed the model to prioritize finding all malignant cases.
*   **Result:** The final, tuned model achieved **100% recall** for the malignant class, successfully eliminating all False Negatives. This demonstrated a practical understanding of the **Precision-Recall trade-off**, accepting a few manageable False Positives in exchange for maximum patient safety.

### 2. Iris Species Classification

*   **Notebook:** [`iris-classification-project/iris_prediction.ipynb`](./iris-classification-project/iris_prediction.ipynb)
*   **Objective:** To build a model that can accurately classify the species of an iris flower (*setosa*, *versicolor*, or *virginica*) based on its sepal and petal measurements.
*   **Model Used:** K-Nearest Neighbors (KNN)
*   **Key Steps & Findings:**
    *   **EDA & Visualization:** Investigated feature distributions and relationships, identifying petal dimensions as highly predictive.
    *   **Robust Evaluation:** Demonstrated the weakness of a single train-test split by showing how "lucky splits" can give misleading accuracy scores.
    *   **Hyperparameter Tuning with Cross-Validation:** Used 10-fold cross-validation to find the optimal `k` (n_neighbors), resulting in a more reliable and generalizable model.
*   **Result:** The final model, tuned with robust methods, achieved an average cross-validated accuracy of **96.0%**.

---

*(More projects, including deep learning applications, will be added here as they are completed...)*
