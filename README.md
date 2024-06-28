# Phishing Website Detection

## Project Overview

Phishing attacks pose a significant threat to internet users, as malicious websites impersonate legitimate ones to steal sensitive information such as usernames, passwords, and credit card details. This project aims to build a machine learning model that can accurately detect phishing websites, enhancing cybersecurity and protecting users from such attacks.

## Data Source and Description

The dataset used for this project is publicly available and was sourced from the Phishing Websites Data Repository. It contains 11,054 instances and 31 features, which include various characteristics of URLs, such as the presence of IP addresses, the length of URLs, and the use of special characters. The target variable indicates whether a website is legitimate or phishing.

**Dataset Citation**:
Dheeru, D., & Karra Taniskidou, E. (2017). UCI Machine Learning Repository [https://archive.ics.uci.edu/ml/datasets/Phishing+Websites]. Irvine, CA: University of California, School of Information and Computer Science.

## Project Steps

1. **Data Loading and Preprocessing**
    - Imported necessary libraries.
    - Loaded the dataset.
    - Removed duplicates and handled missing values.

2. **Exploratory Data Analysis (EDA)**
    - Visualized the distribution of the target variable.
    - Created a correlation heatmap to explore relationships between features.

3. **Data Splitting and Scaling**
    - Separated features and target variable.
    - Split the dataset into training and testing sets using a 70-30 split.
    - Applied standard scaling to normalize the features.

4. **Model Training and Evaluation**
    - Defined multiple machine learning models including Logistic Regression, K-Nearest Neighbors, Decision Tree, Random Forest, AdaBoost, and Support Vector Machine.
    - Trained and evaluated each model using accuracy and ROC AUC scores.
    - Visualized confusion matrices for each model.

5. **Hyperparameter Tuning**
    - Performed hyperparameter tuning on the best-performing model (Random Forest) using GridSearchCV.
    - Retrained the Random Forest model with optimal hyperparameters and evaluated its performance.

6. **Dimensionality Reduction with PCA**
    - Applied Principal Component Analysis (PCA) to reduce dimensionality.
    - Retrained the best model on the PCA-transformed data and evaluated its performance.

## Model Performance

- **Logistic Regression**
  - Accuracy: 92.3%
  - ROC AUC: 97.7%

- **K-Nearest Neighbors**
  - Accuracy: 63.5%
  - ROC AUC: 66.3%

- **Decision Tree**
  - Accuracy: 94.6%
  - ROC AUC: 94.4%

- **Random Forest**
  - Accuracy: 97.2%
  - ROC AUC: 99.6%

- **AdaBoost**
  - Accuracy: 91.2%
  - ROC AUC: 96.5%

- **Support Vector Machine**
  - Accuracy: 56.0%
  - ROC AUC: 92.8%

**Best Model**: Random Forest
- **Optimized Model Accuracy**: 94.3%
- **Optimized Model ROC AUC**: 98.9%

**PCA Model Performance**:
- **PCA Model Accuracy**: 60.9%
- **PCA Model ROC AUC**: 65.4%

## Conclusion

The Phishing Website Detection project successfully utilized various machine learning models to identify phishing websites, with the Random Forest classifier achieving the highest performance. Key findings highlight the significance of features such as IP addresses and URL characteristics in detecting phishing attempts. Future improvements could include advanced feature engineering and the exploration of more sophisticated models like XGBoost to further enhance detection accuracy.

## Repository Structure

- `Phishing_Website_Detection_EDA_Modeling_Analysis.ipynb`: Jupyter Notebook containing the entire analysis and modeling process.
- `phishing.csv`: The dataset used for the project.
- `README.md`: This README file.

## How to Use

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/phishing-website-detection.git
    cd phishing-website-detection
    ```

2. Ensure you have the necessary dependencies installed. You can use the following command to install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Open the Jupyter Notebook and run the cells to reproduce the analysis and results:
    ```bash
    jupyter notebook Phishing_Website_Detection_EDA_Modeling_Analysis.ipynb
    ```

## Acknowledgements

Special thanks to the UCI Machine Learning Repository for providing the dataset used in this project.

---

Feel free to contribute to the project by forking the repository and submitting pull requests. If you have any questions or need further details, please open an issue in the repository.

Thank you for your interest in the Phishing Website Detection project!
