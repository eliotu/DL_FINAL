# **Deep Learning Financial Covariance and Return Prediction**

This project aims to explore the correlation between assets and whether it can help to pre- dict asset returns.

## **Project Overview**

Financial predictions, such as stock returns and risk estimations, are crucial for informed decision-making in trading and portfolio management. This project aims to:

1. Predict **asset returns** for a 1-hour horizon.
2. Forecast the **covariance matrix** of asset returns for the same time period.

The project uses dual neural network (NN) architectures:

- **LSTM-based models** for covariance prediction.
- **Baseline models** like MLPs and StockMixer for return predictions.

By combining these models, the project assesses whether covariance insights can enhance return predictions.

---

## **Repository Structure**

- ### **Cholesky_Lstm/**

  - Implements an **LSTM model** to predict Cholesky decompositions of realized covariance matrices.
  - **Execution**:
    ```
    python Cholesky_Lstm/scripts/train_cholesky_model.py
    ```
    Users can select between two LSTM variants to test their performance.

- ### **Stock-Mixer_Model/**

  - Contains the **baseline model** for stock return predictions using MLP architectures.
  - **Execution**:
    ```
    python Stock_Mixer_Model/src/train.py
    ```

- ### **Combined_model/**

  - Integrates results from `Cholesky_Lstm` and `Stock-Mixer_Model` to create enhanced prediction architectures.
  - **StockMixer_cov_loaded/**:
    Stores covariance results from `Cholesky_Lstm` in a structured dataframe format.
  - Users can switch between `StockMixer_new1` and `StockMixer_new2` models to compare their performance.
  - **Execution**:
    Modify configurations as needed and execute corresponding scripts in the folder.
    ```
    python Combined_model/StockMixer_cov_loaded/src/train_with_covariance.py  
    ```

- ### **data/**

  - Contains datasets for training and testing models.
  - **Dataset Link**: [Download Here](https://polybox.ethz.ch/index.php/s/2pkmbJI1mAEXSTv)

- ### **models/**
  - Stores pre-trained models, especially for the `Cholesky_Lstm`, for quick evaluation.

---

## **Key Features**

- **High-frequency financial data processing** to capture temporal dependencies.
- **Covariance prediction using LSTMs**, ensuring positive semidefinite matrices via Cholesky decomposition.
- **Baseline and combined model evaluation**, leveraging insights from covariance predictions to improve return forecasts.
- **Metrics**:
  - Root Mean Squared Error (RMSE)
  - Mean Absolute Error (MAE)
  - Information Coefficient (IC)
  - Rank Information Coefficient (RIC)
  - Precision@N
  - Sharpe Ratio

---

## **How to Run**

1. Clone the repository:
   ```
   git clone <repository-link>
   ```
2. Navigate to the project folder:
   ```
   cd <project-folder>
   ```
3. Install dependencies:

4. Execute scripts from the respective folders as described above.

---

## **Acknowledgments**

This project is based on methodologies and concepts from various studies, including:

- Bucci (2019) for covariance prediction using Cholesky decompositions.
- Fan and Shen (2024) for StockMixer MLP architectures.
- Sako et al. (2022) for LSTM efficiency in time-series forecasting.

Special thanks to the ETH Zürich Deep Learning Class of 2024-2025 for their guidance and support.
