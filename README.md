# KNN Play Predictor

A Machine Learning project that uses the **K-Nearest Neighbors (KNN)** classification algorithm to predict whether a person can **Play (Yes/No)** based on **Weather** and **Temperature**.

This project demonstrates the complete Machine Learning workflow including data loading, data preprocessing, label encoding, model training, prediction, and accuracy calculation.

---

## 📌 Project Overview

The dataset contains information about weather conditions and temperature along with the target variable `Play`.

The model uses:

- **Weather** as a feature
- **Temperature** as a feature
- **Play** as the target variable

The categorical values are converted into numerical values using **LabelEncoder**, and the KNN algorithm is used for classification.

---

## 📊 Dataset

The dataset contains the following columns:

| Feature | Description |
|---|---|
| Weather | Weather condition |
| Temperature | Temperature condition |
| Play | Target variable |

### Weather Values

- Sunny
- Overcast
- Rainy

### Temperature Values

- Hot
- Mild
- Cool

### Target Values

- Yes
- No

### Sample Dataset

| Weather | Temperature | Play |
|---|---|---|
| Sunny | Hot | No |
| Sunny | Hot | No |
| Overcast | Hot | Yes |
| Rainy | Mild | Yes |
| Rainy | Cool | Yes |
| Rainy | Cool | No |
| Overcast | Cool | Yes |
| Sunny | Mild | No |
| Sunny | Cool | Yes |
| Rainy | Mild | Yes |

---

## 🤖 Algorithm Used

### K-Nearest Neighbors (KNN)

KNN is a supervised Machine Learning classification algorithm.

In this project:

1. The dataset is loaded.
2. Categorical data is converted into numerical data.
3. KNN model is trained.
4. User provides Weather and Temperature.
5. The trained model predicts whether `Play` is **Yes** or **No**.
6. Accuracy is calculated for different values of K.

The default value of K used for prediction is:

```text
K = 3
