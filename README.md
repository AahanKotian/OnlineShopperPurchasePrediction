## Online Shopper Purchase Intention Prediction

**Predicting whether an online visitor will complete a purchase using Neural Networks (TensorFlow & PyTorch)**

https://via.placeholder.com/800x200?text=Ecommerce+Purchase+Prediction

## 📋 Project Overview

This project develops and compares **Multi-Layer Perceptron (MLP)** models implemented in both **TensorFlow** and **PyTorch** to predict whether an online shopping session will end in a purchase.

**Dataset**: UCI Online Shoppers Purchasing Intention Dataset  
- **12,330 sessions** collected over a 1-year period  
- **18 features** (10 numerical + 8 categorical) describing user behavior (Administrative, Informational, ProductRelated pages, Bounce Rate, Exit Rate, Page Values, Special Day, etc.)  
- **Target**: `Revenue` (binary: `True` = purchase completed, `False` = no purchase)  
- **Severe class imbalance**: **84.5%** no-purchase vs. **15.5%** purchase sessions

**Business Problem**  
E-commerce platforms lose significant revenue by failing to identify high-intent visitors in real time. This model enables better prioritization of marketing efforts, personalized offers, live chat support, and website interventions for sessions likely to convert.

**Key Challenge**  
Highly imbalanced data makes overall **accuracy** misleading. The focus must be on **Recall** and **F1-score** for the minority (purchase) class to avoid missing valuable conversions.

## 🚀 Key Features

- Full end-to-end pipeline: Exploratory Data Analysis → Data Preprocessing → Feature Scaling → Modeling → Evaluation
- Dual framework implementation (TensorFlow/Keras and PyTorch) demonstrating framework flexibility
- Proper handling of class imbalance with business-aware metric selection
- Regularization techniques: Early Stopping, Batch Normalization, Dropout layers
- Comparative performance analysis between the two deep learning frameworks

## 🛠 Tech Stack

- **Language**: Python
- **Deep Learning Frameworks**: TensorFlow / Keras, PyTorch
- **Data Science Stack**: pandas, scikit-learn, NumPy, Matplotlib, Seaborn
- **Environment**: Jupyter Notebook

## 📊 Results

| Model              | Accuracy | Macro F1 | Purchase Recall | Purchase F1 |
|--------------------|----------|----------|-----------------|-------------|
| TensorFlow MLP     | ~90%     | ~0.78    | ~0.56           | ~0.62       |
| PyTorch MLP        | ~89.5%   | ~0.76    | ~0.54           | ~0.60       |

**Key Insights**:
- Both models achieve strong overall accuracy, but performance on the critical **purchase class** remains moderate due to severe class imbalance.
- High no-purchase accuracy can hide poor recall on actual buyers — a common pitfall in real-world e-commerce applications.
- **Business Recommendation**: Tune the decision threshold lower, apply class weights, or use oversampling techniques (SMOTE / ADASYN) to improve purchase recall while monitoring precision trade-offs.
