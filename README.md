# Online Shopper Purchase Intention Prediction

**Predicting whether an online visitor will make a purchase using Neural Networks (TensorFlow + PyTorch)**

![Banner Image](https://via.placeholder.com/800x200?text=Ecommerce+Purchase+Prediction)  
*(Add a relevant banner or confusion matrix / ROC curve here)*

## 📋 Project Overview
This project builds and compares **Multi-Layer Perceptron (MLP)** models in both **TensorFlow** and **PyTorch** to predict purchase intent from the UCI Online Shoppers Purchasing Intention Dataset.

**Business Problem**: E-commerce companies lose revenue when they fail to identify high-intent visitors. This model helps prioritize marketing, personalization, or interventions for sessions likely to convert.

**Key Challenge**: Highly imbalanced dataset (~85% no-purchase).

**Why it matters**: Focus on **Recall** for the minority (purchase) class rather than just accuracy.

## 🚀 Key Features
- End-to-end pipeline: EDA → Preprocessing → Modeling → Evaluation
- Dual implementation (TensorFlow & PyTorch) to demonstrate framework flexibility
- Handling class imbalance with awareness of business impact
- Early stopping, BatchNorm, Dropout for regularization

## 🛠 Tech Stack
- **Languages**: Python
- **Frameworks**: TensorFlow, PyTorch
- **Libraries**: pandas, scikit-learn, matplotlib, seaborn, numpy
- **Environment**: Jupyter Notebook

## 📊 Results

| Model          | Accuracy | Macro F1 | Purchase Recall | Purchase F1 |
|----------------|----------|----------|-----------------|-------------|
| TensorFlow MLP | ~90%    | ~0.78   | ~0.56          | ~0.62      |
| PyTorch MLP    | [Add]   | [Add]   | [Add]          | [Add]      |

**Insights**:
- Good overall accuracy but room for improvement on minority class recall.
- Business recommendation: Adjust decision threshold or use techniques like SMOTE / class weights.

*(Add confusion matrices, ROC curves, or training loss plots here — use GitHub markdown image links)*

## 📁 Project Structure
