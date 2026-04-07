# OnlineShopperPurchasePrediction
This is a project using the Online Shoppers Purchasing Intention dataset with both TensorFlow and PyTorch. This dataset contains features of user sessions on an online shopping website, with the goal of predicting whether a user will make a purchase. Here is the dataset: https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset. 

Explanation of Results (Ready to put in README)

Business Context

The dataset contains features extracted from online shopping sessions (Administrative, Informational, ProductRelated pages, BounceRates, ExitRates, PageValues, VisitorType, etc.). The goal is to predict whether the session resulted in a purchase (Revenue = True).
This is a highly imbalanced problem (~84.5% No Purchase vs 15.5% Purchase). In a real business setting, correctly identifying potential buyers (high Recall on the minority class) is often more valuable than overall accuracy.
Model Architecture (Both Frameworks)

Similar MLP structure used in both:
Input layer → 128 neurons (ReLU + BatchNorm + Dropout 0.3)
Hidden layer → 64 neurons (ReLU + BatchNorm + Dropout 0.2)
Hidden layer → 32 neurons (ReLU)
Output layer → 1 neuron (Sigmoid) for binary classification

Optimizer: Adam
Loss: Binary Crossentropy
Early stopping based on validation loss

Key Results
TensorFlow Model Performance (on Test Set):

Accuracy: ~90%
Macro F1-score: ~0.78
Purchase class (minority): Precision ~0.71, Recall ~0.56, F1 ~0.62

Observations from Training:

The model converged reasonably well.
Validation loss and accuracy stabilized after ~15–20 epochs.
Some signs of slight overfitting (training accuracy higher than validation), which is common with imbalanced data.
