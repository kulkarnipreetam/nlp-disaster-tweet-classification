# NLP Disaster Tweet Classification

This project is focused on predicting whether a tweet is related to a disaster using Natural Language Processing (NLP) techniques. The model is fine-tuned on BERT (Bidirectional Encoder Representations from Transformers) for sequence classification to classify tweets into disaster-related or non-disaster categories. The project includes data preprocessing, model fine-tuning, and generating predictions for a Kaggle competition.

## Project Overview

The main components of this project include:
- **Tweet Length and Word Count Distribution Analysis**: Initial exploratory data analysis (EDA) was performed to understand the distribution of tweet lengths and word counts in the dataset.
- **Model Fine-Tuning**: A pre-trained BERT model (`bert-base-uncased`) was fine-tuned on the provided disaster tweet dataset.
- **Prediction and Submission**: The fine-tuned model was used to make predictions on the test dataset, and results were submitted to a Kaggle competition.

## Key Steps

1. **Data Preprocessing**: 
   - Cleaned and preprocessed the raw tweet data, including the removal of URLs.

2. **Exploratory Data Analysis (EDA)**:
   - Analyzed tweet lengths and word counts to understand the distribution of text data.
   - Visualizations provided insight into the range and frequency of tweet lengths.

3. **Model Training**:
   - Fine-tuned a BERT model using the preprocessed training data.
   - Used AdamW optimizer with learning rate scheduling and warmup steps to improve training efficiency.
   - Evaluated model performance after each epoch using test accuracy.

4. **Prediction and Submission**:
   - Used the fine-tuned model to generate predictions on the test dataset.
   - Prepared a submission file for the Kaggle competition.
## Training Results

The model was trained for 5 epochs with the following results:

- **Epoch 1/5**:
  - **Train Loss**: 0.5012
  - **Test Accuracy**: 84.11%
  
- **Epoch 2/5**:
  - **Train Loss**: 0.3663
  - **Test Accuracy**: 84.83%
  
- **Epoch 3/5**:
  - **Train Loss**: 0.3114
  - **Test Accuracy**: 83.26%
  
- **Epoch 4/5**:
  - **Train Loss**: 0.2515
  - **Test Accuracy**: 81.16%
  
- **Epoch 5/5**:
  - **Train Loss**: 0.1967
  - **Test Accuracy**: 82.93%

### Analysis

- **Training Loss**: The training loss decreased consistently from 0.5012 in the first epoch to 0.1967 in the fifth epoch, indicating that the model was learning and improving its performance on the training data.
  
- **Test Accuracy**: The test accuracy initially increased from 84.11% in the first epoch to a peak of 84.83% in the second epoch. However, it then decreased to 81.16% by the fourth epoch and slightly improved to 82.93% in the final epoch.

### Observations

- The model achieved its highest test accuracy of 84.83% during the second epoch. Despite this peak, the test accuracy exhibited fluctuations in subsequent epochs.
- The decreasing trend in test accuracy in later epochs may suggest potential overfitting or instability in the model’s performance on the test data.

### Next Steps

- **Hyperparameter Tuning**: Experiment with different hyperparameters, such as learning rates and batch sizes, to stabilize and improve model performance.
- **Regularization Techniques**: Implement regularization methods to mitigate overfitting and enhance model generalization.
- **Further Epochs**: Consider training for additional epochs or applying early stopping based on validation performance to better capture the optimal model state.

## Important Notice

The code in this repository is proprietary and protected by copyright law. Unauthorized copying, distribution, or use of this code is strictly prohibited. By accessing this repository, you agree to the following terms:

- **Do Not Copy:** You are not permitted to copy any part of this code for any purpose.
- **Do Not Distribute:** You are not permitted to distribute this code, in whole or in part, to any third party.
- **Do Not Use:** You are not permitted to use this code, in whole or in part, for any purpose without explicit permission from the owner.

If you have any questions or require permission, please contact the repository owner.

Thank you for your cooperation.

