# Loan Approval Prediction Model
![image](https://github.com/user-attachments/assets/4b42220c-4bf5-4156-87e2-21cea575d034)

This repository contains a machine learning model built to predict loan approval status based on applicant data. The dataset is pre-processed and trained using a Random Forest Classifier to classify loan applications as either "Approved" or "Not Approved."

## Project Overview

Using a dataset that includes applicant demographics and financial details, the model identifies patterns that influence loan approval. By training a classifier, the model provides quick loan eligibility predictions based on a series of input criteria.

## Dataset

The dataset used in this project includes features such as:
- **Applicant Income**
- **Loan Amount**
- **Credit History**
- **Employment Status**
- **Property Area**
- And other demographic attributes

## Requirements

To run this code, ensure you have the following libraries installed:

```bash
pip install pandas numpy scikit-learn
```

## Model Training and Testing

The model is trained using a `RandomForestClassifier` from `scikit-learn`. Categorical columns in the dataset are encoded, and numerical columns are scaled before training to improve the model's performance.

### Training the Model
The model is trained on a subset of the data, and evaluation metrics such as accuracy, precision, recall, and a confusion matrix are calculated on a test set.

## Predicting Loan Approval

To predict loan approval for a new applicant, input the applicant’s details into the `applicant_data` dictionary and follow these steps to generate a prediction:

1. Convert the applicant data into a DataFrame.
2. Apply feature scaling to match the training data.
3. Use the trained model to predict the loan status.

### Example Prediction

The following example shows how to input applicant data and run a prediction:

```python
# Example applicant data
applicant_data = {
    'Gender': [1],               # Male
    'Married': [1],              # Married
    'Dependents': [0],           # No dependents
    'Education': [0],            # Graduate
    'Self_Employed': [0],        # Not self-employed
    'ApplicantIncome': [5000],   # Monthly income of 5000
    'CoapplicantIncome': [2000], # Coapplicant income of 2000
    'LoanAmount': [150],         # Loan amount requested
    'Loan_Amount_Term': [360],   # Loan term in months
    'Credit_History': [1],       # Good credit history
    'Property_Area': [2]         # Urban property area
}

# Convert to DataFrame
applicant_df = pd.DataFrame(applicant_data)

# Scale the necessary columns
applicant_df[['ApplicantIncome', 'LoanAmount']] = scaler.transform(applicant_df[['ApplicantIncome', 'LoanAmount']])

# Make the prediction
loan_prediction = model.predict(applicant_df)

# Display the result
if loan_prediction[0] == 1:
    print("Loan Approved")
else:
    print("Loan Not Approved")
```

In this example:
- If the model outputs `1`, the loan is **Approved**.
- If the model outputs `0`, the loan is **Not Approved**.

## Results Interpretation

The model uses various features to determine whether the applicant meets the loan eligibility criteria. Key predictors include **ApplicantIncome**, **Credit_History**, and **LoanAmount**. 

### Example Result
For the applicant data above:
- `Loan Approved` if the applicant meets the lender’s criteria, such as having a good credit history, sufficient income, and manageable loan request.
- `Loan Not Approved` if the applicant fails to meet the required thresholds.

## Repository Structure

- `FutureInterns_DS_03_Loan_Approval_Prediction_Model.ipynb`: Jupyter notebook with all code, from data cleaning to model training and prediction.
- `README.md`: Overview of the project and example usage.

## Conclusion

This model demonstrates a basic approach to automating loan eligibility checks using machine learning. It is easily adaptable to different loan datasets, making it a versatile tool for financial decision-making.

## Future Improvements

- Implementing additional models for better accuracy.
- Incorporating more complex feature engineering and parameter tuning.
- Improving the user interface for non-technical users.
