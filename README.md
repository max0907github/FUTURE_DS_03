# Loan Approval Prediction Model
![image](https://github.com/user-attachments/assets/4b42220c-4bf5-4156-87e2-21cea575d034)

This repository contains a predictive model for determining loan approvals based on applicant data. The model uses various applicant features such as income, education level, marital status, and more to classify applicants as either approved or not approved for a loan.

## Project Overview

In this project, we built a machine learning model that leverages features from applicant profiles to predict the likelihood of loan approval. The data processing, scaling, and predictive analysis have been performed in Python, and the entire workflow is available in a Jupyter notebook.

## Dataset

The dataset for this project includes information on loan applicants, such as income, loan amount, credit history, and demographics. You can find the dataset on my GitHub using the respiratory link. (provide the actual path or link to the data file in your repository or a public data source if available).

### Data Fields

The dataset includes the following fields:

- `Loan_ID`: Unique identifier for each loan applicant
- `Gender`: Gender of the applicant
- `Married`: Marital status
- `Dependents`: Number of dependents
- `Education`: Education level (Graduate/Not Graduate)
- `Self_Employed`: Employment status
- `ApplicantIncome`: Applicant's income
- `CoapplicantIncome`: Co-applicant's income
- `LoanAmount`: Loan amount requested
- `Loan_Amount_Term`: Term of the loan
- `Credit_History`: Credit history meets guidelines or not
- `Property_Area`: Location of the property
- `Loan_Status`: Approval status of the loan (Y/N)

## Requirements

To run this project, you’ll need the following dependencies:

- Python 3.x
- pandas
- numpy
- scikit-learn
- Jupyter Notebook

You can install these requirements using:
```bash
pip install -r requirements.txt
```

## Model Training and Prediction

The project uses a Jupyter notebook (`FutureInterns_DS_03_Loan_Approval_Prediction_Model.ipynb`) to preprocess the data, train the model, and make predictions. Here’s a breakdown of the workflow:

1. **Data Loading and Preprocessing**: The data is loaded and cleaned, handling missing values and encoding categorical variables.
2. **Feature Scaling**: Scaling is applied to the numerical fields for normalization.
3. **Model Training**: A classification model (e.g., Logistic Regression, Decision Tree) is trained on the processed data.
4. **Prediction**: The trained model is used to predict loan approvals for new applicants.

## Running the Model

To run the model and make predictions, open the notebook in Jupyter:

```bash
jupyter notebook FutureInterns_DS_03_Loan_Approval_Prediction_Model.ipynb
```

Follow the steps in the notebook to load the data, preprocess it, and generate predictions.

## Results

The model outputs a prediction for each applicant: either "Loan Approved" or "Loan Not Approved."

## License

This project is open-source and available under the MIT License.

--- 

Make sure to replace `path_to_data` with the actual link to your data if available publicly. If you need more guidance on adding specific commands or extra details, feel free to ask!
