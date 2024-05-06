#!/usr/bin/python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols


def analyze_data_from_csv(file_path):
    # Load data from CSV
    df = pd.read_csv(file_path)
    print("Columns in the DataFrame:", df.columns)
    exit
    
    # Fitting the regression model
    model = ols('Annual_Revenue ~ Marketing_Spend', data=df).fit()

    # a. Explained variance
    r_squared = model.rsquared
    explanation = "The R-squared value is the proportion of the variance in the dependent variable that is predictable from the independent variable."

    # b. Hypotheses and conclusion
    null_hypothesis = "H0: There is no relationship between marketing spend and annual revenue."
    alternative_hypothesis = "HA: There is a relationship between marketing spend and annual revenue."
    conclusion = "Reject H0 if p-value < 0.05, otherwise fail to reject H0."

    # c. Plotting annual revenue vs marketing spend
    plt.figure(figsize=(8, 6))
    plt.scatter(df['Marketing_Spend'], df['Annual_Revenue'],
                color='blue', label='Data points')
    # Adding a trend line
    slope, intercept = np.polyfit(
        df['Marketing_Spend'], df['Annual_Revenue'], 1)
    plt.plot(df['Marketing_Spend'], slope*df['Marketing_Spend'] +
             intercept, color='red', label=f'Fit: {slope:.2f}x + {intercept:.2f}')
    plt.xlabel('Marketing_Spend')
    plt.ylabel('Annual Revenue')
    plt.title(
        f'Annual Revenue vs. Marketing_Spend\nR-squared: {r_squared:.2f}')
    plt.legend()
    plt.show()

    # d. Regression assumptions
    residuals = model.resid
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Plot for residuals (should be randomly distributed)
    sm.qqplot(residuals, line='45', fit=True, ax=ax[0])
    ax[0].set_title('Q-Q Plot of Residuals')

    # Residuals vs fitted values (should not show patterns)
    ax[1].scatter(model.fittedvalues, residuals)
    ax[1].axhline(y=0, color='red', linestyle='--')
    ax[1].set_xlabel('Fitted values')
    ax[1].set_ylabel('Residuals')
    ax[1].set_title('Residuals vs Fitted Values')
    plt.show()

    return {
        'R_squared': r_squared,
        'Explanation': explanation,
        'Null_Hypothesis': null_hypothesis,
        'Alternative_Hypothesis': alternative_hypothesis,
        'Conclusion': conclusion
    }


# To use this function, provide the path to your CSV file as follows:
file_path = 'q7-data.csv'
results = analyze_data_from_csv(file_path)
print(results)
