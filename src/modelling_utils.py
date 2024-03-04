
'''
This script aims to provide functions that will turn the modelling process easier
'''

'''
Importing libraries
'''

# Data manipulation and visualization.
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Modeling.
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss, r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import statsmodels.api as sm

# Debugging.
from src.exception import CustomException
import sys

# Artifacts.
import os
import pickle

# Warnings.
from warnings import filterwarnings
filterwarnings('ignore')


class LogisticRegressionWithPvalues:
    '''
    Custom logistic regression class with p-values for coefficient significance.

    Attributes:
    alpha (float): Regularization parameter. Default is 0.
    method (str): Regularization method. Default is 'l1'.
    model: Fitted logistic regression model.

    Methods:
    __init__(alpha=0, method='l1'):
        Initializes the LogisticRegressionWithPvalues instance.

        Parameters:
        - alpha (float): Regularization parameter. Default is 0.
        - method (str): Regularization method ('l1'). Default is 'l1'.

    fit(X, y):
        Fit the regularized logistic regression model.

        Parameters:
        - X (DataFrame): Input features for model fitting.
        - y (array-like): Target variable for model fitting.

        Raises:
        - CustomException: If an exception occurs during fitting.

    predict(X):
        Predict probabilities using the fitted model.

        Parameters:
        - X (DataFrame): Input features for prediction.

        Returns:
        - predicted_probabilities: Predicted probabilities.

        Raises:
        - CustomException: If an exception occurs during prediction.

    get_result_table():
        Get a summary table with beta coefficients, p-values, and Wald statistics.

        Returns:
        - result: Summary table with Beta Coefficient, P-Value, and Wald Statistic.

        Raises:
        - CustomException: If an exception occurs during result table creation.
    '''
    def __init__(self, alpha=0, method='l1'):
        self.alpha = alpha
        self.method = method
        self.model = None

    def fit(self, X, y):
        try:
            # Add a constant to the data, which will be the intercept, and reshape y.
            X_copy = X.copy()
            X_copy = sm.add_constant(X_copy)
            y_reshaped = y.values.reshape(-1,1)

            # Fit the regularized logistic regression model.
            self.model = sm.Logit(y_reshaped, X_copy).fit_regularized(alpha=self.alpha, method=self.method)
        
        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, X):
        try:
            # Add a constant to the data, which will multiply the intercept.
            X_copy = X.copy()
            X_copy = sm.add_constant(X_copy)
            
            # Predicting probabilities.
            predicted_probabilities = self.model.predict(X_copy)
            
            return predicted_probabilities
        
        except Exception as e:
            raise CustomException(e, sys)
    
    def get_summary_table(self):
        try:
            # Collect beta coefficients, p-values and Wald statistics in a summary table.
            summary = self.model.summary2().tables[1]
            summary['Wald'] = summary['z'] ** 2
            summary_table = summary[['Coef.', 'P>|z|', 'Wald']]
            summary_table.columns = ['Beta Coefficient', 'P-Value', 'Wald Statistic']
            summary_table = summary_table.sort_index()   
            return summary_table
        
        except Exception as e:
            raise CustomException(e, sys)
        

def evaluate_credit_scoring_model(y_train, y_test, train_probas, test_probas, plot=False): 
    '''
    Evaluate the performance of a credit scoring model using AUC, Gini, KS and Brier metrics on both training and test sets.

    Parameters:
    - y_train (pd.Series): Actual target values for the training set. 1 for non-default and 0 for default.
    - y_test (pd.Series): Actual target values for the test set. 1 for non-default and 0 for default
    - train_probas (np.ndarray): Predicted probabilities of being good for the training set.
    - test_probas (np.ndarray): Predicted probabilities of being good for the test set.
    - plot (bool, optional): Whether to plot the ROC curve. Default is False.

    Returns:
    - pd.DataFrame: A DataFrame containing evaluation metrics for both the training and test sets.

    Raises:
    - CustomException: An exception is raised if an error occurs during the evaluation process.
    
    Example:
    ```python
    model_metrics = evaluate_credit_scoring_model(y_train, y_test, train_probas, test_probas, plot=True)
    print(model_metrics)
    ```
    '''
    try:  
        # Obtain roc curve and auc score on test set.
        fpr, tpr, thresholds = roc_curve(y_test, test_probas)
        roc_auc_test = roc_auc_score(y_test, test_probas)
        
        # Obtain gini index on test set.
        gini_test = 2 * roc_auc_test - 1
        
        if plot:
        # Plot roc curve for test.
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(fpr, tpr, label=f'TEST ROC AUC = {roc_auc_test:.2f}', color='#461220')
            ax.plot([0, 1], [0, 1], linestyle='--', color='#8a817c')  # Random guessing line.
            ax.set_xlabel('False Positive Rate', fontsize=10.8, labelpad=20)
            ax.set_ylabel('True Positive Rate', fontsize=10.8, labelpad=20)
            ax.set_xticks(ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0], labels=['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'])
            ax.set_yticks(ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0], labels=['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'])
            ax.set_title('Receiver Operating Characteristic (ROC) Curve', fontweight='bold', fontsize=12, pad=20)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#8a817c')
            ax.spines['bottom'].set_color('#8a817c')
            ax.grid(False)
            ax.legend()
        
        # Obtain KS on test set.
        test_scores = pd.DataFrame()
        test_scores['actual'] = y_test.reset_index(drop=True)
        test_scores['probability_of_default'] = 1 - test_probas
        sorted_test_scores = test_scores.sort_values(by=['probability_of_default'], ascending=False)
        sorted_test_scores['cum_bad'] = (1 - sorted_test_scores['actual']).cumsum() / (1 - sorted_test_scores['actual']).sum()
        sorted_test_scores['cum_good'] = sorted_test_scores['actual'].cumsum() / sorted_test_scores['actual'].sum()
        sorted_test_scores['ks'] = np.abs(sorted_test_scores['cum_good'] - sorted_test_scores['cum_bad'])
        ks_statistic_test = sorted_test_scores['ks'].max()
        
        # Obtain Brier Score on test set.
        brier_score_test = brier_score_loss(y_test, test_probas)
        
        # Obtain roc-auc score, gini index, ks statistic and brier score on train set.
        roc_auc_train = roc_auc_score(y_train, train_probas)
        gini_train = 2 * roc_auc_train - 1
        brier_score_train = brier_score_loss(y_train, train_probas)
        
        train_scores = pd.DataFrame()
        train_scores['actual'] = y_train.reset_index(drop=True)
        train_scores['probability_of_default'] = 1 - train_probas
        sorted_train_scores = train_scores.sort_values(by=['probability_of_default'], ascending=False)
        sorted_train_scores['cum_bad'] = (1 - sorted_train_scores['actual']).cumsum() / (1 - sorted_train_scores['actual']).sum()
        sorted_train_scores['cum_good'] = sorted_train_scores['actual'].cumsum() / sorted_train_scores['actual'].sum()
        sorted_train_scores['ks'] = np.abs(sorted_train_scores['cum_good'] - sorted_train_scores['cum_bad'])
        ks_statistic_train = sorted_train_scores['ks'].max()
        
        # Construct a DataFrame with metrics for train and test.
        model_metrics = pd.DataFrame({
                                    'Metric': ['KS', 'AUC', 'Gini', 'Brier'],
                                    'Train Value': [ks_statistic_train, roc_auc_train, gini_train, brier_score_train],
                                    'Test Value': [ks_statistic_test, roc_auc_test, gini_test, brier_score_test],
                                    })
        
        return model_metrics
    
    except Exception as e:
        raise CustomException(e, sys)
    

def deciles_scores_analysis(y_train, y_test, train_probas, test_probas):
    '''
    Analyzes and plots the bad rate and cumulative bad rate per decile obtained from the predicted probabilities of the credit scoring model.

    Parameters:
    - y_train (pd.Series): Actual values for the training set. 1 is non-default, 0 is default.
    - y_test (pd.Series): Actual values for the test set. 1 is non-default, 0 is default.
    - train_probas (numpy.ndarray): Predicted probabilities of being good for the training set.
    - test_probas (numpy.ndarray): Predicted probabilities of being good for the test set.

    Returns:
    - train_score_table (pd.DataFrame): Table with bad rate, volume, and cumulative bad rate per decile for the training set.
    - test_score_table (pd.DataFrame): Table with bad rate, volume, and cumulative bad rate per decile for the test set.
    
    Raises:
    - CustomException: If any unexpected error occurs during the execution.
    '''
    try:
        # Add some noise to the predicted probabilities and round them to avoid duplicate problems in bin limits.
        noise = np.random.uniform(0, 0.0001, size=train_probas.shape)
        train_probas += noise
        train_probas = round(train_probas, 10)
        
        # Create a DataFrame with the predicted probabilities of being good and actual values for train.
        train_df = pd.DataFrame({'probability': train_probas, 'actual': y_train.reset_index(drop=True)})
        
        # Sort the train_df by probabilities.
        train_df = train_df.sort_values(by='probability', ascending=True)
        
        # Calculate the deciles.
        train_df['decile'] = pd.qcut(train_df['probability'], q=10, labels=False, duplicates='drop')
        
        # Calculate the bad rate per decile.
        train_decile_df = train_df.groupby(['decile'])['actual'].mean().reset_index()
        train_decile_df['bad_rate'] = 1 - train_decile_df['actual']
        
        # Add some noise to the predicted probabilities and round them to avoid duplicate problems in bin limits.
        noise = np.random.uniform(0, 0.0001, size=test_probas.shape)
        test_probas += noise
        test_probas = round(test_probas, 10)
        
        # Create a DataFrame with the predicted probabilities of being good and actual values for test.
        test_df = pd.DataFrame({'probability': test_probas, 'actual': y_test.reset_index(drop=True)})
        
        # Sort the test_df by probabilities.
        test_df = test_df.sort_values(by='probability', ascending=True)
        
        # Calculate the deciles.
        test_df['decile'] = pd.qcut(test_df['probability'], q=10, labels=False, duplicates='drop')
        
        # Calculate the bad rate per decile.
        test_decile_df = test_df.groupby(['decile'])['actual'].mean().reset_index()
        test_decile_df['bad_rate'] = 1 - test_decile_df['actual']
        
        # Obtain a table with bad rate, volume and cumulative bad rate per decile for train.
        train_df['bad_probability'] = 1 - train_df['probability']
        train_scores_table = train_df.groupby('decile').agg(
            min_score=pd.NamedAgg(column='bad_probability', aggfunc='min'),
            max_score=pd.NamedAgg(column='bad_probability', aggfunc='max'),
            bad_rate=pd.NamedAgg(column='actual', aggfunc='mean'),
            volume=pd.NamedAgg(column='actual', aggfunc='size')
        ).reset_index()
        train_scores_table['bad_rate'] = 1 - train_scores_table['bad_rate']
        train_scores_table['cum_bad_rate'] = train_scores_table['bad_rate'].cumsum() / train_scores_table['bad_rate'].sum()
        
        # Obtain a table with bad rate, volume and cumulative bad rate per decile for train.
        test_df['bad_probability'] = 1 - test_df['probability']
        test_scores_table = test_df.groupby('decile').agg(
            min_score=pd.NamedAgg(column='bad_probability', aggfunc='min'),
            max_score=pd.NamedAgg(column='bad_probability', aggfunc='max'),
            bad_rate=pd.NamedAgg(column='actual', aggfunc='mean'),
            volume=pd.NamedAgg(column='actual', aggfunc='size')
        ).reset_index()
        test_scores_table['bad_rate'] = 1 - test_scores_table['bad_rate']
        test_scores_table['cum_bad_rate'] = test_scores_table['bad_rate'].cumsum() / test_scores_table['bad_rate'].sum()
    
        # Plot scores ordering and cumulative bad rate per decile for training and test sets.

        fig, ax = plt.subplots(figsize=(20, 4))

        # Ordering per decile.
        bar_width = 0.35  
        r1 = np.arange(len(train_decile_df))
        r2 = [x + bar_width + 0.015 for x in r1]

        ax.bar(r1, train_decile_df['bad_rate'], color='#8a817c', width=bar_width, label='Train')
        ax.bar(r2, test_decile_df['bad_rate'], color='#461220', width=bar_width, label='Test')
        ax.set_title('Ordering per Decile', fontweight='bold', fontsize=12)
        ax.set_xticks([r + bar_width/2 for r in range(len(train_decile_df))], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        ax.set_xlabel('Decile', labelpad=10)
        ax.set_ylabel('Bad Rate')
        ax.set_yticks([])
        ax.legend()
        ax.grid(False)
        
        # Annotate bad_rate on top of the bars for training set.
        for i, value in enumerate(train_decile_df['bad_rate']):
           ax.text(r1[i], value + 0.001, f'{value:.2%}', ha='center', va='bottom', color='#8a817c')

        # Annotate bad_rate on top of the bars for test set.
        for i, value in enumerate(test_decile_df['bad_rate']):
           ax.text(r2[i], value + 0.001, f'{value:.2%}', ha='center', va='bottom', color='#461220')
        
        # Cumulative Bad Rate per decile.
        fig, ax = plt.subplots(figsize=(20, 6))

        r1 = np.arange(len(train_scores_table))
        r2 = [x + bar_width + 0.02 for x in r1]

        ax.bar(r1, train_scores_table['cum_bad_rate'], color='#8a817c', width=bar_width, label='Train')
        ax.bar(r2, test_scores_table['cum_bad_rate'], color='#461220', width=bar_width, label='Test')
        ax.set_title('Cumulative Bad Rate per Decile', fontweight='bold', fontsize=12, pad=20)
        ax.set_xticks([r + bar_width/2 for r in range(len(train_scores_table))], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        ax.set_xlabel('Decile', labelpad=10)
        ax.set_ylabel('Cumulative Bad Rate')
        ax.set_yticks([])
        ax.legend(loc='upper right', bbox_to_anchor=(1, 1.1))
        ax.grid(False)

        # Annotate cum_bad_rate on top of the bars for training set.
        for i, value in enumerate(train_scores_table['cum_bad_rate']):
            ax.text(r1[i], value + 0.001, f'{value:.2%}', ha='center', va='bottom', color='#8a817c')

        # Annotate cum_bad_rate on top of the bars for test set.
        for i, value in enumerate(test_scores_table['cum_bad_rate']):
            ax.text(r2[i], value + 0.001, f'{value:.2%}', ha='center', va='bottom', color='#461220')
    
        # Adjust layout to prevent clipping of titles and labels.
        plt.tight_layout()

        # Show the plot.
        plt.show()

        return train_scores_table, test_scores_table
    except Exception as e:
        raise CustomException(e, sys)
    

def probability_scores_ordering(y_train, y_test, train_probas, test_probas):
    '''
    Order and visualize the probability scores in deciles for both training and test sets.

    Parameters:
    - y_train (pd.Series): Actual target values for the training set. 1 is non-default and 0 is default.
    - y_test (pd.Series): Actual target values for the test set. 1 is non-default and 0 is default.
    - train_probas (np.ndarray): Predicted probabilities of being good for the training set.
    - test_probas (np.ndarray): Predicted probabilities of being good for the test set.

    Returns:
    - None: Plots the probability scores ordering for both training and test sets.

    Raises:
    - CustomException: An exception is raised if an error occurs during the execution.
    
    Example:
    ```python
    probability_scores_ordering(y_train, y_test, train_probas, test_probas)
    ```
    '''
    try:
        # Add some noise to the predicted probabilities and round them to avoid duplicate problems in bin limits.
        noise = np.random.uniform(0, 0.0001, size=train_probas.shape)
        train_probas += noise
        train_probas = round(train_probas, 10)
        
        # Create a DataFrame with the predicted probabilities of being good and actual values for train.
        train_df = pd.DataFrame({'probabilities': train_probas, 'actual': y_train.reset_index(drop=True)})
        
        # Sort the train_df by probabilities.
        train_df = train_df.sort_values(by='probabilities', ascending=True)
        
        # Calculate the deciles.
        train_df['deciles'] = pd.qcut(train_df['probabilities'], q=10, labels=False, duplicates='drop')
        
        # Calculate the bad rate per decile.
        train_decile_df = train_df.groupby(['deciles'])['actual'].mean().reset_index()
        train_decile_df['bad_rate'] = 1 - train_decile_df['actual']
        
        # Add some noise to the predicted probabilities and round them to avoid duplicate problems in bin limits.
        noise = np.random.uniform(0, 0.0001, size=test_probas.shape)
        test_probas += noise
        test_probas = round(test_probas, 10)
        
        # Create a DataFrame with the predicted probabilities of being good and actual values for test.
        test_df = pd.DataFrame({'probabilities': test_probas, 'actual': y_test.reset_index(drop=True)})
        
        # Sort the test_df by probabilities.
        test_df = test_df.sort_values(by='probabilities', ascending=True)
        
        # Calculate the deciles.
        test_df['deciles'] = pd.qcut(test_df['probabilities'], q=10, labels=False, duplicates='drop')
        
        # Calculate the bad rate per decile.
        test_decile_df = test_df.groupby(['deciles'])['actual'].mean().reset_index()
        test_decile_df['bad_rate'] = 1 - test_decile_df['actual']
        
        # Plot probability scores ordering for train and test sets.
        # Plot bar graph of deciles vs event rate.
        fig, ax = plt.subplots(1, 2, figsize=(20, 4))
        ax[0].bar(train_decile_df['deciles'], train_decile_df['bad_rate'], color='#8a817c')
        ax[0].set_title('Probability Scores Ordering - Train')
        ax[0].set_xticks(range(10), [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        ax[0].set_xlabel('Score Bins', labelpad=15)
        ax[0].set_ylabel('Bad Rate', labelpad=15)
        ax[0].grid(False)
        
        ax[1].bar(test_decile_df['deciles'], test_decile_df['bad_rate'], color='#461220')
        ax[1].set_title('The probability scores follow an order - Test')
        ax[1].set_xticks(range(10), [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        ax[1].set_xlabel('Score Bins', labelpad=15)
        ax[1].set_ylabel('Bad Rate', labelpad=15)
        ax[1].grid(False)
    
    except Exception as e:
        raise CustomException(e, sys)
    

def create_scorecard(referece_categories, summary_table):
    '''
    Creates a credit scorecard based on a summary table containing the independent variables, their dummies and corresponding coefficients, p-values and Wald statistics.

    Parameters:
    - referece_categories (dict): A dictionary containing reference categories for independent variables.
    - summary_table (pd.DataFrame): A summary table with information about dummy variables, their coefficients, p-values and Wald statistics.

    Returns:
    - pd.DataFrame: The scorecard containing integer and easily interpretable scores for each dummy variable.
    - float: The minimum possible score (default is 300).
    - float: The maximum possible score (default is 850).

    Raises:
    - CustomException: An exception indicating an error during the scorecard creation process.
    '''

    try:
        # Obtain a scorecard df for reference categories, each with 0 coefficient.
        referece_categories_scorecard = {key: key+'_'+value for key, value in referece_categories.items()}
        scorecard_reference_categories = pd.DataFrame()
        scorecard_reference_categories['Dummy'] = list(referece_categories_scorecard.values())
        scorecard_reference_categories['Beta Coefficient'] = 0
        scorecard_reference_categories['P-Value'] = 'reference category'
        scorecard_reference_categories['Wald Statistic'] = 'reference category'
        scorecard_reference_categories = scorecard_reference_categories.reset_index(drop=True)
        scorecard_reference_categories.index += 87
        
        # Concatenate the reference categories scorecard with the summary table.
        summary_table_scorecard = summary_table.reset_index().rename(columns={'index': 'Dummy'})
        scorecard = pd.concat([summary_table_scorecard, scorecard_reference_categories])
        scorecard = scorecard.sort_values(by=['Dummy']).reset_index(drop=True)
        
        # Define minimum and maximum desired scores.
        min_score = 300
        max_score = 850
        
        # Obtain the independent variable for each dummy.
        scorecard['Independent Variable'] = scorecard['Dummy'].str.split('_').apply(lambda x: ''.join(x[0]))
        
        # Obtain the minimum and maximum scoring formula values, calculating scores for each dummy.
        min_sum_coef = scorecard.groupby(['Independent Variable'])['Beta Coefficient'].min().sum()
        max_sum_coef = scorecard.groupby(['Independent Variable'])['Beta Coefficient'].max().sum()
        scorecard['Score'] = scorecard['Beta Coefficient'] * (max_score - min_score) / (max_sum_coef - min_sum_coef)
        
        # Calculate the intercept score.
        intercept = scorecard.loc[scorecard['Dummy'] == 'const', 'Beta Coefficient'].values[0]
        scorecard.loc[scorecard['Dummy'] == 'const', 'Score'] = ((intercept - min_sum_coef) / (max_sum_coef - min_sum_coef))  * (max_score - min_score) + min_score
        
        # Round and obtain the minimum and maximum possible score.
        scorecard['Score'] = scorecard['Score'].round()
        min_possible_score = scorecard.groupby(['Independent Variable'])['Score'].min().sum()
        max_possible_score = scorecard.groupby(['Independent Variable'])['Score'].max().sum()
        
        # Drop independent variable auxiliar column.
        scorecard = scorecard.drop(columns=['Independent Variable'])
        
        return scorecard, min_possible_score, max_possible_score
    
    except Exception as e:
        raise CustomException(e, sys)
    

def compute_scores(X, y, probas, scorecard):
    '''
    Compute scores based on the provided input features, actual labels, probabilities, and a scorecard.

    Parameters:
    - X: pandas DataFrame
    Input features, including dummy variables.

    - y: pandas Series
    Actual labels (target variable), 1 is non-default, 0 is default.

    - probas: array-like
    Probabilities of being good.

    - scorecard: pandas DataFrame
    Scorecard containing information about dummies, their scores, and intercept.

    Returns:
    - pandas DataFrame
    DataFrame containing actual labels, probabilities of being good, and calculated scores.

    Raises:
    - CustomException: If any error occurs during the computation.

    The function computes scores by selecting relevant dummies from the input features based on the scorecard.
    It performs element-wise multiplication and sums along the rows, adding the intercept score.
    The resulting scores are then combined with actual labels and probabilities in a DataFrame.
    '''
    try:
        # Select only dummies and put X in the same order as the scorecard dummies.
        scorecard_dummies = scorecard.loc[~(scorecard['Dummy'] == 'const') & ~(scorecard['P-Value'] == 'reference category')]
        X_dummies = X[scorecard_dummies['Dummy'].values]
        
        # Set 'Dummy' as the index in scorecard for faster lookups.
        scorecard_dummies = scorecard_dummies.set_index('Dummy')
        
        # Perform element-wise multiplication and sum along the rows.
        intercept_score = scorecard.loc[scorecard['Dummy'] == 'const', 'Score'].values[0]
        scores = intercept_score + X_dummies.mul(scorecard_dummies['Score']).sum(axis=1)
        
        # Construct a dataframe with actual values, probabilities of being good, and calculated scores.
        scores_df = pd.DataFrame({
            'Actual': y.reset_index(drop=True),
            'Probability of Default (PD)': 1 - probas,
            'Score': scores 
        })
    
        return scores_df
    
    
    except Exception as e:
        raise CustomException(e, sys)
    

def compare_actual_predicted_regression(y_true, y_pred):
    '''
    Compares actual and predicted values and calculates the residuals for a regression model.

    Args:
    y_true : The true target values.
    y_pred : The predicted target values.

    Returns:
    pandas.DataFrame: A dataframe containing the actual, predicted, and residual values.

    Raises:
    CustomException: An error occurred during the comparison process.
    '''
    try:
        actual_pred_df = pd.DataFrame({'Actual': np.round(y_true, 2),
                                    'Predicted': np.round(y_pred, 2), 
                                    'Residual': np.round(np.abs(y_pred - y_true), 2)})
        return actual_pred_df
    except Exception as e:
        raise CustomException(e, sys)
    

def evaluate_regressor(y_true, y_pred, y_train, model_name):
    '''
    Evaluates a regression model based on various metrics and plots.

    Args:
    y_true : The true target values.
    y_pred : The predicted target values.
    y_train : The actual target values from the training set.
    model_name (str): The name of the regression model.

    Returns:
    pandas.DataFrame: A dataframe containing the evaluation metrics.

    Raises:
    CustomException: An error occurred during the evaluation process.
    '''
    try:
        mae = round(mean_absolute_error(y_true, y_pred), 4)
        mse = round(mean_squared_error(y_true, y_pred), 4)
        rmse = round(np.sqrt(mse), 4)
        mape = round(100 * mean_absolute_percentage_error(y_true, y_pred), 4)
        
        # Metrics
        print(f'Mean Absolute Error (MAE): {mae}')
        print(f'Mean Absolute Percentage Error (MAPE): {mape}')
        print(f'Mean Squared Error (MSE): {mse}')
        print(f'Root Mean Squared Error (RMSE): {rmse}')

        # Obtaining a dataframe of the metrics.
        df_results = pd.DataFrame({'Model': model_name, 'MAE': mae, 'MAPE': mape, 'RMSE': rmse}, index=['Results'])

        # Residual Plots
        
        # Distribution of the residuals
        plt.figure(figsize=(5, 3))
        sns.distplot((y_true - y_pred))
        plt.title('Residuals Distribution')
        plt.grid(False)
        plt.show()

        return df_results

    except Exception as e:
        raise CustomException(e, sys)
        

class DiscretizerCombiner(BaseEstimator, TransformerMixin):
    '''
    DiscretizerCombiner
    
    Custom scikit-learn transformer for discretizing and bundling continuous feature categories based on predefined mappings.
    It discretizes and bundles category ranges in numerical continuous features based on similar WoE and balances the number of observations.
    
    Attributes:
    category_mapping (dict): Dictionary mapping features to lists of category bins.
    debug (bool): Flag indicating whether to print debugging information during transformation.
    
    Methods:
    fit(X, y=None):
        Fit the transformer on the input data X.
        
        Parameters:
        - X (DataFrame): Input data for fitting the transformer.
        - y (array-like, optional): Target variable. Default is None.
        
        Returns:
        - self: Returns the instance itself.
        
        Raises:
        - CustomException: If an exception occurs during fitting.
    
    transform(X):
        Transform the input data X by discretizing and bundling feature categories based on predefined mappings.
        
        Parameters:
        - X (DataFrame): Input data for transformation.
        
        Returns:
        - transformed_data: The transformed data containing only categorical features.
        
        Raises:
        - CustomException: If an exception occurs during transformation.
    '''
    def __init__(self, debug=False):
        self.category_mapping = {
                                'int_rate': [7, 10, 12, 14, 16, 18, 22],
                                'loan_amnt': [7400, 14300, 21200, 28100],
                                'dti': [4, 8, 12, 16, 20, 28],
                                'annual_inc': [20000, 40000, 60000, 75000, 90000, 120000, 150000],
                                'mths_since_earliest_cr_line': [151, 226, 276, 401],
                                'revol_bal': [2000, 6000, 12000, 22000, 30000, 36000, 40000],
                                'tot_cur_bal': [80000, 140000, 200000, 240000, 280000, 340000, 400000],
                                'mths_since_last_delinq': [4, 7, 22, 37, 74],
                                'open_acc': [6, 12, 21],
                                'total_acc': [8, 15, 24, 36],
                                }
        self.debug = debug
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        try:
            X_copy = X.copy()
            
            for feature, category_bins in self.category_mapping.items():
                # Create bins based on the provided category_bins
                bins = [-float('inf')] + category_bins + [float('inf')]
        
                # Create labels for the bins
                labels = []
                
                # Process the first bin separately
                first_bin_label = f'<={category_bins[0] / 1000:.1f}K' if category_bins[0] >= 1000 else f'<={category_bins[0]:.1f}'
                labels.append(first_bin_label)

                # Process the middle bins
                for i in range(1, len(category_bins)):
                    lower_bound = category_bins[i-1] / 1000 if category_bins[i-1] >= 1000 else category_bins[i-1]
                    upper_bound = category_bins[i] / 1000 if category_bins[i] >= 1000 else category_bins[i]
                    bin_label = f'{lower_bound:.1f}K-{upper_bound:.1f}K' if category_bins[i] >= 1000 else f'{lower_bound:.1f}-{upper_bound:.1f}'
                    labels.append(bin_label)
                
                # Process the last bin separately
                last_bin_label = f'>{category_bins[-1] / 1000:.1f}K' if category_bins[-1] >= 1000 else f'>{category_bins[-1]:.1f}'
                labels.append(last_bin_label)

                # Discretize and bundle categories of the variable.
                X_copy[feature] = pd.cut(X_copy[feature], bins=bins, labels=labels, include_lowest=False, right=True)
                X_copy[feature] = X_copy[feature].astype(str)
                
                # Inform which feature is being transformed, the original categories and the discretized and bundled categories.
                if self.debug:
                    print(f'Discretize and bundle categories of {feature}.')
                    print(f'Original range: {round(X[feature].min())} to {round(X[feature].max())}.')
                    print(f'Discretized and bundled categories: {X_copy[feature].unique().tolist()}.')
                    print()
            
            return X_copy
        
        except Exception as e:
            raise(CustomException(e, sys))


class CatCombiner(BaseEstimator, TransformerMixin):
    '''
    CatCombiner
    
    Custom scikit-learn transformer for bundling categories based on predefined mappings.
    It bundles categories of categorical features contained in lists based on a previous 
    assessment of similar WoE categories, balancing the number of observations. Both categorical
    and discrete variables with not many domains are suitable. Categories of categorical variables
    are bundled in the form 'A_B_C_D', and discrete variables are bundled in the form '1-3'.
    
    Attributes:
    category_mapping (dict): Dictionary mapping features to lists of category groups.
    debug (bool): Flag indicating whether to print debugging information during transformation.
    
    Methods:
    fit(X, y=None):
        Fit the transformer on the input data X.
        
        Parameters:
        - X (DataFrame): Input data for fitting the transformer.
        - y (array-like, optional): Target variable. Default is None.
        
        Returns:
        - self: Returns the instance itself.
        
        Raises:
        - CustomException: If an exception occurs during fitting.
    
    transform(X):
        Transform the input data X using the predefined category mappings.
        
        Parameters:
        - X (DataFrame): Input data for transformation.
        
        Returns:
        - transformed_data: The transformed data with categorical and discrete variables determined categories bundled.
        
        Raises:
        - CustomException: If an exception occurs during transformation.
    '''

    def __init__(self, debug=False):
        self.category_mapping = {
                                'grade': [],
                                'home_ownership': [['OTHER', 'NONE', 'RENT', 'ANY']],
                                'purpose': [
                                            ['small_business', 'educational', 'renewable_energy', 'moving'],
                                            ['other', 'house', 'medical', 'vacation'],
                                            ['wedding', 'home_improvement', 'major_purchase', 'car'],
                                            ],
                                'addr_state': [
                                            ['NE', 'IA', 'NV', 'HI', 'FL'],
                                            ['AL', 'NM', 'NJ'],
                                            ['OK', 'MO', 'MD', 'NC'],
                                            ['AR', 'TN', 'MI', 'UT', 'VA', 'LA', 'PA', 'AZ', 'OH', 'RI', 'KY', 'DE', 'IN'],
                                            ['MA', 'SD', 'GA', 'MN', 'WI', 'WA', 'OR', 'IL', 'CT'],
                                            ['MS', 'MT', 'SC', 'VT', 'KS', 'CO', 'AK', 'NH', 'WV', 'WY', 'ID', 'DC', 'ME'],
                                            ],
                                'initial_list_status': [],
                                'verification_status': [],
                                'sub_grade': [
                                            ['G1', 'F5', 'G5', 'G3', 'G2', 'F4', 'F3', 'G4', 'F2'],
                                            ['E5', 'F1', 'E4', 'E3', 'E2'],
                                            ['E1', 'D5', 'D4'],
                                            ['D3', 'D2', 'D1'],
                                            ['C5', 'C4', 'C3'],
                                            ['C2', 'C1', 'B5'],
                                            ['B4', 'B3'],
                                            ['B2', 'B1'],
                                            ['A5', 'A4'],
                                            ['A3', 'A2', 'A1']
                                            ],
                                'term': [],
                                'emp_length': [
                                            [1, 3],
                                            [4, 6],
                                            [7, 9]
                                            ],
                                'inq_last_6mths': [
                                                [4, 33]
                                                ],
                                }
        self.debug = debug
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        try:
            X_copy = X.copy()
            
            for feature, category_groups in self.category_mapping.items():
                for category_group in category_groups:
                    # Verify whether the categories are strings, from a categorical feature, bundling with '_'.
                    if all(isinstance(element, str) for element in category_group):
                        bundled_category = '_'.join(category_group)
                        to_replace = category_group
                    # If the categories are integers, from a discrete numerical feature, bundle with '-'.
                    else:
                        bundled_category = f'{category_group[0]}-{category_group[1]}'
                        to_replace = range(category_group[0], category_group[1] + 1)
                    
                    # Bundle the categories of the variable.
                    X_copy[feature] = X_copy[feature].replace(to_replace, bundled_category)
                    
                # Make sure each and every final category will be a string, in an object data type.
                X_copy[feature] = X_copy[feature].astype(str)
                
                # Inform which feature is being transformed, the original categories and the bundled categories.
                if self.debug:
                    print(f'Bundle categories of {feature}.')
                    print(f'Original categories: {X[feature].unique().tolist()}.')
                    print(f'Bundled categories: {X_copy[feature].unique().tolist()}.')
                    print()
            
            return X_copy

        except Exception as e:
            raise(CustomException(e, sys))
        

class CatImputer(BaseEstimator, TransformerMixin):
    '''
    CatImputer
    
    Custom scikit-learn transformer for imputing missing values in categorical features. 
    It imputes missing values in features when these missing values are treated as another 
    category of the feature. This is applicable when the missing values are not at random 
    (MNAR) or when they represent a specific value, such as zero.
    
    Attributes:
    impute_mapping (dict): Dictionary mapping features to impute values.
    missing (str): String representation of missing values.
    
    Methods:
    fit(X, y=None):
        Fit the transformer on the input data X.
        
        Parameters:
        - X (DataFrame): Input data for fitting the transformer.
        - y (array-like, optional): Target variable. Default is None.
        
        Returns:
        - self: Returns the instance itself.
        
        Raises:
        - CustomException: If an exception occurs during fitting.
    
    transform(X):
        Transform the input data X by imputing missing values based on predefined mappings.
        
        Parameters:
        - X (DataFrame): Input data for transformation.
        
        Returns:
        - transformed_data: The transformed data.
        
        Raises:
        - CustomException: If an exception occurs during transformation.
    '''

    def __init__(self):
        self.impute_mapping = {
                                'mths_since_last_delinq': 'never_delinquent',
                                'tot_cur_bal': 'missing',
                              }
        self.missing = 'nan'
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        try:
            X_copy = X.copy()
            
            # Impute each feature's corresponding missing with the determined value.
            for feature, impute_value in self.impute_mapping.items():
                X_copy[feature] = X_copy[feature].replace(self.missing, impute_value)
        
            return X_copy
        
        except Exception as e:
            raise(CustomException(e, sys))
    

class CatOneHotEncoder(BaseEstimator, TransformerMixin):
    '''
    CatOneHotEncoder
    
    Custom scikit-learn transformer for one-hot encoding features. It receives data containing
    only categorical features. The numerical continuous features have been discretized, and their
    categories bundled based on similar Weight of Evidence (WoE) and a balancing of the number of 
    observations. Additionally, categorical features have been bundled with the same criteria. Then, 
    it creates dummy variables for each of these categories, removing the reference categories (those
    with the highest credit risk, lowest WoE), and creates a dataframe with each dummy name in the 
    format of 'original_independent_variable_cat'. It handles unknown categories that might arise and
    converts everything to the int8 data type to optimize memory.
    
    Attributes:
    reference_categories (list): List of reference categories to be dropped by OneHotEncoder.
    encoder (OneHotEncoder): OneHotEncoder object for feature encoding.
    
    Methods:
    fit(X, y=None):
        Fit the OneHotEncoder on the input data X.
        
        Parameters:
        - X (DataFrame): Input data for fitting the encoder.
        - y (array-like, optional): Target variable. Default is None.
        
        Returns:
        - self: Returns the instance itself.
        
        Raises:
        - CustomException: If an exception occurs during fitting.
    
    transform(X):
        Transform the input data X using the fitted encoder.
        
        Parameters:
        - X (DataFrame): Input data for transformation.
        
        Returns:
        - one_hot_df (DataFrame): Transformed DataFrame with one-hot encoded features.
        
        Raises:
        - CustomException: If an exception occurs during transformation.
    '''
    def __init__(self):
        self.reference_categories = [
                                    '>28.1K', '60', '>22.0', 'G',
                                    'G1_F5_G5_G3_G2_F4_F3_G4_F2', 
                                    '0', 'OTHER_NONE_RENT_ANY', '<=20.0K',
                                    'Verified', 'small_business_educational_renewable_energy_moving',
                                    'NE_IA_NV_HI_FL', '>28.0', '4-33', '<=4.0',
                                    '<=6.0', '<=2.0K', '<=8.0', 'f', 'missing', '<=151.0'
                                    ]
        self.encoder = OneHotEncoder(drop=self.reference_categories, 
                                    sparse_output=False,
                                    dtype=np.int8,
                                    handle_unknown='ignore',
                                    feature_name_combiner='concat')
        
    def fit(self, X, y=None):
        try:
            self.encoder.fit(X)
            return self
        
        except Exception as e:
            raise(CustomException(e, sys))
        
    def transform(self, X):
        try:
            # One-hot encode the columns.
            X_one_hot = self.encoder.transform(X)
            
            # Create a dataframe for the one-hot encoded data.
            one_hot_df = pd.DataFrame(X_one_hot, columns=self.encoder.get_feature_names_out())
            
            return one_hot_df
        
        except Exception as e:
            raise(CustomException(e, sys))
        

def save_object(file_path, object):
    '''
    Save a Python object to a binary file using pickle serialization.

    This function takes an object and a file path as input and saves the object to the specified file using pickle
    serialization. If the directory of the file does not exist, it will be created.

    Args:
        file_path (str): The path to the file where the object will be saved.
        object_to_save: The Python object that needs to be saved.

    Raises:
        CustomException: If any exception occurs during the file saving process, a custom exception is raised with
                         the original exception details.

    Example:
        save_object("saved_object.pkl", my_data)

    Note:
        This function uses pickle to serialize the object. Be cautious when loading pickled data, as it can pose
        security risks if loading data from untrusted sources.
    '''

    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, 'wb') as file_object:
            pickle.dump(object, file_object)
    
    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    '''
    Load a Python object from a binary file using pickle deserialization.

    This function reads and deserializes a Python object from the specified binary file using pickle. It returns the
    loaded object.

    Args:
        file_path (str): The path to the file from which the object will be loaded.

    Returns:
        object: The Python object loaded from the file.

    Raises:
        CustomException: If any exception occurs during the file loading process, a custom exception is raised with
                         the original exception details.
    '''

    try:
        with open(file_path, 'rb') as file_object:
            return pickle.load(file_object)
        
    except Exception as e:
        raise CustomException(e, sys)