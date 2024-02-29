
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

# Debugging.
from src.exception import CustomException
import sys

# Warnings.
from warnings import filterwarnings
filterwarnings('ignore')


class WoeDiscretizerBundler(BaseEstimator, TransformerMixin):
    '''
    WoeDiscretizerBundler
    
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


class WoeBundler(BaseEstimator, TransformerMixin):
    '''
    WoeBundler
    
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
        

class MissingImputerCategory(BaseEstimator, TransformerMixin):
    '''
    MissingImputerCategory
    
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
    

class OneHotFeatureEncoder(BaseEstimator, TransformerMixin):
    '''
    OneHotFeatureEncoder
    
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
            # One-hot encoding the columns.
            X_one_hot = self.encoder.transform(X)
            
            # Creating a dataframe for the one-hot encoded data.
            one_hot_df = pd.DataFrame(X_one_hot, columns=self.encoder.get_feature_names_out())
            
            return one_hot_df
        
        except Exception as e:
            raise(CustomException(e, sys))
        