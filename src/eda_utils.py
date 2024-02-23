'''
This script aims to provide functions that will turn the exploratory data analysis (EDA) process easier. 
'''


'''
Importing libraries
'''

# Data manipulation and visualization.
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Debugging.
from src.exception import CustomException
import sys

# Warnings.
from warnings import filterwarnings
filterwarnings('ignore')


def sns_plots(data, features, histplot=True, countplot=False,     
              barplot=False, barplot_y=None, boxplot=False, 
              boxplot_x=None, outliers=False, kde=False, 
              hue=None, rotation=None, color='#8d0801'):
    '''
    Generate Seaborn plots for visualization.

    This function generates various types of Seaborn plots based on the provided
    data and features. Supported plot types include histograms, count plots,
    bar plots, box plots, and more.

    Args:
        data (DataFrame): The DataFrame containing the data to be visualized.
        features (list): A list of feature names to visualize.
        histplot (bool, optional): Generate histograms. Default is True.
        countplot (bool, optional): Generate count plots. Default is False.
        barplot (bool, optional): Generate bar plots. Default is False.
        barplot_y (str, optional): The name of the feature for the y-axis in bar plots.
        boxplot (bool, optional): Generate box plots. Default is False.
        boxplot_x (str, optional): The name of the feature for the x-axis in box plots.
        outliers (bool, optional): Show outliers in box plots. Default is False.
        kde (bool, optional): Plot Kernel Density Estimate in histograms. Default is False.
        hue (str, optional): The name of the feature to use for color grouping. Default is None.
        rotation (int, optional): The xticks rotation for high cardinality features plots. Default is None.
        color (str, optional): The color of the plot. Default is '#8d0801'.

    Returns:
        None

    Raises:
        CustomException: If an error occurs during the plot generation.

    '''
    
    try:
        # Getting num_features and num_rows and iterating over the sublot dimensions.
        num_features = len(features)
        num_rows = num_features // 3 + (num_features % 3 > 0)  

        fig, axes = plt.subplots(num_rows, 3, figsize=(20, 5*num_rows))  

        for i, feature in enumerate(features):
            row = i // 3  
            col = i % 3  

            ax = axes[row, col] if num_rows > 1 else axes[col] 
            
            if countplot:
                # Plotting countplot and adding the counts at the top of each bar.
                sns.countplot(data=data, x=feature, hue=hue, ax=ax, color=color)
                for container in ax.containers:
                    ax.bar_label(container)

            elif barplot:
                # Plotting barplot and adding the averages at the top of each bar.
                ax = sns.barplot(data=data, x=feature, y=barplot_y, hue=hue, ax=ax, ci=None, color=color)
                for container in ax.containers:
                    ax.bar_label(container)

            elif boxplot:
                # Plotting multivariate boxplot.
                sns.boxplot(data=data, x=boxplot_x, y=feature, showfliers=outliers, ax=ax, color=color)

            elif outliers:
                # Plotting univariate boxplot.
                sns.boxplot(data=data, x=feature, ax=ax, color=color)

            else:
                # Plotting histplot.
                sns.histplot(data=data, x=feature, hue=hue, kde=kde, ax=ax, color=color)

            ax.set_title(feature)  
            ax.set_xlabel('')  
            
        # Adjusting xticks rotation for high cardinality variables.
        if rotation is not None:
            for ax in axes.flat:
                ax.tick_params(axis='x', rotation=rotation)
        
        # Removing unused axes.
        if num_features < len(axes.flat):
            for j in range(num_features, len(axes.flat)):
                fig.delaxes(axes.flat[j])

        plt.tight_layout()
    
    except Exception as e:
        raise CustomException(e, sys)


def check_outliers(data, features):
    '''
    Check for outliers in the given dataset features.

    This function calculates and identifies outliers in the specified features
    using the Interquartile Range (IQR) method.

    Args:
        data (DataFrame): The DataFrame containing the data to check for outliers.
        features (list): A list of feature names to check for outliers.

    Returns:
        tuple: A tuple containing three elements:
            - outlier_indexes (dict): A dictionary mapping feature names to lists of outlier indexes.
            - outlier_counts (dict): A dictionary mapping feature names to the count of outliers.
            - total_outliers (int): The total count of outliers in the dataset.

    Raises:
        CustomException: If an error occurs while checking for outliers.

    '''
    
    try:
    
        outlier_counts = {}
        outlier_indexes = {}
        total_outliers = 0
        
        for feature in features:
            Q1 = data[feature].quantile(0.25)
            Q3 = data[feature].quantile(0.75)
            
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            feature_outliers = data[(data[feature] < lower_bound) | (data[feature] > upper_bound)]
            outlier_indexes[feature] = feature_outliers.index.tolist()
            outlier_count = len(feature_outliers)
            outlier_counts[feature] = outlier_count
            total_outliers += outlier_count
        
        print(f'There are {total_outliers} outliers in the dataset.')
        print()
        print(f'Number (percentage) of outliers per feature: ')
        print()
        for feature, count in outlier_counts.items():
            print(f'{feature}: {count} ({round(count/len(data)*100, 2)})%')

        return outlier_indexes, outlier_counts, total_outliers
    
    except Exception as e:
        raise CustomException(e, sys)
    

def default_analysis(cat_variable, variable_name, data, continuous=False, not_ordered=False):
    '''
    Perform default analysis on a discrete or discretized continuous variable.

    This function analyzes the default behavior based on a discrete or a discretized continuous variable, providing insights such as the proportion
    of good and bad borrowers per category, weight of evidence (WoE), information value (IV), and other relevant statistics.

    Args:
    - cat_variable: Discrete or discretized continuous variable to be analyzed.
    - variable_name: Name of the variable (used for creating a new variable indicating the categories).
    - data: DataFrame containing the data for analysis.
    - continuous: Whether the input variable is continuous. Default is False.
    - not_ordered: Whether the variable has an ordinal/continuous relationship. Default is False.

    Returns:
    - DataFrame: Analysis results including counts, proportions, weight of evidence and information value.

    Raises:
    - CustomException: If an error occurs during the analysis.

    Example:
    ```python
    result = default_analysis(data['loan_amnt'], 'loan_amnt', data, continuous=True)
    ```
    '''
    try:
        # Obtaining the discrete variable (already cut) and a copy of the original data.
        cat_name = f'{variable_name}_cat'
        default_analysis_df = data.copy()
        default_analysis_df[cat_name] = cat_variable
        
        if continuous:
            default_analysis_df[cat_name] = default_analysis_df[cat_name].apply(lambda x: f'{round(x.left)}-{round(x.right)}')
        
        # Grouping the data and adding the interesting columns.
        grouped_n_obs = default_analysis_df.groupby([cat_name])[['default']].count().reset_index().rename(columns={'default': 'n_obs'})
        grouped_n_obs['obs_proportion (%)'] = grouped_n_obs['n_obs'] / grouped_n_obs['n_obs'].sum()
        good_proportion = default_analysis_df.groupby([cat_name])[['default']].mean().reset_index().rename(columns={'default': 'good_row (%)'}).drop(columns=[cat_name])
        default_analysis_df = pd.concat([grouped_n_obs, good_proportion], axis=1)
        default_analysis_df['bad_row (%)'] = (1 - default_analysis_df['good_row (%)']) 
        default_analysis_df['n_good'] = default_analysis_df['good_row (%)'] * default_analysis_df['n_obs'] 
        default_analysis_df['n_bad'] = default_analysis_df['bad_row (%)'] * default_analysis_df['n_obs']
        default_analysis_df['good_col (%)'] = default_analysis_df['n_good'] / default_analysis_df['n_good'].sum() * 100
        default_analysis_df['bad_col (%)'] = default_analysis_df['n_bad'] / default_analysis_df['n_bad'].sum() * 100
        default_analysis_df['g/b'] = default_analysis_df['good_row (%)'] / default_analysis_df['bad_row (%)']
        default_analysis_df['woe'] = np.log((default_analysis_df['n_good'] / default_analysis_df['n_good'].sum()) / \
                                            (default_analysis_df['n_bad'] / default_analysis_df['n_bad'].sum()))
        if not_ordered:
            default_analysis_df = default_analysis_df.sort_values(by=['woe'])
        default_analysis_df['iv'] = ((default_analysis_df['n_good'] / default_analysis_df['n_good'].sum()) - \
                                    (default_analysis_df['n_bad'] / default_analysis_df['n_bad'].sum())) * \
                                    default_analysis_df['woe']
        default_analysis_df['obs_proportion (%)'] = default_analysis_df['obs_proportion (%)'] * 100
        default_analysis_df['good_row (%)'] = default_analysis_df['good_row (%)'] * 100
        default_analysis_df['bad_row (%)'] = default_analysis_df['bad_row (%)'] * 100
        default_analysis_df = default_analysis_df.round(2)
        default_analysis_df.loc[len(default_analysis_df.index)] = ['total', default_analysis_df['n_obs'].sum(), 100, 
                                                                round((default_analysis_df['n_good'].sum() / default_analysis_df['n_obs'].sum() * 100), 2),
                                                                round((default_analysis_df['n_bad'].sum() / default_analysis_df['n_obs'].sum() * 100), 2), default_analysis_df['n_good'].sum(),
                                                                default_analysis_df['n_bad'].sum(), 100, 100, '-', '-', default_analysis_df['iv'].sum()]
        default_analysis_df.index = default_analysis_df[cat_name]
        default_analysis_df = default_analysis_df.drop(columns=[cat_name])
        
        return default_analysis_df
    except Exception as e:
        raise CustomException(e, sys)


def plot_woe_bad_rate_by_variable(data, variable_name, figsize=(20, 5), rotation=None):
    '''
    Plot Weight of Evidence (WoE) and Bad Rate for each category of a variable.

    This function creates a subplot with two plots:
    - The first plot displays the WoE values for each category.
    - The second plot shows the composition of Good and Bad Rates for each category.

    Args:
    - data: DataFrame containing WoE and Bad Rate information.
    - variable_name: Name of the variable for labeling.
    - figsize: Tuple specifying the size of the figure. Default is (20, 5).
    - rotation: Rotation angle for x-axis labels in the first plot. Default is None.

    Returns:
    - None

    Raises:
    - CustomException: If an error occurs during the plotting.

    Example:
    ```python
    plot_woe_bad_rate_by_variable(my_data, 'CreditScore', figsize=(15, 6), rotation=45)
    ```

    '''
    try:
        fig, ax = plt.subplots(1, 2, figsize=figsize)
        
        categories = data.reset_index().iloc[:-1, 0]
        woe = data.reset_index().iloc[:-1, -2].astype('float')
        good_row = data.reset_index().iloc[:-1, 3].astype('float')
        bad_row = data.reset_index().iloc[:-1, 4].astype('float')

        ax[0].plot(categories, woe, marker='o', linestyle='--')
        ax[0].set_title(f'WoE by {variable_name}')
        ax[0].set_xticks(categories.unique().tolist(), categories.unique().tolist(), rotation=rotation)
        
        good_bars = ax[1].barh(y=categories, width=good_row, color='#8a817c', label='Good')
        bad_bars = ax[1].barh(y=categories, width=bad_row, left=good_row, color='#8d0801', label='Bad')

        # Annotate percentage values inside each bar
        for good_bar, bad_bar, good_rate, bad_rate in zip(good_bars, bad_bars, good_row, bad_row):
            x_position_good = good_bar.get_x() + good_bar.get_width() / 2
            x_position_bad = bad_bar.get_x() + bad_bar.get_width() / 2  
            y_position = good_bar.get_y() + good_bar.get_height() / 2

            ax[1].text(x_position_good, y_position, f'{good_rate}%', ha='center', va='center', color='white', fontsize=10)
            ax[1].text(x_position_bad, y_position, f'{bad_rate}%', ha='center', va='center', color='white', fontsize=10)

        ax[1].set_title(f'Default rate by {variable_name}')
        ax[1].xaxis.set_visible(False)
        ax[1].invert_yaxis()
        ax[1].grid(False)
        ax[1].legend(bbox_to_anchor=(1, 1.1), loc='center')
    except Exception as e:
        raise CustomException(e, sys)
