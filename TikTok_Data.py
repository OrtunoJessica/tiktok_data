#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 15:10:01 2023

@author: jessicaortuno
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

df1 = pd.read_csv("free_tiktok_scraper_dataset.csv")

# Select columns of interest
tiktok_data = df1[["authorMeta/nickName", "authorMeta/verified",
                    "commentCount", "playCount", "shareCount",
                    "videoMeta/duration"]]

# Rename columns
tiktok_data.columns = ["Nickname", "Verified", "CommentCount",
                       "PlayCount", "ShareCount", "Duration"]

# Display the resulting data frame
print(tiktok_data)

#Summary statistics
print(tiktok_data.describe())

# Display the first few rows of the resulting data frame
print(tiktok_data.head())

# Linear regression to compare possible measures of popularity
CommentShare = sm.OLS.from_formula('CommentCount ~ ShareCount + PlayCount', data=tiktok_data).fit()

# Display regression summary
print(CommentShare.summary())

# Plotting the regression
sns.regplot(x='ShareCount', y='CommentCount', data=tiktok_data, scatter_kws={'alpha': 0.3})
plt.title(f'Adj R2 = {CommentShare.rsquared_adj:.5f}, Intercept = {CommentShare.params[0]:.5f}, Slope = {CommentShare.params[1]:.5f}, P = {CommentShare.pvalues[1]:.5f}')
plt.show()

# Linear regression to compare possible measures of popularity
CommentShare = sm.OLS.from_formula('CommentCount ~ ShareCount + PlayCount', data=tiktok_data).fit()

# Residual plot
residuals = CommentShare.resid
fitted_values = CommentShare.fittedvalues

# Scatter plot of residuals against fitted values
sns.scatterplot(x=fitted_values, y=residuals, alpha=0.3)
plt.axhline(y=0, color='red', linestyle='--')  # Add a horizontal line at y=0 for reference
plt.title('Residuals vs Fitted Values')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.show()

# Linear regression to compare possible measures of popularity
CommentShare = sm.OLS.from_formula('CommentCount ~ ShareCount + PlayCount', data=tiktok_data).fit()

# Standardized residuals
standardized_residuals = CommentShare.get_influence().resid_studentized_internal

# QQ plot of standardized residuals
sm.qqplot(standardized_residuals, line='45', fit=True)
plt.title('QQ Plot of Standardized Residuals')
plt.xlabel('Theoretical Quantiles')
plt.ylabel('Standardized Residuals')
plt.xlim(-4, 4)
plt.show()

# Standardized residuals
standardized_residuals = CommentShare.get_influence().resid_studentized_internal

# Square root of absolute values of standardized residuals
sqrt_residuals = abs(standardized_residuals)**0.5

# Scale-location plot
plt.figure(figsize=(10, 6))  # Adjust the figure size
plt.scatter(CommentShare.fittedvalues, sqrt_residuals, alpha=0.3)
sns.regplot(x=CommentShare.fittedvalues, y=sqrt_residuals, scatter=False, ci=None, line_kws={'color': 'red'})
plt.title('Scale Location Plot')
plt.xlabel('Fitted Values')
plt.ylabel('Square Root of Standardized Residuals')
plt.show()