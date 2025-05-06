#!/usr/bin/env python3
"""
Exploratory Data Analysis of Dementia Risk Factors
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind

# 1. Data Loading and Initial Exploration
df = pd.read_csv('dementia-excel.csv')
print("\n=== Initial Data Exploration ===")
print(df.head())
print(df.info())
print(df.describe())

# 2. Data Cleaning
print("\n=== Missing Values per Column ===")
print(df.isnull().sum())

categorical_cols = [
    'Gender', 'Physical Activity Level', 'Smoking Status',
    'Alcohol Consumption', 'Diabetes', 'Hypertension',
    'Cholesterol Level', 'Family History of Alzheimer’s',
    'Depression Level', 'Sleep Quality', 'Dietary Habits',
    'Air Pollution Exposure', 'Employment Status',
    'Marital Status', 'Genetic Risk Factor (APOE-ε4 allele)',
    'Social Engagement Level', 'Income Level', 'Stress Levels',
    'Urban vs Rural Living', 'Alzheimer’s Diagnosis'
]

for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].astype('category')

# 3. Exploratory Data Analysis
# Demographic Analysis

# Age distribution by diagnosis
plt.figure(figsize=(10, 6))
sns.boxplot(x='Alzheimer’s Diagnosis', y='Age', data=df)
plt.title('Age Distribution by Alzheimer\'s Diagnosis')
plt.show()

# Gender distribution
plt.figure(figsize=(8, 5))
df['Gender'].value_counts().plot(kind='bar')
plt.title('Gender Distribution')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()

# Education level analysis
plt.figure(figsize=(12, 6))
sns.boxplot(x='Alzheimer’s Diagnosis', y='Education Level', data=df)
plt.title('Education Level by Alzheimer\'s Diagnosis')
plt.show()

# Health Indicators Analysis

# BMI analysis
plt.figure(figsize=(12, 6))
sns.boxplot(x='Alzheimer’s Diagnosis', y='BMI', data=df)
plt.title('BMI Distribution by Alzheimer\'s Diagnosis')
plt.show()

# Physical activity analysis
plt.figure(figsize=(10, 6))
pd.crosstab(df['Physical Activity Level'], df['Alzheimer’s Diagnosis']).plot(kind='bar')
plt.title('Physical Activity Level vs Alzheimer\'s Diagnosis')
plt.xlabel('Physical Activity Level')
plt.ylabel('Count')
plt.show()

# Comorbidities analysis
comorbidities = ['Diabetes', 'Hypertension']
for comorbidity in comorbidities:
    plt.figure(figsize=(8, 5))
    pd.crosstab(df[comorbidity], df['Alzheimer’s Diagnosis']).plot(kind='bar')
    plt.title(f'{comorbidity} vs Alzheimer\'s Diagnosis')
    plt.xlabel(comorbidity)
    plt.ylabel('Count')
    plt.show()

# Lifestyle Factors

# Smoking status
plt.figure(figsize=(10, 6))
pd.crosstab(df['Smoking Status'], df['Alzheimer’s Diagnosis']).plot(kind='bar')
plt.title('Smoking Status vs Alzheimer\'s Diagnosis')
plt.xlabel('Smoking Status')
plt.ylabel('Count')
plt.show()

# Alcohol consumption
plt.figure(figsize=(10, 6))
pd.crosstab(df['Alcohol Consumption'], df['Alzheimer’s Diagnosis']).plot(kind='bar')
plt.title('Alcohol Consumption vs Alzheimer\'s Diagnosis')
plt.xlabel('Alcohol Consumption')
plt.ylabel('Count')
plt.show()

# Sleep quality
plt.figure(figsize=(10, 6))
pd.crosstab(df['Sleep Quality'], df['Alzheimer’s Diagnosis']).plot(kind='bar')
plt.title('Sleep Quality vs Alzheimer\'s Diagnosis')
plt.xlabel('Sleep Quality')
plt.ylabel('Count')
plt.show()

# 4. Correlation Analysis
df_corr = df.copy()
for col in categorical_cols:
    if col in df_corr.columns:
        df_corr[col] = df_corr[col].cat.codes

plt.figure(figsize=(16, 12))
sns.heatmap(df_corr.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Heatmap of Variables')
plt.show()

# Focus on correlation with diagnosis
diagnosis_corr = df_corr.corr()['Alzheimer’s Diagnosis'].sort_values(ascending=False)
print("\n=== Correlation with Alzheimer's Diagnosis ===")
print(diagnosis_corr)

# 5. Statistical Analysis
print("\n=== Statistical Comparison (T-tests) ===")
numerical_cols = ['Age', 'Education Level', 'BMI', 'Cognitive Test Score']
for col in numerical_cols:
    if col in df.columns:
        diagnosed = df[df['Alzheimer’s Diagnosis'] == 'Yes'][col]
        non_diagnosed = df[df['Alzheimer’s Diagnosis'] == 'No'][col]
        t_stat, p_val = ttest_ind(diagnosed, non_diagnosed, nan_policy='omit')
        print(f'{col}: t-statistic = {t_stat:.3f}, p-value = {p_val:.4f}')

# 6. Risk Factor Identification
top_risk_factors = diagnosis_corr[1:11]  # exclude diagnosis itself
plt.figure(figsize=(10, 6))
top_risk_factors.plot(kind='bar')
plt.title('Top Correlated Factors with Alzheimer\'s Diagnosis')
plt.ylabel('Correlation Coefficient')
plt.show()

print("\nEDA and analysis complete.")
