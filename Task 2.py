import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid')
file_path = "C:/Users/heman/OneDrive/Documents/prodiology internship/TAsk 2/gender_submission.csv"
titanic = pd.read_csv(file_path)
titanic.columns = titanic.columns.str.strip().str.lower()
print("Column names after standardization:")
print(titanic.columns)
print("First few rows:")
print(titanic.head())    
print("\nGeneral information:")
print(titanic.info())    
print("\nSummary statistics:")
print(titanic.describe())    
if 'age' in titanic.columns:
    titanic['age'].fillna(titanic['age'].median(), inplace=True)
if 'embarked' in titanic.columns:
    titanic['embarked'].fillna(titanic['embarked'].mode()[0], inplace=True)
if 'cabin' in titanic.columns:
    titanic.drop('cabin', axis=1, inplace=True)
if 'survived' in titanic.columns:
    sns.countplot(x='survived', data=titanic)
    plt.title("Survival Count")
    plt.xlabel("Survived")
    plt.ylabel("Count")
    plt.show()
if 'sex' in titanic.columns and 'survived' in titanic.columns:
    sns.countplot(x='sex', hue='survived', data=titanic)
    plt.title("Survival by Gender")
    plt.xlabel("Sex")
    plt.ylabel("Count")
    plt.legend(title='Survived')
    plt.show()
if 'age' in titanic.columns:
    sns.histplot(titanic['age'], kde=True)
    plt.title("Distribution of Age")
    plt.xlabel("Age")
    plt.ylabel("Frequency")
    plt.show()
numerical_columns = ['age', 'fare', 'sibsp', 'parch']
numerical_columns = [col for col in numerical_columns if col in titanic.columns]
if numerical_columns:
    sns.pairplot(titanic[numerical_columns])
    plt.suptitle("Pairplot of Numerical Variables")
    plt.subplots_adjust(top=0.95)
    plt.show()
if numerical_columns:
    corr = titanic[numerical_columns].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title("Correlation Heatmap")
    plt.show()
print("\nEDA completed successfully.")