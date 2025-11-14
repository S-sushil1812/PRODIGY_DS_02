import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load dataset
df = pd.read_csv(r"C:\Users\sushi\OneDrive\Desktop\Task\Task_no_2\train.csv")
print(df.head())

# --- Preview Data ---
print(df.shape)
print(df.info())

#Step 2:Data cleaning
#Fill missing 'Age' with median 
df['Age']= df['Age'].fillna(df['Age'].median())
#Fill missing 'Embarked' with mode
df['Embarked']= df['Embarked'].fillna(df['Embarked'].mode()[0])
# Drop 'Cabin' since it has too many missing values
df.drop('Cabin', axis=1, inplace=True)

print("\nRecheck missing values after cleaning:",df.isnull().sum())

# Step 3: Exploratory Data Analysis (EDA)
df.describe(include='all')

print("\na).Univariate Analysis \n 1.Gender Distribution:")
sns.countplot(x='Sex', data=df)
plt.title('Gender Distribution')
plt.show()

print("\n 2.Survival distribution:")
sns.countplot(x='Survived', data=df)
plt.title('Survival Distribution (0 = Died, 1 = Survived)')
plt.show()

print("\n 3.Age distribution:")
sns.histplot(df['Age'], bins=30, kde=True, color='blue')
plt.title('Age Distribution')
plt.show()

print("\nb).Bivariate Analysis:\n 1.Survival vs Gender:")
sns.countplot(x='Sex', hue='Survived', data=df)
plt.title('Survival count by Gender')
plt.show()

print("\n 2.Survival vs Passenger Class:")
sns.countplot(x='Pclass', hue='Survived', data=df)
plt.title('Survival count by Passenger Class')
plt.show()

print("\n 3.Age vs Survival:")
sns.boxplot(x='Survived', y='Age', data=df)
plt.title('Age distribution by Survival Status')
plt.show()

# --- Correlation Heatmap ---
print("\n📈 Correlation Analysis")
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# --- Key Insights ---
print("""
🔍 Key Insights:
1️⃣ Females had a much higher survival rate than males.
2️⃣ 1st class passengers had higher chances of survival.
3️⃣ Younger passengers had slightly better survival chances.
4️⃣ Fare is positively correlated with survival (higher fare → higher chance).
""")

# Step 4: Save cleaned dataset
#df.to_csv(r"C:\Users\sushi\OneDrive\Desktop\Task\Task_no_2\cleaned_titanic.csv", index=False)

