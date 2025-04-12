# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set visualization style
sns.set(style="whitegrid", palette="pastel")
%matplotlib inline

# Step 1: Load the dataset
print("Step 1: Loading the Data")
file_path = r"C:\Users\dhira\OneDrive\Desktop\Accidental_Drug_Related_Deaths_2012-2023 python .csv"
data = pd.read_csv(file_path)
print(f"Dataset shape: {data.shape}")
print("\nFirst 5 rows:")
print(data.head())
print("\nColumns:", data.columns.tolist())

# Finding: Dataset has 65 columns (Age, Sex, Heroin, Fentanyl, cities, etc.).

# Step 2: Check data types and missing values
print("\nStep 2: Checking Data Types and Missing Values")
print("\nData Types:")
print(data.dtypes)
print("\nMissing Values:")
missing_values = data.isnull().sum()
missing_percentage = (missing_values / len(data)) * 100
print(pd.DataFrame({'Missing Values': missing_values, 'Percentage': missing_percentage}))

# Finding: 'Date' is text, drugs are 'Y' or empty, some columns have many missing values.

# Step 3: Clean the data
print("\nStep 3: Cleaning the Data")

# Convert 'Date' to datetime
data['Date'] = pd.to_datetime(data['Date'], errors='coerce')

# Fill missing text columns with 'Unknown'
text_cols = ['Sex', 'Race', 'Residence City', 'Death City', 'Cause of Death', 'Manner of Death']
for col in text_cols:
    data[col].fillna('Unknown', inplace=True)

# Fill missing drug columns with 'N'
drug_cols = ['Heroin', 'Cocaine', 'Fentanyl', 'Oxycodone', 'Ethanol', 'Benzodiazepine']
for col in drug_cols:
    data[col].fillna('N', inplace=True)

# Fill missing Age with median and ensure float
data['Age'] = data['Age'].fillna(data['Age'].median()).astype(float)

# Remove duplicates
data = data.drop_duplicates()

print(f"Shape after cleaning: {data.shape}")

# Finding: Dataset retains >2000 rows, 65 columns, cleaned for analysis.

# Step 4: Calculate summary statistics
print("\nStep 4: Summary Statistics")
print("\nNumerical Column (Age):")
print(data['Age'].describe())
print("\nCategorical Columns:")
for col in ['Sex', 'Race', 'Manner of Death']:
    print(f"\n{col} counts:")
    print(data[col].value_counts())
print("\nAverage Age by Sex:")
print(data.groupby('Sex')['Age'].mean())

# Finding: Age averages ~40, more males, mostly White, accidents dominate.

# Step 5: Create visualizations
print("\nStep 5: Visualizations")

# Histogram: Age distribution
plt.figure(figsize=(8, 4))
plt.hist(data['Age'], bins=20, color='red', edgecolor='black', alpha=0.5)
plt.title('Histogram of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.savefig(r"C:\Users\dhira\OneDrive\Desktop\age_histogram.png", dpi=300)
plt.show()

# Finding: Most ages are 30-50.

# Frequency Polygon: Age by Sex
plt.figure(figsize=(8, 4))
sns.histplot(data=data, x='Age', bins=20, kde=True, hue='Sex', palette='Set2')
plt.title('Age Distribution by Sex')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.savefig(r"C:\Users\dhira\OneDrive\Desktop\age_frequency_polygon.png", dpi=300)
plt.show()

# Finding: Males have more deaths, similar age range.

# Bar Plot: Deaths by Sex
plt.figure(figsize=(8, 4))
bars = sns.countplot(x='Sex', data=data, color='skyblue')
for bar in bars.patches:
    height = bar.get_height()
    bars.annotate(f'{int(height)}', 
                  (bar.get_x() + bar.get_width() / 2, height),
                  ha='center', va='bottom')
plt.title('Number of Deaths by Sex')
plt.xlabel('Sex')
plt.ylabel('Number of Deaths')
plt.savefig(r"C:\Users\dhira\OneDrive\Desktop\sex_bar_plot.png", dpi=300)
plt.show()

# Finding: More males than females.

# Pie Chart: Race distribution
race_counts = data['Race'].value_counts().head(3)
plt.figure(figsize=(8, 4))
plt.pie(race_counts, labels=race_counts.index, autopct='%1.1f%%', 
        colors=['lightcoral', 'lightgreen', 'lightblue'])
plt.title('Top Races in Drug-Related Deaths')
plt.savefig(r"C:\Users\dhira\OneDrive\Desktop\race_pie_chart.png", dpi=300)
plt.show()

# Finding: Whites are the largest group.

# Box Plot: Age by Sex
plt.figure(figsize=(8, 4))
sns.boxplot(x='Sex', y='Age', data=data, palette='Set3')
plt.title('Age Distribution by Sex')
plt.xlabel('Sex')
plt.ylabel('Age')
plt.savefig(r"C:\Users\dhira\OneDrive\Desktop\age_box_plot.png", dpi=300)
plt.show()

# Finding: Ages 30-50, some outliers.

# Line Plot: Deaths by year
data['Year'] = data['Date'].dt.year
year_counts = data['Year'].value_counts().sort_index()
plt.figure(figsize=(8, 4))
plt.plot(year_counts.index, year_counts.values, marker='o', linestyle='--', color='purple')
plt.title('Drug Deaths Over Time')
plt.xlabel('Year')
plt.ylabel('Number of Deaths')
plt.grid(True)
plt.savefig(r"C:\Users\dhira\OneDrive\Desktop\year_line_plot.png", dpi=300)
plt.show()

# Finding: Deaths increase over time.

# Grouped Bar Chart: Top drugs by Sex
top_drugs = ['Heroin', 'Cocaine', 'Fentanyl']
drug_sex = data.groupby('Sex')[top_drugs].apply(lambda x: (x == 'Y').sum()).T
plt.figure(figsize=(8, 4))
drug_sex.plot(kind='bar', color=['pink', 'lightblue'])
plt.title('Top Drugs Used by Sex')
plt.xlabel('Drug')
plt.ylabel('Number of Cases')
plt.legend(title='Sex')
plt.xticks(rotation=0)
plt.savefig(r"C:\Users\dhira\OneDrive\Desktop\drug_sex_bar_plot.png", dpi=300)
plt.show()

# Finding: Males use more drugs.

# Scatter Plot: Age vs Drug Count
data['Drug_Count'] = data[drug_cols].apply(lambda x: (x == 'Y').sum(), axis=1).astype(float)
plt.figure(figsize=(8, 4))
sns.scatterplot(x='Age', y='Drug_Count', hue='Sex', data=data, alpha=0.5, palette='coolwarm')
plt.title('Age vs Number of Drugs')
plt.xlabel('Age')
plt.ylabel('Number of Drugs')
plt.savefig(r"C:\Users\dhira\OneDrive\Desktop\age_drug_scatter.png", dpi=300)
plt.show()

# Finding: Males tend to use more drugs.

# Pair Plot: Age and Drug Count
plt.figure(figsize=(8, 4))
sns.pairplot(data[['Age', 'Drug_Count', 'Sex']], hue='Sex', palette='Set2')
plt.suptitle('Age vs Number of Drugs by Sex', y=1.02)
plt.savefig(r"C:\Users\dhira\OneDrive\Desktop\pair_plot.png", dpi=300)
plt.show()

# Finding: Shows age and drug use patterns.

# Step 6: Detect outliers
print("\nStep 6: Outlier Detection")
q1 = data['Age'].quantile(0.25)
q3 = data['Age'].quantile(0.75)
iqr = q3 - q1
outliers = data[(data['Age'] < q1 - 1.5 * iqr) | (data['Age'] > q3 + 1.5 * iqr)]
print(f"Number of Age outliers: {len(outliers)}")

# Finding: Some young or old ages, but valid.

# Step 7: Calculate correlations
print("\nStep 7: Correlation")
drug_binary = data[drug_cols].apply(lambda x: x.map({'Y': 1, 'N': 0}))
corr_matrix = drug_binary.corr()
plt.figure(figsize=(6, 4))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Between Drugs')
plt.savefig(r"C:\Users\dhira\OneDrive\Desktop\correlation_heatmap.png", dpi=300)
plt.show()

# Finding: Heroin and Fentanyl often used together.

# Step 8: Save cleaned dataset
print("\nStep 8: Saving Cleaned Data")
output_path = r"C:\Users\dhira\OneDrive\Desktop\drug_overdose_cleaned.csv"
data.to_csv(output_path, index=False)
print(f"Cleaned dataset saved as: {output_path}")

# Step 9: Summarize key findings
print("\nKey Findings:")
print("- Data cleaned: Fixed missing values, kept >2000 rows.")
print("- Trend: Drug deaths rising, Fentanyl leads.")
print("- Demographics: Males, Whites, ages 30-50 most affected.")
print("- Drugs: Fentanyl, Heroin, Cocaine top causes.")
print("- Correlation: Heroin and Fentanyl linked.")
print("- Outliers: Young/old ages valid.")