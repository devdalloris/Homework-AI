import pandas as pd
import matplotlib.pyplot as plt

#Understand the Dataset
df_titanic = pd.read_csv('lesson-1/homework/titanic.csv') # Corrected line
print("First few rows \n",df_titanic.head())
print(df_titanic.describe())
print(df_titanic.info())

#Summary statistics
#mean, median, mode
print('Survived column mean: ', df_titanic['Survived'].mean())
print('Survived column median: ', df_titanic['Survived'].median())
print('Survived column mode: ', df_titanic['Survived'].mode())
print()
print('Age column median: ', df_titanic['Age'].median())
print('Age column mode: ', df_titanic['Age'].mode())
print('Age column mean: ', df_titanic['Age'].mean())

#Unique values 
print("Unique values in 'Sex' column: ",df_titanic['Sex'].unique())
print("Unique values in 'Embarked' column: ",df_titanic['Embarked'].unique())

#missing values
print("Missing values: \n", df_titanic.isnull())
print("The most missing values in column: \n", df_titanic.isnull().sum())

#data visualization
#histogram to analyze Age column
plt.figure(figsize=(10, 6)) # Set figure size for better readability
plt.hist(df_titanic['Age'], bins=20, edgecolor='black', alpha=0.7)
plt.title('Distribution of Age (Histogram)', fontsize=16)
plt.xlabel('Age', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(axis='y', alpha=0.75)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout() # Adjust layout to prevent labels from overlapping
plt.show()

#barchart to analyze sex column
plt.figure(figsize=(8, 6)) # Set figure size
sex_counts = df_titanic['Sex'].value_counts()
sex_counts.plot(kind='bar', color=['skyblue', 'lightcoral'], edgecolor='black')
# Add title and labels
plt.title('Male/Female Ratio (Sex Distribution)', fontsize=16)
plt.xlabel('Gender', fontsize=12)
plt.ylabel('Number of Passengers', fontsize=12)

# Rotate x-axis labels if needed (for better readability)
plt.xticks(rotation=0, fontsize=10)
plt.yticks(fontsize=10)

# Add grid for better readability
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout() # Adjust layout to prevent labels from overlapping
plt.show()


# --- 2. Count the occurrences of each 'Survived' status ---
survived_counts = df_titanic['Survived'].value_counts()

# Map numerical labels to meaningful strings for plotting
survived_labels = {0: 'Not Survived', 1: 'Survived'}
survived_counts.index = survived_counts.index.map(survived_labels)


# --- 3. Create the Pie Chart for Survived Distribution ---
plt.figure(figsize=(8, 8)) # Set figure size for a good pie chart
plt.pie(survived_counts,
        labels=survived_counts.index,
        autopct='%1.1f%%', # Format percentages to one decimal place
        startangle=90,     # Start the first slice at the top
        colors=['lightcoral', 'lightgreen'], # Colors for Not Survived and Survived
        wedgeprops={'edgecolor': 'black'}) # Add black edges to slices
plt.title('Distribution of Survived (Target Variable)', fontsize=16)
plt.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.
plt.tight_layout()
plt.show()
