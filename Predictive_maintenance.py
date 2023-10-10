#Import Dependecies 

import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from scipy.stats import chi2_contingency
import plotly.express as px

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from imblearn.under_sampling import RandomUnderSampler  # Assuming you have imbalanced-learn installed

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score


#read csv file

df = pd.read_csv('C:\\Users\\FloMartin-ResolveMin\\Desktop\\git\\Predictive_maintenance\\predictive_maintenance_dataset.csv')

print(df)

# check for missing values

missing_data = df.isnull().sum()

# remove duplicates columns

df = df.loc[:, ~df.columns.duplicated()]


# check the numbers of entries and unique devices

num_entries = df.shape[0]
num_unique_devices = df['device'].nunique()

print(f"Number of entries (rows): {num_entries}")
print(f"Number of unique devices: {num_unique_devices}")

# print summary statistics

print(df.describe())

#check the class imbalance

class_counts = df['failure'].value_counts()

# Perform a chi-squared test for independence
contingency_table = pd.crosstab(df['failure'], df['metric1'])

chi2, p, dof, expected = chi2_contingency(contingency_table)

print("Chi-Squared Statistic:", chi2)
print("P-value:", p)

# Check if the result is statistically significant (you can set your significance level)
alpha = 0.05
if p < alpha:
    print("There is a significant relationship between variable1 and variable2.")
else:
    print("There is no significant relationship between variable1 and variable2.")


# data visualisation
# visualise the total number of failures per month

df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.month

monthly_fail = df.groupby('month')['failure'].sum().reset_index()

plt.figure(figsize=(20, 12))
sns.barplot(x='month', y='failure', data=monthly_fail, color='midnightblue')
plt.title('Aggregated device failures by month')
plt.xlabel('Month')
plt.ylabel('Total number of failures')
plt.ylim(0, 100)
plt.xticks(rotation=0)
plt.savefig('device_failures_by_month.png')
plt.show()

# Check devices with multiple failures
device_failure_counts = df.groupby('device')['failure'].sum()
devices_with_multiple_failures = device_failure_counts[device_failure_counts >= 1]
num_devices_with_multiple_failures = len(devices_with_multiple_failures)

print(f"Number of devices with multiple failures: {num_devices_with_multiple_failures}")

# Calculate the correlation matrix
correlation_matrix = df.corr(numeric_only=True)


# Create a heatmap to visualize the correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')

# Save the correlation matrix as an image
plt.savefig('correlation_matrix.png')

plt.show()

# Data visualization with Plotly px
fig = px.bar(monthly_fail, x='month', y='failure', color='month',
             labels={'failure': 'Total number of failures'},
             title='Aggregated device failures by month')

# Customize the layout (optional)
fig.update_layout(
    xaxis_title='Month',
    yaxis_title='Total number of failures',
    xaxis=dict(tickmode='linear'),
    yaxis=dict(range=[0, 100])
)

# Customize the layout with larger font size
fig.update_layout(
    xaxis_title='Month',
    yaxis_title='Total number of failures',
    xaxis=dict(
        tickmode='linear',
        title=dict(
            text='Month',
            font=dict(
                size=25  # Adjust the font size as needed
            )
        ),
        tickfont=dict(
            size=25  # Adjust the font size of tick labels as needed
        )
    ),
    yaxis=dict(
        range=[0, 100],
        title=dict(
            text='Total number of failures',
            font=dict(
                size=25  # Adjust the font size as needed
            )
        ),
        tickfont=dict(
            size=25  # Adjust the font size of tick labels as needed
        )
    ),
    title=dict(
        text='Aggregated device failures by month',
        font=dict(
            size=35  # Adjust the title font size as needed
        )
    )
)

# Save the interactive visualization as an HTML file
fig.write_html('device_failures_by_month_interactive.html')

# Display the interactive chart in the notebook (optional)
fig.show()

# DATA PREP


# Select columns and convert 'failure' to a categorical variable
df['failure'] = df['failure'].astype('category')

# Set seed and split data into training and testing sets
X = df.drop('failure', axis=1)
y = df['failure']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Drop the 'date' column from both X_train and X_test
X_train = X_train.drop('date', axis=1)
X_test = X_test.drop('date', axis=1)

# Identify non-numeric columns
non_numeric_cols = X_train.select_dtypes(exclude=['number']).columns.tolist()

# One-hot encode non-numeric columns
X_train = pd.get_dummies(X_train, columns=non_numeric_cols)
X_test = pd.get_dummies(X_test, columns=non_numeric_cols)

# Set seed and create cross-validation folds
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

# Create and train a Random Forest Classifier
rf_classifier = RandomForestClassifier(random_state=1)

# Evaluate the model using cross-validation
cross_val_scores = cross_val_score(rf_classifier, X_train, y_train, cv=skf, scoring='accuracy')

print("Cross-validation scores:", cross_val_scores)
print("Mean CV Accuracy:", cross_val_scores.mean())

# Fit the model to the entire training dataset
rf_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_classifier.predict(X_test)

# Evaluate the model on the test set
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)



print(accuracy)
print(report)
print(conf_matrix)