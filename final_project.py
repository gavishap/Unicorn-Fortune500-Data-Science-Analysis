
# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
import plotly.express as px
import numpy as np

# Load the data
df1 = pd.read_csv('Unicorn_Companies.csv')
df2 = pd.read_csv('Fortune 500 Companies.csv')

# Display the first 5 rows 
print(df1.head())
print(df2.head())

# Print the column names
print("\nColumn Names for df1:\n", df1.columns.to_list())
print("\nColumn Names for df2:\n", df2.columns.to_list())

# Print the data types
print("\nData Types for df1:\n", df1.dtypes)
print("\nData Types for df2:\n", df2.dtypes)

# Print information about missing values
print("\nMissing Values in df1:\n", df1.isnull().sum())
print("\nMissing Values in df2:\n", df2.isnull().sum())

# Calculate the percentage of missing values 
missing_values_percentage_df1 = df1.isnull().sum().sum() / (df1.shape[0] * df1.shape[1]) * 100
missing_values_percentage_df2 = df2.isnull().sum().sum() / (df2.shape[0] * df2.shape[1]) * 100
print("\nPercentage of Missing Values in df1:", missing_values_percentage_df1)
print("\nPercentage of Missing Values in df2:", missing_values_percentage_df2)

# If the percentage of missing values is less than 10%, then this data file is not good for this project
if missing_values_percentage_df1 < 10:
    print("\ndf1 is not suitable for this project as it has less than 10% missing values.")
if missing_values_percentage_df2 < 10:
    print("\ndf2 is not suitable for this project as it has less than 10% missing values.")

# Print the first 5 lines
print("\nFirst 5 Lines of df1:")
print(df1.head())
print("\nFirst 5 Lines of df2:")
print(df2.head())

# Print the last 5 lines
print("\nLast 5 Lines of df1:")
print(df1.tail())
print("\nLast 5 Lines of df2:")
print(df2.tail())

# Print 5 random lines from the middle
print("\nRandom 5 Lines from Middle of df1:")
print(df1.sample(5))
print("\nRandom 5 Lines from Middle of df2:")
print(df2.sample(5))

# Print statistics describing the data
print("\nDescriptive Statistics of df1:")
print(df1.describe(include='all'))
print("\nDescriptive Statistics of df2:")
print(df2.describe(include='all'))

# Perform a Full Outer Join on the two dataframes using 'Company' in df1 and 'name' in df2 as the common key column
merged_df = pd.merge(df1, df2, left_on='Company', right_on='name', how='outer')

# Check for any missing values in the merged dataframe
print("\nMissing Values in Merged DataFrame:\n", merged_df.isnull().sum())

# Display the first few rows of the merged dataframe
print(merged_df.head())

# Create new dataframes by selecting columns from the merged dataframe
df1_new = merged_df[['Company', 'Valuation ($B)', 'Date Joined', 'Country', 'City', 'Founded Year']]
df2_new = merged_df[['Industry', 'Select Inverstors', 'Financial Stage', 'Total Raised', 'Investors Count', 'Deal Terms']]

# Display the first few rows of the new dataframes
print(df1_new.head())
print(df2_new.head())

# Identify numeric and string columns in df1_new
numeric_cols_df1 = df1_new.select_dtypes(include=np.number).columns.tolist()
string_cols_df1 = df1_new.select_dtypes(include='object').columns.tolist()

# Identify numeric and string columns in df2_new
numeric_cols_df2 = df2_new.select_dtypes(include=np.number).columns.tolist()
string_cols_df2 = df2_new.select_dtypes(include='object').columns.tolist()

# For each numeric column, remove rows with bad values
for col in numeric_cols_df1:
    df1_new = df1_new[pd.to_numeric(df1_new[col], errors='coerce').notnull()]
for col in numeric_cols_df2:
    df2_new = df2_new[pd.to_numeric(df2_new[col], errors='coerce').notnull()]

# For each string column, remove rows with non-string values
for col in string_cols_df1:
    df1_new = df1_new[df1_new[col].apply(lambda x: isinstance(x, str))]
for col in string_cols_df2:
    df2_new = df2_new[df2_new[col].apply(lambda x: isinstance(x, str))]


# Replace missing values using the mean of the values in the column for numeric columns
for col in numeric_cols_df1:
    df1_new[col].fillna(df1_new[col].mean(), inplace=True)
for col in numeric_cols_df2:
    df2_new[col].fillna(df2_new[col].mean(), inplace=True)

# Replace missing values using the mode of the values in the column for string columns
for col in string_cols_df1:
    df1_new[col].fillna(df1_new[col].mode()[0], inplace=True)
for col in string_cols_df2:
    df2_new[col].fillna(df2_new[col].mode()[0], inplace=True)

# Convert 'Valuation ($B)' to numerical values
df1_new['Valuation ($B)'] = df1_new['Valuation ($B)'].replace({'\$': '', 'B': 'e9', 'M': 'e6', 'K': 'e3', '': 'e0'}, regex=True).map(pd.eval).astype(float)
# Normalize a numeric data column in df1_new
# Replace 'numeric_column' with the name of the numeric column you want to normalize
df1_new['Valuation ($B)_norm'] = df1_new['Valuation ($B)'].apply(lambda x: abs(x) / df1_new['Valuation ($B)'].abs().max())

# Print all duplicate rows in df1_new
duplicate_rows_df1 = df1_new[df1_new.duplicated()]
print("Duplicate Rows in df1_new:")
print(duplicate_rows_df1)

# Print all duplicate rows in df2_new
duplicate_rows_df2 = df2_new[df2_new.duplicated()]
print("Duplicate Rows in df2_new:")
print(duplicate_rows_df2)

# Drop duplicate rows in df1_new
df1_new.drop_duplicates(inplace=True)

# Drop duplicate rows in df2_new
df2_new.drop_duplicates(inplace=True)

import matplotlib.pyplot as plt
import seaborn as sns

# Convert Investors Count, Deal Terms to numeric
df2_new['Investors Count'] = pd.to_numeric(df2_new['Investors Count'], errors='coerce')
df2_new['Deal Terms'] = pd.to_numeric(df2_new['Deal Terms'], errors='coerce')

# 1. Bar plot of Country
plt.figure(figsize=(10, 6))
sns.countplot(x='Country', data=df1_new)
plt.title('Bar Plot of Country')
plt.xticks(rotation=90)
plt.show()

# 2. Bar plot of Industry
plt.figure(figsize=(10, 6))
sns.countplot(x='Industry', data=df2_new)
plt.title('Bar Plot of Industry')
plt.xticks(rotation=90)
plt.show()

# 3. Histogram of Investors Count
plt.figure(figsize=(10, 6))
sns.histplot(df2_new['Investors Count'], bins=30)
plt.title('Histogram of Investors Count')
plt.show()

# 4. Histogram of Deal Terms
plt.figure(figsize=(10, 6))
sns.histplot(df2_new['Deal Terms'], bins=30)
plt.title('Histogram of Deal Terms')
plt.show()

# 5. Box plot of Investors Count grouped by Industry
plt.figure(figsize=(10, 6))
sns.boxplot(x='Industry', y='Investors Count', data=df2_new)
plt.title('Box Plot of Investors Count Grouped by Industry')
plt.xticks(rotation=90)
plt.show()

# 6. Box plot of Deal Terms grouped by Industry
plt.figure(figsize=(10, 6))
sns.boxplot(x='Industry', y='Deal Terms', data=df2_new)
plt.title('Box Plot of Deal Terms Grouped by Industry')
plt.xticks(rotation=90)
plt.show()

# 7. Heatmap of correlations
plt.figure(figsize=(10, 6))
sns.heatmap(df2_new[['Investors Count', 'Deal Terms']].corr(), annot=True, cmap='coolwarm')
plt.title('Heatmap of Correlations')
plt.show()

# 8. Violin plot of Investors Count grouped by Industry
plt.figure(figsize=(10, 6))
sns.violinplot(x='Industry', y='Investors Count', data=df2_new)
plt.title('Violin Plot of Investors Count Grouped by Industry')
plt.xticks(rotation=90)
plt.show()

# 9. Violin plot of Deal Terms grouped by Industry
plt.figure(figsize=(10, 6))
sns.violinplot(x='Industry', y='Deal Terms', data=df2_new)
plt.title('Violin Plot of Deal Terms Grouped by Industry')
plt.xticks(rotation=90)
plt.show()

# 10. Scatter plot of Investors Count vs Deal Terms
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Investors Count', y='Deal Terms', data=df2_new)
plt.title('Scatter Plot of Investors Count vs Deal Terms')
plt.show()

# 11. # Scatter plot of Valuation ($B)_norm vs Founded Year
# Convert 'Founded Year' to numeric
df1_new['Founded Year'] = pd.to_numeric(df1_new['Founded Year'], errors='coerce')

# Scatter plot of Valuation ($B)_norm vs Founded Year
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Founded Year', y='Valuation ($B)_norm', data=df1_new)
plt.title('Scatter Plot of Normalized Valuation vs Founded Year')
plt.show()


# 12. Bar plot of Founded Year
plt.figure(figsize=(10, 6))
sns.countplot(x='Founded Year', data=df1_new)
plt.title('Bar Plot of Founded Year')
plt.xticks(rotation=90)
plt.show()

# 13. Box plot of Founded Year grouped by Country
plt.figure(figsize=(10, 6))
sns.boxplot(x='Country', y=pd.to_numeric(df1_new['Founded Year'], errors='coerce'), data=df1_new)
plt.title('Box Plot of Founded Year Grouped by Country')
plt.xticks(rotation=90)
plt.show()

"""קורלציה בין שווי למספר עובדים:

יש קורלציה חיובית בין שווי למספר העובדים.
עם זאת, ישנם מספר חריגים בגרף. חריגים אלה הם חברות שיש להן שווי גבוה אך מספר עובדים יחסית נמוך.

קורלציה בין שווי לתנאי העסקה:

ישנה קורלציה חיובית בין שווי לתנאי העסקה.
יש הרבה שינויים בתנאי העסקה, אפילו לחברות עם שוויים דומים.
התפלגות תנאי העסקה מוטה לימין, מה שאומר שישנן מספר חברות עם תנאי עסקה מאוד מועדפים.
מגמה במספר החברות שנוסדו לאורך הזמן:

מספר החברות שנוסדו גדל לאורך הזמן.
היו מספר התפרצויות במספר החברות שנוסדו, כמו בתחילת שנות ה-2000 ותחילת שנות ה-2010.
מספר החברות שנוסדו עשוי להיות מקורלצי לתנאים כלכליים.
התפלגות החברות במדינות שונות:

ארצות הברית יש להן את מרבית החברות, אחריהן מגיעות סין והודו.
אסטוניה, הונג קונג, ומקסיקו יש להן מספר החברות הכי נמוך.
יש טווח רחב במספר החברות במדינות שונות.
מגמה במספר החברות שנוסדו במדינות שונות לאורך הזמן:

מספר החברות שנוסדו בארצות הברית הולך וגדל לאורך הזמן, עם מספר התפרצויות בתחילת שנות ה-2000 ותחילת שנות ה-2010.
מספר החברות שנוסדו בסין גם הולך וגדל לאורך הזמן, אך בקצב יותר איטי מאשר בארצות הברית.
מספר החברות שנוסדו בהודו הולך וגדל לאורך הזמן, אך בקצב יותר איטי מאשר בארצות הברית או סין.

"""

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Fill NaN values with mean of the column
df1_new = df1_new.fillna(df1_new.mean())
# Select numeric columns
df_numeric = df1_new.select_dtypes(include=np.number)



# Standardize the features
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_numeric)

# Compute the sum of squared distances for different numbers of clusters
ssd = []
K = range(1, 15)
for k in K:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(df_scaled)
    ssd.append(kmeans.inertia_)

# Plot the elbow curve
plt.plot(K, ssd, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum of Squared Distances')
plt.title('Elbow Method For Optimal k')
plt.show()

"""מסקנה:

המספר האופטימלי של אשכולות למאגר נתונים זה הוא 4.
נקודות הנתונים מתאגדות בצורה טובה יותר לארבעה קבוצות מאשר ליותר מארבעה קבוצות.

"""

from sklearn.linear_model import LinearRegression

# Convert 'Founded Year' to numeric
df1_new['Founded Year'] = pd.to_numeric(df1_new['Founded Year'], errors='coerce')

# Select two columns, replace 'Valuation ($B)_norm' and 'Founded Year' with your selected columns
x = df1_new[['Valuation ($B)_norm']].dropna()
y = df1_new.loc[x.index, 'Founded Year']

# Fit a linear regression model
model = LinearRegression()
model.fit(x, y)

# Plot the scatter plot and regression line
plt.scatter(x, y, color='blue')
plt.plot(x, model.predict(x), color='red')
plt.title('Regression Line')
plt.xlabel('Valuation ($B)_norm')
plt.ylabel('Founded Year')
plt.show()

"""יש קורלציה חיובית בין השווי לשנת ההקמה. משמעות הדבר היא ש, בממוצע, חברות שנוסדו בתקופה האחרונה משוויות בסכום גבוה יותר.
קו הרגרסיה הליניארית אינו מתאים באופן מושלם לנתונים. משמעות הדבר היא שישנן השתנותים בנתונים שלא ניתן להסביר על ידי הקשר הליניארי.
מניח הקו של הרגרסיה הליניארית הוא חיובי. משמעות הדבר היא שעבור כל מיליארד דולר נוסף בשווי, שנת ההקמה מתמרה בקצב של כ-1.5 שנים.
"""

# Compute the correlation matrix
corr = df1_new.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.title('Correlation Matrix Heatmap')
plt.show()

"""ישנן כמה זוגות משתנים שיש להם קורלציה חזקה, כמו שווי לשווי (נורמה של SB).
יש גם כמה זוגות משתנים שיש להם קורלציה חלשה או שלילית, כמו שווי לשנת ההקמה.
חשוב לזכור שקורלציה אינה שווה לגרימה. רק מכיוון ששני משתנים מתקשרים זה לא אומר שאחד גורם לשני.

"""