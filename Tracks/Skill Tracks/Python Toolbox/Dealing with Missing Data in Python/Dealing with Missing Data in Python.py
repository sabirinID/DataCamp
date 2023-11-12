## Steps for treating missing values

# B, D, C, A

## Null value operations

try:
  # Print the sum of two None's
  print("Add operation output of 'None': ", None + None)

except TypeError:
  # Print if error
  print("'None' does not support Arithmetic Operations!!")

try:
  # Print the sum of two np.nan's
  print("Add operation output of 'np.nan': ", np.nan + np.nan)

except TypeError:
  # Print if error
  print("'np.nan' does not support Arithmetic Operations!!")

try:
  # Print the output of logical OR of two None's
  print("OR operation output of 'None': ", None or None)

except TypeError:
  # Print if error
  print("'None' does not support Logical Operations!!")

try:
  # Print the output of logical OR of two np.nan's
  print("OR operation output of 'np.nan': ", np.nan or np.nan)

except TypeError:
  # Print if error
  print("'np.nan' does not support Logical Operations!!")

# None is of type NoneType while np.nan is of float. This allows np.nan to have both arithmetic and logical operations.

## Finding Null values

try:
  # Print the comparison of two 'None's
  print("'None' comparison output: ", None == None)

except TypeError:
  # Print if error
  print("'None' does not support this operation!!")

try:
  # Print the comparison of two 'np.nan's
  print("'np.nan' comparison output: ", np.nan == np.nan)

except TypeError:
  # Print if error  
  print("'np.nan' does not support this operation!!")

try:
  # Check if 'None' is 'NaN'
  print("Is 'None' same as nan? ", np.isnan(None))

except TypeError:
  # Print if error
  print("Function 'np.isnan()' does not support this Type!!")

try:
  # Check if 'np.nan' is 'NaN'
  print("Is 'np.nan' same as nan? ", np.isnan(np.nan))

except TypeError:
  # Print if error
  print("Function 'np.isnan()' does not support this Type!!")

# To check for None, use the comparison operator == and to check for 'NaN' use np.isnan(). Fortunately, pandas intrinsically converts None to NaN values, so that you don't need to worry!

## Detecting missing values

# Read the dataset 'college.csv'
college = pd.read_csv('college.csv')
print(college.head())

# Print the info of college
print(college.info())

# Store unique values of 'csat' column to 'csat_unique'
csat_unique = college.csat.unique()

# Print the sorted values of csat_unique
print(np.sort(csat_unique))

## Replacing missing values

# Read the dataset 'college.csv' with na_values set to '.'
college = pd.read_csv('college.csv', na_values='.')
print(college.head())

# Print the info of college
print(college.info())

## Replacing hidden missing values

# Print the description of the data
print(diabetes.describe())

# Store all rows of column 'BMI' which are equal to 0 
print(diabetes.BMI[diabetes.BMI == 0])

# Set the 0 values of column 'BMI' to np.nan
diabetes.BMI[diabetes.BMI == 0] = np.nan

# Print the 'NaN' values in the column BMI
print(diabetes.BMI[diabetes.BMI.isna()])

## Analyzing missingness percentage

# Load the air-quality.csv dataset
airquality = pd.read_csv('air-quality.csv', parse_dates=['Date'], index_col='Date')

# Create a nullity DataFrame airquality_nullity
airquality_nullity = airquality.isnull()
print(airquality_nullity.head())

# Calculate the number of missing values by summing airquality_nullity
missing_values_sum = airquality_nullity.sum()
print('Total Missing Values:\n', missing_values_sum)

# Calculate percentage of missing values
missing_values_percent = airquality_nullity.mean() * 100
print('Percentage of Missing Values:\n', missing_values_percent)

## Visualize missingness

# Import missingno as msno
import missingno as msno

# Plot amount of missingness
msno.bar(airquality)

# Display bar chart of missing values
display("/usr/local/share/datasets/bar_chart.png")

# Import missingno as msno
import missingno as msno

# Plot nullity matrix of airquality
msno.matrix(airquality)

# Display nullity matrix
display("/usr/local/share/datasets/matrix.png")

# Import missingno as msno
import missingno as msno

# Plot nullity matrix of airquality with frequency 'M'
msno.matrix(airquality, freq='M')

# Display nullity matrix
display("/usr/local/share/datasets/matrix_frequency.png")

# Import missingno as msno
import missingno as msno

# Plot the sliced nullity matrix of airquality with frequency 'M'
msno.matrix(airquality.loc['May-1976':'Jul-1976'], freq='M')

# Display nullity matrix
display("/usr/local/share/datasets/matrix_sliced.png")

## Guess the missingness type

# Import missingno as msno
import missingno as msno

# Visualize the missingness summary
msno.matrix(diabetes)

# Display nullity matrix
display("/usr/local/share/datasets/matrix_diabetes.png")

# 1 and 3
# You're right! While Serum_Insulin is definitely not missing at random, Glucose is most likely missing completely at random.

## Deduce MNAR

# Import missingno as msno
import missingno as msno

# Sort diabetes dataframe on 'Serum_Insulin'
sorted_values = diabetes.sort_values('Serum_Insulin')

# Visualize the missingness summary of sorted
msno.matrix(sorted_values)

# Display nullity matrix
display("/usr/local/share/datasets/matrix_sorted.png")

## Finding correlations in your data

# Import missingno
import missingno as msno

# Plot missingness heatmap of diabetes
msno.heatmap(diabetes)

# Show plot
plt.show()

# Import missingno
import missingno as msno

# Plot missingness heatmap of diabetes
msno.heatmap(diabetes)

# Plot missingness dendrogram of diabetes
msno.dendrogram(diabetes)

# Show plot
plt.show()

## Identify the missingness type

# MNAR
# You're right. Skin_Fold is not missing at random

## Fill dummy values

def fill_dummy_values(df):
  df_dummy = df.copy(deep=True)
  for col_name in df_dummy:
    col = df_dummy[col_name]
    # Calculate column range
    col_range = col.max() - col.min()
  return df_dummy

def fill_dummy_values(df):
  df_dummy = df.copy(deep=True)
  for col_name in df_dummy:
    col = df_dummy[col_name]
    # Calculate column range
    col_range = col.max() - col.min()
    # Calculate the number of missing values in column 
    col_null = col.isnull()
    num_nulls = col_null.sum()
  return df_dummy

def fill_dummy_values(df):
  df_dummy = df.copy(deep=True)
  for col_name in df_dummy:
    col = df_dummy[col_name]
    # Calculate column range
    col_range = col.max() - col.min()
    # Calculate number of missing values in column 
    col_null = col.isnull()    
    num_nulls = col_null.sum()
    # Create random values of size num_nulls
    dummy_values = (rand(num_nulls) - 2)
  return df_dummy

def fill_dummy_values(df, scaling_factor=0.075):
  df_dummy = df.copy(deep=True)
  for col_name in df_dummy:
    col = df_dummy[col_name]
    # Calculate column range
    col_range = col.max() - col.min()
    # Calculate number of missing values in column 
    col_null = col.isnull()
    num_nulls = col_null.sum()
    # Scale the random values to scaling_factor times col_range
    dummy_values = (rand(num_nulls) - 2) * scaling_factor * col_range + col.min()
    col[col_null] = dummy_values
  return df_dummy

## Generate scatter plot with missingness

# Use OR operation to combine Skin_Fold and BMI nullity
nullity = diabetes['Skin_Fold'].isnull() | diabetes['BMI'].isnull()

# Fill dummy values in diabetes_dummy
diabetes_dummy = fill_dummy_values(diabetes)

# Create a scatter plot of BMI versus Skin_Fold
diabetes_dummy.plot(x='Skin_Fold', y='BMI', kind='scatter', alpha=0.5,
                    # Set color to nullity of BMI and Skin_Fold
                    c=nullity, 
                    cmap='rainbow')

plt.show()

## Delete MCAR

# Visualize the missingness of diabetes prior to dropping missing values
msno.matrix(diabetes)

# Display nullity matrix
display("/usr/local/share/datasets/matrix_diabetes.png")

# Visualize the missingness of diabetes prior to dropping missing values
msno.matrix(diabetes)

# Print the number of missing values in Glucose
print(diabetes.Glucose.isna().sum())

# Visualize the missingness of diabetes prior to dropping missing values
msno.matrix(diabetes)

# Print the number of missing values in Glucose
print(diabetes['Glucose'].isnull().sum())

# Drop rows where 'Glucose' has a missing value
diabetes.dropna(subset=['Glucose'], how='any', inplace=True)

# Visualize the missingness of diabetes after dropping missing values
msno.matrix(diabetes)

display("/usr/local/share/datasets/glucose_dropped.png")

## Will you delete?

# Visualize the missingness in the data
msno.matrix(diabetes)

# Display nullity matrix
display("/usr/local/share/datasets/matrix_diabetes.png")

# Visualize the missingness in the data
msno.matrix(diabetes)

# Visualize the correlation of missingness between variables
msno.heatmap(diabetes)

# Show heatmap
plt.show()

# ... (MCAR) ... delete it.

# Visualize the missingness in the data
msno.matrix(diabetes)

# Visualize the correlation of missingness between variables
msno.heatmap(diabetes)

# Show heatmap
plt.show()

# Drop rows where 'BMI' has a missing value
diabetes.dropna(subset=['BMI'], how='any', inplace=True)

## Mean & median imputation

# Make a copy of diabetes
diabetes_mean = diabetes.copy(deep=True)

# Create mean imputer object
mean_imputer = SimpleImputer(strategy='mean')

# Impute mean values in the DataFrame diabetes_mean
diabetes_mean.iloc[:, :] = mean_imputer.fit_transform(diabetes_mean)

# Make a copy of diabetes
diabetes_median = diabetes.copy(deep=True)

# Create median imputer object
median_imputer = SimpleImputer(strategy='median')

# Impute median values in the DataFrame diabetes_median
diabetes_median.iloc[:, :] = median_imputer.fit_transform(diabetes_median)

## Mode and constant imputation

# Make a copy of diabetes
diabetes_mode = diabetes.copy(deep=True)

# Create mode imputer object
mode_imputer = SimpleImputer(strategy="most_frequent")

# Impute using most frequent value in the DataFrame mode_imputer
diabetes_mode.iloc[:, :] = mode_imputer.fit_transform(diabetes_mode)

# Make a copy of diabetes
diabetes_constant = diabetes.copy(deep=True)

# Create constant imputer object
constant_imputer = SimpleImputer(strategy="constant", fill_value=0)

# Impute missing values to 0 in diabetes_constant
diabetes_constant.iloc[:, :] = constant_imputer.fit_transform(diabetes_constant)

## Visualize imputations

# Set nrows and ncols to 2
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
nullity = diabetes.Serum_Insulin.isnull()+diabetes.Glucose.isnull()

# Create a dictionary of imputations
imputations = {'Mean Imputation': diabetes_mean, 'Median Imputation': diabetes_median, 
               'Most Frequent Imputation': diabetes_mode, 'Constant Imputation': diabetes_constant}

# Loop over flattened axes and imputations
for ax, df_key in zip(axes.flatten(), imputations):
    # Select and also set the title for a DataFrame
    imputations[df_key].plot(x='Serum_Insulin', y='Glucose', kind='scatter', 
                          alpha=0.5, c=nullity, cmap='rainbow', ax=ax, 
                          colorbar=False, title=df_key)
plt.show()

## Filling missing time-series data

# Print prior to imputing missing values
print(airquality[30:40])

# Fill NaNs using forward fill
airquality.fillna(method='ffill', inplace=True)

# Print after imputing missing values
print(airquality[30:40])

# Print prior to imputing missing values
print(airquality[30:40])

# Fill NaNs using backward fill
airquality.fillna(method='bfill', inplace=True)

# Print after imputing missing values
print(airquality[30:40])

## Impute with interpolate method

# Print prior to interpolation
print(airquality[30:40])

# Interpolate the NaNs linearly
airquality.interpolate(method='linear', inplace=True)

# Print after interpolation
print(airquality[30:40])

# Print prior to interpolation
print(airquality[30:40])

# Interpolate the NaNs quadratically
airquality.interpolate(method='quadratic', inplace=True)

# Print after interpolation
print(airquality[30:40])

# Print prior to interpolation
print(airquality[30:40])

# Interpolate the NaNs with nearest value
airquality.interpolate(method='nearest', inplace=True)

# Print after interpolation
print(airquality[30:40])

## Visualize forward fill imputation

# Impute airquality DataFrame with ffill method
ffill_imputed = airquality.fillna(method='ffill')

# Plot the imputed DataFrame ffill_imputed in red dotted style 
ffill_imputed['Ozone'].plot(color='red', marker='o', linestyle='dotted', figsize=(30, 5))

plt.show()

# Impute airquality DataFrame with ffill method
ffill_imputed = airquality.fillna(method='ffill')

# Plot the imputed DataFrame ffill_imputed in red dotted style 
ffill_imputed['Ozone'].plot(color='red', marker='o', linestyle='dotted', figsize=(30, 5))

# Plot the airquality DataFrame with title
airquality['Ozone'].plot(title='Ozone', marker='o', figsize=(30, 5))

plt.show()

## Visualize backward fill imputation

# Impute airquality DataFrame with bfill method
bfill_imputed = airquality.fillna(method='bfill')

# Plot the imputed DataFrame bfill_imputed in red dotted style 
bfill_imputed['Ozone'].plot(color='red', marker='o', linestyle='dotted', figsize=(30, 5))

plt.show()

# Impute airquality DataFrame with bfill method
bfill_imputed = airquality.fillna(method='bfill')

# Plot the imputed DataFrame bfill_imputed in red dotted style 
bfill_imputed['Ozone'].plot(color='red', marker='o', linestyle='dotted', figsize=(30, 5))

# Plot the airquality DataFrame with title
airquality['Ozone'].plot(title='Ozone', marker='o', figsize=(30, 5))

plt.show()

## Plot interpolations

# Set nrows to 3 and ncols to 1
fig, axes = plt.subplots(3, 1, figsize=(30, 20))

# Create a dictionary of interpolated DataFrames for looping 
interpolations = {'Linear Interpolation': linear_interp, 'Quadratic Interpolation': quadratic_interp, 
                  'Nearest Interpolation': nearest_interp}

# Loop over axes and interpolations
for ax, df_key in zip(axes, interpolations):
  # Select and also set the title for a DataFrame
  interpolations[df_key].Ozone.plot(color='red', marker='o', 
                                 linestyle='dotted', ax=ax)
  airquality.Ozone.plot(title=df_key + ' - Ozone', marker='o', ax=ax)
  
plt.show()

## KNN imputation

# Import KNN from fancyimpute
from fancyimpute import KNN

# Copy diabetes to diabetes_knn_imputed
diabetes_knn_imputed = diabetes.copy(deep=True)

# Initialize KNN
knn_imputer = KNN()

# Impute using fit_tranform on diabetes_knn_imputed
diabetes_knn_imputed.iloc[:, :] = knn_imputer.fit_transform(diabetes_knn_imputed)

## MICE imputation

# Import IterativeImputer from fancyimpute
from fancyimpute import IterativeImputer

# Copy diabetes to diabetes_mice_imputed
diabetes_mice_imputed = diabetes.copy(deep=True)

# Initialize IterativeImputer
mice_imputer = IterativeImputer()

# Impute using fit_tranform on diabetes_mice_imputed
diabetes_mice_imputed.iloc[:, :] = mice_imputer.fit_transform(diabetes_mice_imputed)

## Ordinal encoding of a categorical column

# Set col_name to 'ambience'
col_name = 'ambience'
# Create Ordinal encoder
ambience_ord_enc = OrdinalEncoder()

# Select non-null values of ambience column in users
ambience = users[col_name]
ambience_not_null = ambience[ambience.notnull()]

# Reshape ambience_not_null to shape (-1, 1)
reshaped_vals = ambience_not_null.values.reshape(-1, 1)

# Select the non-null values for the column col_name in users and store the encoded values
encoded_vals = ambience_ord_enc.fit_transform(reshaped_vals)
users.loc[ambience.notnull(), col_name] = np.squeeze(encoded_vals)

## Ordinal encoding of a DataFrame

# Create an empty dictionary ordinal_enc_dict
ordinal_enc_dict = {}

for col_name in users:
    # Create Ordinal encoder for col
    ordinal_enc_dict[col_name] = OrdinalEncoder()
    col = users[col_name]
    
    # Select non-null values of col
    col_not_null = col[col.notnull()]
    reshaped_vals = col_not_null.values.reshape(-1, 1)
    encoded_vals = ordinal_enc_dict[col_name].fit_transform(reshaped_vals)
    
    # Select the non-null values for the column col_name in users and store the encoded values
    users.loc[col.notnull(), col_name] = np.squeeze(encoded_vals)

## KNN imputation of categorical values

# Create KNN imputer
KNN_imputer = KNN()

# Impute 'users' DataFrame. It is rounded to get integer values
users_KNN_imputed.iloc[:, :] = np.round(KNN_imputer.fit_transform(users))

# Loop over the column names in 'users'
for col_name in users:
    
    # Reshape the column data
    reshaped = users_KNN_imputed[col_name].values.reshape(-1, 1)
    
    # Select the column's Encoder and perform inverse transform on 'reshaped'
    users_KNN_imputed[col_name] = ordinal_enc_dict[col_name].inverse_transform(reshaped)

## Analyze the summary of linear model

# Add constant to X and set X & y values to fit linear model
X = sm.add_constant(diabetes_cc.iloc[:, :-1])
y = diabetes_cc['Class']
lm = sm.OLS(y, X).fit()

# Print summary of lm
print('\nSummary: ', lm.summary())

# Print Adjusted R squared score of lm
print('\nAdjusted R-squared score: ', lm.rsquared_adj)

# Print the params of lm
print('\nCoefficcients:\n', lm.params)

## Comparing and choosing the best adjusted R-squared

# Store the Adj. R-squared scores of the linear models
rsquared_df = pd.DataFrame({'Mean Imputation': lm_mean.rsquared_adj, 
                          'KNN Imputation': lm_KNN.rsquared_adj, 
                          'MICE Imputation': lm_MICE.rsquared_adj}, 
                         index=['Adj. R-squared'])

# Neatly print the Adj. R-squared scores in the console
print(rsquared_df)

rsquared_df = {'Mean Imputation': lm_mean.rsquared_adj, 
             'KNN Imputation': lm_KNN.rsquared_adj, 
             'MICE Imputation': lm_MICE.rsquared_adj}

# Find the maximum Adjusted R-squared score
best_imputation = max(rsquared_df, key=rsquared_df.get)

print("The best imputation technique is: ", best_imputation)

## Comparing density plots

# Plot graphs of imputed DataFrames and the complete case
diabetes_cc['Skin_Fold'].plot(kind='kde', c='red', linewidth=3)
diabetes_mean_imputed['Skin_Fold'].plot(kind='kde')
diabetes_knn_imputed['Skin_Fold'].plot(kind='kde')
diabetes_mice_imputed['Skin_Fold'].plot(kind='kde')

# Create labels for the four DataFrames
labels = ['Baseline (Complete Case)', 'Mean Imputation', 'KNN Imputation', 'MICE Imputation']
plt.legend(labels)

# Set the x-label as Skin Fold
plt.xlabel('Skin Fold')

plt.show()

## Conclusion