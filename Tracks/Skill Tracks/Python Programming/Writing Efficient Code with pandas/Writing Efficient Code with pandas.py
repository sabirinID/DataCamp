## What does time.time() measure?
# What does the time.time() function exactly measure?

# The time since a predefined date and time set by the operating system.
# Congratulations! The time.time() function shows the time (in seconds) since a pre-defined time, which in Unix-based systems is January 1, 1970.

## Measuring time I
"""
In the lecture slides, you saw how the time.time() function can be loaded and used to assess the time required to perform a basic mathematical operation.

Now, you will use the same strategy to assess two different methods for solving a similar problem: calculate the sum of squares of all the positive integers from 1 to 1 million (1,000,000).

Similar to what you saw in the video, you will compare two methods; one that uses brute force and one more mathematically sophisticated.

In the function formula, we use the standard formula

N * (N + 1)(2N +1) / 6

where N=1,000,000.

In the function brute_force we loop over each number from 1 to 1 million and add it to the result.
"""

# Calculate the result of the problem using formula() and print the time required
N = 1000000
fm_start_time = time.time()
first_method = formula(N)
print("Time using formula: {} sec".format(time.time() - fm_start_time))

# Calculate the result of the problem using brute_force() and print the time required
sm_start_time = time.time()
second_method = brute_force(N)
print("Time using the brute force: {} sec".format(time.time() - sm_start_time))

## Measuring time II
"""
As we discussed in the lectures, in the majority of cases, a list comprehension is faster than a for loop.

In this demonstration, you will see a case where a list comprehension and a for loop have so small difference in efficiency that choosing either method will perform this simple task instantly.

In the list words, there are random words downloaded from the Internet. We are interested to create another list called listlet in which we only keep the words that start with the letter b.

In case you are not familiar with dealing with strings in Python, each string has the .startswith() attribute, which returns a True/False statement whether the string starts with a specific letter/phrase or not.
"""

# Store the time before the execution
start_time = time.time()

# Execute the operation
letlist = [wrd for wrd in words if wrd.startswith('b')]

# Store and print the difference between the start and the current time
total_time_lc = time.time() - start_time
print('Time using list comprehension: {} sec'.format(total_time_lc))

# Store the time before the execution
start_time = time.time()

# Execute the operation
letlist = []
for wrd in words:
    if wrd.startswith('b'):
        letlist.append(wrd)
        
# Print the difference between the start and the current time
total_time_fl = time.time() - start_time
print('Time using for loop: {} sec'.format(total_time_fl))

## Row selection: loc[] vs iloc[]
"""
A big part of working with DataFrames is to locate specific entries in the dataset. You can locate rows in two ways:

By a specific value of a column (feature).
By the index of the rows (index). In this exercise, we will focus on the second way.

If you have previous experience with pandas, you should be familiar with the .loc and .iloc indexers, which stands for 'location' and 'index location' respectively. In most cases, the indices will be the same as the position of each row in the Dataframe (e.g. the row with index 13 will be the 14th entry).

While we can use both functions to perform the same task, we are interested in which is the most efficient in terms of speed.
"""

# Define the range of rows to select: row_nums
row_nums = range(0, 1000)

# Select the rows using .loc[] and row_nums and record the time before and after
loc_start_time = time.time()
rows = poker_hands.loc[row_nums]
loc_end_time = time.time()

# Print the time it took to select the rows using .loc
print("Time using .loc[]: {} sec".format(loc_end_time - loc_start_time))

# Select the rows using .iloc[] and row_nums and record the time before and after
iloc_start_time = time.time()
rows = poker_hands.iloc[row_nums]
iloc_end_time = time.time()

# Print the time it took to select the rows using .iloc
print("Time using .iloc[]: {} sec".format(iloc_end_time - iloc_start_time))

# If you need to select specific rows of a DataFrame, which function is more efficient, it terms of speed?
# .iloc[]
# Congratulations! There is an explanation for that, .iloc() takes advantages of the sorted position of each rows, simplifying the computations needed.

## Column selection: .iloc[] vs by name
"""
In the previous exercise, you saw how the .loc[] and .iloc[] functions can be used to locate specific rows of a DataFrame (based on the index). Turns out, the .iloc[] function performs a lot faster (~ 2 times) for this task!

Another important task is to find the faster function to select the targeted features (columns) of a DataFrame. In this exercise, we will compare the following:

using the index locator .iloc()
using the names of the columns While we can use both functions to perform the same task, we are interested in which is the most efficient in terms of speed.

In this exercise, you will continue working with the poker data which is stored in poker_hands. Take a second to examine the structure of this DataFrame by calling poker_hands.head() in the console!
"""

# Use .iloc to select the first, fourth, fifth, seventh and eighth column and record the times before and after
iloc_start_time = time.time()
cols = poker_hands.iloc[:,[0,3,4,6,7]]
iloc_end_time = time.time()

# Print the time it took
print("Time using .iloc[] : {} sec".format(iloc_end_time - iloc_start_time))

# Use simple column selection to select the first, fourth, fifth, seventh and eighth column and record the times before and after
names_start_time = time.time()
cols = poker_hands[['S1', 'S2', 'R2', 'R3', 'S4']]
names_end_time = time.time()

# Print the time it took
print("Time using selection by name : {} sec".format(names_end_time - names_start_time))

# If you need to select a specific column (or columns) of a DataFrame, which function is more efficient, it terms of speed?
# Simple columns selection
# Congratulations! There is an explanation for that: with .iloc we need to specify both the rows and columns needed, and this takes more time! It might seem insignificant here, but it will make a huge difference in bigger datasets!

## Random row selection
"""
In this exercise, you will compare the two methods described for selecting random rows (entries) with replacement in a pandas DataFrame:

The built-in pandas function .random()
The NumPy random integer number generator np.random.randint()

Generally, in the fields of statistics and machine learning, when we need to train an algorithm, we train the algorithm on the 75% of the available data and then test the performance on the remaining 25% of the data.

For this exercise, we will randomly sample the 75% percent of all the played poker hands available, using each of the above methods, and check which method is more efficient in terms of speed.
"""

# Extract number of rows in dataset
N=poker_hands.shape[0]

# Select and time the selection of the 75% of the dataset's rows
rand_start_time = time.time()
poker_hands.iloc[np.random.randint(low=0, high=N, size=int(0.75 * N))]
print("Time using Numpy: {} sec".format(time.time() - rand_start_time))

# Select and time the selection of the 75% of the dataset's rows using sample()
samp_start_time = time.time()
poker_hands.sample(int(0.75 * N), axis=0, replace = True)
print("Time using .sample: {} sec".format(time.time() - samp_start_time))

# Between np.random.randint() and .sample(), which one is faster when selecting random rows from a pandas DataFrame?
# .sample()
# Congratulations! You found the most efficient way to sample random rows from a pandas DataFrame, and it's always the built-in function.

## Random column selection
"""
In the previous exercise, we examined two ways to select random rows from a pandas DataFrame. We can use the same functions to randomly select columns in a pandas DataFrame.

To randomly select 4 columns out of the poker dataset, you will use the following two functions:

The built-in pandas function .sample()
The NumPy random integer number generator np.random.randint()
"""

# Extract number of columns in dataset
D = poker_hands.shape[1]

# Select and time the selection of 4 of the dataset's columns using NumPy
np_start_time = time.time()
poker_hands.iloc[:, np.random.randint(low=0, high=D, size=4)]
print("Time using NumPy's random.randint(): {} sec".format(time.time() - np_start_time))

# Select and time the selection of 4 of the dataset's columns using pandas
pd_start_time = time.time()
poker_hands.sample(4, axis=1)
print("Time using pandas's .sample(): {} sec".format(time.time() - pd_start_time))

# Between np.random.randint() and .sample(), which one is faster when selecting random columns from a pandas DataFrame?
# .sample()
# Congratulations! You found the most efficient way to sample random columns from a pandas DataFrame.

## Replacing scalar values I
"""
In this exercise, we will replace a list of values in our dataset by using the .replace() method with another list of desired values.

We will apply the functions in the poker_hands DataFrame. Remember that in the poker_hands DataFrame, each row of columns R1 to R5 represents the rank of each card from a player's poker hand spanning from 1 (Ace) to 13 (King). The Class feature classifies each hand as a category, and the Explanation feature briefly explains each hand.

The poker_hands DataFrame is already loaded for you, and you can explore the features Class and Explanation.

Remember you can always explore the dataset and see how it changes in the IPython Shell, and refer to the slides in the Slides tab.
"""

# Replace Class 1 to -2 
poker_hands['Class'].replace(1, -2, inplace=True)
# Class 2 to -3
poker_hands['Class'].replace(2, -3, inplace=True)

print(poker_hands[['Class', 'Explanation']])

## Replace scalar values II
"""
As discussed in the video, in a pandas DataFrame, it is possible to replace values in a very intuitive way: we locate the position (row and column) in the Dataframe and assign in the new value you want to replace with. In a more pandas-ian way, the .replace() function is available that performs the same task.

You will be using the names DataFrame which includes, among others, the most popular names in the US by year, gender and ethnicity.

Your task is to replace all the babies that are classified as FEMALE to GIRL using the following methods:

intuitive scalar replacement
using the .replace() function
"""

start_time = time.time()

# Replace all the entries that has 'FEMALE' as a gender with 'GIRL'
names['Gender'].loc[names['Gender'] == 'FEMALE'] = 'GIRL'

print("Time using .loc[]: {} sec".format(time.time() - start_time))

start_time = time.time()

# Replace all the entries that has 'FEMALE' as a gender with 'GIRL'
names['Gender'].replace('FEMALE', 'GIRL', inplace=True)

print("Time using .replace(): {} sec".format(time.time() - start_time))

# Which of the two methods presented in the previous exercises is the most efficient when replacing a scalar value?
# Using .replace() was faster.
# Congratulations! You found the most efficient way to replace a scalar value on a pandas DataFrame! Exciting, isn't it?

## Replace multiple values I
"""
In this exercise, you will apply the .replace() function for the task of replacing multiple values with one or more values. You will again use the names dataset which contains, among others, the most popular names in the US by year, gender and Ethnicity.

Thus you want to replace all ethnicities classified as black or white non-hispanics to non-hispanic. Remember, the ethnicities are stated in the dataset as follows: 
['BLACK NON HISP', 'BLACK NON HISPANIC', 'WHITE NON HISP' , 'WHITE NON HISPANIC'] 
and should be replaced to 'NON HISPANIC'
"""

start_time = time.time()

# Replace all non-Hispanic ethnicities with 'NON HISPANIC'
names['Ethnicity'].loc[(names["Ethnicity"] == 'BLACK NON HISPANIC') | 
                      (names["Ethnicity"] == 'BLACK NON HISP') | 
                      (names["Ethnicity"] == 'WHITE NON HISPANIC') | 
                      (names["Ethnicity"] == 'WHITE NON HISP')] = 'NON HISPANIC'

print("Time using .loc[]: sec".format(time.time() - start_time))

start_time = time.time()

# Replace all non-Hispanic ethnicities with 'NON HISPANIC'
names['Ethnicity'].replace(['BLACK NON HISPANIC', 'BLACK NON HISP', 'WHITE NON HISPANIC', 'WHITE NON HISP'], 'NON HISPANIC', inplace=True)

print("Time using .replace(): {} sec".format(time.time() - start_time))

## Replace multiple values II
"""
As discussed in the video, instead of using the .replace() function multiple times to replace multiple values, you can use lists to map the elements you want to replace one to one with those you want to replace them with.

As you have seen in our popular names dataset, there are two names for the same ethnicity. We want to standardize the naming of each ethnicity by replacing

'ASIAN AND PACI' to 'ASIAN AND PACIFIC ISLANDER'
'BLACK NON HISP' to 'BLACK NON HISPANIC'
'WHITE NON HISP' to 'WHITE NON HISPANIC'

In the DataFrame names, you are going to replace all the values on the left by the values on the right.
"""

start_time = time.time()

# Replace ethnicities as instructed
names['Ethnicity'].replace(['BLACK NON HISP','WHITE NON HISP', 'ASIAN AND PACI'], ['BLACK NON HISPANIC','WHITE NON HISPANIC','ASIAN AND PACIFIC ISLANDER'], inplace=True)

print("Time using .replace(): {} sec".format(time.time() - start_time))

## Replace single values I
"""
In this exercise, we will apply the following replacing technique of replacing multiple values using dictionaries on a different dataset.

We will apply the functions in the data DataFrame. Each row represents the rank of 5 cards from a playing card deck, spanning from 1 (Ace) to 13 (King) (features R1, R2, R3, R4, R5). The feature 'Class' classifies each row to a category (from 0 to 9) and the feature 'Explanation' gives a brief explanation of what each class represents.

The purpose of this exercise is to categorize the two types of flush in the game ('Royal flush' and 'Straight flush') under the 'Flush' name.
"""

# Replace Royal flush or Straight flush to Flush
poker_hands.replace({'Royal flush':'Flush', 'Straight flush':'Flush'}, inplace=True)
print(poker_hands['Explanation'].head())

## Replace single values II
"""
For this exercise, we will be using the names DataFrame. In this dataset, the column 'Rank' shows the ranking of each name by year. For this exercise, you will use dictionaries to replace the first ranked name of every year as 'FIRST', the second name as 'SECOND' and the third name as 'THIRD'.

You will use dictionaries to replace one single value per key.

You can already see the first 5 names of the data, which correspond to the 5 most popular names for all the females belonging to the 'ASIAN AND PACIFIC ISLANDER' ethnicity in 2011.
"""    

# Replace the number rank by a string
names['Rank'].replace({1:'FIRST', 2:'SECOND', 3:'THIRD'}, inplace=True)
print(names.head())

## Replace multiple values III
"""
As you saw in the video, you can use dictionaries to replace multiple values with just one value, even from multiple columns. To show the usefulness of replacing with dictionaries, you will use the names dataset one more time.

In this dataset, the column 'Rank' shows which rank each name reached every year. You will change the rank of the first three ranked names of every year to 'MEDAL' and those from 4th and 5th place to 'ALMOST MEDAL'.

You can already see the first 5 names of the data, which correspond to the 5 most popular names for all the females belonging to the 'ASIAN AND PACIFIC ISLANDER' ethnicity in 2011.
"""    

# Replace the rank of the first three ranked names to 'MEDAL'
names.replace({'Rank': {1:'MEDAL', 2:'MEDAL', 3:'MEDAL'}}, inplace=True)

# Replace the rank of the 4th and 5th ranked names to 'ALMOST MEDAL'
names.replace({'Rank': {4:'ALMOST MEDAL', 5:'ALMOST MEDAL'}}, inplace=True)
print(names.head())

## Most efficient method for scalar replacement
# If you want to replace a scalar value with another scalar value, which technique is the most efficient??

# Replace using dictionaries.
# Congratulations! Looking through a list requires a pass in every element of the list, while looking at a dictionary directs instantly to the key that matches the entry.

## Create a generator for a pandas DataFrame
"""
As you've seen in the video, you can easily create a generator out of a pandas DataFrame. Each time you iterate through it, it will yield two elements:

the index of the respective row
a pandas Series with all the elements of that row

You are going to create a generator over the poker dataset, imported as poker_hands. Then, you will print all the elements of the 2nd row, using the generator.

Remember you can always explore the dataset and see how it changes in the IPython Shell, and refer to the slides in the Slides tab.
"""

# Create a generator over the rows
generator = poker_hands.iterrows()

# Access the elements of the 2nd row
first_element = next(generator)
second_element = next(generator)
print(first_element, '\n' , second_element)

## The iterrows() function for looping
"""
You just saw how to create a generator out of a pandas DataFrame. You will now use this generator and see how to take advantage of that method of looping through a pandas DataFrame, still using the poker_hands dataset.

Specifically, we want the sum of the ranks of all the cards, if the index of the hand is an odd number. The ranks of the cards are located in the odd columns of the DataFrame.
"""

data_generator = poker_hands.iterrows()

for index, values in data_generator:
  	# Check if index is odd
    if index % 2 != 0:
      	# Sum the ranks of all the cards
        hand_sum = sum([values[1], values[3], values[5], values[7], values[9]])

## .apply() function in every cell
"""
As you saw in the lesson, you can use .apply() to map a function to every cell of the DataFrame, regardless the column or the row.

You're going to try it out on the poker_hands dataset. You will use .apply() to square every cell of the DataFrame. The native Python way to square a number n is n**2.
"""

# Define the lambda transformation
get_square = lambda x: x**2

# Apply the transformation
data_sum = poker_hands.apply(get_square)
print(data_sum.head())

## .apply() for rows iteration
"""
.apply() is a very useful to iterate through the rows of a DataFrame and apply a specific function.

You will work on a subset of the poker_hands dataset, which includes only the rank of all the five cards of each hand in each row (this subset is generated for you in the script). You're going to get the variance of every hand for all ranks, and every rank for all hands.
"""

# Define the lambda transformation
get_variance = lambda x: np.var(x)

# Apply the transformation
data_tr = poker_hands[['R1', 'R2', 'R3', 'R4', 'R5']].apply(get_variance, axis=1)
print(data_tr.head())

get_variance = lambda x: np.var(x)

# Apply the transformation
data_tr = poker_hands[['R1', 'R2', 'R3', 'R4', 'R5']].apply(get_variance, axis=0)
print(data_tr.head())

## Why vectorization in pandas is so fast?
"""
As you probably noticed in this lesson, we achieved a massive improvement using some form of vectorization.

Where does this improvement come from?
"""

# Fewer operations are required due to optimization in pandas.
# Congratulations! Vectorization is always the fastest option, and now you know why. Extra tip: It uses C code the optimal way!

## pandas vectorization in action
"""
In this exercise, you will apply vectorization over pandas series to:

calculate the mean rank of all the cards in each hand (row)
calculate the mean rank of each of the 5 cards in each hand (column)

You will use the poker_hands dataset once again to compare both methods' efficiency.
"""

# Calculate the mean rank in each hand
row_start_time = time.time()
mean_r = poker_hands[['R1', 'R2', 'R3', 'R4', 'R5']].mean(axis=1)
print("Time using pandas vectorization for rows: {} sec".format(time.time() - row_start_time))
print(mean_r.head())

# Calculate the mean rank of each of the 5 card in all hands
col_start_time = time.time()
mean_c = poker_hands[['R1', 'R2', 'R3', 'R4', 'R5']].mean(axis=0)
print("Time using pandas vectorization for columns: {} sec".format(time.time() - col_start_time))
print(mean_c.head())

## Best method of vectorization
"""
So far, you have encountered two vectorization methods:

Vectorization over pandas Series
Vectorization over Numpy ndarrays

While these two methods outperform all the other methods, when can vectorization over NumPy ndarrays be used to replace vectorization over pandas Series?
"""

# When operations like indexing or data type are not used.
# Congratulations! Vectorization over NumPy ndarrays is always the fastest option, and now you know why!

## Vectorization methods for looping a DataFrame
"""
Now that you're familiar with vectorization in pandas and NumPy, you're going to compare their respective performances yourself.

Your task is to calculate the variance of all the hands in each hand using the vectorization over pandas Series and then modify your code using the vectorization over Numpy ndarrays method.
"""

# Calculate the variance in each hand
start_time = time.time()
poker_var = poker_hands[['R1','R2','R3','R4','R5']].var(axis=1)
print("Time using pandas vectorization: {} sec".format(time.time() - start_time))
print(poker_var.head())

# Calculate the variance in each hand
start_time = time.time()
poker_var = poker_hands[['R1','R2','R3','R4','R5']].values.var(axis=1, ddof=1)
print("Time using NumPy vectorization: {} sec".format(time.time() - start_time))
print(poker_var[0:5])

## The min-max normalization using .transform()
"""
A very common operation is the min-max normalization. It consists in rescaling our value of interest by deducting the minimum value and dividing the result by the difference between the maximum and the minimum value. For example, to rescale student's weight data spanning from 160 pounds to 200 pounds, you subtract 160 from each student's weight and divide the result by 40 (200 - 160).

You're going to define and apply the min-max normalization to all the numerical variables in the restaurant data. You will first group the entries by the time the meal took place (Lunch or Dinner) and then apply the normalization to each group separately.

Remember you can always explore the dataset and see how it changes in the IPython Shell, and refer to the slides in the Slides tab.
"""

# Define the min-max transformation
min_max_tr = lambda x: (x - x.min()) / (x.max() - x.min())

# Group the data according to the time
restaurant_grouped = restaurant_data.groupby('time')

# Apply the transformation
restaurant_min_max_group = restaurant_grouped.transform(min_max_tr)
print(restaurant_min_max_group.head())

## Transforming values to probabilities
"""
In this exercise, we will apply a probability distribution function to a pandas DataFrame with group related parameters by transforming the tip variable to probabilities.

The transformation will be a exponential transformation. The exponential distribution is defined as

    e^-λ*x * λ 

where λ (lambda) is the mean of the group that the observation x belongs to.

You're going to apply the exponential distribution transformation to the size of each table in the dataset, after grouping the data according to the time of the day the meal took place. Remember to use each group's mean for the value of λ.

In Python, you can use the exponential as np.exp() from the NumPy library and the mean value as .mean().
"""

# Define the exponential transformation
exp_tr = lambda x: np.exp(-x.mean()*x) * x.mean()

# Group the data according to the time
restaurant_grouped = restaurant_data.groupby('time')

# Apply the transformation
restaurant_exp_group = restaurant_grouped['tip'].transform(exp_tr)
print(restaurant_exp_group.head())

## Validation of normalization
"""
For this exercise, we will perform a z-score normalization and verify that it was performed correctly.

A distinct characteristic of normalized values is that they have a mean equal to zero and standard deviation equal to one.

After you apply the normalization transformation, you can group again on the same variable, and then check the mean and the standard deviation of each group.

You will apply the normalization transformation to every numeric variable in the poker_grouped dataset, which is the poker_hands dataset grouped by Class.
"""

zscore = lambda x: (x - x.mean()) / x.std()

# Apply the transformation
poker_trans = poker_grouped.transform(zscore)
print(poker_trans.head())

zscore = lambda x: (x - x.mean()) / x.std()

# Apply the transformation
poker_trans = poker_grouped.transform(zscore)

# Re-group the grouped object
poker_regrouped = poker_trans.groupby(poker_hands['Class'])

# Print each group's means and standard deviation
print(np.round(poker_regrouped.mean(), 3))
print(poker_regrouped.std())

## When to use transform()?
# The .transform() function applies a function to all members of each group. Which of the following transformations would produce the same results in the whole dataset regardless of groupings?

# lambda x: np.random.randint(0,10)
# Congratulations! You know when to and when not to use the .transform() function.

## Identifying missing values
"""
The first step before missing value imputation is to identify if there are missing values in our data, and if so, from which group they arise.

For the same restaurant_data data you encountered in the lesson, an employee erased by mistake the tips left in 65 tables. The question at stake is how many missing entries came from tables that smokers where present vs tables with no-smokers present.

Your task is to group both datasets according to the smoker variable, count the number or present values and then calculate the difference.

We're imputing tips to get you to practice the concepts taught in the lesson. From an ethical standpoint, you should not impute financial data in real life, as it could be considered fraud.
"""

# Group both objects according to smoke condition
restaurant_nan_grouped = restaurant_nan.groupby('smoker')

# Store the number of present values
restaurant_nan_nval = restaurant_nan_grouped['tip'].count()

# Print the group-wise missing entries
print(restaurant_nan_grouped['total_bill'].count() - restaurant_nan_nval)

## Missing value imputation
"""
As the majority of the real world data contain missing entries, replacing these entries with sensible values can increase the insight you can get from our data.

In the restaurant dataset, the "total_bill" column has some missing entries, meaning that you have not recorded how much some tables have paid. Your task in this exercise is to replace the missing entries with the median value of the amount paid, according to whether the entry was recorded on lunch or dinner (time variable).
"""

# Define the lambda function
missing_trans = lambda x: x.fillna(x.median())

# Group the data according to time
restaurant_grouped = restaurant_data.groupby('time')

# Apply the transformation
restaurant_impute = restaurant_grouped.transform(missing_trans)
print(restaurant_impute.head())

## When to use filtration?
# When applying the filter() function on a grouped object, what you can use as a criterion for filtering?

# The number of missing values of a feature.
# The numerical mean of a feature.
# The numerical mean of more than one feature.
# Congratulations! You know when to and when not to use the filter function!

## Data filtration
"""
As you noticed in the video lesson, you may need to filter your data for various reasons.

In this exercise, you will use filtering to select a specific part of our DataFrame:

by the number of entries recorded in each day of the week
by the mean amount of money the customers paid to the restaurant each day of the week
"""

# Filter the days where the count of total_bill is greater than $40
total_bill_40 = restaurant_data.groupby('day').filter(lambda x: x['total_bill'].count() > 40)

# Print the number of tables where total_bill is greater than $40
print('Number of tables where total_bill is greater than $40:', total_bill_40.shape[0])

# Filter the days where the count of total_bill is greater than $40
total_bill_40 = restaurant_data.groupby('day').filter(lambda x: x['total_bill'].count() > 40)

# Select only the entries that have a mean total_bill greater than $20
total_bill_20 = total_bill_40.groupby('day').filter(lambda x: x['total_bill'].mean() > 20)

# Print days of the week that have a mean total_bill greater than $20
print('Days of the week that have a mean total_bill greater than $20:', total_bill_20.day.unique())

# After applying the .filter() operation on total_bill_20 in Step 2 in the Console, how many entries (rows) does the last DataFrame you created (total_bill_20) have?
# 163
# Great job! You just helped a restaurant to adjust their opening times according to profit. Not surprisingly, it looks like that the week-end is the most profitable time of the week!

## Congratulations!