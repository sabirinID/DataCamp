# Introduction to Julia

## Julia as a calculator

# Add the comment on the line below
# Hours worked this week

# Add the hours worked
39 + 37.5 + 40 + 42

## Printing

# Hours worked this week

# Print the number of weeks worked
println(4)

# Print the sum of the hours worked each week
println(39 + 37.5 + 40 + 42)

## Assigning variables

# Create variable monday_distance
monday_distance = 4500

# Create variable monday_time
monday_time = 28.9

# Create variable wednesday_distance
wednesday_distance = 6000

# Create variable wednesday_time
wednesday_time = 37.1

## Calculating with variables

# Calculate the total distance in miles
total_distance = (monday_distance + wednesday_distance)/1609

# Calculate the total run time in hours
total_time = (monday_time + wednesday_time)/60

# Calculate the average speed
average_speed = total_distance/total_time

# Print the average speed
println(average_speed)

## Finding the data type

# Print the type of runtime_1
println(typeof(runtime_1))

# Print the type of runtime_2
println(typeof(runtime_2))

# Calculate the average run time
average_runtime = (runtime_1 + runtime_2)/2

# Print the type of average_runtime
println(typeof(average_runtime))

## Converting types

a_float = 3.0

# Convert a_float to be an integer and print its new type
a_int = Int64(a_float)

println(typeof(a_int))

b_int = 3

# Convert b_int into a float
b_float = Float64(b_int)

println("b_float is a ", typeof(b_float))

c_int = 10

# Convert c_int into a string
c_string = string(c_int)

println("c_string is a ", typeof(c_string))

d_string = "3.1415"

# Convert d_string into a float
d_float = parse(Float64, d_string)

println("d_float is a ", typeof(d_float))

## String interpolation

# Index drama grade
drama = grades[end]

# Print drama grade with interpolation
println("Your drama grade is $drama")

# Print mathematics grade with interpolation and parentheses
println("Your mathematics grade is $(grades[1])")

## Slicing strings

# Slice out the customer-id
customer_id = order_data[16:20]

# Slice out the order-number
order_number = order_data[38:46]

## Creating arrays

# Create array of the days of the week
week_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

# Create array of hours worked
hours_worked = [9, 7.5, 8, 9.5, 7]

# Print the type of the week days arrays
println(typeof(week_days))

# Print the type of the hours worked array
println(typeof(hours_worked))

## Indexing arrays

# Select the first run time
first_runtime = runtimes[1]

# Select the last run time
last_runtime = runtimes[end]

# Calculate the difference
time_diff = first_runtime - last_runtime

println("Last run time was $time_diff minutes faster")

## Slicing arrays

x[2:4]

x[3:end-1]

x[end-2:end]

## Modifying arrays

# Create array of zeros
x = zeros(Float64, 100)

# Set the second value to 12
x[2] = 12

# Set the values from the 3rd to 6th index
x[3:6] = [3,4,5,6]

## Fibonacci sequence

# Calculate the next number in the sequence
next_number = fibonacci[end-1] + fibonacci[end]

# Add this new number to the sequence
push!(fibonacci, next_number)

println(fibonacci)

## Appending arrays

runtimes = [35.1, 34.0, 34.31, 32.8, 32.04, 33.66, 32.41, 32.32, 33.37, 31.4, 31.4];
new_runtimes = [30.44, 31.21, 30.38, 30.52, 30.2];

# Remove the duplicated value
duplicated_value = pop!(runtimes)

# Append new runtimes and new_runtimes
append!(runtimes, new_runtimes)

println("Duplicated value $duplicated_value")
println("All run times: $runtimes")

## Finding the array length

x = 792

## Array operating fluency

# Multiply x by 3
a = 3 .* x

# Subtract 5 from x
b = x .- 5

# Add x and y
c = x .+ y

# Divide x by y
d = x ./ y

## Operating on body temperatures

# Subtract 32 from each element of body_temps_f
body_temps_sub = body_temps_f .− 32

# Multiply each element in body_temps_sub by 5/9
body_temps_c = body_temps_sub .* (5/9)

# Sort the temperatures from lowest to highest
sorted_body_temps_c = sort(body_temps_c)

# Print the 5 lowest temperatures
println(sorted_body_temps_c[1:5])

## Check input data type

# Check if n is the correct type
if typeof(n)==Int64
	# Print n is the correct type
	println("n is the correct type")
# Use an else statement to print if n is not the correct type
else
	println("This script only works on integers")
# Finish the conditional statement
end

println(n)

## If-else practice

# Print x if it is positive, otherwise print -x
if x>0
	println(x)
else
	println(-x)
end

# Complete the conditional statement to find the largeest value
if x>y
	larger = x
else
	larger = y
end

println("Maximum from x=$x and y=$y is $larger")

# Complete the conditional statement to check if your script has finished
if finished
	println("Script has finished")
else
	println("Script still running")
end

## Conditioning on body temperature

# Print a warning message if the temperature is too low
if body_temp < 35.9
	println("Seems a little cold")
# Print a warning message if the temperature is too high
elseif body_temp > 37.6 
    println("Might be a fever")
# Print the message if the temperature is normal
else
	println("The patient's temperature seems normal")
# Finish the conditional block
end

println("temperature = $body_temp")

## Writing a function for strings

# Create the get_mathgrade function
function get_mathgrade(grades)
	return grades[1]
end

# Use the function on mygrades
println(get_mathgrade(mygrades))

# Use the function on grades_array
println(get_mathgrade.(grades_array))

## Writing a function with multiple arguments

# Create the get_gradenumber function
function get_gradenumber(grades, n)
	return grades[n]
end

# Use the function on mygrades to extract the history grade
println(get_gradenumber(mygrades, 2))

# Use the function on grades_array to extract the history grades
println(get_gradenumber.(grades_array, 2))

## Absolute value

# Begin the absolute function
function absolute(x)
	# Write an if-statement to return absolute value of x
	if x>=0
    	return x
    else
    	return -x
    end
end

# Use the function on residuals
println(absolute.(residuals))

## Modifying arrays

# Write a grade mutating function
function topstudent!(x)
	x[4] = "AAAA"
end

println("Your previous grades were $(grades_array[4])")

# Call the function on grades_array
topstudent!(grades_array)

println("Your new grades are $(grades_array[4])")

## Everyone wins

# Write a grade mutating function
function topstudents!(x)
	x .= "AAAA"
end

# Call the function on grades_array
topstudents!(grades_array)

println("The new grades are $(grades_array)")

## Multiple dispatch

# Write the negative function
function negative(x)
	return -x
end

# Write negative function for Bool data type
function negative(x::Bool)
	if x
    	return false
    else
    	return true
    end
end

## Importing packages

# Import the Statistics package
import Statistics

# Calculate median
m = Statistics.median(x)

println("Median of x is $m")

# Import the Statistics package
using Statistics

# Calculate standard deviation
m = std(x)

println("Standard deviation of x is $m")

## Using the Statistics package

# Import the Statistics package
import Statistics as sts

# Calculate the mean body temperature
mean_body_temp_c = sts.mean(body_temps_c)

# Print the mean body temperature
println("The mean body temperature is ", mean_body_temp_c)

# Calculate the mean heart rate
mean_heart_rate = sts.mean(heart_rates)

println("The mean heart rate is $mean_heart_rate")

## Loading and examining data

# Import packages
using CSV
using DataFrames

# Load the CSV file
file_contents = CSV.File("patients.csv")

# Convert the data to a DataFrame
df_patients = DataFrame(file_contents)

# Print the first 5 rows of the DataFrame
println(first(df_patients, 5))

## Creating a DataFrame

# Extract the mathematics grades
math_grades = get_gradenumber.(grades_array, 1)

# Create the DataFrame
df_grades = DataFrame(
	mathematics=math_grades
)

println(first(df_grades, 5))

# Create the DataFrame
df_grades = DataFrame(
	mathematics=get_gradenumber.(grades_array, 1),
)

println(first(df_grades, 5))

# Create the DataFrame
df_grades = DataFrame(
	mathematics=get_gradenumber.(grades_array, 1),
    history=get_gradenumber.(grades_array, 2), 
    science=get_gradenumber.(grades_array, 3), 
    drama=get_gradenumber.(grades_array, 4),
)

println(first(df_grades, 5))

## DataFrame properties

# Load the book review data
df_books = DataFrame(CSV.File("books.csv"))

# Print column names
println(names(df_books))

# Find number of rows and columns
println(size(df_books))

## Indexing DataFrames

# Select the body temperature column
body_temps = df_patients[:, "bodytemp"]

println(body_temps)

# Select the third row of df_grades
third_grades = df_grades[3, :]

println(third_grades)

# Select the 
book_title = df_books[710, "title"]

println("The book is $book_title")

## Slicing DataFrames

# Slice the first 6 columns
df_narrow = df_books[:, 1:6]

# Slice the 10th to 20th rows
df_short = df_narrow[10:20, :]

println(df_short)

## Sorting patients

# Sort the data by heart rate
df_byheart = sort(df_patients, "heartrate")

# Print the first 5 rows
println(df_byheart[1:5, :])

# Sort the data by body temperature
df_bytemp = sort(df_patients, "bodytemp", rev=true)

# Print the first 5 rows
println(df_bytemp[1:5, :])

## Literary analysis

# Find the total number of ratings
total_reviews = sum(df_books.ratings_count)

# Find the earliest publication year
earliest_year = minimum(df_books.original_publication_year)

println("Total number of reviews is $total_reviews")
println("Earliest year of publication is $earliest_year")

## Describing patient data

# Summarize the DataFrame
println(describe(df_patients))

heartrate = 73.7615

bodytemp = 100.8

## Standardize heart rate

# Find the mean heart rate
mean_hr = mean(df_patients.heartrate)

# Find the standard deviation of heart rates
std_hr = std(df_patients.heartrate)

# Calculate the normalized array of heart rates
norm_heartrate = (df_patients.heartrate .- mean_hr) ./ std_hr

# Add the normalized heartrate to the DataFrame
df_patients[:, "norm_heartrate"] = norm_heartrate

println(last(df_patients, 5))

## Constructing filters

press 3 # You're on fire! But in a good way.

## Filtered body temp

# Filter to where the sex is female
df_female = filter(row -> row.sex == "female", df_patients)

# Filter to where the sex is male
df_male = filter(row -> row.sex == "male", df_patients)

# Calculate mean body temperature for females
female_temp = mean(df_female.bodytemp)

# Calculate mean body temperature for males
male_temp = mean(df_male.bodytemp)

println("Body temperatures of females is: $female_temp F")
println("Body temperatures of males is: $male_temp F")

## Classic books

# Filter to books which were published before 1900
df_old_books = filter(row -> row.original_publication_year < 1900, df_books)

# Sort these books by rating
df_old_books_sorted = sort(df_old_books, "average_rating", rev=true)

# Print the 5 top-rated old books
println(df_old_books_sorted[1:5, :])

## Final thoughts