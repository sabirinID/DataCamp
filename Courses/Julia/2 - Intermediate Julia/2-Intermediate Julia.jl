# Intermediate Julia

## For loops question

press 4 # That's right! The for loop will iterate over every element in the names vector.

## Loop over a vector

# Loop over stock_tickers and print each element
for value in stock_tickers
    println(value)
end

## Loops and enumerate

# Loop over stock_tickers printing the index and item
for (index, item) in enumerate(stock_tickers)
	println(index, " ", item)
end

## Looping over nested structures

# Loop over stocks, printing the ticker and the price
for (ticker, price) in stocks
    println("The price of 1 ", ticker, " share is ", price, ".")
end

## Writing a while loop

# Initialise stock_required and stock_owned
stock_required = 10
stock_owned = 0

# Check if stock_required is not zero, and if true, purchase one share
while stock_required != 0
	stock_required = stock_required - 1
    # Update the stock_owned variable and print both stock_owned and stock_required
	stock_owned = stock_owned + 1
    println("You own ", stock_owned, " shares. You need ", stock_required, " more.")
end

## While loops for iteration

ticker = 1
# Use a while loop to iterate over ticker
while ticker <= 4
	# Print each ticker in stock_tickers
	println(stock_tickers[ticker])
    # Increment ticker
    ticker = ticker + 1
end

ticker = 1
# Use a while loop to iterate over the length of stock_tickers
while ticker <= length(stock_tickers)
	# Print each ticker in stock_tickers
	println(stock_tickers[ticker])
    # Increment ticker
    ticker = ticker + 1
end

## Ranges - iteration

press 3 # Perfect! The step size means that we will print 1, 3, 5, 7, and 9.

## Defining ranges

# Define the range in the variable x
x = 1:10

# Print the type of the range x
println(typeof(x))

# Define the range and store it in the variable y
y = 1:2:10

# Print the type of the range y
println(typeof(y))

## Looping over ranges

# Define a range with start=1, stop=100, step=5
my_range = 1:5:100

# Iterate over my_range using a for loop, print results
for i in my_range
	println(i)
end

# Define a range with start=1, stop=100, step=5
my_range = 1:5:100

# Use the iterator n to iterate over my_range
n = 1
while n <= length(my_range)
  println(my_range[n])
  n = n + 1
end

## Splat unpacking

# Unpack my_range using the splat operator
my_range = 0:10:50
println([my_range...])

## Create, index, and slice a tuple

# Create a tuple containing the emergency information
emergency_information = ("1990-05-24", "Los Angeles", "O+")

# Print the third element of the emergency_information tuple
emergency_information = ("1990-05-24", "Los Angeles", "O+")
println(emergency_information[3])

# Print the second and third elements in the tuple
println(emergency_information[2:3])

## Create a NamedTuple for a person

# Create a NamedTuple with the names and values in the instructions
emergency_information = (birthdate="1990-05-24", birthlocation="Los Angeles", bloodtype="O+")
# Print the birthdate, birthlocation and bloodtype of the tuple
println(emergency_information.birthdate)
println(emergency_information.birthlocation)
println(emergency_information.bloodtype)

## Create untyped dict, iterate

# Create a dictionary with three keys and assign the given values to each key
apple_stock_data = Dict("price" => 131.86, "prev_close" => 131.99, "volume" => 55000000)

# Use a for loop to iterate over the dictionary
for i in apple_stock_data
    println(i)
end

## Create typed dict, iterate

# Create a typed dictionary with keys as String and values as Float64
apple_stock_data = Dict{String, Float64}("price" => 131.86, "prev_close" => 131.99, "volume" => 55000000)

# Print the keys and values in apple_stock_data
println(keys(apple_stock_data))
println(values(apple_stock_data))

## Modify keys/values in dict, use get()

apple_stock_data = Dict{String, Float64}("price" => 131.86, "prev_close" => 131.99, "volume" => 55000000)

# Change the price and volume values
apple_stock_data["price"] = 132.55
apple_stock_data["volume"] = 70000000

# Use get() to get the price key in apple_stock_data
println(get(apple_stock_data, "price", "Unknown key"))

## Create 1D and 2D arrays

# Create a vector containing the values one through six
my_array = [1, 2, 3, 4, 5, 6]

# Create a 3x2 matrix containing the values one through six
my_array = [1 2; 3 4; 5 6]

## Index and slice a 2D array

# Create a 3x3 array
my_array = [1 2 3 ; 4 5 6 ; 7 8 9]

# Print the instructed values in my_array
println(my_array[1, 2])
println(my_array[3, 1])
println(getindex(my_array, 3, 2))

# Print select elements of the array my_array by slicing
println(my_array[2, :])
println(my_array[:, 3])

## Array merging

# Create the arrays stocks_yesterday and stocks_today
stocks_yesterday = ["AAPL" "MSFT";"TSLA" "MSFT"]
stocks_today = ["V" "AAPL";"GOOG" "AMZN"]

# Concatenate the stocks_yesterday and stocks_today arrays
stocks_master = [stocks_yesterday ; stocks_today]

## Define a structure

# Define a structure called Employee with three variables
struct Employee
    name
    location
    age
end

# Create an instance of your structure called employee with sample data
employee = Employee("Peter", "Sydney", 30)

# Print the three values of the struct object you created
println(employee.name)
println(employee.location)
println(employee.age)

## Mutable and typed structs

# Create a mutable struct Employee
mutable struct Employee
    name
    location
    age
end

# Create a mutable struct Employee, adding type constraints to each field
mutable struct Employee
    name::String
    location::String
    age::Int
end

## Built-in time macro

press 1 # Perfect! This is why the BenchmarkTools package exists.

## Timing a function

# This function will square each number from 1 to 10 and push it to a vector
function my_function()
    x = Vector{Int}()
    for i in 1:10
      push!(x, i^2)
    end
    return println(x)
  end
  
# Time my_function using the base time macro
@time my_function()

# Time my_function using the benchmark macro in BenchmarkTools
@benchmark my_function samples=1000

## Positional arguments recap

# Define my_profit with two positional arguments
function my_profit(previous_price, current_price)
	return current_price - previous_price
end

# Call my_profit
my_profit(100, 105)

## Default arguments

# Define my_profit with two positional arguments and a default argument
function my_profit(previous_price, current_price, fees=2)
	return current_price - previous_price - fees
end

my_profit(100, 105)

## Type declarations

# Define my_profit with three type-restricted arguments
function my_profit(previous_price::Float64, current_price::Float64, fees::Int64=2)
	return current_price - previous_price - fees
end

# Call my_profit, passing in values of the correct data type
my_profit(100.00, 105.00)

## Keyword arguments

# Define my_profit with keyword arguments and a default argument
function my_profit(; previous_price::Float64, current_price::Float64, fees::Int64=2)
	return current_price - previous_price - fees
end

# Call my_profit
my_profit(current_price=100.0, previous_price=105.0)

## Variable number of arguments

# Define stock_to_buy, accepting varargs
function stock_to_buy(stocks...)
	println(stocks)
end

# Call stock_to_buy with four arguments
stock_to_buy("AAPL", "MSFT", "GOOG", "TSLA")

## Writing your own functions

# Create a function calculate_order_cost that takes in items
function calculate_order_cost(items...)

end

# Create a function calculate_order_cost that takes in items
function calculate_order_cost(items...)
	total_cost = 0
    # Loop over the restaurant menu
    for (key, value) in restaurant_menu
    
    end
    println("Total order cost is $total_cost.")
end

# Create a function calculate_order_cost that takes in items
function calculate_order_cost(items...)
	total_cost = 0
    # Loop over the restaurant menu
    for (key, value) in restaurant_menu
    	# Check if the item name is in the customer order items
    	if key in items
        	total_cost = total_cost + value
        end
    end
    println("Total order cost is $total_cost.")
end

# Create a function calculate_order_cost that takes in items
function calculate_order_cost(items...)
	total_cost = 0
    # Loop over the restaurant menu
    for (key, value) in restaurant_menu
    	# Check if the item name is in the customer order items
    	if key in items
        	total_cost = total_cost + value
        end
    end
    println("Total order cost is $total_cost.")
end

# Call the function with the given input for testing
calculate_order_cost("Meat", "Chocolate")
calculate_order_cost("Meat", "Apple", "Bread")

## Writing your own functions - structs

mutable struct Employee
    name::String
    location::String
    age::Int
    
    # Create a function that defaults the location to "Sydney"
    function Employee(name, age)
		new(name, "Sydney", age)
    end
    
end

# Define a function employee_birthday to increment age by one
function employee_birthday(employee::Employee)
	employee.age = employee.age + 1
end

# Create an instance of Employee called tim, and call employee_birthday on tim
tim = Employee("Tim", 30)
employee_birthday(tim)

## Multiple dispatch

# Create three functions to handle any input, String input, and Bool input
function largest_value(x, y, z)
    maximum((x, y, z))
end
  
function largest_value(x::String, y::String, z::String)
    map(length, (x, y, z))
end
  
function largest_value(x::Bool, y::Bool, z::Bool)
    x, y, z
end
  
# Un-comment this test case to test your function
#println(largest_value("12", "24", "36"))

## Anonymous functions

# Create an anonymous function and evaluate it at x=6
(x -> x^2 + 6*x + 4)(6)

# Use map to map the given values of x and y to an anonymous function
map((x, y) -> x^2 + 6x - y + 4, [1, 2, 3], [2, 2, 2])

## Filtering DataFrames

# Use an anonymous function to filter the tuple nums
nums = (1, 2, 3, 4, 5, 6, 7, 8)
filter(x -> x%2==0, nums)

# Filter stock_data for the row with the largest volume traded
filter!("Volume" => n -> n == maximum(stock_data.Volume), stock_data)

# Filter stock_data for any rows where the closing price is over 175
filter!("Adj Close" => n -> n > 175, stock_data)

## Importing Python and R

# Import PythonCall
using PythonCall

# Import math using PythonCall
pymath = pyimport("math")

# Import RCall
using RCall

# Import the base R package using RCall
@rimport base as r_base

## Python functions in Julia

using PythonCall
pymath = pyimport("math")

# Define a vector from -3 to 3
x = [-3, -2, -1, 0, 1, 2, 3]

# Print the absolute value of the second value in x using pymath
println(pymath.fabs(x[2]))

# Print the sixth value of x raised to the power of the seventh value of x
println(pymath.pow(x[6], x[7]))

## R functions in Julia

using RCall
@rimport base as r_base

# Define two vectors, x and y, with values one to four and five to eight
x = [1, 2, 3, 4]
y = [5, 6, 7, 8]

# Reverse the order of the values in y using R
y = r_base.rev(y)

# Plot x and y using R
r_base.plot(x, y)

## Renaming columns

# Show the first row of our DataFrame
println(first(video_game_data))

# Show the first row of our DataFrame
println(first(video_game_data))

# Rename the North_American_Sales column to NA_Sales
rename!(video_game_data, Dict("North_American_Sales" => "NA_Sales"))

## Missing data

# Describe the DataFrame to find columns with missing values
describe(sales_df)

# Count the number of rows in the DataFrame
println(nrow(sales_df))

# Drop rows with missing values from the JP_Sales column in the DataFrame
dropmissing!(sales_df, :"JP_Sales")

# Count the number of rows in the DataFrame to confirm they are dropped
println(nrow(sales_df))

## Advanced missing data

# Define a function replace_missing that takes one argument, the name of the column we want to modify
function replace_missing(column_name)
	# Calculate the average of all non-missing values in the column
	mean_value = mean(skipmissing(sales_df[!, column_name]))
    # Replace missing values with the mean_value
    replace!(sales_df[!, column_name], missing => mean_value)
end

## Congratulations!