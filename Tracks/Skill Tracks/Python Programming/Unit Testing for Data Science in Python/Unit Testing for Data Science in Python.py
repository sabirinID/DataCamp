## How frequently is a function tested?
"""
Many data scientists do not think much about testing, and just do it in the manual way when necessary. But once you see the big picture i.e. the life cycle of a function over the entire project, you appreciate how important testing really is and how frequently you need to test things.

Which of the following is true about testing?
"""

# A function is tested after the first implementation and then any time the function is modified, which happens mainly when new bugs are found, new features are implemented or the code is refactored.
# Exactly! If the project goes on for a few years, you may end up testing the same function over a hundred times because of new bugs, new feature requests and refactoring!

## Manual testing
"""
The function row_to_list(), which you met in the video lesson, has the following expected return values for the arguments listed below.

Argument	        Expected return value	Explanation
"2,081\t314,942\n"	["2,081", "314,942"]	Correct row format
"\t293,410\n"	    None	                Missing area
"1,463238,765\n"	None	                Missing tab separator

row_to_list() has been defined and imported for you. Your job is to test the function manually in the IPython console.

While testing manually, notice how many times you have to repeat the same steps! The point is to experience the inefficiency of manual testing.
"""

# Call row_to_list() in the IPython console on the three arguments listed in the table. Do the actual return values match the expected return values listed in the table?
# No, ... for the argument "\t293,410\n" ... None.

# In the last step, you discovered a bug in our implementation of row_to_list(). Good job!
# We have implemented a corresponding bug fix in a new function row_to_list_bugfix(). Call row_to_list_bugfix() in the IPython console on the three arguments listed in the table. Do the actual return values now match the expected return values listed in the table?
# Yes, the implementation ...
# Well done! Did you notice how manual testing involves repeating the same steps over and over in the IPython console? In this exercise, you just went through a single bug discovery and fixing phase. Just imagine doing this a hundred times over the entire life cycle of row_to_list(), including new feature implementation and refactoring phases! Unit testing can automate these repetitive steps, so that testing becomes easier, and you will learn it in the next lesson ;-)

## Your first unit test using pytest
"""
The data file containing housing area and prices uses commas as thousands separators, e.g. "2,081" or "314,942", as you can see in the IPython Shell.

The convert_to_int() function takes a comma separated integer string as argument, and returns the integer. Therefore, the expected return value of convert_to_int("2,081") is the integer 2081.

This function is defined in the module preprocessing_helpers.py. But it is not known if the function is working properly.
"""

# Import the pytest package
import pytest

# Import the function convert_to_int()
from preprocessing_helpers import convert_to_int

# Complete the unit test name by adding a prefix
def test_on_string_with_one_comma():
  pass

# Import the pytest package
import pytest

# Import the function convert_to_int()
from preprocessing_helpers import convert_to_int

# Complete the unit test name by adding a prefix
def test_on_string_with_one_comma():
  # Complete the assert statement
  assert convert_to_int("2,081") == 2081

## Running unit tests
"""
The tests that you wrote in the previous exercise have been written to a test module test_convert_to_int.py. Try running the tests in the IPython console.

What is the correct IPython console command to run the tests in this test module?
"""

# !pytest test_convert_to_int.py
# Congratulations! You just wrote and ran your first unit test. You probably saw that running the test produced a lot of output. This output contains information about a bug in convert_to_int(). You should be able to read and understand this output in order to bug fix efficiently. This will be covered in the next video lesson. Jump right in!

## What causes a unit test to fail?
"""
In the test result report, the character ., as shown below, stands for a passing test. A passing test is good news as it means that your function works as expected. The character F stands for a failing test. A failing test is bad news as this means that something is broken.

test_row_to_list.py .F.                                                  [100%]

Which of the following describes best why a unit test fails?
"""

# An exception is raised when running the unit test. This could be an AssertionError raised by the assert statement or another exception, e.g. NameError, which is raised before the assert statement can run.
# Exactly! If you get an AssertionError, this means the function has a bug and you should fix it. If you get another exception, e.g. NameError, this means that something else is wrong with the unit test code and you should fix it so that the assert statement can actually run.

## Spotting and fixing bugs
"""
To find bugs in functions, you need to follow a four step procedure.

1. Write unit tests.
2. Run them.
3. Read the test result report and spot the bugs.
4. Fix the bugs.

In a previous exercise, you wrote a unit test for the function convert_to_int(), which is supposed to convert a comma separated integer string like "2,081" to the integer 2081. You also ran the unit test and discovered that it is failing.

In this exercise, you will read the test result report from that exercise in detail, and then spot and fix the bug. This would equip you with all basic skills to start using unit tests for your projects.

The convert_to_int() function is defined in the file preprocessing_helpers.py. The unit test is available in the test module test_convert_to_int.py.
"""

# Run the unit test in the test module test_convert_to_int.py in the IPython console. Read the test result report and spot the bug.
# Which of the following describes the bug in the function convert_to_int(), if any?

# ... the integer 2081 ... the string "2,081".

def convert_to_int(string_with_comma):
    # Fix this line so that it returns an int, not a str
    return int(string_with_comma.replace(",", ""))

## Benefits of unit testing
"""
You have been invited to a meeting where company executives are discussing whether developers should write unit tests. The CEO is unsure, and asks you about the benefits that unit testing might bring. In your response, which of the following benefits should you include?

1. Time savings, leading to faster development of new features.
2. Better user experience due to faster code execution.
3. Improved documentation, which will help new colleagues understand the code base better.
4. More user trust in the software product.
5. Better user experience due to improved visualizations.
6. Better user experience due to reduced downtime.
"""

# 1, 3, 4 and 6
# You steered the CEO in the right direction! Time savings and reduced downtime are the major benefits of unit testing, while improved documentation and more user trust are great side effects.

## Unit tests as documentation
"""
Assume that you are a new collaborator of our linear regression project on housing area and prices.

While inspecting the project, you come across a function mystery_function() in the feature module. You want to figure out what this function does. As you know, reading the unit tests might give you the answer quickly!

The unit tests for the function is available in the test module test_mystery_function.py. You can read it, and any other file that you encounter, by using the !cat command in the IPython shell.

Having read the unit tests, can you guess what mystery_function() does?
"""

# It converts data in a data file into a NumPy array.
# You guessed it right and you didn't even take a look at the function definition! This is why - when onboarding new colleagues - it is a good idea to tell them to look at the unit tests if they are not sure about a function's purpose. In Chapter 2, you will see more functions from the feature and models module, and write more advanced

## Write an informative test failure message
"""
The test result reports become a lot easier to read when you make good use of the optional message argument of the assert statement.

In a previous exercise, you wrote a test for the convert_to_int() function. The function takes an integer valued string with commas as thousand separators e.g. "2,081" as argument and should return the integer 2081.

In this exercise, you will rewrite the test called test_on_string_with_one_comma() so that it prints an informative message if the test fails.
"""

import pytest
from preprocessing_helpers import convert_to_int

def test_on_string_with_one_comma():
    test_argument = "2,081"
    expected = 2081
    actual = convert_to_int(test_argument)
    # Format the string with the actual return value
    message = "convert_to_int('2,081') should return the int 2081, but it actually returned {0}".format(actual)
    # Write the assert statement which prints message on failure
    assert actual == expected, message

# The test that you wrote was written to a test module called test_convert_to_int.py. Run the test in the IPython console and read the test result report.
# Which of the following is true?

# The test fails because convert_to_int('2,081') returns None and not the integer 2081.
# That's right! It is a lot easier to understand the custom message that you wrote than the automatic messages that pytest prints. Therefore, it is recommended that you add custom failure messages to all assert statements that you write in the future.

## Testing float return values
"""
The get_data_as_numpy_array() function (which was called mystery_function() in one of the previous exercises) takes two arguments: the path to a clean data file and the number of data columns in the file . An example file has been printed out in the IPython console. It contains three rows.

The function converts the data into a 3x2 NumPy array with dtype=float64. The expected return value has been stored in a variable called expected. Print it out to see it.

The housing areas are in the first column and the housing prices are in the second column. This array will be the features that will be fed to the linear regression model for learning.

The return value contains floats. Therefore you have to be especially careful when writing unit tests for this function.
"""   

import numpy as np
import pytest
from as_numpy import get_data_as_numpy_array

def test_on_clean_file():
  expected = np.array([[2081.0, 314942.0],
                       [1059.0, 186606.0],
  					   [1148.0, 206186.0]
                       ]
                      )
  actual = get_data_as_numpy_array("example_clean_data.txt", num_columns=2)
  message = "Expected return value: {0}, Actual return value: {1}".format(expected, actual)
  # Complete the assert statement
  assert actual == pytest.approx(expected), message

## Testing with multiple assert statements
"""
You're now going to test the function split_into_training_and_testing_sets() from the models module.

It takes a n x 2 NumPy array containing housing area and prices as argument. To see an example argument, print the variable example_argument in the IPython console.

The function returns a 2-tuple of NumPy arrays (training_set, testing_set). The training set contains int(0.75 * n) (approx. 75%) randomly selected rows of the argument array. The testing set contains the remaining rows.

Print the variable expected_return_value in the IPython console. example_argument had 6 rows. Therefore the training array has int(0.75 * 6) = 4 of its rows and the testing array has the remaining 2 rows.

numpy as np, pytest and split_into_training_and_testing_sets have been imported for you.
"""

def test_on_six_rows():
    example_argument = np.array([[2081.0, 314942.0], [1059.0, 186606.0],
                                 [1148.0, 206186.0], [1506.0, 248419.0],
                                 [1210.0, 214114.0], [1697.0, 277794.0]]
                                )
    # Fill in with training array's expected number of rows
    expected_training_array_num_rows = int(0.75 * example_argument.shape[0])

def test_on_six_rows():
    example_argument = np.array([[2081.0, 314942.0], [1059.0, 186606.0],
                                [1148.0, 206186.0], [1506.0, 248419.0],
                                [1210.0, 214114.0], [1697.0, 277794.0]]
                               )
    # Fill in with training array's expected number of rows
    expected_training_array_num_rows = 4
    # Fill in with testing array's expected number of rows
    expected_testing_array_num_rows = example_argument.shape[0] - expected_training_array_num_rows

def test_on_six_rows():
    example_argument = np.array([[2081.0, 314942.0], [1059.0, 186606.0],
                                 [1148.0, 206186.0], [1506.0, 248419.0],
                                 [1210.0, 214114.0], [1697.0, 277794.0]]
                                )
    # Fill in with training array's expected number of rows
    expected_training_array_num_rows = 4
    # Fill in with testing array's expected number of rows
    expected_testing_array_num_rows = 2
    actual = split_into_training_and_testing_sets(example_argument)
    # Write the assert statement checking training array's number of rows
    assert actual[0].shape[0] == expected_training_array_num_rows, "The actual number of rows in the training array is not {}".format(expected_training_array_num_rows)

def test_on_six_rows():
    example_argument = np.array([[2081.0, 314942.0], [1059.0, 186606.0],
                                 [1148.0, 206186.0], [1506.0, 248419.0],
                                 [1210.0, 214114.0], [1697.0, 277794.0]]
                                )
    # Fill in with training array's expected number of rows
    expected_training_array_num_rows = 4
    # Fill in with testing array's expected number of rows
    expected_testing_array_num_rows = 2
    actual = split_into_training_and_testing_sets(example_argument)
    # Write the assert statement checking training array's number of rows
    assert actual[0].shape[0] == expected_training_array_num_rows, "The actual number of rows in the training array is not {}".format(expected_training_array_num_rows)
    # Write the assert statement checking testing array's number of rows
    assert actual[1].shape[0] == expected_testing_array_num_rows, "The actual number of rows in the testing array is not {}".format(expected_testing_array_num_rows)

## Practice the context manager
"""
In pytest, you can test whether a function raises an exception by using a context manager. Let's practice your understanding of this important context manager, the with statement and the as clause.

At any step, feel free to run the code by pressing the "Run Code" button and check if the output matches your expectations.
"""
"""
import pytest

# Fill in with a context manager that will silence the ValueError
with pytest.raises(ValueError):
    raise ValueError

import pytest

try:
    # Fill in with a context manager that raises Failed if no OSError is raised
    with pytest.raises(OSError):
        raise ValueError
except:
    print("pytest raised an exception because no OSError was raised in the context.")

import pytest

# Store the raised ValueError in the variable exc_info
with pytest.raises(ValueError) as exc_info:
    raise ValueError("Silence me!")

import pytest

with pytest.raises(ValueError) as exc_info:
    raise ValueError("Silence me!")
# Check if the raised ValueError contains the correct message
assert exc_info.match("Silence me!")
"""
## Unit test a ValueError
"""
Sometimes, you want a function to raise an exception when called on bad arguments. This prevents the function from returning nonsense results or hard-to-interpret exceptions. This is an important behavior which should be unit tested.

Remember the function split_into_training_and_testing_sets()? It takes a NumPy array containing housing area and prices as argument. The function randomly splits the array row wise into training and testing arrays in the ratio 3:1, and returns the resulting arrays in a tuple.

If the argument array has only 1 row, the testing array will be empty. To avoid this situation, you want the function to not return anything, but raise a ValueError with the message "Argument data_array must have at least 2 rows, it actually has just 1".
"""

import numpy as np
import pytest
from train import split_into_training_and_testing_sets

def test_on_one_row():
    test_argument = np.array([[1382.0, 390167.0]])
    # Fill in with a context manager for checking ValueError
    with pytest.raises(ValueError):
      split_into_training_and_testing_sets(test_argument)

import numpy as np
import pytest
from train import split_into_training_and_testing_sets

def test_on_one_row():
    test_argument = np.array([[1382.0, 390167.0]])
    # Store information about raised ValueError in exc_info
    with pytest.raises(ValueError) as exc_info:
      split_into_training_and_testing_sets(test_argument)

import numpy as np
import pytest
from train import split_into_training_and_testing_sets

def test_on_one_row():
    test_argument = np.array([[1382.0, 390167.0]])
    # Store information about raised ValueError in exc_info
    with pytest.raises(ValueError) as exc_info:
      split_into_training_and_testing_sets(test_argument)
    expected_error_msg = "Argument data_array must have at least 2 rows, it actually has just 1"
    # Check if the raised ValueError contains the correct message
    assert str(exc_info.value) == expected_error_msg

# The test test_on_one_row() was written to the test module test_split_into_training_and_testing_sets.py. Run the test in the IPython console and read the test result report. Does the test pass or fail?
# The test passes.
# That's correct! Congratulations on writing your first unit test that checks for exceptions. In the next lesson, you will find out that it is good practice to include a few tests of this type for every function that you test.

## Testing well: Boundary values
"""
Remember row_to_list()? It takes a row containing housing area and prices e.g. "2,041\t123,781\n" and returns the data as a list e.g. ["2,041", "123,781"].

A row can be mapped to a 2-tuple (m, n), where m is the number of tab separators. n is 1 if the row has any missing values, and 0 otherwise.

For example,

"123\t456\n" 
 (1, 0).
"\t456\n" 
 (1, 1).
"\t456\t\n" 
 (2, 1).

The function only returns a list for arguments mapping to (1, 0). All other tuples correspond to invalid rows, with either more than one tab or missing values. The function returns None in all these cases. See the plot.

This mapping shows that the function has normal behavior at (1, 0), and special behavior everywhere else.
"""

# Which are the boundary values for this function, according to the plot?
# (0, 0), (2, 0) and (1, 1)

import pytest
from preprocessing_helpers import row_to_list

def test_on_no_tab_no_missing_value():    # (0, 0) boundary value
    # Assign actual to the return value for the argument "123\n"
    actual = row_to_list("123\n")
    assert actual is None, "Expected: None, Actual: {0}".format(actual)

import pytest
from preprocessing_helpers import row_to_list

def test_on_no_tab_no_missing_value():    # (0, 0) boundary value
    # Assign actual to the return value for the argument "123\n"
    actual = row_to_list("123\n")
    assert actual is None, "Expected: None, Actual: {0}".format(actual)
    
def test_on_two_tabs_no_missing_value():    # (2, 0) boundary value
    actual = row_to_list("123\t4,567\t89\n")
    # Complete the assert statement
    assert actual is None, "Expected: None, Actual: {0}".format(actual)

import pytest
from preprocessing_helpers import row_to_list

def test_on_no_tab_no_missing_value():    # (0, 0) boundary value
    # Assign actual to the return value for the argument "123\n"
    actual = row_to_list("123\n")
    assert actual is None, "Expected: None, Actual: {0}".format(actual)
    
def test_on_two_tabs_no_missing_value():    # (2, 0) boundary value
    actual = row_to_list("123\t4,567\t89\n")
    # Complete the assert statement
    assert actual is None, "Expected: None, Actual: {0}".format(actual)
    
def test_on_one_tab_with_missing_value():    # (1, 1) boundary value
    actual = row_to_list("\t4,567\n")
    # Format the failure message
    assert actual is None, "Expected: None, Actual: {0}".format(actual)

## Testing well: Values triggering special logic
"""
Look at the plot. The boundary values of row_to_list() are now marked in orange. The normal argument is marked in green and the values triggering special behavior are marked in blue.

In the last exercise, you wrote tests for boundary values. In this exercise, you are going to write tests for values triggering special behavior, in particular, (0, 1) and (2, 1). These are values triggering special logic since the function returns None instead of a list.
"""   

import pytest
from preprocessing_helpers import row_to_list

def test_on_no_tab_with_missing_value():    # (0, 1) case
    # Assign to the actual return value for the argument "\n"
    actual = row_to_list("\n")
    # Write the assert statement with a failure message
    assert actual is None, "Expected: None, Actual: {0}".format(actual)
    
def test_on_two_tabs_with_missing_value():    # (2, 1) case
    # Assign to the actual return value for the argument "123\t\t89\n"
    actual = row_to_list("123\t\t89\n")
    # Write the assert statement with a failure message
    assert actual is None, "Expected: None, Actual: {0}".format(actual)

## Testing well: Normal arguments
"""
This time, you will test row_to_list() with normal arguments i.e. arguments mapping to the tuple (1, 0). The plot is provided to you for reference.

Remembering that the best practice is to test for two to three normal arguments, you will write two tests in this exercise.
"""   

# How many normal arguments is it recommended to test?
# At least two or three.

import pytest
from preprocessing_helpers import row_to_list

def test_on_normal_argument_1():
    actual = row_to_list("123\t4,567\n")
    # Fill in with the expected return value for the argument "123\t4,567\n"
    expected = ["123", "4,567"]
    assert actual == expected, "Expected: {0}, Actual: {1}".format(expected, actual)

import pytest
from preprocessing_helpers import row_to_list

def test_on_normal_argument_1():
    actual = row_to_list("123\t4,567\n")
    # Fill in with the expected return value for the argument "123\t4,567\n"
    expected = ["123", "4,567"]
    assert actual == expected, "Expected: {0}, Actual: {1}".format(expected, actual)
    
def test_on_normal_argument_2():
    actual = row_to_list("1,059\t186,606\n")
    expected = ["1,059", "186,606"]
    # Write the assert statement along with a failure message
    assert actual == expected, "Expected: {0}, Actual: {1}".format(expected, actual)

# The tests for boundary values, values triggering special behavior and normal arguments have been written to a test module test_row_to_list.py. Run the tests in the IPython shell. Which bugs does the function have?
# The function does not have any bugs.
# Well done! You tested the function row_to_list() on boundary values, values triggering special behavior and normal arguments. All the tests are passing. So you can be quite confident that the function is correctly coded! Note that this function does not have bad arguments, so you did not write any tests for that. Also note how mapping the arguments to tuples enabled us to categorize the arguments easily. Use this trick for other functions whenever applicable ;-)

## TDD: Tests for normal arguments
"""
In this and the following exercises, you will implement the function convert_to_int() using Test Driven Development (TDD). In TDD, you write the tests first and implement the function later.

Normal arguments for convert_to_int() are integer strings with comma as thousand separators. Since the best practice is to test a function for two to three normal arguments, here are three examples with no comma, one comma and two commas respectively.

Argument value	Expected return value
"756"	        756
"2,081"	        2081
"1,034,891"	    1034891

Since the convert_to_int() function does not exist yet, you won't be able to import it. But you will use it in the tests anyway. That's how TDD works.

pytest has already been imported for you.
"""

def test_with_no_comma():
    actual = convert_to_int("756")
    # Complete the assert statement
    assert actual == 756, "Expected: 756, Actual: {0}".format(actual)
    
def test_with_one_comma():
    actual = convert_to_int("2,081")
    # Complete the assert statement
    assert actual == 2081, "Expected: 2081, Actual: {0}".format(actual)
    
def test_with_two_commas():
    actual = convert_to_int("1,034,891")
    # Complete the assert statement
    assert actual == 1034891, "Expected: 1034891, Actual: {0}".format(actual)

## TDD: Requirement collection
"""
What should convert_to_int() do if the arguments are not normal? In particular, there are three special argument types:

1. Arguments that are missing a comma e.g. "178100,301".
2. Arguments that have the comma in the wrong place e.g. "12,72,891".
3. Float valued strings e.g. "23,816.92".

Also, should convert_to_int() raise an exception for specific argument values?

When your boss asked you to implement the function, she didn't say anything about these cases! But since you want to write tests for special and bad arguments as a part of TDD, you go and ask your boss.

She says that convert_to_int() should return None for every special argument and there are no bad arguments for this function.

pytest has been imported for you.
"""

# Give a name to the test for an argument with missing comma
def test_on_string_with_missing_comma():
    actual = convert_to_int("178100,301")
    assert actual is None, "Expected: None, Actual: {0}".format(actual)
    
def test_on_string_with_incorrectly_placed_comma():
    # Assign to the actual return value for the argument "12,72,891"
    actual = convert_to_int("12,72,891")
    assert actual is None, "Expected: None, Actual: {0}".format(actual)
    
def test_on_float_valued_string():
    actual = convert_to_int("23,816.92")
    # Complete the assert statement
    assert actual is None, "Expected: None, Actual: {0}".format(actual)

# The tests for normal and special arguments have been written to a test module test_convert_to_int.py. Run it in the IPython console and read the test result report. What happens?
# All tests are not passing. Did you run the tests using the !pytest command and read the test result report carefully?
# That test is failing, but not because convert_to_int() returns the wrong value. Did you run the tests using the !pytest command and read the test result report carefully?
# All tests are failing ...
# Yes! In TDD, the first run of the tests always fails with a NameError or ImportError because the function does not exist yet. In the next exercise, you will implement the function and fix this. But before you move on, notice how thinking about special and bad arguments crystallized the requirements for the function. This will help us immensely in implementing the function in the coming exercise.

## TDD: Implement the function
"""
convert_to_int() returns None for the following:

1. Arguments with missing thousands comma e.g. "178100,301". If you split the string at the comma using "178100,301".split(","), then the resulting list ["178100", "301"] will have at least one entry with length greater than 3 e.g. "178100".

2. Arguments with incorrectly placed comma e.g. "12,72,891". If you split this at the comma, then the resulting list is ["12", "72", "891"]. Note that the first entry is allowed to have any length between 1 and 3. But if any other entry has a length other than 3, like "72", then there's an incorrectly placed comma.

3. Float valued strings e.g. "23,816.92". If you remove the commas and call int() on this string i.e. int("23816.92"), you will get a ValueError.
"""

def convert_to_int(integer_string_with_commas):
    comma_separated_parts = integer_string_with_commas.split(",")
    for i in range(len(comma_separated_parts)):
        # Write an if statement for checking missing commas
        if len(comma_separated_parts[i]) > 3:
            return None

def convert_to_int(integer_string_with_commas):
    comma_separated_parts = integer_string_with_commas.split(",")
    for i in range(len(comma_separated_parts)):
        # Write an if statement for checking missing commas
        if len(comma_separated_parts[i]) > 3:
            return None
        # Write the if statement for incorrectly placed commas
        if i != 0 and len(comma_separated_parts[i]) != 3:
            return None
        
def convert_to_int(integer_string_with_commas):
    comma_separated_parts = integer_string_with_commas.split(",")
    for i in range(len(comma_separated_parts)):
        # Write an if statement for checking missing commas
        if len(comma_separated_parts[i]) > 3:
            return None
        # Write the if statement for incorrectly placed commas
        if i != 0 and len(comma_separated_parts[i]) != 3:
            return None
    integer_string_without_commas = "".join(comma_separated_parts)
    try:
        return int(integer_string_without_commas)
    # Fill in with a ValueError
    except ValueError:
        return None
    
# Now that you have implemented the convert_to_int() function, let's run the tests in the test module test_convert_to_int.py again. Run it the IPython console and read the test result report. Did you implement the function correctly, or are there any bugs?
# All tests are passing and ...
# Yes! All tests are passing and you nailed the implementation! Congratulations are also due on finshing Chapter 2. You've learned a lot, and in the next Chapter, you will learn several best practices that will take your testing to the next level.

## Place test modules at the correct location
"""
A data science project without visualization is like pizza without cheese, right? But this has been fixed by creating a package called visualization under the top level application directory src.

src/                                    # All application code lives here
|-- visualization/                      # Package for visualization
    |-- __init__.py
    |-- plots.py                        # Module for plotting
In the package, there is a Python module plots.py, which contain functions related to plotting. These functions should be tested in a test module test_plots.py.

According to pytest guidelines, where should you place this test module within the project structure?
"""

# tests/visualization/test_plots.py.
# Wow, you have become good at organizing tests! Placing it in this location gives us two advantages: easier navigation within the tests folder and the possibility of having identically named test modules distinguished by the parent mirror package.

## Create a test class
"""
Test classes are containers inside test modules. They help separate tests for different functions within the test module, and serve as a structuring tool in the pytest framework.

Test classes are written in CamelCase e.g. TestMyFunction as opposed to tests, which are written using underscores e.g. test_something().

You met the function split_into_training_and_testing_sets() in Chapter 2, and wrote some tests for it. One of these tests was called test_on_one_row() and it checked if the function raises a ValueError when passed a NumPy array with only one row.

In this exercise you are going to create a test class for this function. This test class will hold the test test_on_one_row().
"""

import pytest
import numpy as np

from models.train import split_into_training_and_testing_sets

# Declare the test class
class TestSplitIntoTrainingAndTestingSets(object):
    # Fill in with the correct mandatory argument
    def test_on_one_row(self):
        test_argument = np.array([[1382.0, 390167.0]])
        with pytest.raises(ValueError) as exc_info:
            split_into_training_and_testing_sets(test_argument)
        expected_error_msg = "Argument data_array must have at least 2 rows, it actually has just 1"
        assert exc_info.match(expected_error_msg)

## One command to run them all
"""
One of your colleagues pushed some changes to the functions row_to_list(), convert_to_int(), get_data_as_numpy_array() and split_into_training_and_testing_sets(). That means that you have to run all the tests again to figure out if something got broken as a result.

The current working directory in the IPython console is the tests directory, which contains all the tests in the same layout as described in the video. You can, at any time, run the tests in the IPython console using the appropriate command.
"""

# In the IPython console, what is the correct command for running all tests contained in the tests folder?
# !pytest

# When you run all tests with the command !pytest, how many of them pass and how may fail?
# Passing: 15, Failing: 1

# Assuming that you simply want to answer the binary question "Are all tests passing" without wasting time and resources, what is the correct command to run all tests till the first failure is encountered?
# !pytest -x
# The -k flag is used for selecting a subset of tests whose node ID matches a particular pattern. It does not stop test execution on the first failure.

# When you ran the tests using the !pytest -x command, how many tests ran in total before test execution stopped because of the first failing test?
# 15

## Running test classes
"""
When you ran the !pytest command in the last exercise, the test test_on_six_rows() failed. This is a test for the function split_into_training_and_testing_sets(). This means that this function is broken.

Short recap in case you forgot: this function takes a NumPy array containing housing area and prices as argument. The function randomly splits the argument array into training and testing arrays in the ratio 3:1, and returns the resulting arrays in a tuple.

A quick look revealed that during the code update, someone inadvertently changed the split from 3:1 to 9:1. This has to be changed back and the unit tests for the function, which now lives in the test class TestSplitIntoTrainingAndTestingSets, needs to be run again. Are you up to the challenge?
"""    

import numpy as np

def split_into_training_and_testing_sets(data_array):
    dim = data_array.ndim
    if dim != 2:
        raise ValueError("Argument data_array must be two dimensional. Got {0} dimensional array instead!".format(dim))
    num_rows = data_array.shape[0]
    if num_rows < 2:
        raise ValueError("Argument data_array must have at least 2 rows, it actually has just {0}".format(num_rows))
    # Fill in with the correct float
    num_training = int(0.75 * data_array.shape[0])
    permuted_indices = np.random.permutation(data_array.shape[0])
    return data_array[permuted_indices[:num_training], :], data_array[permuted_indices[num_training:], :]

# Now let's see if that modification fixed the broken function. The current working directory in the IPython console is the tests folder that contains all tests. The test class TestSplitIntoTrainingAndTestingSets resides in the test module tests/models/test_train.py.
# What is the correct command to run all the tests in this test class using node IDs?

# The :: separator is only used to separate the test module path from the test class name. It should be used as a path separator.
# The -k option is to run a test class using keyword expressions, but in this step, you should run the test using node IDs.
# Since the test class is not a file, it needs to be separated from the test module path using a special separator, which is ::.
# !pytest models/test_train.py::TestSplitIntoTrainingAndTestingSets

# What is the correct command to run only the previously failing test test_on_six_rows() using node IDs?
# !pytest -k "SplitInto"
# That's correct! The -k flag is really useful, because it helps you select tests and test classes by typing only a unique part of its name. This saves a lot of typing, and you must admit that TestSplitIntoTrainingAndTestingSets is a horrendously long name! In your projects, you will often run tests with the node IDs and the -k flag because you are often not interested in running all tests, but only a subset depending on the functions you are currently working on.

## Mark a test class as expected to fail
"""
A new function model_test() is being developed and it returns the accuracy of a given linear regression model on a testing dataset. Test Driven Development (TDD) is being used to implement it. The procedure is: write tests first and then implement the function.

A test class TestModelTest has been created within the test module models/test_train.py. In the test class, there are two unit tests called test_on_linear_data() and test_on_one_dimensional_array(). But the function model_test() has not been implemented yet.

Throughout this exercise, pytest and numpy as np will be imported for you.
"""

# Run the tests in the test class TestModelTest in the IPython console. What is the outcome?
# The tests fail with NameError ...

# Mark the whole test class as "expected to fail"
@pytest.mark.xfail
class TestModelTest(object):
    def test_on_linear_data(self):
        test_input = np.array([[1.0, 3.0], [2.0, 5.0], [3.0, 7.0]])
        expected = 1.0
        actual = model_test(test_input, 2.0, 1.0)
        message = "model_test({0}) should return {1}, but it actually returned {2}".format(test_input, expected, actual)
        assert actual == pytest.approx(expected), message
        
    def test_on_one_dimensional_array(self):
        test_input = np.array([1.0, 2.0, 3.0, 4.0])
        with pytest.raises(ValueError) as exc_info:
            model_test(test_input, 1.0, 1.0)

# Add a reason for the expected failure
@pytest.mark.xfail(reason="Using TDD, model_test() has not yet been implemented")
class TestModelTest(object):
    def test_on_linear_data(self):
        test_input = np.array([[1.0, 3.0], [2.0, 5.0], [3.0, 7.0]])
        expected = 1.0
        actual = model_test(test_input, 2.0, 1.0)
        message = "model_test({0}) should return {1}, but it actually returned {2}".format(test_input, expected, actual)
        assert actual == pytest.approx(expected), message
        
    def test_on_one_dimensional_array(self):
        test_input = np.array([1.0, 2.0, 3.0, 4.0])
        with pytest.raises(ValueError) as exc_info:
            model_test(test_input, 1.0, 1.0)

## Mark a test as conditionally skipped
"""
In Python 2, there was a built-in function called xrange(). In Python 3, xrange() was removed. Therefore, if any test uses xrange(), it's going to fail with a NameError in Python 3.

Remember the function get_data_as_numpy_array()? You saw it in Chapter 2. It converted data in a preprocessed data file into a NumPy array.

range() has been deliberately replaced with the obsolete xrange() in the function. Evil laughter! But no worries, it will be changed back after you're done with this exercise.

You wrote a test called test_on_clean_file() for this function. This test currently resides in a test class TestGetDataAsNumpyArray inside the test module features/test_as_numpy.py.

pytest, numpy as np and get_data_as_numpy_array() has been imported for you.
"""

# Run the tests in the test class TestGetDataAsNumpyArray in the IPython console. What is the outcome?
# The test test_on_clean_file() fails with a NameError ...

# Import the sys module
import sys

class TestGetDataAsNumpyArray(object):
    # Mark as skipped if Python version is greater than 2.7
    @pytest.mark.skipif(sys.version_info > (2, 7), reason="Skipped on Python versions greater than 2.7")
    def test_on_clean_file(self):
        expected = np.array([[2081.0, 314942.0],
                             [1059.0, 186606.0],
                             [1148.0, 206186.0]
                             ]
                            )
        actual = get_data_as_numpy_array("example_clean_data.txt", num_columns=2)
        message = "Expected return value: {0}, Actual return value: {1}".format(expected, actual)
        assert actual == pytest.approx(expected), message

# Import the sys module
import sys

class TestGetDataAsNumpyArray(object):
    # Add a reason for skipping the test
    @pytest.mark.skipif(sys.version_info > (2, 7), reason="Works only on Python 2.7 or lower")
    def test_on_clean_file(self):
        expected = np.array([[2081.0, 314942.0],
                             [1059.0, 186606.0],
                             [1148.0, 206186.0]
                             ]
                            )
        actual = get_data_as_numpy_array("example_clean_data.txt", num_columns=2)
        message = "Expected return value: {0}, Actual return value: {1}".format(expected, actual)
        assert actual == pytest.approx(expected), message

## Reasoning in the test result report
"""
In the last exercises, you marked the test class TestModelTest in the test module models/test_train.py as expected to fail. You also marked the test test_on_clean_file() in the test class TestGetDataAsNumpyArray belonging to the test module features/test_as_numpy.py as skipped if the Python version is greater than 2.7.

In both cases, you provided a reason argument which detailed why they are expected to fail or skipped. In this exercise, your job is to make this reason show up in the test result report when you run all tests in the IPython console.

Feel free to run the !pytest command with different options and flags in the IPython console while doing the exercise.
"""

# What is the command that would only show the reason for expected failures in the test result report?
# !pytest -rx

# What is the command that would only show the reason for skipped tests in the test result report?
# !pytest -rs

# What is the command that would show the reason for both skipped tests and tests that are expected to fail in the test result report?
# !pytest -rsx
# Seems like you have become a pro at the pytest command line tool. Congratulations!

## Build failing
"""
In the GitHub repository of a Python package, you see the following badge:

What can you, as a user, conclude from this badge?
"""

# The package has bugs, which is either causing installation to error out or some of the unit tests in the test suite to fail.
# That's correct! Since a build failing badge is indicative of bugs, the maintainer of any package should strive to keep this badge green ("passing").

## What does code coverage mean?
"""
In a Github repository of a Python package, you see the following badge

What does it mean?
"""

# The test suite tests about 85% of the application code.
# You got that right! This brings us to the end of Chapter 3. Congratulations on coming this far! In the next Chapter, you are going to dive into advanced topics in unit testing and look at some data science specific unit testing tricks. See you there :-)

## Use a fixture for a clean data file
"""
In the video, you saw how the preprocess() function creates a clean data file.

The get_data_as_numpy_array() function takes the path to this clean data file as the first argument and the number of columns of data as the second argument. It returns a NumPy array holding the data.

In a previous exercise, you wrote the test test_on_clean_file() without using a fixture. That's bad practice! This time, you'll use the fixture clean_data_file(), which

creates a clean data file in the setup,
yields the path to the clean data file,
removes the clean data file in the teardown.

The contents of the clean data file that you will use for testing is printed in the IPython console.

pytest, os, numpy as np and get_data_as_numpy_array() have been imported for you.
"""

# Add a decorator to make this function a fixture
@pytest.fixture
def clean_data_file():
    file_path = "clean_data_file.txt"
    with open(file_path, "w") as f:
        f.write("201\t305671\n7892\t298140\n501\t738293\n")
    yield file_path
    os.remove(file_path)
    
# Pass the correct argument so that the test can use the fixture
def test_on_clean_file(clean_data_file):
    expected = np.array([[201.0, 305671.0], [7892.0, 298140.0], [501.0, 738293.0]])
    # Pass the clean data file path yielded by the fixture as the first argument
    actual = get_data_as_numpy_array(clean_data_file, 2)
    assert actual == pytest.approx(expected), "Expected: {0}, Actual: {1}".format(expected, actual) 

## Write a fixture for an empty data file
"""
When a function takes a data file as an argument, you need to write a fixture that takes care of creating and deleting that data file. This exercise will test your ability to write such a fixture.

get_data_as_numpy_array() should return an empty numpy array if it gets an empty data file as an argument. To test this behavior, you need to write a fixture empty_file() that does the following.

Creates an empty data file empty.txt relative to the current working directory in setup.
Yields the path to the empty data file.
Deletes the empty data file in teardown.

The fixture will be used by the test test_on_empty_file(), which is available for you to see in the script.

os, pytest, numpy as np and get_data_as_numpy_array have been imported for you.
"""

@pytest.fixture
def empty_file():
    # Assign the file path "empty.txt" to the variable
    file_path = "empty.txt"
    open(file_path, "w").close()
    # Yield the variable file_path
    yield file_path
    # Remove the file in the teardown
    os.remove(file_path)
    
def test_on_empty_file(self, empty_file):
    expected = np.empty((0, 2))
    actual = get_data_as_numpy_array(empty_file, 2)
    assert actual == pytest.approx(expected), "Expected: {0}, Actual: {1}".format(expected, actual)

# The test test_on_empty_file() was added to the test class TestGetDataAsNumpyArray, which lives in the test module tests/features/test_as_numpy.py. The fixture empty_file() was also written to this test module.
# Remembering that the current working directory in the IPython console is tests, run the test test_on_empty_file(). What is the outcome?
# The test passes
# Right you are! Notice that fixtures like empty_file() are quite reusable, since any function which accepts data files as arguments needs to be tested with an empty file.

## Fixture chaining using tmpdir
"""
The built-in tmpdir fixture is very useful when dealing with files in setup and teardown. tmpdir combines seamlessly with user defined fixture via fixture chaining.

In this exercise, you will use the power of tmpdir to redefine and improve the empty_file() fixture that you wrote in the last exercise and get some experience with fixture chaining.
"""

import pytest

@pytest.fixture
# Add the correct argument so that this fixture can chain with the tmpdir fixture
def empty_file(tmpdir):
    # Use the appropriate method to create an empty file in the temporary directory
    file_path = tmpdir.join("empty.txt")
    open(file_path, "w").close()
    yield file_path

# In what order will the setup and teardown of empty_file() and tmpdir be executed?
# Setup of tmpdir -> setup of empty_file() -> teardown of empty_file()
# Well ordered! Notice how you didn't have to write any teardown code to delete the empty file, because tmpdir is going to take care of that for us in its teardown step, which is executed last.

## Program a bug-free dependency
"""
In the video, row_to_list() was mocked. But preprocess() has another dependency convert_to_int(). Generally, its best to mock all dependencies of the function under test. It's your job to mock convert_to_int() in this and the following exercises.

The raw data file used in the test is printed in the IPython console. The second row "1,767565,112\n" is dirty, so row_to_list() will filter it out. The rest will be converted to lists and convert_to_int() will process the areas and prices.

The mocked convert_to_int() should process these areas and prices correctly. Here is the dictionary holding the correct return values.

{"1,801": 1801, "201,411": 201411, "2,002": 2002, "333,209": 333209, "1990": None, "782,911": 782911, "1,285": 1285, "389129": None}
"""

# Define a function convert_to_int_bug_free
def convert_to_int_bug_free(comma_separated_integer_string):
    # Assign to the dictionary holding the correct return values
    return_values = {"1,801": 1801,
                     "201,411": 201411,
                     "2,002": 2002,
                     "333,209": 333209,
                     "1990": None,
                     "782,911": 782911,
                     "1,285": 1285,
                     "389129": None,
                     }
    # Return the correct result using the dictionary return_values
    return return_values[comma_separated_integer_string]

## Mock a dependency
"""
Mocking helps us replace a dependency with a MagicMock() object. Usually, the MagicMock() is programmed to be a bug-free version of the dependency. To verify whether the function under test works properly with the dependency, you simply check whether the MagicMock() is called with the correct arguments and in the right order.

In the last exercise, you programmed a bug-free version of the dependency data.preprocessing_helpers.convert_to_int in the context of the test test_on_raw_data(), which applies preprocess() on a raw data file. The data file is printed out in the IPython console.

pytest, unittest.mock.call, preprocess raw_and_clean_data_file and convert_to_int_bug_free has been imported for you.
"""

# Add the correct argument to use the mocking fixture in this test
def test_on_raw_data(self, raw_and_clean_data_file, mocker):
    raw_path, clean_path = raw_and_clean_data_file

# Add the correct argument to use the mocking fixture in this test
def test_on_raw_data(self, raw_and_clean_data_file, mocker):
    raw_path, clean_path = raw_and_clean_data_file
    # Replace the dependency with the bug-free mock
    convert_to_int_mock = mocker.patch("data.preprocessing_helpers.convert_to_int",
                                       side_effect=convert_to_int_bug_free)
    preprocess(raw_path, clean_path)
    # Check if preprocess() called the dependency correctly
    assert convert_to_int_mock.call_args_list == [
        call("1,801"),
        call("201,411"),
        call("2,002"),
        call("333,209"),
        call("1990"),
        call("782,911"),
        call("1,285"),
        call("389129")
    ]
    with open(clean_path, "r") as f:
        lines = f.readlines()
    first_line = lines[0]
    assert first_line == "1801\\t201411\\n"
    second_line = lines[1]
    assert second_line == "2002\\t333209\\n" 

# The test that you wrote was written to the test class TestPreprocess in the test module data/test_preprocessing_helpers.py. The same test module also contains the test class TestConvertToInt.
# Run the tests in TestPreprocess and TestConvertToInt. Based on the test result report, which of the following is correct?
# Some tests for convert_to_int() fail ...
# Wow! You are turning into a seasoned tester! The results tell us that preprocess() is defined correctly and is bug-free. But one of its dependencies convert_to_int() has bugs. This kind of precise result is only possible using mocking.

## Testing on linear data
"""
The model_test() function, which measures how well the model fits unseen data, returns a quantity called 
 which is very difficult to compute in the general case. Therefore, you need to find special testing sets where computing 
 is easy.

One important special case is when the model fits the testing set perfectly. This happens when the testing set is perfectly linear. One such testing set is printed out in the IPython console for you.

In this special case, model_test() should return 1.0 if the model's slope and intercept match the testing set, because 1.0 is usually the highest possible value that r^2 can take.

Remember that for data points (xn, yn), the slope is (y2-y1) / (x2-x1) and the intercept is y1 - slope * x1.
"""

import numpy as np
import pytest
from models.train import model_test

def test_on_perfect_fit():
    # Assign to a NumPy array containing a linear testing set
    test_argument = np.array([[1.0, 3.0], [2.0, 5.0], [3.0, 7.0]])
    # Fill in with the expected value of r^2 in the case of perfect fit
    expected = 1.0
    # Fill in with the slope and intercept of the model
    actual = model_test(test_argument, slope=2.0, intercept=1.0)
    # Complete the assert statement
    assert actual == pytest.approx(expected), "Expected: {0}, Actual: {1}".format(expected, actual)

## Testing on circular data
"""
Another special case where it is easy to guess the value of r^2 is when the model does not fit the testing dataset at all. In this case, r^2 takes its lowest possible value 0.0.

The plot shows such a testing dataset and model. The testing dataset consists of data arranged in a circle of radius 1.0. The x and y co-ordinates of the data is shown on the plot. The model corresponds to a straight line y=0.

As one can easily see, the straight line does not fit the data at all. In this particular case, the value of r^2 is known to be 0.0.

Your job is to write a test test_on_circular_data() for the function model_test() that performs this sanity check. pytest, numpy as np, model_test, sin, cos and pi have been imported for you.
"""    

def test_on_circular_data(self):
    theta = pi/4.0
    # Assign to a NumPy array holding the circular testing data
    test_argument = np.array([[1.0, 0.0], [cos(theta), sin(theta)],
                              [0.0, 1.0],
                              [cos(3 * theta), sin(3 * theta)],
                              [-1.0, 0.0],
                              [cos(5 * theta), sin(5 * theta)],
                              [0.0, -1.0],
                              [cos(7 * theta), sin(7 * theta)]]
                             )
    # Fill in with the slope and intercept of the straight line
    actual = model_test(test_argument, slope=0.0, intercept=0.0)
    # Complete the assert statement
    assert actual == pytest.approx(0.0)

def test_on_circular_data(self):
    theta = pi/4.0
    # Assign to a NumPy array holding the circular testing data
    test_argument = np.array([[1.0, 0.0], [cos(theta), sin(theta)],
                              [0.0, 1.0],
                              [cos(3 * theta), sin(3 * theta)],
                              [-1.0, 0.0],
                              [cos(5 * theta), sin(5 * theta)],
                              [0.0, -1.0],
                              [cos(7 * theta), sin(7 * theta)]]
                             )
    # Fill in with the slope and intercept of the straight line
    actual = model_test(test_argument, slope=0.0, intercept=0.0)

def test_on_circular_data(self):
    theta = pi/4.0
    # Assign to a NumPy array holding the circular testing data
    test_argument = np.array([[1.0, 0.0], [cos(theta), sin(theta)],
                              [0.0, 1.0],
                              [cos(3 * theta), sin(3 * theta)],
                              [-1.0, 0.0],
                              [cos(5 * theta), sin(5 * theta)],
                              [0.0, -1.0],
                              [cos(7 * theta), sin(7 * theta)]]
                             )
    # Fill in with the slope and intercept of the straight line
    actual = model_test(test_argument, slope=0.0, intercept=0.0)
    # Complete the assert statement
    assert actual == pytest.approx(0.0)

# The tests test_on_perfect_fit() and test_on_circular_data() that you wrote in the last two exercises has been written to the test class TestModelTest in the test module models/test_train.py. Run the test class in the IPython console. What is the outcome?
# The sanity checks are all passing
# That's correct! model_test() seems to pass all the sanity checks. While this function is complicated and cannot be well tested, these sanity checks greatly reduce the chance of having a bug in it.

## Generate the baseline image
"""
In this exercise, you will get one step closer to the real thing. During this whole course, you've built a library of tests using a Python script and an IPython console. In real life, you're more likely to use an IDE (Integrated Development Environment), that lets you write scripts in the language you want, organize them into your directories, and execute shell commands. Basically, an IDE increases your productivity by gathering the most common activities of software development into a single application: writing source code, executing, and debugging.

Here, you can see the directory you've built on the left pane. The upper right pane is where you will write your Python scripts, and the bottom right pane is a shell console, which replaces the IPython console you've used so far.

Parts of an integrated development environment

In this exercise, you will test the function introduced in the video get_plot_for_best_fit_line() on another set of test arguments. Here is the test data.

1.0    3.0
2.0    8.0
3.0    11.0

The best fit line that the test will draw follows the equation y = 5x - 2. Two points, (1.0, 3.0) and (2.0, 8.0) will fall on the line. The point (3.0, 11.0) won't. The title of the plot will be "Test plot for almost linear data".

The test is called test_plot_for_almost_linear_data() and it's your job to complete the test and generate the baseline image. pytest, numpy as np and get_plot_for_best_fit_line has been imported for you.
"""

import pytest
import numpy as np

from visualization.plots import get_plot_for_best_fit_line

class TestGetPlotForBestFitLine(object):
    # Add the pytest marker which generates baselines and compares images
    @pytest.mark.mpl_image_compare
    def test_plot_for_almost_linear_data(self):
        slope = 5.0
        intercept = -2.0
        x_array = np.array([1.0, 2.0, 3.0])
        y_array = np.array([3.0, 8.0, 11.0])
        title = "Test plot for almost linear data"
        # Return the matplotlib figure returned by the function under test
        return get_plot_for_best_fit_line(slope, intercept, x_array, y_array, title)
            
# After completing the test_plots.py script, run the following command in the shell:
# pytest --mpl-generate-path /home/repl/workspace/project/tests/visualization/baseline -k "test_plot_for_almost_linear_data"

## Run the tests for the plotting function
"""
Shortly after the baseline image was generated, one of your colleagues modified the plotting function. You have to run the tests in order to check whether the function still works as expected.

Remember the following:

The tests were housed in a test class TestGetPlotForBestFitLine in the test module visualization/test_plots.py. You can specify this test class in the pytest command by either using its node ID or the -k command line flag.
To ensure plots are compared to the baseline during testing, the pytest command must include a special command line flag that comes from the pytest-mpl package.
"""

# This is the correct command. To submit, click "Run this file" followed by "Submit Answer". 
# pytest -k "TestGetPlotForBestFitLine" --mpl

## Fix the plotting function
"""
In the last exercise, pytest saved the baseline images, actual images, and images containing the pixelwise difference in a temporary folder. The difference image for one of the tests test_on_almost_linear_data() is shown below.

The black areas are where the actual image and the baseline matches. The white areas are where they don't match.

This clearly tells us that something is wrong with the axis labels. Take a look at the plots section to see the baseline (plot 1/2) and the actual plot (plot 2/2). Based on that, it's your job to fix the plotting function.
"""

import matplotlib.pyplot as plt
import numpy as np

def get_plot_for_best_fit_line(slope, intercept, x_array, y_array, title):
    fig, ax = plt.subplots()
    ax.plot(x_array, y_array, ".")
    ax.plot([0, np.max(x_array)], [intercept, slope * np.max(x_array) + intercept], "-")
    # Fill in with axis labels so that they match the baseline
    ax.set(xlabel="area (square feet)", ylabel="price (dollars)", title=title)
    return fig

# Now that you have fixed the function, run all the tests in the tests directory, remembering that the current working directory in the IPython console is tests. What is the outcome?
# All 25 tests pass
# That is amazing! The linear regression project is now well-tested with 25 tests and they are all passing, thanks to your efforts throughout the course. With the skills and techniques you learned here, you can now go and test your own data science projects!

## Congratulations