*COURSE*

# Introduction to Julia

[Click to view website →](https://app.datacamp.com/learn/courses/introduction-to-julia)

---

## Course Description

Julia is a new and exciting programming language designed from its foundations to be the ideal language for scientific computing, machine learning, and data mining. This course will give you a running start in your journey with Julia.


We'll start from the very basics of Julia, so you can follow along if you have never programmed before. By the end of this course, you'll be familiar with coding in Julia, understanding the basics, including data types and structures, the functions and packages, and how to use DataFrames to work with tabular data.

---

- Julia
- 4 hours

---

## Chapters
1. Julia basics

Take your first steps towards coding with Julia and learn what makes this language unique. Learn about variables and data types, and perform simple calculations using Julia scripts and the interactive console.

2. Data structures

Learn how to process text data using strings and how to use arrays to process large amounts of data quickly and efficiently. Julia's inbuilt arrays are both powerful and easy to use.

3. Functions and packages

Learn to write code that can make its own decisions using conditional expressions and write functions so you can reuse your code. Plus, learn Julia's powerful function features like broadcasting and multiple dispatch.

4. DataFrames

The DataFrames package is the definitive way to work with tabular data in Julia. You'll use this package to load CSV files and analyze and process this data to get the insights you need.

### 1. Julia basics
#### Getting started with Julia

1. Getting started with Julia
00:00 - 00:08
Hi, and welcome to this introductory course on programming with Julia. I'm James, and I'll be leading you through this course.

2. What is Julia?
00:08 - 00:29
Julia is an open-source programming language, and compared to other languages used in data science, it is much newer. Julia is a general-purpose programming language that can be used for almost anything we would like to program. However, Julia was designed to be the ideal language for scientific computing, machine learning, and data mining, so these areas are where it shines.

3. Why create Julia?
00:29 - 00:58
Julia was created by Jeff Bezanson, Stefan Karpinski, Viral B. Shah, and Alan Edelman. It was made to be simple to learn but deep enough that you should never outgrow it. It was designed to incorporate the best features of other scientific programming languages. To be as good at general programming as Python, as good at statistics as R, and as good at linear algebra as MATLAB. All while being as fast as compiled languages like C.

1 https://julialang.org/blog/2012/02/why-we-created-julia
4. This course
00:58 - 01:20
In this course, we will cover the basics of the Julia language and won't assume you know any other programming languages. We will cover all the syntax and concepts you need to know so you can get started working with data in Julia. If you are already familiar with a language like Python, MATLAB, or R, you will learn the basics you need to translate your programming knowledge into Julia.

5. Installing Julia
01:20 - 01:33
If you want to install Julia on your own computer, you can download it from this link. However, you do not need to have it installed for this course. You can run Julia online from a browser here.

6. Scripts vs. the console
01:33 - 01:47
There are two different ways to run Julia code. The first way is using the console. On the DataCamp platform, you will see the console under the output tab. Here you can type commands and immediately see the results.

7. Scripts vs. the console
01:47 - 01:55
Sometimes we don't want to see the results because they are too long. Then we can use a semicolon to hide them.

8. Scripts vs. the console
01:55 - 02:26
The other way to run Julia code is by writing scripts. You will see a script located in the script-dot-jl tab. A script is a sequence of Julia commands inside a file. You can add many commands to this file, and they will be executed in order from top to bottom. Using a script means you can keep all the commands saved and reuse them repeatedly instead of manually typing them into the console each time. In the exercises you will often be working on a script.

9. Simple calculations and printing
02:26 - 02:50
When we write our code into a script we will not see the output from each line like in the console. Instead we can choose which lines to print using the println function. We write println and inside the parentheses we place whatever we want the script to print. In this example we print the number two and the sum of one plus two.

10. Comments
02:50 - 03:02
Inside this script, there are two lines that begin with a hash symbol. These are called comments, and allow you to annotate and add notes to your code. Julia will ignore these lines.

11. Comments
03:02 - 03:08
If we don't use the hash symbols, it will cause an error as Julia tries to run these lines as code.

12. Multi-line comments
03:08 - 03:22
We can spread comments over multiple lines using the multi-line comment syntax. We start the comment with hash-equals, with no space between them, and end it with equals-hash, again without a space.

13. Multi-line comments
03:22 - 03:27
Everything between these symbols will be ignored by Julia.

14. Let's practice!
03:27 - 03:31
Now, let's get started coding in Julia.

#### Using the console
The console to the right allows you to interactively run lines of Julia code. You can type a command and press enter to run it.

Use the console to find the difference between 39221 and 1892.

Instructions

Possible answers
- 38229
- 37329 ✅
- 37229
- 38339

> Great work! Using the console can be useful when you are testing whether some code works or doing a very quick calculation. However, once you close the console, all of the lines of code you wrote will be gone. If you think you will want to run the code multiple times, it is better to have these lines saved in a script.

#### Julia as a calculator
Writing your code into a script instead of the console means you will have it saved for later. This allows you to rerun the code, modify it and run it with different inputs, or check for any mistakes in the code you used earlier.

Here you are writing a script to keep track of your weekly hours worked. You have been adding to it every day to get a total.

Instructions

- Add the comment `Hours worked this week` to line 2 of the script.
- Add all the numbers 39, 37.5, 40, and 42 together to calculate your total hours worked.

```{julia}
# Add the comment on the line below
# Hours worked this week

# Add the hours worked
39 + 37.5 + 40 + 42
```

> Nice work! When you hit the submit answer button, this script was run, and Julia calculated the total of the numbers. However, you cannot see the result of these numbers like you did using the console. To see the results of the calculation, you need to use a print statement.

#### Printing
When working with scripts, print statements can help you see the output of your code. You can use one or more print statements in the same script, and everything inside the statements will be printed to the console.

You are working on the same script from the previous exercise to track your weekly working hours. Now you want to add printing to see the output of your total hours worked this month.

Instructions

- Print the number of weeks worked this month, which is 4.
- Print the sum of hours worked each week, which is 39 + 37.5 + 40 + 42.

```{julia}
# Hours worked this week

# Print the number of weeks worked
println(4)

# Print the sum of the hours worked each week
println(39 + 37.5 + 40 + 42)
```

> Great! Printing is really important to get output from your scripts, but it can also be really useful to check for any mistakes in your code, or to track how fast Julia is progressing through your script.

#### Variables

1. Variables
00:00 - 00:08
In the previous exercises you were calculating sums and printing the answers all at once in a single line of code.

2. Assigning variables
00:08 - 00:36
In order to carry out more complex tasks, we need to use variables and split our calculations across multiple lines. Variables allow us to store values under a name. In this example, we store the value of three under the variable name x. After this, we can print x and see the value assigned to it. We can also perform further calculations with x, such as multiplying it by two and printing the results of this.

3. Calculating with variables
00:36 - 01:30
Using variables makes our programs neater and more reusable. Say we wanted to write a script to find our average speed when going for a run. If we are training to get faster, we might want to use the script repeatedly to track our progress. In this script, we assign values to a variable named distance, which stores the distance we ran in meters, and a variable named time, which holds the number of minutes we took to complete the run. We want to calculate our average speed over the run in miles per hour, so we convert the distance to miles and the time to hours and assign these to new variables. Finally, we print our speed for this run. Keeping all of this in a script means we can calculate our new speed when we have new run times. Perhaps after a few weeks, we have gotten a bit faster, and now we can complete the same distance in less time.

4. Calculating with variables
01:30 - 01:37
We can return to the script, update the time taken and rerun it to find our new average speed.

5. Naming variables
01:37 - 02:07
In the previous script, we named our variables using Julia's best practices. In Julia, variables should all start with a letter, although we can use Unicode symbols too. When the variable name is a word, the letters should all be lowercase, and when the name is multiple words, we should separate them with an underscore. We can leave out the underscores if the name is still readable. Variable names can also use numbers after the first letter, but they cannot begin with a number.

6. More operations using variables
02:07 - 03:09
Using variables allows us to run more complex operations. In the previous exercises, we used the addition and subtraction operators. In this lesson, we have used multiply and divide. Another primary operator we should know is the power operator, which raises a number to a power. Sometimes we want to use a few of these operations in a single expression. The operations are resolved in the same as the order used in mathematics. However, we can use brackets to control the order in which the operations are resolved. For example, we could simplify our script to calculate our running speed to this expression. We divide the distance by one-thousand-six-hundred-and-nine and divide this by the time divided by sixty. The expressions inside the brackets are computed first, giving the same answer as before. If we didn't use the brackets in this expression, we would divide the distance by each of the three other quantities, which wouldn't give the correct answer.

7. Let's practice!
03:09 - 03:14
Now, let's practice using variables in the exercises.

#### Valid variable names
Some variable names will cause your code to raise an error, while others are not in keeping with the recommended form. Identifying and replacing these with better names that adhere to best practices will help you write more readable and neat code.

Instructions

Which of the following variable names adhere to best practices?

- β, speed, hours_worked, time_day_2

> Great work! It is best practice to name your variables according to these rules. This will make your code more readable and easier for you and others to use.

#### Assigning variables
Assigning variables means you can write more complex code. Using variables also makes the code you write more interpretable since you are labeling the pieces of information rather than having them as raw pieces of data.

You are writing a script to calculate your average running pace from your last two runs. You want to assign your run times and distances to variables so you can process them later.

Instructions

- Assign the value 4500 to the variable `monday_distance`.
- Assign the value 28.9 to the variable `monday_time`.
- Assign the value 6000 to the variable `wednesday_distance`.
- Assign the value 37.1 to the variable `wednesday_time`.

```{julia}
# Create variable monday_distance
monday_distance = 4500

# Create variable monday_time
monday_time = 28.9

# Create variable wednesday_distance
wednesday_distance = 6000

# Create variable wednesday_time
wednesday_time = 37.1
```

> Assignment complete! With these values assigned, you can write code to analyze this data without having to know the actual numbers. This means the code you write will be more general and easier to use again in the future.

#### Calculating with variables
Now that you have assigned your run times and distances to variables, you want to process these times to calculate your average speed over the two runs.

You can calculate the speed using:

MathJax Original Source - TeX Commands
\text{speed} = \frac{\text{distance}}{\text{time}}

The variables monday_distance, wednesday_distance, monday_time, and wednesday_time are all available in your environment.

Instructions

- Calculate the total miles run by adding monday_distance and wednesday_distance and dividing their total by 1609.
- Calculate the total number of hours spent running by adding monday_time and wednesday_time and dividing their total by 60.
- Calculate the average run speed across the two runs in miles per hour.
- Print the average run speed.
