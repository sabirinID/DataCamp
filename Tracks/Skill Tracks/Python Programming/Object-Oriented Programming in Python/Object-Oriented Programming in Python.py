## OOP termininology
# That was a lot of terminology at once -- classes, objects, methods, attributes… Before you start writing your own object-oriented code, make sure you have a solid grasp on the main concepts of OOP.

# True
# Attributes
# Methods
# Encapsulation
# Great job! Classes and objects both have attributes and methods, but the difference is that a class is an abstract template, while an object is a concrete representation of a class.

## Exploring object interface
"""
The best way to learn how to write object-oriented code is to study the design of existing classes. You've already learned about exploration tools like type() and dir().

Another important function is help(): calling help(x) in the console will show the documentation for the object or class x.

Most real world classes have many methods and attributes, and it is easy to get lost, so in this exercise, you will start with something simpler. We have defined a class, and created an object of that class called mystery. Explore the object in the console using the tools that you learned.
"""

# What class does the mystery object have?
# __main__.Employee

# Print the mystery employee's name
print(mystery.name)

# Print the mystery employee's salary
print(mystery.salary)

# Give the mystery employee a raise of $2500
mystery.give_raise(2500)

# Print the salary again
print(mystery.salary)

## Understanding class definitions
"""
Objects and classes consist of attributes (storing the state) and methods (storing the behavior).

Before you get to writing your own classes, you need to understand the basic structure of the class, and how attributes in the class definition relate to attributes in the object. In this exercise, you have a class with one method and one attribute, as well as the object of that class.
"""

# Arrange the code blocks in the order that will produce the output 6 when run.
# class, def, self.count, mc, mc.set_count, mc.count, print

## Create your first class
"""
Time to write your first class! In this exercise, you'll start building the Employee class you briefly explored in the previous lesson. You'll start by creating methods that set attributes, and then add a few methods that manipulate them.

As mentioned in the first video, an object-oriented approach is most useful when your code involves complex interactions of many objects. In real production code, classes can have dozens of attributes and methods with complicated logic, but the underlying structure is the same as with the most simple class.

Your classes in this course will only have a few attributes and short methods, but the organizational principles behind the them will be directly translatable to more complicated code.
"""

# Create an empty class Employee
class Employee :
    pass

# Create an object emp of class Employee 
emp = Employee()

# Include a set_name method
class Employee:
  
  def set_name(self, new_name):
    self.name = new_name
  
# Create an object emp of class Employee  
emp = Employee()

# Use set_name() on emp to set the name of emp to 'Korel Rossi'
emp.set_name('Korel Rossi')

# Print the name of emp
print(emp.name)

class Employee:
  
  def set_name(self, new_name):
    self.name = new_name
  
  # Add set_salary() method
  def set_salary(self, new_salary):
      self.salary = new_salary
  
  
# Create an object emp of class Employee  
emp = Employee()

# Use set_name to set the name of emp to 'Korel Rossi'
emp.set_name('Korel Rossi')

# Set the salary of emp to 50000
emp.set_salary(50000)

## Using attributes in class definition
"""
In the previous exercise, you defined an Employee class with two attributes and two methods setting those attributes. This kind of method, aptly called a setter method, is far from the only possible kind. Methods are functions, so anything you can do with a function, you can also do with a method. For example, you can use methods to print, return values, make plots, and raise exceptions, as long as it makes sense as the behavior of the objects described by the class (an Employee probably wouldn't have a pivot_table() method).

In this exercise, you'll go beyond the setter methods and learn how to use existing class attributes to define new methods. The Employee class and the emp object from the previous exercise are in your script pane.
"""

class Employee:
    def set_name(self, new_name):
        self.name = new_name

    def set_salary(self, new_salary):
        self.salary = new_salary 
  
emp = Employee()
emp.set_name('Korel Rossi')
emp.set_salary(50000)

# Print the salary attribute of emp
print(emp.salary)

# Increase salary of emp by 1500
emp.salary = emp.salary + 1500

# Print the salary attribute of emp again
print(emp.salary)

class Employee:
    def set_name(self, new_name):
        self.name = new_name

    def set_salary(self, new_salary):
        self.salary = new_salary 

    # Add a give_raise() method with raise amount as a parameter
    def give_raise(self, given_rais_sal):
        self.salary += given_rais_sal

emp = Employee()
emp.set_name('Korel Rossi')
emp.set_salary(50000)

print(emp.salary)
emp.give_raise(1500)
print(emp.salary)

class Employee:
    def set_name(self, new_name):
        self.name = new_name

    def set_salary(self, new_salary):
        self.salary = new_salary 

    def give_raise(self, amount):
        self.salary = self.salary + amount

    # Add monthly_salary method that returns 1/12th of salary attribute
    def monthly_salary(self): 
        return self.salary/12.0
    
emp = Employee()
emp.set_name('Korel Rossi')
emp.set_salary(50000)

# Get monthly salary of emp and assign to mon_sal
mon_sal = emp.monthly_salary()

# Print mon_sal
print(mon_sal)

## Correct use of __init__
"""
Python allows you to run custom code - for example, initializing attributes - any time an object is created: you just need to define a special method called __init__(). Use this exercise to check your understanding of __init__() mechanics!

Which of the code blocks will NOT return an error when run?
"""

# 2
# Exactly right! The printout, however, will be 5, while the person writing the code likely expected 0. That's why documentation is important!

## Add a class constructor
"""
In this exercise, you'll continue working on the Employee class. Instead of using the methods like set_salary() that you wrote in the previous lesson, you will introduce a constructor that assigns name and salary to the employee at the moment when the object is created.

You'll also create a new attribute -- hire_date -- which will not be initialized through parameters, but instead will contain the current date.

Initializing attributes in the constructor is a good idea, because this ensures that the object has all the necessary attributes the moment it is created.
"""

class Employee:
    # Create __init__() method
    def __init__(self, name, salary=0):
        # Create the name and salary attributes
        self.name = name
        self.salary = salary
    
    # From the previous lesson
    def give_raise(self, amount):
        self.salary += amount

    def monthly_salary(self):
        return self.salary/12
        
emp = Employee("Korel Rossi")
print(emp.name)
print(emp.salary)

class Employee:
  
    def __init__(self, name, salary=0):
        self.name = name
        # Modify code below to check if salary is positive
        if salary >= 0 : 
            self.salary = salary 
        else: 
            self.salary = 0 
            print("Invalid salary!")
   # ...Other methods omitted for brevity ...
      
emp = Employee("Korel Rossi", -1000)
print(emp.name)
print(emp.salary)

# Import datetime from datetime
from  datetime import datetime

class Employee:
    
    def __init__(self, name, salary=0):
        self.name = name
        if salary > 0:
          self.salary = salary
        else:
          self.salary = 0
          print("Invalid salary!")
          
        # Add the hire_date attribute and set it to today's date
        self.hire_date = datetime.today()
        
   # ...Other methods omitted for brevity ...
      
emp = Employee("Korel Rossi", -1000)
print(emp.name)
print(emp.salary)

## Write a class from scratch
"""
You are a Python developer writing a visualization package. For any element in a visualization, you want to be able to tell the position of the element, how far it is from other elements, and easily implement horizontal or vertical flip .

The most basic element of any visualization is a single point. In this exercise, you'll write a class for a point on a plane from scratch.
"""

# For use of np.sqrt
import numpy as np

class Point:
    """ A point on a 2D plane
    
   Attributes
    ----------
    x : float, default 0.0. The x coordinate of the point        
    y : float, default 0.0. The y coordinate of the point
    """
    def __init__(self, x=0.0, y=0.0):
      self.x = x
      self.y = y
      
    def distance_to_origin(self):
      """Calculate distance from the point to the origin (0,0)"""
      return np.sqrt(self.x ** 2 + self.y ** 2)
    
    def reflect(self, axis):
      """Reflect the point with respect to x or y axis."""
      if axis == "x":
        self.y = - self.y
      elif axis == "y":
        self.x = - self.x
      else:
        print("The argument axis only accepts values 'x' and 'y'!")

## Class-level attributes
"""
Class attributes store data that is shared among all the class instances. They are assigned values in the class body, and are referred to using the ClassName. syntax rather than self. syntax when used in methods.

In this exercise, you will be a game developer working on a game that will have several players moving on a grid and interacting with each other. As the first step, you want to define a Player class that will just move along a straight line. Player will have a position attribute and a move() method. The grid is limited, so the position of Player will have a maximal value.
"""

# Create a Player class
class Player: 
    
    # Class Attributes 
    MAX_POSITION = 10 
    
    def __init__(self):
        self.position = 0

# Print Player.MAX_POSITION       
print(Player.MAX_POSITION)

# Create a player p and print its MAX_POSITITON
p = Player()
print(p.MAX_POSITION)

class Player:
    MAX_POSITION = 10
    
    def __init__(self):
        self.position = 0

    # Add a move() method with steps parameter
    def move(self, steps):
        
        if(self.position + steps < Player.MAX_POSITION):
            self.position += steps
        else: 
            self.position = Player.MAX_POSITION 

    # This method provides a rudimentary visualization in the console    
    def draw(self):
        drawing = "-" * self.position + "|" +"-"*(Player.MAX_POSITION - self.position)
        print(drawing)

p = Player(); p.draw()
p.move(4); p.draw()
p.move(5); p.draw()
p.move(3); p.draw()

## Changing class attributes
"""
You learned how to define class attributes and how to access them from class instances. So what will happen if you try to assign another value to a class attribute when accessing it from an instance? The answer is not as simple as you might think!

The Player class from the previous exercise is pre-defined. Recall that it has a position instance attribute, and MAX_SPEED and MAX_POSITION class attributes. The initial value of MAX_SPEED is 3.
"""

# Create Players p1 and p2
p1 = Player()
p2 = Player()

print("MAX_SPEED of p1 and p2 before assignment:")
# Print p1.MAX_SPEED and p2.MAX_SPEED
print(p1.MAX_SPEED)
print(p2.MAX_SPEED)

# Assign 7 to p1.MAX_SPEED
p1.MAX_SPEED = 7

print("MAX_SPEED of p1 and p2 after assignment:")
# Print p1.MAX_SPEED and p2.MAX_SPEED
print(p1.MAX_SPEED)
print(p2.MAX_SPEED)

print("MAX_SPEED of Player:")
# Print Player.MAX_SPEED
print(Player.MAX_SPEED)

p1, p2 = Player(), Player()

print("MAX_SPEED of p1 and p2 before assignment:")
# Print p1.MAX_SPEED and p2.MAX_SPEED
print(p1.MAX_SPEED)
print(p2.MAX_SPEED)

# ---MODIFY THIS LINE--- 
Player.MAX_SPEED = 7

print("MAX_SPEED of p1 and p2 after assignment:")
# Print p1.MAX_SPEED and p2.MAX_SPEED
print(p1.MAX_SPEED)
print(p2.MAX_SPEED)

print("MAX_SPEED of Player:")
# Print Player.MAX_SPEED
print(Player.MAX_SPEED)

## Alternative constructors
"""
Python allows you to define class methods as well, using the @classmethod decorator and a special first argument cls. The main use of class methods is defining methods that return an instance of the class, but aren't using the same code as __init__().

For example, you are developing a time series package and want to define your own class for working with dates, BetterDate. The attributes of the class will be year, month, and day. You want to have a constructor that creates BetterDate objects given the values for year, month, and day, but you also want to be able to create BetterDate objects from strings like 2020-04-30.

You might find the following functions useful:

.split("-") method will split a string at"-" into an array, e.g. "2020-04-30".split("-") returns ["2020", "04", "30"],
int() will convert a string into a number, e.g. int("2019") is 2019 .
"""

class BetterDate:    
    # Constructor
    def __init__(self, year, month, day):
      # Recall that Python allows multiple variable assignments in one line
      self.year, self.month, self.day = year, month, day
    
    # Define a class method from_str
    @classmethod
    def from_str(cls, datestr):
        # Split the string at "-" and convert each part to integer
        parts = datestr.split("-")
        year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
        # Return the class instance
        return BetterDate(year, month, day)
        
bd = BetterDate.from_str('2020-04-30')   
print(bd.year)
print(bd.month)
print(bd.day)

# import datetime from datetime
from datetime import datetime

class BetterDate:
    def __init__(self, year, month, day):
      self.year, self.month, self.day = year, month, day
      
    @classmethod
    def from_str(cls, datestr):
        year, month, day = map(int, datestr.split("-"))
        return cls(year, month, day)
      
    # Define a class method from_datetime accepting a datetime object
    @classmethod
    def from_datetime(cls,datetime):
        return BetterDate(datetime.year, datetime.month, datetime.day)

# You should be able to run the code below with no errors: 
today = datetime.today()     
bd = BetterDate.from_datetime(today)   
print(bd.year)
print(bd.month)
print(bd.day)

## Understanding inheritance
"""
Inheritance is a powerful tool of object-oriented languages that allows you to customize functionality of existing classes without having to re-implement methods from scratch.

In this exercise you'll check your understanding of the basics of inheritance. In the questions, we'll make use of the following two classes:

class Counter:
    def __init__(self, count):
       self.count = count

    def add_counts(self, n):
       self.count += n

class Indexer(Counter):
   pass
"""

# False
# Every, If ... error, Inheritance

## Create a subclass
"""
The purpose of child classes -- or sub-classes, as they are usually called - is to customize and extend functionality of the parent class.

Recall the Employee class from earlier in the course. In most organizations, managers enjoy more privileges and more responsibilities than a regular employee. So it would make sense to introduce a Manager class that has more functionality than Employee.

But a Manager is still an employee, so the Manager class should be inherited from the Employee class.
"""

class Employee:
  MIN_SALARY = 30000    

  def __init__(self, name, salary=MIN_SALARY):
      self.name = name
      if salary >= Employee.MIN_SALARY:
        self.salary = salary
      else:
        self.salary = Employee.MIN_SALARY
        
  def give_raise(self, amount):
      self.salary += amount      
        
# Define a new class Manager inheriting from Employee
class Manager(Employee):
    pass 

# Define a Manager object
mng = Manager("Debbie Lashko", 86500)

# Print mng's name
print(mng.name)

class Employee:
  MIN_SALARY = 30000    

  def __init__(self, name, salary=MIN_SALARY):
      self.name = name
      if salary >= Employee.MIN_SALARY:
        self.salary = salary
      else:
        self.salary = Employee.MIN_SALARY
  def give_raise(self, amount):
    self.salary += amount      
        
# MODIFY Manager class and add a display method
class Manager(Employee):
  
  def display(self):
      print("Manager "+ self.name)

mng = Manager("Debbie Lashko", 86500)
print(mng.name)

# Call mng.display()
mng.display()

## Method inheritance
"""
Inheritance is powerful because it allows us to reuse and customize code without rewriting existing code. By calling methods of the parent class within the child class, we reuse all the code in those methods, making our code concise and manageable.

In this exercise, you'll continue working with the Manager class that is inherited from the Employee class. You'll add new data to the class, and customize the give_raise() method from Chapter 1 to increase the manager's raise amount by a bonus percentage whenever they are given a raise.

A simplified version of the Employee class, as well as the beginning of the Manager class from the previous lesson is provided for you in the script pane.
"""

class Employee:
    def __init__(self, name, salary=30000):
        self.name = name
        self.salary = salary

    def give_raise(self, amount):
        self.salary += amount

        
class Manager(Employee):
  # Add a constructor 
    def __init__(self, name, salary=50000, project= None):

        # Call the parent's constructor   
        Employee.__init__(self, name, salary)

        # Assign project attribute
        self.project = project

    def display(self):
        print("Manager ", self.name)

class Employee:
    def __init__(self, name, salary=30000):
        self.name = name
        self.salary = salary

    def give_raise(self, amount):
        self.salary += amount

        
class Manager(Employee):
    def display(self):
        print("Manager ", self.name)

    def __init__(self, name, salary=50000, project=None):
        Employee.__init__(self, name, salary)
        self.project = project

    # Add a give_raise method
    def give_raise(self, amount, bonus=1.05):
        
        Employee.give_raise(self, amount * bonus)
    

mngr = Manager("Ashta Dunbar", 78500)
mngr.give_raise(1000)
print(mngr.salary)
mngr.give_raise(2000, bonus=1.03)
print(mngr.salary)

## Inheritance of class attributes

"""In the beginning of this chapter, you learned about class attributes and methods that are shared among all the instances of a class. How do they work with inheritance?

In this exercise, you'll create subclasses of the Player class from the first lesson of the chapter, and explore the inheritance of class attributes and methods.

The Player class has been defined for you. Recall that the Player class had two class-level attributes: MAX_POSITION and MAX_SPEED, with default values 10 and 3.
"""

# Create a Racer class and set MAX_SPEED to 5
class Racer(Player):
    MAX_SPEED = 5
    
# Create a Player and a Racer objects
p = Player()
r = Racer()

print("p.MAX_SPEED = ", p.MAX_SPEED)
print("r.MAX_SPEED = ", r.MAX_SPEED)

print("p.MAX_POSITION = ", p.MAX_POSITION)
print("r.MAX_POSITION = ", r.MAX_POSITION)

# Which of the following statements about inheritance of class attributes is correct?
# Class attributes CAN be inherited, and ... CAN be overwritten ...
# Correct! But notice that the value of MAX_SPEED in Player was not affected by the changes to the attribute of the same name in Racer.

## Customizing a DataFrame
"""
In your company, any data has to come with a timestamp recording when the dataset was created, to make sure that outdated information is not being used. You would like to use pandas DataFrames for processing data, but you would need to customize the class to allow for the use of timestamps.

In this exercise, you will implement a small LoggedDF class that inherits from a regular pandas DataFrame but has a created_at attribute storing the timestamp. You will then augment the standard to_csv() method to always include a column storing the creation date.

Tip: all DataFrame methods have many parameters, and it is not sustainable to copy all of them for each method you're customizing. The trick is to use variable-length arguments *args and **kwargsto catch all of them.
"""

# Import pandas as pd
import pandas as pd

# Define LoggedDF inherited from pd.DataFrame and add the constructor
class LoggedDF( pd.DataFrame ):
    
    def __init__(self, *args, **kwargs):
        
        pd.DataFrame.__init__(self,*args, **kwargs)
        self.created_at = datetime.today()
    
    
ldf = LoggedDF({"col1": [1,2], "col2": [3,4]})
print(ldf.values)
print(ldf.created_at)

# Import pandas as pd
import pandas as pd

# Define LoggedDF inherited from pd.DataFrame and add the constructor
class LoggedDF(pd.DataFrame):
  
  def __init__(self, *args, **kwargs):
    pd.DataFrame.__init__(self, *args, **kwargs)
    self.created_at = datetime.today()
    
  def to_csv(self, *args, **kwargs):
    # Copy self to a temporary DataFrame
    temp = self.copy()
    
    # Create a new column filled with self.created at
    temp["created_at"] = self.created_at
    
    # Call pd.DataFrame.to_csv on temp with *args and **kwargs
    pd.DataFrame.to_csv(temp, *args, **kwargs)

## Overloading equality
"""
When comparing two objects of a custom class using ==, Python by default compares just the object references, not the data contained in the objects. To override this behavior, the class can implement the special __eq__() method, which accepts two arguments -- the objects to be compared -- and returns True or False. This method will be implicitly called when two objects are compared.

The BankAccount class from the previous chapter is available for you in the script pane. It has one attribute, balance, and a withdraw() method. Two bank accounts with the same balance are not necessarily the same account, but a bank account usually has an account number, and two accounts with the same account number should be considered the same.
"""

class BankAccount:
     # MODIFY to initialize a number attribute
    def __init__(self, number, balance=0):
        self.balance = balance
        self.number = number
      
    def withdraw(self, amount):
        self.balance -= amount 

    # Define __eq__ that returns True if the number attributes are equal 
    def __eq__(self, other):
        return self.number == other.number    
    
acct1 = BankAccount(123, 1000)
acct2 = BankAccount(123, 1000)
acct3 = BankAccount(456, 1000)
print(acct1 == acct2)
print(acct1 == acct3)
    
## Checking class equality
"""
In the previous exercise, you defined a BankAccount class with a number attribute that was used for comparison. But if you were to compare a BankAccount object to an object of another class that also has a number attribute, you could end up with unexpected results.

For example, consider two classes


class Phone:
  def __init__(self, number):
     self.number = number

  def __eq__(self, other):
    return self.number == \
          other.number

pn = Phone(873555333)

class BankAccount:
  def __init__(self, number):
     self.number = number

  def __eq__(self, other):
    return self.number == \
           other.number

acct = BankAccount(873555333)
Running acct == pn will return True, even though we're comparing a phone number with a bank account number.

It is good practice to check the class of objects passed to the __eq__() method to make sure the comparison makes sense.
"""

class BankAccount:
    def __init__(self, number, balance=0):
        self.number, self.balance = number, balance
      
    def withdraw(self, amount):
        self.balance -= amount 

    # MODIFY to add a check for the type()
    def __eq__(self, other):
        return (self.number == other.number) and (type(self) == type(other))    

acct = BankAccount(873555333)      
pn = Phone(873555333)
print(acct == pn)

## Comparison and inheritance
"""
What happens when an object is compared to an object of a child class? Consider the following two classes:

class Parent:
    def __eq__(self, other):
        print("Parent's __eq__() called")
        return True

class Child(Parent):
    def __eq__(self, other):
        print("Child's __eq__() called")
        return True
The Child class inherits from the Parent class, and both implement the __eq__() method that includes a diagnostic printout.
"""

# Which __eq__() method will be called when the following code is run?

p = Parent()
c = Child()

p == c 

# Feel free to experiment in the console -- the classes have already been defined for you.
# Child's __eq__() method will be called
# Correct! Python always calls the _child's_ __eq__() method when comparing a child object to a parent object.

## String formatting review
"""
Before you start defining custom string representations for objects, make sure you are comfortable working with strings and formatting them. If you need a refresher, take a minute to look through the official Python tutorial on string formatting.

In this exercise, consider the following code

my_num = 5
my_str = "Hello"

f = ...
print(f)
where the definition for f is missing.

Here are a few possible variants for the definition of f:

1.


f = "my_num is {0}, and my_str is {1}.".format(my_num, my_str)
 
2.


f = "my_num is {}, and my_str is \"{}\".".format(my_num, my_str)
3.


f = "my_num is {n}, and my_str is '{s}'.".format(n=my_num, s=my_str)
4.


f = "my_num is {my_num}, and my_str is '{my_str}'.".format()
"""

# Pick the definition of f that will make the code above print exactly the following:

my_num is 5, and my_str is "Hello".

# There is only one correct answer! Feel free to use the script pane or console to experiment.
# 2
# Great work! To recap: to format a string with variables, you can either use keyword arguments in .format ('Insert {n} here'.format(n=num)), refer to them by position index explicitly (like 'Insert {0} here'.format(num)) or implicitly (like 'Insert {} here'.format(num)). You can use double quotation marks inside single quotation marks and the way around, but to nest the same set of quotation marks, you need to escape them with a slash like \".

## String representation of objects
"""
There are two special methods in Python that return a string representation of an object. __str__() is called when you use print() or str() on an object, and __repr__() is called when you use repr() on an object, print the object in the console without calling print(), or instead of __str__() if __str__() is not defined.

__str__() is supposed to provide a "user-friendly" output describing an object, and __repr__() should return the expression that, when evaluated, will return the same object, ensuring the reproducibility of your code.

In this exercise, you will continue working with the Employee class from Chapter 2.
"""

class Employee:
    def __init__(self, name, salary=30000):
        self.name, self.salary = name, salary
            
    # Add the __str__() method
    def __str__(self):
        output = """
        Employee name:{name}
        Employee salary:{salary}
        """.format(name = self.name, salary = self.salary)
        return output

emp1 = Employee("Amar Howard", 30000)
print(emp1)
emp2 = Employee("Carolyn Ramirez", 35000)
print(emp2)

class Employee:
    def __init__(self, name, salary=30000):
        self.name, self.salary = name, salary
      

    def __str__(self):
        s = "Employee name: {name}\nEmployee salary: {salary}".format(name=self.name, salary=self.salary)      
        return s
      
    # Add the __repr__method  
    def __repr__(self):
        
        return "Employee(\"{name}\", {salary})".format(name = self.name, salary = self.salary)   

emp1 = Employee("Amar Howard", 30000)
print(repr(emp1))
emp2 = Employee("Carolyn Ramirez", 35000)
print(repr(emp2))

## Catching exceptions
"""
Before you start writing your own custom exceptions, let's make sure you have the basics of handling exceptions down.

In this exercise, you are given a function invert_at_index(x, ind) that takes two arguments, a list x and an index ind, and inverts the element of the list at that index. For example invert_at_index([5,6,7], 1) returns 1/6, or 0.166666 .

Try running the code as-is and examine the output in the console. There are two unsafe operations in this function: first, if the element that we're trying to invert has the value 0, then the code will cause a ZeroDivisionError exception. Second, if the index passed to the function is out of range for the list, the code will cause a IndexError. In both cases, the script will be interrupted, which might not be desirable.
"""

# MODIFY the function to catch exceptions
def invert_at_index(x, ind):
    
    try : 
       return 1/x[ind]
    
    except ZeroDivisionError: 
        print("Cannot divide by zero!")
    except IndexError:
            print("Index out of range!")
 
a = [5,6,0,7]

# Works okay
print(invert_at_index(a, 1))

# Potential ZeroDivisionError
print(invert_at_index(a, 2))

# Potential IndexError
print(invert_at_index(a, 5))

## Custom exceptions
"""
You don't have to rely solely on built-in exceptions like IndexError: you can define your own exceptions more specific to your application. You can also define exception hierarchies. All you need to define an exception is a class inherited from the built-in Exception class or one of its subclasses.

In Chapter 1, you defined an Employee class and used print statements and default values to handle errors like creating an employee with a salary below the minimum or giving a raise that is too big. A better way to handle this situation is to use exceptions. Because these errors are specific to our application (unlike, for example, a division by zero error which is universal), it makes sense to use custom exception classes.
"""

# Define SalaryError inherited from ValueError
class SalaryError(ValueError):
    pass

# Define BonusError inherited from SalaryError
class BonusError(SalaryError):
    pass

class SalaryError(ValueError): pass
class BonusError(SalaryError): pass

class Employee:
  MIN_SALARY = 30000
  MAX_RAISE = 5000

  def __init__(self, name, salary = 30000):
    self.name = name
    
    # If salary is too low
    if self.salary < Employee.MIN_SALARY:
      # Raise a SalaryError exception
      raise SalaryError("Salary is too low!")

    self.salary = salary

class SalaryError(ValueError): pass
class BonusError(SalaryError): pass

class Employee:
  MIN_SALARY = 30000
  MAX_BONUS = 5000

  def __init__(self, name, salary = 30000):
    self.name = name    
    if salary < Employee.MIN_SALARY:
      raise SalaryError("Salary is too low!")      
    self.salary = salary
    
  # Rewrite using exceptions  
  def give_bonus(self, amount):
    if amount > Employee.MAX_BONUS:
       raise BonusError("The bonus amount is too high!")  
        
    elif self.salary + amount <  Employee.MIN_SALARY:
       raise SalaryError("The salary after bonus is too low!")
      
    else:  
      self.salary += amount

## Handling exception hierarchies
"""
Previously, you defined an Employee class with a method get_bonus() that raises a BonusError and a SalaryError depending on parameters. But the BonusError exception was inherited from the SalaryError exception. How does exception inheritance affect exception handling?

The Employee class has been defined for you. It has a minimal salary of 30000 and a maximal bonus amount of 5000.
"""

# Experiment with the following code
"""
emp = Employee("Katze Rik", salary=50000)
try:
  emp.give_bonus(7000)
except SalaryError:
  print("SalaryError caught!")

try:
  emp.give_bonus(7000)
except BonusError:
  print("BonusError caught!")

try:
  emp.give_bonus(-100000)
except SalaryError:
  print("SalaryError caught again!")

try:
  emp.give_bonus(-100000)
except BonusError:
  print("BonusError caught again!")  
"""
# and select the statement which is TRUE about handling parent/child exception classes:
# ... will ...

# Experiment with two pieces of code:
"""
emp = Employee("Katze Rik",\
                    50000)
try:
  emp.give_bonus(7000)
except SalaryError:
  print("SalaryError caught")
except BonusError:
  print("BonusError caught")
      

emp = Employee("Katze Rik",\
                    50000)
try:
  emp.give_bonus(7000)
except BonusError:
  print("BonusError caught")
except SalaryError:
  print("SalaryError caught")
"""
# (one catches BonusError before SalaryError, and the other -SalaryError before BonusError)
# Select the statement which is TRUE about the order of except blocks:
# It's ... for a child exception ...
# Exactly! It's better to list the except blocks in the increasing order of specificity, i.e. children before parents, otherwise the child exception will be called in the parent except block.

## Polymorphic methods
"""
To design classes effectively, you need to understand how inheritance and polymorphism work together.

In this exercise, you have three classes - one parent and two children - each of which has a talk() method. Analyze the following code:

class Parent:
    def talk(self):
        print("Parent talking!")     

class Child(Parent):
    def talk(self):
        print("Child talking!")          

class TalkativeChild(Parent):
    def talk(self):
        print("TalkativeChild talking!")
        Parent.talk(self)


p, c, tc = Parent(), Child(), TalkativeChild()

for obj in (p, c, tc):
    obj.talk()
What is the output of the code above?
"""

# 4
# Great job! Polymorphism ensures that the exact method called is determined dynamically based on the instance. What do you think would happen if Child did not implement talk()?

## Square and rectangle
"""
The classic example of a problem that violates the Liskov Substitution Principle is the Circle-Ellipse problem, sometimes called the Square-Rectangle problem.

By all means, it seems like you should be able to define a class Rectangle, with attributes h and w (for height and width), and then define a class Square that inherits from the Rectangle. After all, a square "is-a" rectangle!

Unfortunately, this intuition doesn't apply to object-oriented design.
"""

# Define a Rectangle class
class Rectangle:
    
    def __init__(self, h, w):
        self.h = h
        self.w = w

# Define a Square class
class Square(Rectangle):
    
    def __init__(self, w):
        
        Rectangle.__init__(self, w,w)

# The classes are defined for you. Experiment with them in the console.
# For example, in the console or the script pane, create a Square object with side length 4. Then try assigning 7 to the h attribute.
# What went wrong with these classes?
# The 4x4 ...

class Rectangle:
    def __init__(self, w,h):
        self.w, self.h = w,h

# Define set_h to set h      
    def set_h(self, h):
        self.h = h
      
# Define set_w to set w          
    def set_w(self, w):
        self.w = w 
      
      
class Square(Rectangle):
    def __init__(self, w):
        self.w, self.h = w, w 

# Define set_h to set w and h
    def set_h(self, h):
        self.h = h
        self.w = h

# Define set_w to set w and h      
    def set_w(self, w):
        self.w = w 
        self.h = w

# Later in this chapter you'll learn how to make these setter methods run automatically when attributes are assigned new values, don't worry about that for now, just assume that when we assign a value to h of a square, now the w attribute will be changed accordingly.
# How does using these setter methods violate Liskov Substitution principle?
# Each ...
# Correct! Remember that the substitution principle requires the substitution to preserve the oversall state of the program. An example of a program that would fail when this substitution is made is a unit test for a setter functions in Rectangle class.

## Attribute naming conventions
"""
In Python, all data is public. Instead of access modifiers common in languages like Java, Python uses naming conventions to communicate the developer's intention to class users, shifting the responsibility of safe class use onto the class user.

Python uses underscores extensively to signal the purpose of methods and attributes. In this exercise, you will match a use case with the appropriate naming convention.
"""

# _name     : A helper ...
# __name    : A 'version' attribute ...
# __name__  : A method ...

## Using internal attributes
"""
In this exercise, you'll return to the BetterDate class of Chapter 2.

You decide to add a method that checks the validity of the date, but you don't want to make it a part of BetterDate's public interface.

The class BetterDate is available in the script pane.
"""

# Add class attributes for max number of days and months
class BetterDate:
    _MAX_DAYS = 31
    _MAX_MONTHS = 12
    
    def __init__(self, year, month, day):
        self.year, self.month, self.day = year, month, day
        
    @classmethod
    def from_str(cls, datestr):
        year, month, day = map(int, datestr.split("-"))
        return cls(year, month, day)
        
    # Add _is_valid() checking day and month values
    def _is_valid(self):
        return (self.day <= BetterDate._MAX_DAYS) and \
               (self.month <= BetterDate._MAX_MONTHS)
        
bd1 = BetterDate(2020, 4, 30)
print(bd1._is_valid())

bd2 = BetterDate(2020, 6, 45)
print(bd2._is_valid())

## What do properties do?
"""
You could think of properties as attributes with built-in access control. They are especially useful when there is some additional code you'd like to execute when assigning values to attributes.

Which of the following statements is NOT TRUE about properties?
"""

# Properties can prevent creation of new attributes via assignment
# This statement is indeed not true! Properties control only one specific attribute that they're attached to. There are ways to prevent creating new attributes, but doing so would go against the "we're all adults here" principle.

## Create and set properties
"""
There are two parts to defining a property:

first, define an "internal" attribute that will contain the data;
then, define a @property-decorated method whose name is the property name, and that returns the internal attribute storing the data.

If you'd also like to define a custom setter method, there's an additional step:

define another method whose name is exactly the property name (again), and decorate it with @prop_name.setter where prop_name is the name of the property. The method should take two arguments -- self (as always), and the value that's being assigned to the property.

In this exercise, you'll create a balance property for a Customer class - a better, more controlled version of the balance attribute that you worked with before.
"""

# Create a Customer class
class Customer:
    def __init__(self,name, new_bal):
        self.name = name 
        if new_bal <0:
            raise ValueError
        else:
            self._balance = new_bal

class Customer:
    def __init__(self, name, new_bal):
        self.name = name
        if new_bal < 0:
           raise ValueError("Invalid balance!")
        self._balance = new_bal  
    
    # Add a decorated balance() method returning _balance
    @property
    def balance(self):
        return self._balance
    
class Customer:
    def __init__(self, name, new_bal):
        self.name = name
        if new_bal < 0:
           raise ValueError("Invalid balance!")
        self._balance = new_bal  

    # Add a decorated balance() method returning _balance        
    @property
    def balance(self):
        return self._balance
     
    # Add a setter balance() method
    @balance.setter
    def balance(self, new_balance):
        # Validate the parameter value
        if new_balance < 0: 
            raise ValueError
        else:
           self._balance = new_balance
        
        # Print "Setter method is called"
        print("Setter method is called")

class Customer:
    def __init__(self, name, new_bal):
        self.name = name
        if new_bal < 0:
           raise ValueError("Invalid balance!")
        self._balance = new_bal  

    # Add a decorated balance() method returning _balance        
    @property
    def balance(self):
        return self._balance

    # Add a setter balance() method
    @balance.setter
    def balance(self, new_bal):
        # Validate the parameter value
        if new_bal < 0:
           raise ValueError("Invalid balance!")
        self._balance = new_bal
        print("Setter method called")

# Create a Customer        
cust = Customer("Belinda Lutz", 2000)

# Assign 3000 to the balance property
cust.balance = 3000

# Print the balance property
print(cust.balance)

## Read-only properties
"""
The LoggedDF class from Chapter 2 was an extension of the pandas DataFrame class that had an additional created_at attribute that stored the timestamp when the DataFrame was created, so that the user could see how out-of-date the data is.

But that class wasn't very useful: we could just assign any value to created_at after the DataFrame was created, thus defeating the whole point of the attribute! Now, using properties, we can make the attribute read-only.

The LoggedDF class from Chapter 2 is available for you in the script pane.
"""

import pandas as pd
from datetime import datetime

# LoggedDF class definition from Chapter 2
class LoggedDF(pd.DataFrame):
    def __init__(self, *args, **kwargs):
        pd.DataFrame.__init__(self, *args, **kwargs)
        self.created_at = datetime.today()

    def to_csv(self, *args, **kwargs):
        temp = self.copy()
        temp["created_at"] = self.created_at
        pd.DataFrame.to_csv(temp, *args, **kwargs)   

# Instantiate a LoggedDF called ldf
ldf = LoggedDF({"col1": [1,2], "col2":[3,4]}) 

# Assign a new value to ldf's created_at attribute and print
ldf.created_at = '2035-07-13'
print(ldf.created_at)

import pandas as pd
from datetime import datetime

# MODIFY the class to use _created_at instead of created_at
class LoggedDF(pd.DataFrame):
    def __init__(self, *args, **kwargs):
        pd.DataFrame.__init__(self, *args, **kwargs)
        self._created_at = datetime.today()
    
    def to_csv(self, *args, **kwargs):
        temp = self.copy()
        temp["created_at"] = self._created_at
        pd.DataFrame.to_csv(temp, *args, **kwargs)   
    
    # Add a read-only property: _created_at
    @property  
    def created_at(self):
        return self._created_at

# Instantiate a LoggedDF called ldf
ldf = LoggedDF({"col1": [1,2], "col2":[3,4]}) 

# What happens when you assign '2035-07-13' to ldf.created_at?
# ... is read-only
# You've put it all together! Notice that the to_csv() method in the original class was using the original created_at attribute. After converting the attribute into a property, you could replace the call to self.created_at with the call to the internal attribute that's attached to the property, or you could keep it as self.created_at, in which case you'll now be accessing the property. Either way works!

## Congratulations!