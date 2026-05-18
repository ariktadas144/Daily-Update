#Data types, Variables, Conditional Statements, Loops (for/while), OOP basics

#1. Data types
x = 10 #int
y = 1.2 #float
z = "Hello" #string
a = True #boolean
b = None #None Data type
list1 = [1, 2, 3, 4, 5] #list
tuple1=(1,2,3,4,5) #tuple
dict1= {"Name":"James", "age":30} #dictionary
set1= {1,2,3} #set

#2. Variables
name = "Alice"
age = 25
height = 5.5

#3. Conditional Statements
if age >= 18:            
    print("Eligible to vote.")
else:
    print("Not eligible to vote.")
#Eligible to vote.

#4. Loops
#For loop
for i in range(5):
    print(i)
#0
#1
#2
#3
#4

#While loop
count=0
while count < 5:
    print(count)
    count += 1
#0
#1
#2
#3
#4

#5. OOP basics
#Class and Object
class Person:
    def __init__(self, marks):
        self.marks = marks

    def grade(self):
        if self.marks>=90:
            return "S"
        elif self.marks>=80:
            return "A"
        elif self.marks>=70:
            return "B"
        elif self.marks>=60:
            return "C"
        elif self.marks>=50:
            return "D"
        else:
            return "F"
person1=Person(85)
result = person1.grade()
print("Grade: ", result)
#Grade:  A

#Inheritance
class Student(Person):
    def __init__(self, marks, name):
        super().__init__(marks)
        self.name = name
student1 = Student(92, "Alice")
print("Name: ", student1.name)
print("Grade: ", student1.grade())
#Name:  Alice
#Grade:  S

#Encapsulation
class BankAccount:
    def __init__(self, balance):
        self.balance = balance  

    def deposit(self, amount):
        self.balance += amount

    def withdraw(self, amount):
        if amount <= self.balance:
            self.balance -= amount
        else:
            print("Insufficient funds")

    def get_balance(self):
        return self.balance
account = BankAccount(1000)
account.deposit(500)    
account.withdraw(200)
print("Balance: ", account.get_balance())
#Balance:  1300

#Polymorphism
class Shape:
    def area(self):
        pass
class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius

    def area(self):
        return 3.14 * self.radius ** 2
class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def area(self):
        return self.width * self.height
circle = Circle(5)
rectangle = Rectangle(4, 6)
print("Circle Area: ", circle.area())
print("Rectangle Area: ", rectangle.area())
#Circle Area:  78.5
#Rectangle Area:  24
