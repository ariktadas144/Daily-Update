#Slicing, Indexing, Data Structures, List/Dict Comprehensions, Lambda, User-Defined Functions, Exception Handling

#1. Slicing
string1="The weather is sunny today"
print(string1[15:20]) #sunny
list1= [2,4,6,8,10]
print(list1[0:4]) #[2, 4, 6, 8]

#2. Indexing
string2="Hello, World!"
print(string2[7]) #W
list2= [1,3,5,7,9]
print(list2[2]) #5

#3. Data Structures
#List
list3= [1,2,3,4,5]
#Dictionary
dict2= {"Name":"Alice", "age":25}
#Set
set2= {1,2,3,4,5}
#Tuple
tuple2= (1,2,3,4,5)

#4. List/Dict Comprehensions
#List comprehension
cubed=[x**3 for x in range(5)]
print(cubed) #[0, 1, 8, 27, 64]
#Dict comprehension
cubed_dict={x:x**3 for x in range(5)}
print(cubed_dict) #{0: 0, 1: 1, 2: 8, 3: 27, 4: 64}

#5. Lambda
print((lambda x, y: x*y)(5,2)) #10

#6. User-Defined Functions
def greet(name):
    return f"Hello, {name}!"
print(greet("James"))
#Hello, James!

#7. Exception Handling
try:
    result=10/0
except ZeroDivisionError:
    print("Cannot divide by zero.") 
#Cannot divide by zero.
