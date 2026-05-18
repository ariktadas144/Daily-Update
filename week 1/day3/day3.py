#Map, Filter, Reduce, REST API basics, NumPy – arrays, statistical functions, indexing

from functools import reduce
import numpy as np

#Map
list1=[1,2,3,4,5]
print(list(map(lambda x:x**2, list1))) #[1, 4, 9, 16, 25]

#Filter
print(list(filter(lambda x:x%2==0, list1))) #[2, 4]

#Reduce
factorial=reduce(lambda x, y: x*y, list1)
print(factorial) #120

#REST API basics
#in rest_api.py

#Numpy- arrays
array=np.array([1,2,3,4,5]) #!D numpy array
print(array) #[1 2 3 4 5]
array1=np.array([[1,2,3],[4,5,6],[7,8,9]]) #2D numpy array
print(array1)
#[[1 2 3]
# [4 5 6]
# [7 8 9]]
array2=np.ones(10)
array2=array2.astype(int)
print(array2) #[1 1 1 1 1 1 1 1 1 1]

#Numpy- statistical functions
print(np.min(array)) #1
print(np.max(array)) #5
print(np.sum(array)) #15
print(np.mean(array)) #3.0
print(np.median(array)) #3.0
print(np.std(array)) #1.4142135623730951

#Numpy- indexing
print(array[2]) #3

#Numpy- slicing
#Basic slicing (View)
b=array[0:3]
b[0]=100
print(array)
print(b)
#[100   2   3   4   5]
#[100   2   3]

#Fancy slicing (Copy)
ar=np.array([1,2,3,4,5])
b1=ar[0:3].copy()
b1[0]=100
print(ar)
print(b1)
#[1 2 3 4 5]
#[100   2   3]
