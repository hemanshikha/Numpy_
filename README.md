# Numpy_

• Introduction to NumPy:
NumPy (Numerical Python) is a Python library used for numerical computations, matrix operations, and data analysis.

It provides an n-dimensional array which is faster and more efficient than Python list.

 •Importing NumPy
 
    import Numpy as np
    
 • Creating NumPy Arraya
 
1. From Lists or Tuples

    arr = np.array([1, 2, 3, 4])
   print(arr)
   
3. Multi-dimensional Arrays
   
    arr2 = np.array([[1, 2, 3], [4, 5, 6]])
   
5. Using Built-in Functions
   
•Funcions 

np.zeros(shape)	#Creates array with all zeros	

    eg. np.zeros((2,3))
    
np.ones(shape)	#Creates array with all ones

    eg.  np.ones((3,3))
    
np.arange(start, stop, step)	#Creates array with a range

    eg. np.arange(1,10,2)
    
np.linspace(start, stop, num)	#Creates evenly spaced values

    eg. np.linspace(0,1,5)
    
np.eye(n) 	#Identity matrix
  
    eg. np.eye(3)
    
np.random.rand(m,n)	#Random floats 0–1	

    eg. np.random.rand(2,3)
    
np.random.randint(low,high,size)	 #Random integers	

    eg. np.random.randint(1,10,(2,3))
    
• Array Attributes

arr = np.array([[1,2,3],[4,5,6]])

print(arr.ndim)   # Number of dimensions

print(arr.shape)  # Rows and Columns

print(arr.size)   # Total elements

print(arr.dtype)  # Data type

• Array Indexing and Slicing

a = np.array([10,20,30,40,50])

print(a[0])       # 10

print(a[-1])      # 50

print(a[1:4])     # [20 30 40]

b = np.array([[1,2,3],[4,5,6]])

print(b[1,2])     # 6

print(b[:,1])     # 2nd column -> [2 5]

•Mathematical Operations

x = np.array([1,2,3])

y = np.array([4,5,6])

print(x + y)    # [5 7 9]

print(x - y)    # [-3 -3 -3]

print(x * y)    # [4 10 18]

print(x / y)    # [0.25 0.4 0.5]

• Aggregate Functions

np.sum(arr)	#sum of elements
    
    eg. np.sum(a)
    
np.min(arr)	#minimum

    eg. np.min(a)
    
np.max(arr)	#maximum	

    eg. np.max(a)
np.mean(arr)	#mean	

    eg. np.mean(a)
    
np.median(arr)	#median

     eg. np.median(a)
     
np.std(arr)	#standard deviation

    eg. np.std(a)
    
np.var(arr)	 #variance	

    eg. np.var(a)
    
• Reshaping and Flattening

a = np.arange(1,7)

b = a.reshape(2,3)

print(b)

print(b.flatten())   # Converts 2D to 1D

• Stacking Arrays

a = np.array([[1,2],[3,4]])

b = np.array([[5,6],[7,8]])

print(np.hstack((a,b)))   # Horizontal stack

print(np.vstack((a,b)))   # Vertical stack

• Splitting Arrays

arr = np.array([10,20,30,40,50,60])

print(np.split(arr, 3))   # Split into 3 equal parts

Here is a code which includes all the functions of Numpy...

import numpy as np

# Array creation

a = np.array([[1,2,3],[4,5,6]])

b = np.arange(1,7).reshape(2,3)

# Basic info

print("Shape:", a.shape)

print("Dimensions:", a.ndim)

print("Data type:", a.dtype)

# Mathematical operations

print("Sum:", np.sum(a))

print("Mean:", np.mean(a))

print("Max:", np.max(a))

print("Min:", np.min(a))

print("Standard Deviation:", np.std(a))

# Element-wise operation

print("a+b:", a+b)

print("Square root:", np.sqrt(a))

# Reshaping

c = a.reshape(3,2)

print("Reshaped:\n", c)

# Stacking

v = np.vstack((a,b))

h = np.hstack((a,b))

print("Vertical Stack:\n", v)

print("Horizontal Stack:\n", h)

# Matrix operations

x = np.array([[1,2],[3,4]])

y = np.array([[5,6],[7,8]])

print("Dot Product:\n", np.dot(x,y))

print("Transpose:\n", np.transpose(x))

print("Determinant:", np.linalg.det(x))

print("Inverse:\n", np.linalg.inv(x))
