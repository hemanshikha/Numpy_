# Numpy_

• Introduction to NumPy:
NumPy (Numerical Python) is a Python library used for numerical computations, matrix operations, and data analysis.

It provides an n-dimensional array which is faster and more efficient than Python list.

 •Importing NumPy
 
    import Numpy as np
    
 • Creating NumPy Arrays
 
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


# Array Creation

a = np.array([1, 2, 3, 4])

b = np.array([[5, 6, 7], [8, 9, 10]])

print("a =", a)

print("b =\n", b)


# Built-in creation

print("Zeros:\n", np.zeros((2,3)))

print("Ones:\n", np.ones((2,3)))

print("Arange:", np.arange(0,10,2))

print("Linspace:", np.linspace(0,1,5))

print("Identity:\n", np.eye(3))

print("Random Float:\n", np.random.rand(2,3))

print("Random Int:\n", np.random.randint(1,10,(2,3)))



# Array Attributes

print("\nArray b ndim:", b.ndim)

print("Array b shape:", b.shape)

print("Array b size:", b.size)

print("Array b dtype:", b.dtype)



# Indexing and Slicing

print("\nFirst element of a:", a[0])

print("Slice of a[1:3]:", a[1:3])

print("Element b[1,2]:", b[1,2])

print("Second column of b:", b[:,1])



# Mathematical Operations

x = np.array([1,2,3])

y = np.array([4,5,6])

print("\nx + y =", x + y)

print("x - y =", x - y)

print("x * y =", x * y)

print("x / y =", x / y)

print("Square root of x:", np.sqrt(x))

print("Exponent of x:", np.exp(x))

print("Log of x:", np.log(x))

print("Sin of x:", np.sin(x))


#  Aggregate Functions


arr = np.array([10, 20, 30, 40, 50])

print("\nSum:", np.sum(arr))

print("Min:", np.min(arr))

print("Max:", np.max(arr))

print("Mean:", np.mean(arr))

print("Median:", np.median(arr))

print("Std Dev:", np.std(arr))

print("Variance:", np.var(arr))



# Reshape and Flatten

r = np.arange(1,7)

reshaped = r.reshape(2,3)

print("\nReshaped:\n", reshaped)

print("Flattened:", reshaped.flatten())



# Stacking


p = np.array([[1,2],[3,4]])

q = np.array([[5,6],[7,8]])

print("\nHorizontal Stack:\n", np.hstack((p,q)))

print("Vertical Stack:\n", np.vstack((p,q)))



# Splitting


s = np.array([10,20,30,40,50,60])

print("\nSplit into 3 parts:", np.split(s,3))



#  Matrix Operations


A = np.array([[1,2],[3,4]])

B = np.array([[5,6],[7,8]])

print("\nDot Product:\n", np.dot(A,B))

print("Transpose:\n", np.transpose(A))

print("Determinant:", np.linalg.det(A))

print("Inverse:\n", np.linalg.inv(A))



# Logical and Comparison


c = np.array([1,2,3,4,5])

print("\nc > 3:", c > 3)

print("Any element > 3?", np.any(c > 3))

print("All elements > 0?", np.all(c > 0))

print("Where >3:", np.where(c > 3))


# Copy vs View

orig = np.array([10,20,30])

copy_arr = orig.copy()

view_arr = orig.view()

orig[0] = 99

print("\nOriginal:", orig)

print("Copy:", copy_arr)

print("View:", view_arr)


# Random & Extra Functions

print("\nRandom normal distribution:\n", np.random.randn(2,2))

print("Unique elements:", np.unique([1,2,2,3,3,3,4]))

print("Sort:", np.sort([3,1,4,2]))

print("Concatenate:", np.concatenate(([1,2,3],[4,5,6])))

OUTPUT:

a = [1 2 3 4]

b =

 [[ 5  6  7]
 
  [ 8  9 10]]
  
Zeros:

 [[0. 0. 0.]
 
  [0. 0. 0.]]
  
Ones:

 [[1. 1. 1.]
 
  [1. 1. 1.]]
  
Arange: [0 2 4 6 8]

Linspace: [0.   0.25 0.5  0.75 1.  ]

Identity:

 [[1. 0. 0.]
 
  [0. 1. 0.]
  
  [0. 0. 1.]]
  
Random Float:

 [[0.56 0.72 0.35]
 
  [0.84 0.11 0.93]]
  
Random Int:

 [[6 1 5]
 
  [9 4 7]]
  

Array b ndim: 2

Array b shape: (2, 3)

Array b size: 6

Array b dtype: int64


First element of a: 1

Slice of a[1:3]: [2 3]

Element b[1,2]: 10

Second column of b: [6 9]


x + y = [5 7 9]

x - y = [-3 -3 -3]

x * y = [ 4 10 18]

x / y = [0.25 0.4  0.5 ]

Square root of x: [1.         1.41421356 1.73205081]

Exponent of x: [ 2.71828183  7.3890561  20.08553692]

Log of x: [0.         0.69314718 1.09861229]

Sin of x: [0.84147098 0.90929743 0.14112001]


Sum: 150

Min: 10

Max: 50

Mean: 30.0

Median: 30.0

Std Dev: 14.1421356237

Variance: 200.0


Reshaped:

 [[1 2 3]
 
  [4 5 6]]
  
Flattened: [1 2 3 4 5 6]


Horizontal Stack:

 [[1 2 5 6]
 
  [3 4 7 8]]
  
Vertical Stack:

 [[1 2]
 
  [3 4]
  
  [5 6]
  
  [7 8]]
  

Split into 3 parts: [array([10, 20]), array([30, 40]), array([50, 60])]


Dot Product:

 [[19 22]
 
  [43 50]]
  
Transpose:

 [[1 3]
 
  [2 4]]
  
Determinant: -2.0000000000000004

Inverse:

 [[-2.   1. ]
 
  [ 1.5 -0.5]]
  

c > 3: [False False False  True  True]

Any element > 3? True

All elements > 0? True

Where >3: (array([3, 4]),)


Original: [99 20 30]

Copy: [10 20 30]

View: [99 20 30]


Random normal distribution:

 [[ 0.65 -0.22]
 
  [ 1.45 -0.81]]
  
Unique elements: [1 2 3 4]

Sort: [1 2 3 4]

Concatenate: [1 2 3 4 5 6]


          ----------END--------------
          
