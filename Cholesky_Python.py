# Functions #
#Imports: For being able to use different commands
import numpy as np
#Step 1: This allows us to calculate for U (Upper Triangle)
def Matrix_Factorization(N,A):
  U = np.identity(N)
  for irow in range(0,N):
      for icolumn in range(irow,N):
          if icolumn == irow:
              summation = 0
              for e6 in range (0,irow-1):
                  summation = summation + (U[e6][icolumn]**2)
              Uii = A[irow][icolumn] - summation
              U[irow][icolumn] = Uii**0.5
          else:
              summation = 0
              for e7 in range (0,irow-1):
                  summation = summation + (U[e7][icolumn]*U[e7][irow])
              Uij = A[irow][icolumn] - summation
              U[irow][icolumn] = Uij/U[irow][irow]
  return U
 
#Step 2: Calculate Y (U transpose * x) to go onto the next step PROBLEM SOMETHING TO DUE WITH PUTTING THE NUMBER IN THE LIST COMING OUT AS ALWAYS 0
def Forward_Sol(U,N,b):
  Y = np.arange(N).reshape(N,1)
  Utranspose = U.transpose(1, 0)
  for irow in range(0,N):
    summation2 = 0
    for e16 in range(0,irow-1):
      summation2 = summation2 + (Utranspose[irow][e16]*Y[e16][0])
    Yj = b[irow][0]-summation2
    Y[irow][0] = Yj/Utranspose[irow][irow]
  return Y

#Step 3: Works backwards in relation to finding x as opposed to Y
def Backward_Sol(N,U,Y):
  x = np.arange(N).reshape(N,1)
  for irow in range(N,0):
    summation3 = 0
    for e21 in range(irow+1,N):
      summation3 = summation3 + (U[irow][e21]*x[e21][0])
    xj = Y[irow][0] - summation3
    x[irow][0] = xj/U[irow][irow]
  return x
      
# Main Program #
N = 3
A = [
      [2.00, -1.00, 0.00],
      [-1.00, 2.00, -1.00],
      [0.00, -1.00, 1.00]
]
b = [
      [1.00],
      [0.00],
      [0.00]
]
#Step 0: Checking the Matrices
#Step 1: Matrix Factorization Phase
U = Matrix_Factorization(N,A)
#Step 2: Forward Solution Phase
Y = Forward_Sol(U,N,b)
#Step 3: Backward Solution Phase
x = Backward_Sol(N,U,Y)