# Functions #
#Imports: For being able to use different commands
import numpy as np

#Step 0: Checks Matrix A for the conditions of Positive Definite and Symmetric in order to proceed, if not it will multiple both sides by transpose of A
def is_pos_def(A):
  return np.all(np.linalg.eigvals(A) > 0)

def check_symmetry(A):
  Atranspose = A.transpose()
  if (A == Atranspose).all:
    if (A == Atranspose).all():
      print("Matrix A is Symmetric")
      Symmetry = True
    else:
      print("Matrix A is NOT Symmetric")
      Symmetry = False
  else:
    print("Matrix A is NOT Symmetric")
    Symmetry = False
  return Symmetry

def condition(Symmetry,PositiveDefinite,A,b):
  if Symmetry and PositiveDefinite:
    return (A,b)
  else:
    Atranspose = A.transpose()
    SymPosDef = (Atranspose * A, Atranspose * b)
    return SymPosDef

#Step 1: This allows us to calculate for U (Upper Triangle)
def Matrix_Factorization(N,A):
  A = A.tolist()
  U = np.zeros([N,N])
  for irow in range(0,N):
      for icolumn in range(irow,N):
          if icolumn == irow:
              summation = 0
              for e6 in range (0,irow):
                  summation += U[e6][icolumn]**2
              Uii = A[irow][icolumn] - summation
              U[irow][icolumn] = Uii**0.5
          else:
              summation = 0
              for e7 in range (0,irow):
                  summation += U[e7][icolumn]*U[e7][irow]
              Uij = A[irow][icolumn] - summation
              U[irow][icolumn] = Uij/U[irow][irow]
  return U
 
#Step 2: Calculate Y (U transpose * x) to go onto the next step
def Forward_Sol(U,N,b):
  Y = np.ndarray([N,1])
  Utranspose = np.transpose(U)
  for irow in range(0,N):
    summation2 = 0
    for e16 in range(0,irow):
      summation2 += Utranspose[irow][e16]*Y[e16][0]
    Yj = b[irow][0]-summation2
    Y[irow][0] = Yj/Utranspose[irow][irow]
  return Y

#Step 3: Works backwards in relation to finding x as opposed to Y
def Backward_Sol(N,U,Y):
  x = np.ndarray([N,1])
  for irow in range(N-1,-1,-1):
    summation3 = 0
    for e21 in range(irow,N):
      summation3 += U[irow][e21]*x[e21][0]
    xj = Y[irow][0] - summation3
    x[irow][0] = xj/U[irow][irow]
  return x
      
# Main Program #
N = 3
A = np.asarray([
      [2.00, -1.00, 0.00],
      [-1.00, 2.00, -1.00],
      [0.00, -1.00, 1.00]
])
b = np.asarray([
      [1.00],
      [0.00],
      [0.00]
])
#Step 0: Checking the Matrices
PositiveDefinite = is_pos_def(A)
print(PositiveDefinite)
Symmetry = check_symmetry(A)
SymPosDef = condition(Symmetry,PositiveDefinite,A,b)
A = SymPosDef[0]
b = SymPosDef[1]
#Step 1: Matrix Factorization Phase
U = Matrix_Factorization(N,A)
#Step 2: Forward Solution Phase
Y = Forward_Sol(U,N,b)
#Step 3: Backward Solution Phase
x = Backward_Sol(N,U,Y)
print(x)