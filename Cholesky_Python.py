from typing import Tuple
import numpy as np

# ---------------------------- Functions ---------------------------- #

def is_pos_def(A: np.ndarray) -> bool:
  """ Check if a matrix is positive definite.

      Used in step 0 of Cholesky Decomposition
  """
  return np.all(np.linalg.eigvals(A) > 0)

def check_symmetry(A: np.ndarray) -> bool:
  """ Check if a matrix is symmetric.

      Used in step 0 of Cholesky Decomposition
  """
  return (A == A.transpose()).all()

def condition(
    symmetric: bool, positive_definite: bool, A: np.ndarray, 
    b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
  """ Step 0 of Cholesky Decomposition.

      Checks if Matrix A is both positive definite and symmetric. 
      Multiplies both sides by the transpose of A if this condition is not met.
  """
  print(f"Matrix A is {('' if symmetric else 'NOT')} Symmetric")
  print(f"Matrix A is {('' if positive_definite else 'NOT')} Positive Definite")
  if symmetric and positive_definite:
    return (A,b)
  else:
    Atranspose = A.transpose()
    SymPosDef = (Atranspose * A, Atranspose * b)
    return SymPosDef

def Matrix_Factorization(N: int, A: np.ndarray) -> np.ndarray:
  """ Performs Step 1 of Cholesky Decomposition.

      Calculates U (Upper Triangle) of Matrix A.
  """
  A = A.tolist()
  U = np.zeros([N,N])
  for irow in range(0,N):
      for icolumn in range(irow,N):
          if icolumn == irow:
              summation = 0
              for e_6 in range (0,irow):
                  summation += U[e_6][icolumn]**2
              Uii = A[irow][icolumn] - summation
              U[irow][icolumn] = Uii**0.5
          else:
              summation = 0
              for e_7 in range (0,irow):
                  summation += U[e_7][icolumn] * U[e_7][irow]
              Uij = A[irow][icolumn] - summation
              U[irow][icolumn] = Uij / U[irow][irow]
  return U

def Forward_Sol(U: np.ndarray, N: int, b: np.ndarray) -> np.ndarray:
  """ Performs Step 2 of Cholesky Decomposition.

      Calculates Y (Utranspose * x)
  """
  b = b.tolist()
  Y = np.zeros([N, 1])
  Utranspose = np.transpose(U)

  for irow in range(0, N):
    summation_2 = 0
    for e_16 in range(0, irow):
      summation_2 += Utranspose[irow][e_16] * Y[e_16][0]
    Yj = b[irow][0] - summation_2
    Y[irow][0] = Yj / Utranspose[irow][irow]
  
  return Y

def Backward_Sol(N: int, U: np.ndarray, Y: np.ndarray) -> np.ndarray:
  """ Performs Step 3 of Cholesky Decomposition.

      Calculates x by working backwards from Y
  """
  x = np.zeros([N,1])

  for irow in range(N-1, -1, -1):
    summation_3 = 0
    for e_21 in range(irow, N):
      summation_3 += U[irow][e_21] * x[e_21][0]
    xj = Y[irow][0] - summation_3
    x[irow][0] = xj / U[irow][irow]
  
  return x

def main():
  """ Performs Cholesky Decomposition
  """
  # Initialize variables
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
  positive_definite = is_pos_def(A)
  symmetry = check_symmetry(A)

  A, b = condition(symmetry, positive_definite, A, b)

  #Step 1: Matrix Factorization Phase
  U = Matrix_Factorization(N, A)

  #Step 2: Forward Solution Phase
  Y = Forward_Sol(U, N, b)

  #Step 3: Backward Solution Phase
  x = Backward_Sol(N, U, Y)
  print(f"Cholesky Decomposition Result (x): {str(x)}")

# Run Cholesky Decomposition
main()