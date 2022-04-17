% ---------------- MAIN PROGRAM ---------------- %
file1 = fopen('Cholesky_Data.txt','r');
N = fscanf(file1,'%g',1);
A = readmatrix("Cholesky_Data_MatrixA.txt");
b = readmatrix("Cholesky_Data_MatrixB.txt");
%Step 0: Checking the Matrices
[checkA, checkb] = SymmetricPositiveDefinite(A,b);
A = checkA;
b = checkb;
disp('Performing Cholesky Decomposition on Matrix A: ')
disp(A)
disp('and Matrix b: ')
disp(b)
%Step 1: Matrix Factorization Phase
[U] = factorization(N,A);
%Step 2: Forward Solution Phase
[Y] = forward_sol(U,N,b);
%Step 3: Backward Solution Phase
[x] = backward_sol(N,U,Y);
%Step 4: Matlab Answer compared Program Answer
[matlab_answer] = CheckBothAnswers(A,b,x);

% ---------------- FUNCTION PROGRAM ---------------- %
%Step 0: Checking Matrix as Symmetric and Positive Definite
function [checkA, checkb] = SymmetricPositiveDefinite(A,b)
    check = issymmetric(A);
    if check == 1
        disp('Matrix A is Symmetric')
        checkposdef = eig(A);
        isposdef = all(checkposdef>0);
        if isposdef == 1
            disp('Matrix A is Positive Definite')
            checkA = A;
    	    checkb = b;
        else
            disp('Matrix A is not Positive Definite')
            checkA = transpose(A)*A;
            checkb = transpose(A)*b;
        end
    else
        %If the Matrix is not Symmetric, it can't be positive definite
        disp('Matrix A is not Symmetric')
        disp('Matrix A is not Positive Definite')
        checkA = transpose(A)*A;
    	checkb = transpose(A)*b;
    end
end

%Step 1: This allows us to calculate for U (upper triangle)
function [U] = factorization(N,A)
    U = zeros(N,N);
    for irow = 1:1:N
        for icolumn = irow:1:N
            if icolumn == irow
                summation = 0;
                for e6 = 1:1:irow-1
                    summation = summation + (U(e6,icolumn))^2; 
                end
                Uii = A(irow,icolumn) - summation;
                U(irow,icolumn) = Uii^0.5;
            else
                summation = 0;
                for e7 = 1:1:irow-1
                    summation = summation + (U(e7,icolumn)*U(e7,irow));
                end
                Uij = A(irow,icolumn) - summation;
                U(irow,icolumn) = Uij/U(irow,irow);
            end
        end
    end
end

%Step 2: Calculate Y (U transpose * x) to go onto the next step
function [Y] = forward_sol(U,N,b)
    Y = zeros(N,1);
    Utranspose = transpose(U);
    for irow = 1:1:N
        summation2 = 0;
        for e16 = 1:1:irow-1
            summation2 = summation2 + (Utranspose(irow,e16)*Y(e16,1));
        end
        Yj = b(irow,1)-summation2;
        Y(irow, 1) = Yj/Utranspose(irow,irow);
    end
end

%Step 3: Calculates backwards in relation to finding x as opposed to Y
function [x] = backward_sol(N,U,Y)
    x = zeros(N,1);
    for irow = N:-1:1
        summation3 = 0;
        for e21 = irow+1:1:N
            summation3 = summation3 + (U(irow,e21)*x(e21,1));
        end
        xj = Y(irow,1) - summation3;
        x(irow,1) = xj/U(irow,irow);
    end
end

%Step 4: Checking Matlab's Answer vs Cholesky's Decomposition Answer
function [matlab_answer] = CheckBothAnswers(A,b,x)
    matlab_answer = A\b;
    disp('Matlab Result (x):')
    disp(matlab_answer)
    disp('Cholesky Decomposition Result (x):')
    disp(x)
end