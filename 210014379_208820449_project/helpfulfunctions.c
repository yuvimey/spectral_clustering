#include "capi.h"

/*Useful functions and struct - 
This file includes all the useful functions
needed to run the algorithms of:
- Modified Grahm Schmidt
- QR Iteration
- Normalized Spectral Clustering
but does not include kmeans, 
which is in another file*/

int counter = 0; // Counts the number of memory allocations active

void *new_malloc(int len, int sizeofA) {
    /*Malloc with an added counter
    for debugging purposes*/
    counter++;
    void* p = malloc(len * sizeofA);
    if(p == NULL && len*sizeofA != 0){
        fprintf(stderr, "Problem allocating memory!\n");
        counter--;
    }
    return p;
}

void *new_calloc(int len, int sizeofA) {
    /*Calloc with an added counter
    for debugging purposes*/
    counter++;
    void *p = calloc(len, sizeofA);
    if(p == NULL && len*sizeofA != 0){
        fprintf(stderr, "Problem allocating memory!\n");
        counter--;
    }
    return p;
}

void new_free(void * A) {
    /*Free with an added counter
    for debugging purposes*/
    if(A == NULL){
        fprintf(stderr, "Cannot free NULL pointer! continuing...\n");
        return;
    }
    free(A);
    counter--;
}

double VectorMult(double* A, double* B, int d) {
    /*Multiplies arrays of length d*/
    /*---NOT IN USE (Parallel function in Python is more efficient)---*/
	int i;
	double res = 0.0;
	for ( i = 0 ; i < d ; i++) {
		res += A[i] * B[i];
	}
	return res;
}

double ColMult(double* A, double* B, int colA, int colB, int row_length) {
    /*Inner product of two columns of matricies A and B.
    Matrices are not represented here as double arrays of
    size [n][m] but as arrays of size [n * m] and thus
    require special care when it comes to arithmetic*/
    /*---NOT IN USE (Parallel function in Python is more efficient)---*/
	int i;
   double res = 0.0;
   for(i=0; i<row_length; i++){
        res += A[i*row_length + colA]*B[i*row_length + colB];
   }
   return res;
}

double* WeightedAdjGraph(double* A, int n, int d) {
    /*Calculated the weighted adjacent graph of A*/
    /*---NOT IN USE (Parallel function in Python is more efficient)---*/
	int i,j;
	double* W;
	W = (double *) new_calloc(n * n, sizeof(double));
	if(W == NULL){
		return NULL;
	}
	double* tmp;
	for ( i = 0 ; i < n ; i++) {
		for ( j = 0 ; j < n ; j++) {
		    if (i != j) {
		    	tmp = VectorSub(A + i*d, A + j*d, d);
			    W[i * n + j] = (-1.0) * exp( (-1) * sqrt(VectorMult(tmp, tmp, d)) / 2);
			    new_free(tmp);
			}
		}
	}
	return W;
}

double* VectorSub(double* A, double* B, int d) {
    /*Calculates the difference between two "vectors"
    of length d as represented by A and B. This is done
    this way as to allow the representation of matrices
    as single arrays*/
    /*---NOT IN USE (Parallel function in Python is more efficient)---*/
	int k;
	double* C;
	C = (double *) new_malloc(d, sizeof(double));
	if(C == NULL){
		return NULL;
	}
	for ( k = 0 ; k < d ; k++) {
		C[k] = A[k] - B[k];
	}
	return C;
}

double* MatrixMult(double* A, double* B, int n) {
    /*Multiplies two matricies A and B of size n by n*/
    /*---NOT IN USE (Parallel function in Python is more efficient)---*/
	int i,j,h;
	double* C;
	C = (double *) new_malloc(n * n ,sizeof(double));
	if(C == NULL){
		return NULL;
	}
	for ( i = 0 ; i < n ; i++) {
		for ( j = 0 ; j < n ; j++) {
		    C[i * n + j] = 0.0;
			for ( h = 0 ; h < n ; h++) {
				C[i * n + j] += A[i * n + h] * B[h * n + j];
			}
		}
	}
	return C;
}

int Line6InQR(double* A, double* B, double e, int n) {
    /*Line 6 of the QR Iteration algorithm:
    checks if the values of Q1 are close enough
    to the values of Q1 times Q*/
    /*---NOT IN USE (Parallel function in Python is more efficient)---*/
	int i, j;
	int done = 0;
	int end = n * n;
	for (i = 0 ; i < n ; i++) {
		for ( j = 0 ; j < n ; j++) {
			if (fabs(fabs(A[i * n + j]) - fabs(B[i * n + j])) <= e) {
				done += 1;
			}
		}
	}
	if (done == end) {
		return 1;
	}
	return 0;
}


double* DiagDegSqrtMatrix(double* W, int n) {
    /*Returns D, a diagonal matrix where the diagonal
    is 1 over the square root of the sum of the weights
    from the weighted adjacent graph. This graph is given
    to us as W. We will use this to calculate Lnorm.*/
    /*---NOT IN USE (Parallel function in Python is more efficient)---*/
	int i,j;
	double* D;
	D = (double *) new_calloc(n * n, sizeof(double));
	if(D == NULL){
		return NULL;
	}
	for (i = 0 ; i < n ; i++) {
		for ( j = 0 ; j < n ; j++) {
			D[i * n + i] -= W[i * n + j];
		}
		D[i * n + i] = 1.0 / sqrt(D[i *n + i]);
	}
	return D;
}

double* Lnorm(double* W, int n) {
    /*Calculates Lnorm*/
    /*---NOT IN USE (Parallel function in Python is more efficient)---*/
	int i;
	double* lnorm;
	double* D = DiagDegSqrtMatrix(W, n);
	if(D == NULL){
		return NULL;
	}
	double* q1;
	q1 = MatrixMult(W, D, n);
	if(q1 == NULL){
		new_free(D);
		return NULL;
	}
    lnorm = MatrixMult(D, q1, n);
	if(lnorm == NULL){
		new_free(D);
		new_free(q1);
		return NULL;
	}
	for ( i = 0 ; i < n ; i++) {
	    lnorm[i * n + i] = 1.0 + lnorm[i * n + i];
	}
	new_free(D);
	new_free(q1);
	return lnorm;
}

double *buildU(double *A, double *Q, int n, int k) {
    /*Used in the Normalized Spectral Clustering algorithm,
    this function returns a matrix of size n by k whose
    columns are the first k eigenvectors of the calculated Lnorm*/
    /*---NOT IN USE (Parallel function in Python is more efficient)---*/
	int i,j;
    int min_ind = 0;
    double *U;
    U = new_malloc(n * k, sizeof(double));
	if(U == NULL){
		return NULL;
	}
    for(j = 0 ; j < k ; j++){
        for ( i = 0 ; i < n ; i++) {
            if(min_ind == -1){
                if(A[i * n + i] >= 0){
                    min_ind = i;
                }
                continue;
            }
            if ((A[i * n + i] >= 0.0) && (A[i * n + i] < A[min_ind * n + min_ind])) {
                min_ind = i;
            }
        }
        for ( i = 0 ; i < n ; i++) {
            U[i * k + j] = Q[i * n + min_ind];
        }
        A[min_ind * n + min_ind] = -1.0;
        min_ind = -1;
    }
    return U;
}

int compare(const void* a, const void* b) {
    /*Used to sort the eigenvalues in rising order*/
    /*---NOT IN USE (Parallel function in Python is more efficient)---*/
    double A = *(double*)a;
    double B = *(double*)b;
    if ((A - B) < 0){
        return -1;
    }
    if( A == B){
        return 0;
    }
    return 1;
}

int findK(double* A, int n){
    /*Represents the Eigengap Heuristic and
    finds k*/
    /*---NOT IN USE (Parallel function in Python is more efficient)---*/
    int k, i;
	double x;
	int max_index = -1;
	double* eigenvalues;
	eigenvalues = (double *) new_malloc( n, sizeof(double));
	if(eigenvalues == NULL){
		return -1;
	}
	for ( i = 0 ; i < n ; i++) {
		eigenvalues[i] = A[i * n + i];
	}
	qsort(eigenvalues, n, sizeof(double), compare);
	double max = -1.0;
	for ( i = 0 ; i < ceil(n / 2); i++) {
		x = eigenvalues[i+1] - eigenvalues[i];
		if (x > max) {
			max = x;
			max_index = i;
		}
	}
	k = max_index + 1;
	new_free(eigenvalues);
	return k;
}

double* Normalize(double* U, int rows, int cols, double eps){
    /*Used in the Normalized Spectral Clustering,
    this function represents line 5 of the algorithm
    to represent T, the normalized version of U, where U
    is a matrix of size n by k whose columns are the first
    k eigenvectors of Lnorm*/
    /*---NOT IN USE (Parallel function in Python is more efficient)---*/
	int i,j;
    double* T;
    double sum_U_i;
	T = (double *) new_calloc(rows * cols, sizeof(double));
	if(T == NULL){
		return NULL;
	}
	for ( i = 0 ; i < rows ; i++) {
		for ( j = 0 ; j < cols ; j++) {
			T[i * cols + j] = U[i * cols + j];
			
			sum_U_i = sqrt(VectorMult(U + cols*i, U + cols*i, cols));
			if(fabs(sum_U_i) < eps){
			    T[i * cols + j] = 0.0;          // Cases where the vector has length of 0 (less than eps) we set T_ij to 0.0
			}
			else{
			    T[i * cols + j] = T[i * cols + j] / sum_U_i;
			}
		}
	}
	return T;
}