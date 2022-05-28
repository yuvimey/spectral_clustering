#include "capi.h"

/*This file includes the algorithms of:
- Modified Grahm Schmidt
- QR Iteration
- Normalized Spectral Clustering
along with a struct created to allow
for the functions of said algorithms to
return multiple separate values*/


/*ALGO 1 - The Modified Grahm-Schmidt Algorithm*/

struct tuple* ModifiedGrahmSchmidt(double* A, int n, double eps) {
    /*Runs the Modified Grahm Schmidt algorithm and
    returns Q, R*/
    /*---NOT IN USE (Parallel function in Python is more efficient)---*/
	int i,j,l;
	double* U;
	U = (double *) new_malloc(n * n, sizeof(double));
	if(U == NULL){
		return NULL;
	}
	double* R;
	R = (double *) new_calloc(n * n, sizeof(double));
	if(R == NULL){
		new_free(U);
		return NULL;
	}
	double* Q;
	Q = (double *) new_malloc(n * n, sizeof(double));
	if(Q == NULL){
		new_free(U);
		new_free(R);
		return NULL;
	}
	for ( i = 0 ; i < n ; i++) {//U = A
	    for ( j = 0 ; j < n ; j++) {
	        U[i * n + j] = A[i * n + j];
	    }
	}
	for ( i = 0 ; i < n ; i++) {
		R[i * n + i] = sqrt(ColMult(U, U, i, i, n));//Rii = ||Ui||^2

		if (R[i * n + i] == 0.0) {
			perror("ModifiedGrahmSchmidt: R[i][i] == 0.0; divide by zero");
		}
		for ( l = 0 ; l < n ; l++) {
		    if( fabs(R[i * n + i]) > eps){
		        Q[l * n + i] = U[l * n + i]/R[i * n + i];//Qi = Ui / Rii
		    }
		    else{
		        Q[l * n + i] = 0.0;                // Cases where R_ii equals 0 (less than eps) we set Q_li to 0.0
		    }
		}
		for ( j = i + 1 ; j < n ; j++) {
		    R[i * n + j] = ColMult(Q, U, i, j, n);//Rij = (Qi^T)(Uj)
		    for ( l = 0 ; l < n ; l++) {
		        U[l * n + j] -= R[i * n + j] * Q[l * n + i];//Uj = Uj - RijQi
		    }
		}
	}
	new_free(U);
	struct tuple * ret;
	ret = (struct tuple * ) new_malloc(1, sizeof(struct tuple));
	(*ret).A = Q;
	(*ret).B = R;
	return ret;
}

/*ALGO 2 - The QR Iteration Algorithm*/

struct tuple* QRIt(double* A, int n, double e) {//e = epsilon
    /*Calculates the QR Iteration algorithm and
    returns A, Q*/
    /*---NOT IN USE (Parallel function in Python is more efficient)---*/
	int i,j;
	struct tuple * MGS_tuple;
	struct tuple * QR_tuple;
	QR_tuple = (struct tuple*) new_malloc(1, sizeof(struct tuple));
	if(QR_tuple == NULL){
		return NULL;
	}
	double* A1;
	A1 = (double *) new_malloc(n * n, sizeof(double));
	if(A1 == NULL){
		new_free(QR_tuple);
		return NULL;
	}
	double* Q1;
	Q1 = (double *) new_calloc(n * n, sizeof(double));
	if(Q1 == NULL){
		new_free(A1);
		new_free(QR_tuple);
		return NULL;
	}
	double* q;
	for ( i = 0 ; i < n ; i++) {
      Q1[i * n + i] = 1.0;//Q1 = I
	    for ( j = 0 ; j < n ; j++) {
		    A1[i * n + j] = A[i * n + j];//A1 = A
	    }
	}
	for ( i = 0 ; i < n ; i++) {
		MGS_tuple = ModifiedGrahmSchmidt(A1, n, e);
		if (MGS_tuple == NULL){
			new_free(QR_tuple);
			new_free(A1);
			new_free(Q1);
			return NULL;
		}
		new_free(A1);
		double* Q = (*MGS_tuple).A;
		double* R = (*MGS_tuple).B;
		new_free(MGS_tuple);
		A1 = MatrixMult(R, Q, n);//A1 = RQ
		if (A1 == NULL){
			new_free(QR_tuple);
			new_free(Q1);
			new_free(Q);
	        new_free(R);
			return NULL;
		}
		q = MatrixMult(Q1, Q, n);//q = Q1Q, acts as a placeholder
		if (Line6InQR(Q1, q, e, n)) { //Checks if |Q|-|Q*Q1| in [-eps,eps]
			new_free(q);
			new_free(Q);
	        new_free(R);
			(*QR_tuple).A = A1;
			(*QR_tuple).B = Q1;
			return QR_tuple;
		}
        new_free(Q1);
        Q1 = q;//Q1 = Q1Q
        new_free(Q);
	    new_free(R);
	}
	(*QR_tuple).A = A1;
	(*QR_tuple).B = Q1;
	return QR_tuple;
}

/*ALGO 3 - The Normalized Spectral Clustering Algorithm*/

struct tuple* NormSpecClust(double* X, int n, int d, double e, int Random, int K, int max_iter) {
    /*Represents the Normalilized Spectral Clustering algorithm*/
    /*---NOT IN USE (Parallel function in Python is more efficient)---*/

	double* W = WeightedAdjGraph(X, n, d);
	if (W == NULL){
			return NULL;
	}
	double* lnorm = Lnorm(W, n);

	new_free(W);
	if (lnorm == NULL){
			return NULL;
	}
	struct tuple * AQ = QRIt(lnorm, n, e);

	new_free(lnorm);
	if (AQ == NULL){
			return NULL;
	}
	double* A = (*AQ).A;
	double* Q = (*AQ).B;
	new_free(AQ);
	int k;

	if (Random == 1) {//If Random is true, then use the Eigengap Heuristic to calculate k
	    k = findK(A, n);
		if( k == -1){
			new_free(A);
			new_free(Q);
			return NULL;
		}
	}
	else {//else, use the K given
		k = K;
	}

	double *U;
	U = buildU(A, Q, n, k);//columns are the first k eigenvectors of Lnorm
	new_free(A);
	new_free(Q);
	if (U == NULL){
			return NULL;
	}
	double* T;
	T = Normalize(U, n, k, e);//normalized version of U
	new_free(U);
	if (T == NULL){
			return NULL;
	}
	int* cluster_assign = k_means_pp(T, k, n, k, max_iter, e);//run T through kmeans
	new_free(T);
	
	if (cluster_assign == NULL){
			return NULL;
	}
	
	struct tuple* NormSpec_tuple;
	NormSpec_tuple = (struct tuple*) new_malloc(1, sizeof(struct tuple));
	if (NormSpec_tuple == NULL){
			new_free(cluster_assign);
			return NULL;
	}	
	(*NormSpec_tuple).D = cluster_assign;
	(*NormSpec_tuple).C = k;
	return NormSpec_tuple;
}