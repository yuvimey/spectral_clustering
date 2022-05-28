extern int counter;
//DEBUGGINGTOOLKIT
extern int counter;
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
void print_array( int * array, int array_len);
void print_double_array( double * array, int array_len);
void print_double_matrix( double * array, int rows, int cols);

void output_time(double* time_array, int n);
//HELPFULFUNCTIONS
void *new_malloc(int len, int sizeofA);
void *new_calloc(int len, int sizeofA);
void new_free(void * A);
double VectorMult(double* A, double* B, int d);
double ColMult(double* A, double* B, int colA, int colB, int row_length);
double* WeightedAdjGraph(double* A, int n, int d);
double* VectorSub(double* A, double* B, int d);
double* MatrixMult(double* A, double* B, int n);
int Line6InQR(double* A, double* B, double e, int n);
double* DiagDegSqrtMatrix(double* W, int n);
double* Lnorm(double* W, int n);
double *buildU(double *A, double *Q, int n, int k);
int compare(const void* a, const void* b);
int findK(double* A, int n);
double* Normalize(double* U, int rows, int cols, double eps);
//KMEANS
double calculate_distance(double* point1, double* point2 ,int d);
double calculate_Di(double* point, double* centroids_arr, int d, int j);
double* calculate_di_arr(double* points, double* centroids_arr, int n, int d, int j);
int update_clusters(int n, int d, int k, int *cluster_assign, double *centroids_arr, double *obs_array, double eps);
double* initialize_centroids(double* obs_arr, int K, int N, int d);
int* k_means_pp(double* obs_arr, int K, int N, int d, int MAX_ITER, double eps);
//ALGORITHMS
struct tuple {
	double* A;
	double* B;
	int C;
	int* D;
};

struct tuple* ModifiedGrahmSchmidt(double* A, int n, double eps);
struct tuple* QRIt(double* A, int n, double e);
struct tuple* NormSpecClust(double* X, int n, int d, double e, int Random, int K, int max_iter);


