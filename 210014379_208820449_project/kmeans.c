#include "capi.h"

/*k-means - 
This file includes kmeans, kmeans++,
and extra functions created for clarity*/

double calculate_distance(double* point1, double* point2 ,int d) {
    /*Calculates the distance between two points of length d*/
	int i;
	double ret = 0.0;
	for (i = 0 ; i < d ; i++) {
		ret += (point1[i] - point2[i])*(point1[i] - point2[i]);
	}
	return ret;
}

double calculate_Di(double* point, double* centroids_arr, int d, int j) {
    /*Calculates the Di - the distance of xi from centroids*/
	double min_dist = calculate_distance(point, centroids_arr, d);
	double dist;
	int z;
	for (z = 0 ; z < j ; z++) {
		dist = calculate_distance(point, centroids_arr + d*z, d);
		if (dist < min_dist) {
			min_dist = dist;
		}
	}
	return min_dist;
}

double* calculate_di_arr(double* points, double* centroids_arr, int n, int d, int j) {
    /*Calculates the distance array of all the values needed to cluster*/
	double* di_arr;
	di_arr = (double *) new_malloc(n, sizeof(double));//double[n];
	if (di_arr == NULL){
			return NULL;
	}
	int i;
	for (i = 0 ; i < n ; i++) {
		di_arr[i] = calculate_Di(points + d*i, centroids_arr, d, j);
	}
	return di_arr;
}

int update_clusters(int n, int d, int k, int *cluster_assign, double *centroids_arr, double *obs_array, double eps){
    /*Used for kmeans, here we update the clusters during each iteration*/
	int i,j, changed;
	int *cluster_sizes;
	double *new_centroids_arr;
	changed = 0;
	cluster_sizes = new_calloc(k, sizeof(int));
	if (cluster_sizes == NULL){
		return -1;
	}
	new_centroids_arr = new_calloc(k*d, sizeof(double));
	if (new_centroids_arr == NULL){
		new_free(cluster_sizes);
		return -1;
	}
	for(i = 0; i < n; i++){
		for(j = 0; j < d; j++){
			new_centroids_arr[cluster_assign[i]*d + j] += obs_array[i*d + j];
		}
		cluster_sizes[cluster_assign[i]]++;
	}
	for(i = 0; i < k; i++){
		for(j = 0; j < d; j++){
			if(cluster_sizes[i] != 0){ 
				/*If a cluster size is zero, we do not change the centroid.
				This is a precaution, as this case is highly improbable.
				If it is zero, the classification might be inaccurate but
				we chose to still proceed.*/
				new_centroids_arr[i*d + j] = new_centroids_arr[i*d + j] / cluster_sizes[i];
			}
			if(fabs(new_centroids_arr[i*d + j] - centroids_arr[i*d + j]) > eps){
				changed = 1;
			}
			centroids_arr[i*d + j] = new_centroids_arr[i*d + j];
		}
	}
	new_free(new_centroids_arr);
	new_free(cluster_sizes);
	return changed;
}

double* initialize_centroids(double* obs_arr, int K, int N, int d){
    /*This is essentially kmeans++, which is used to
    calculate the initial centroids array for kmeans*/
    srand(0);
	int j, i;
    double* centroids_arr;
	centroids_arr = (double *) new_malloc(K * d, sizeof(double));
	if (centroids_arr == NULL){
		return NULL;
	}
	int index_i = (rand() % N);
	double* di_arr;
	double summ;
	int selected = 0;
	double* probs;
	probs = (double *) new_malloc(N, sizeof(double));
	if (probs == NULL){
		new_free(centroids_arr);
		return NULL;
	}
	double* stacked_probs;
	stacked_probs = (double *) new_malloc(N, sizeof(double));
	if (stacked_probs == NULL){
		new_free(centroids_arr);
		new_free(probs);
		return NULL;
	}
	for ( j = 0 ; j < d ; j++) {
	    centroids_arr[j] = obs_arr[index_i*d + j];
    }
	for ( j = 1 ; j < K ; j++) {
		di_arr = calculate_di_arr(obs_arr, centroids_arr, N, d, j);
		if (di_arr == NULL){
			new_free(centroids_arr);
			new_free(probs);
			new_free(stacked_probs);
			return NULL;
		}
		summ = 0.0;
		for (i = 0 ; i < N ; i++) {
			summ += di_arr[i];
		}
		for ( i = 0 ; i < N ; i++) {
			probs[i] = di_arr[i] / summ;
			stacked_probs[i] = probs[i];
			if (j > 0) {
				stacked_probs[i] += stacked_probs[i - 1];
			}
		}
		double r = fabs((double)rand() / RAND_MAX);
		for ( i = 0 ; i < N ; i++) {
			if (r < stacked_probs[i]) {
				selected = i;
				break;
			}
		}
		for ( i = 0 ; i < d ; i++) {
		    centroids_arr[j * d + i] = obs_arr[selected * d + i];
		}
		new_free(di_arr);
	}
	new_free(probs);
	new_free(stacked_probs);
	return centroids_arr;
}

int* k_means_pp(double* obs_arr, int K, int N, int d, int MAX_ITER, double eps) {
    /*Calculates kmeans using kmeans++*/
	int i,j;
	double* centroids_arr;
	centroids_arr = initialize_centroids(obs_arr, K, N, d);
	if (centroids_arr == NULL){
		return NULL;
	}
	int changed = 1;
	double iter, min_dist, dist_j;
	int min_clust;
	int* cluster_assign;
	cluster_assign = (int *) new_malloc(N, sizeof(int));
	if (cluster_assign == NULL){
		new_free(centroids_arr);
		return NULL;
	}
	iter = 0;
	while(changed == 1 && iter < MAX_ITER){
		changed = 0;
		for ( i = 0 ; i < N ; i++) {
			min_dist = -1.0;
			min_clust = -1;
			for ( j = 0 ; j < K ; j++) {
				dist_j = calculate_distance(centroids_arr + j*d, obs_arr + i*d, d);
				if (min_dist == -1 || min_dist > dist_j) {
					min_dist = dist_j;
					min_clust = j;
				}
			}
			cluster_assign[i] = min_clust;
		}
		changed = update_clusters(N, d, K, cluster_assign, centroids_arr, obs_arr, eps);
		if(changed == -1){
			new_free(centroids_arr);
			new_free(cluster_assign);
			return NULL;
		}
		iter += 1;
	}
	new_free(centroids_arr);
	return cluster_assign;
}
