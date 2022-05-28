import argparse
import numpy as np
import random
from sklearn import datasets
import math
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import capi
import tasks

MAX_ITER = tasks.MAX_ITER
MAX_K_3D = tasks.MAX_K_3D
MAX_N_3D = tasks.MAX_N_3D
MAX_K_2D = tasks.MAX_K_2D
MAX_N_2D = tasks.MAX_N_2D
EPSILON = tasks.EPSIlION

"""
For our max capacity, we found that n = 500 runs
under 5 minutes (on average) for both the two and three dimensional clustering.
We chose k = 12 as we felt the range between 6 to 12 as it best showed 
the impact the value of k has on the quality of the classification (the jaccard measure).
"""

"""
Error handling notes:
- Division by zero: with the exception of that in k-means, 
instead of allowing for objects to be equal to 'inf' we chose to 
make them equal to 0.0, as stated to do in the forums. For the risk
in the update_clusters function, we did not change the centroid and
continued with the program.
- Memory allocation: if memory allocation fails, the function returns NULL
and the program fails.
"""

def main():

    """ 
	The main function of the program,
    assigns clusters to points using k_means_pp and NormSpecClust functions
    generates output txt files using 'generate_output_files' function
    generates pdf file using 'visualize' function.
	"""

    my_parser = argparse.ArgumentParser(description='k-mean and spectral clustering')
    my_parser.add_argument('K', type=int)
    my_parser.add_argument('N', type=int)
    my_parser.add_argument('Random')

    dim = random.randint(2, 3)

    args = my_parser.parse_args()

    rand_int = 0
    if args.Random == "True":
        rand_int = 1
        if dim == 2:
            n = random.randint(MAX_N_2D//2,MAX_N_2D)
            k = random.randint(MAX_K_2D//2,MAX_K_2D)
        else:
            n = random.randint(MAX_N_3D // 2, MAX_N_3D)
            k = random.randint(MAX_K_3D // 2, MAX_K_3D)

    else:
        if (args.K <= 0) or (args.N <= 0) or (args.K >= args.N):
            print("Problem with input!")
            exit(1)
        n = args.N
        k = args.K
    observ_arr = datasets.make_blobs(n_samples=n, n_features=dim, centers=k)

    spectral_cluster_assign, k = NormSpecClust(observ_arr[0], n, k, EPSILON, rand_int, MAX_ITER)

    kmeans_cluster_assign = capi.clustering_capi(k, n, dim, MAX_ITER, EPSILON,observ_arr[0].tolist())

    generate_output_files(observ_arr[1], spectral_cluster_assign, kmeans_cluster_assign,observ_arr[0], n, k)

    visualize(observ_arr, spectral_cluster_assign, kmeans_cluster_assign, observ_arr[1], n, k, dim)


# FUNCTIONS FOR SPECTRAL CLASSIFICATION:

def NormSpecClust(X, n, k, eps, Random, max_iter):

    # Algorithm 3 The Normalized Spectral Clustering Algorithm'
    # returns an array where the i'th variable corresponds to X[i]'s cluster
    # also returns the value k (needed if Random is set to 'True' and the value is recalculated)

    L_norm = calcLnorm(X)   # Lines 1 + 2

    A, Q = QRIteration(L_norm, eps)     # Line 3

    eigenvalues = A.diagonal()
    eigenvectors = Q.transpose()
    eigenvectors = eigenvectors[eigenvalues.argsort()] # sorting eigenvectors via eigenvalues
    eigenvectors = eigenvectors.transpose()


    if Random == 1:
        k = findK(np.sort(eigenvalues))     # Heuristic Eigenmap

    eigenvectors = eigenvectors.transpose()
    eigenvectors = eigenvectors[:k]         # Line 4 eigenvectors := U
    eigenvectors = eigenvectors.transpose()

    eigenvectors = NormalizeVectors(eigenvectors, eps)   # Line 5 eigenvectors := T

    spectral_cluster_assign = capi.clustering_capi(k, n, k, max_iter, eps, eigenvectors.tolist()); # Lines 6 + 7 

    return spectral_cluster_assign, k


def calcLnorm(observ_arr):

    # calculates the L-norm matrix from an array of given points

    weights_mat = squareform(pdist(observ_arr, lambda u, v: math.exp(-(np.linalg.norm(u - v)) / 2)))
    d_mat = np.diag(1 / (np.sqrt(weights_mat.sum(axis=1))))     # d_mat is D^(-1/2)
    if np.isnan(d_mat).any():
        print("Error: calcLnorm : d_mat has invalid (nan) values! ")
        exit(1)
    identity_mat = np.identity(np.shape(d_mat)[0])
    D_W_D = np.dot(np.dot(d_mat, weights_mat), d_mat)   # D_W_D is (D^(-1/2))*W*(D^(-1/2))
    res = identity_mat - D_W_D
    return res
	

def QRIteration(matrix, eps):

    # Algorithm 2 The QR Iteration Algorithm

    A_ = np.copy(matrix)
    Q_ = np.identity(np.shape(A_)[0])
    for i in range(A_.shape[0]):
        Q, R = gramSchmidt(A_, eps)
        A_ = np.matmul(R, Q)
        diff = abs(abs(Q_) - abs(np.matmul(Q_, Q)))
        diff = diff.max()
        if abs(diff) <= eps:        #checking if |Q|-|Q_*Q| in [-eps,eps] via the max value of the difference matrix
            return A_, Q_
        Q_ = np.matmul(Q_, Q)
    return A_, Q_


def gramSchmidt(matrix, eps):

    # Algorithm 1 The Modified Gram-Schmidt Algorithm, returns the QR decomposition of a given matrix

    n = matrix.shape[0]
    u_mat = matrix.copy()                   # matrix := A, u_mat := U
    u_mat = np.transpose(u_mat)             # easier to handle the column vectors as rows
    q_mat = np.zeros((n, n))                # q_mat := Q
    r_mat = np.zeros((n, n))                # r_mat := R
    for i in range(n):
        r_mat[i][i] = (np.linalg.norm(u_mat[i]))
        if r_mat[i][i] <= eps:      #Cases where R_ii is zero (less than epsilon) the value Q_ii is set to zero
            q_mat[i] = q_mat[i]*0.0
        else:
            q_mat[i] = (u_mat[i]) / r_mat[i][i]
        if np.isnan(q_mat).any():
            print("Error: gramSchmidt : q_mat has invalid (nan) values! ")
            exit(1)
        r_mat[i][i+1:] = np.einsum('n,in->i', q_mat[i], u_mat[i+1:])	# Lines 5 and 6 of the algorithm
        u_mat[i+1:] -= np.einsum('j,i->ji', r_mat[i][i+1:], q_mat[i])	# Lines 5 and 7 of the algorithm
    return np.transpose(q_mat), r_mat



def findK(eigenvalues):

    # The Eigengap Heuristic, if Random is set to True, the function returns k using the calculation specified in chapter 4.3

    max_arg = 0
    max_val = 0
    iter_range = eigenvalues.size//2
    if eigenvalues.size % 2 != 0:
        iter_range += 1
    for i in range(iter_range):
        val = abs(eigenvalues[i + 1] - eigenvalues[i])
        if val > max_val:
            max_val = val
            max_arg = i + 1
    return max_arg



def NormalizeVectors(eigenvectors, eps):

    # Returns normalized eigenvectors,
    # Each row is divided by the euclidean norm of the row.
    # Satisfies Line 5 in Algorithm 3 of the project: recieves U returns T

    for i in range(eigenvectors.shape[0]):
        norm = np.linalg.norm(eigenvectors[i])
        if norm != 0 and norm != np.nan:
            euclid_norm_i = np.linalg.norm(eigenvectors[i])
            if(euclid_norm_i <= eps):
                temp = eigenvectors[i]*0         #Cases where the norm is zero (less than epsilon) the vector is set to zero
            else:
                temp = eigenvectors[i] / euclid_norm_i;
                if np.isnan(temp).any():
                    print("NormalizeVectors : couldn't calculate row ")
                    exit(1)
        else:
            temp = eigenvectors[i]
        if np.isnan(temp).any():
            print( "NormalizeVectors : couldn't calculate row ")
            exit(1)
        eigenvectors[i] = temp
    return eigenvectors


# FUNCTIONS TO GENERATE OUTPUT FILES:

def jaccard_measure(original_assign_clusters, other_assign_clusters):

    # Returns the Jaccard measure between the "original" clusters and "other" clusters

    n = len(original_assign_clusters)
    any_pairs = 0
    both_pairs = 0
    for i in range(n):
        for j in range(i+1, n): # calculating pairs via when i < j
            if original_assign_clusters[i] == original_assign_clusters[j]:
                any_pairs += 1
                if other_assign_clusters[i] == other_assign_clusters[j]:
                    both_pairs += 1
            elif other_assign_clusters[i] == other_assign_clusters[j]:
                any_pairs += 1
    if any_pairs == 0:
        return 0.0
    jaccard = both_pairs/any_pairs
    return jaccard


def generate_output_files(sklearn_cluster_assign, spectral_cluster_assign, kmeans_cluster_assign, X, n, k):

    # Generates data.txt and clusters.txt as specified in chapter 6.5

    outputdatastr = ""
    spectral_cluster_strings = ["" for i in range(k)]
    kmeans_cluster_strings = ["" for i in range(k)]
    for i in range(n):
        cluster_index = int(sklearn_cluster_assign[i])
        outputdatastr += np.array2string(X[i], formatter={'float_kind': lambda x: "%.8f" % x}).strip('[').strip(
            ']').replace(" ", ",")
        outputdatastr += "," + str(cluster_index) + "\n"
        spectral_cluster_strings[spectral_cluster_assign[i]] += str(i) + ","
        kmeans_cluster_strings[kmeans_cluster_assign[i]] += str(i) + ","
    file = open("data.txt", 'w')
    file.write(outputdatastr)
    file.close()
    outputclustersstr = "" + str(k) + "\n"
    for i in range(k):
        outputclustersstr += spectral_cluster_strings[i].strip(',') + "\n"
    for i in range(k):
        outputclustersstr += kmeans_cluster_strings[i].strip(',') + "\n"
    file = open("clusters.txt", 'w')
    file.write(outputclustersstr)
    file.close()


def visualize(observations, spectral_cluster_assign, kmeans_cluster_assign, original_cluster_assign, n, k,d):

    #Generates clusters.pdf with the specified graphs and information

    jaccard_kmeans = jaccard_measure(original_cluster_assign, kmeans_cluster_assign)
    jaccard_spectral = jaccard_measure(original_cluster_assign, spectral_cluster_assign)
    fig = plt.figure()
    if d == 3:
        ax = fig.add_subplot(121, projection='3d')
    else:
        ax = fig.add_subplot(121)
    ax.set_title("Normalized Spectral Clustering", fontsize = 10)
    observations = observations[0]
    x = observations[:, 0]
    y = observations[:, 1]
    if d == 3:
        z = observations[:, 2]
        ax.scatter(x, y, z, c=spectral_cluster_assign)
    elif d == 2:
        ax.scatter(x, y, c=spectral_cluster_assign)

    if d == 3:
        ax = fig.add_subplot(122, projection='3d')
    else:
        ax = fig.add_subplot(122)
    ax.set_title("K-means", fontsize = 10)
    x = observations[:, 0]
    y = observations[:, 1]
    if d == 3:
        z = observations[:, 2]
        ax.scatter(x, y, z, c=kmeans_cluster_assign)
    elif d == 2:
        ax.scatter(x, y, c=kmeans_cluster_assign)
    s = "Data was generated from the values:"
    s += "\n" + "n =" + str(n) + ", k=" + str(len(set(original_cluster_assign))) +"\n"
    s += "The k that was used for both algorithms was " + str(k) + "\n"
    s += "The jaccard measure for spectral clustering: " + "%.2f"%jaccard_spectral + "\n"
    s += "The jaccard measure for K-means: " + "%.2f"%jaccard_kmeans + "\n"
    fig.tight_layout(rect=[0, 0.2, 1, 1])
    fig.text(0.55,0, s, fontsize = 10, horizontalalignment = 'center')
    fig.savefig("clusters.pdf")


# FUNCTIONS FOR K-means (NOT IN USE):

def initialize_centroids(obs_arr, K, N, d):

    # selecting initial centriods using the algorithm specified in HW2
    # ---NOT IN USE (Parallel function in C is more efficient)---

    np.random.seed(0)
    i = 1
    index_i = np.random.choice(N, 1)
    centroids_arr = np.array([[np.float64(0.0) for j in range(d)] for i in range(K)])
    centroids_arr[0] = obs_arr[index_i[0]]
    i = 1
    while i < K:
        di_arr = calculate_di_arr(obs_arr, centroids_arr[0:i])
        summ = np.sum(di_arr)
        probs = di_arr / summ
        if np.isnan(probs).any():
            print( "Error: initialize_centroids : couldn't calculate probs")
            exit(1)
        x = np.random.choice(N, 1, p=probs)
        selected = x[0]
        centroids_arr[i] = obs_arr[selected]
        i += 1
    return centroids_arr


def k_means_pp(obs_arr, K, N, d, max_iter):

    # Kmeans++ Algorithm, as specified in HW1 and HW2
    # ---NOT IN USE (Parallel function in C is more efficient)---

    centroids_arr = initialize_centroids(obs_arr, K, N, d)
    changed = True
    iter = 0
    cluster_assign = [0 for i in range(N)]
    while changed is True and iter < max_iter:
        changed = False
        for i in range(N):
            min_dist = -1.0
            min_cent = -1.0
            for j in range(K):
                dist = np.linalg.norm(obs_arr[i] - centroids_arr[j])
                dist = dist * dist
                if min_dist < 0:
                    min_dist = dist
                    min_cent = j
                if dist < min_dist:
                    min_dist = dist
                    min_cent = j
            if cluster_assign[i] != min_cent:
                cluster_assign[i] = min_cent
                changed = True
        counter_arr = np.zeros(K)
        sum_arr = np.zeros((N, d))
        for i in range(N):
            index = int(cluster_assign[i])
            counter_arr[index] += 1
            sum_arr[index] += obs_arr[i]
        for i in range(K):
            centroids_arr[i] = sum_arr[i] / counter_arr[i]
            if np.isnan(centroids_arr).any():
                print("k_means_pp : couldn't calculate centroids_arr ")
                exit(1)
        iter += 1
    return cluster_assign


def calculate_di_arr(points, centroids_arr):

    # Calculates the distance squared from the nearest centroid for each point.
    # ---NOT IN USE (Parallel function in C is more efficient)---

    di_arr = np.array([np.float64(0.0) for i in range(len(points))])
    i = 0
    for point in points:
        di_arr[i] = calculate_Di(point, centroids_arr)
        i += 1
    return di_arr


def calculate_Di(point, centroids_arr):

    # Calculates the distance squared from the nearest centroid for a given point.
    # ---NOT IN USE (Parallel function in C is more efficient)---

    dist = np.float64(np.linalg.norm(point - centroids_arr[0]))
    dist = dist * dist
    min_dist = np.float64(dist)
    for centroid in centroids_arr:
        dist = np.float64(np.linalg.norm(point - centroid))
        dist = dist * dist
        if dist < min_dist:
            min_dist = dist
    return min_dist



if __name__ == '__main__':
    main()
