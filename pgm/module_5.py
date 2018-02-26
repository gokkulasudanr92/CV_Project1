# Module 5 - Factor Analyzer Model
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np, numpy.linalg
from scipy import linalg
from sklearn.decomposition import PCA
from scipy.stats import multivariate_normal
import math
import scipy.special as special
import scipy.optimize as optimize

def near_psd(x, epsilon=0):
    '''
    Calculates the nearest postive semi-definite matrix for a correlation/covariance matrix

    Parameters
    ----------
    x : array_like
      Covariance/correlation matrix
    epsilon : float
      Eigenvalue limit (usually set to zero to ensure positive definiteness)

    Returns
    -------
    near_cov : array_like
      closest positive definite covariance/correlation matrix

    Notes
    -----
    Document source
    http://www.quarchome.org/correlationmatrix.pdf

    '''
    if min(np.linalg.eigvals(x)) > epsilon:
        return x

    # Removing scaling factor of covariance matrix
    n = x.shape[0]
    var_list = np.array([np.sqrt(x[i,i]) for i in xrange(n)])
    y = np.array([[x[i, j]/(var_list[i]*var_list[j]) for i in xrange(n)] for j in xrange(n)])

    # getting the nearest correlation matrix
    eigval, eigvec = np.linalg.eig(y)
    val = np.matrix(np.maximum(eigval, epsilon))
    vec = np.matrix(eigvec)
    T = 1/(np.multiply(vec, vec) * val.T)
    T = np.matrix(np.sqrt(np.diag(np.array(T).reshape((n)) )))
    B = T * vec * np.diag(np.array(np.sqrt(val)).reshape((n)))
    near_corr = B*B.T    

    # returning the scaling factors
    near_cov = np.array([[near_corr[i, j]*(var_list[i]*var_list[j]) for i in xrange(n)] for j in xrange(n)])
    return near_cov

training_faces_folder = "../training/faces/"
faces_images = []
faces_images_rescaled_grayscale = []
faces_size = 1000
faces_tuple_list = []

for filename in os.listdir(training_faces_folder):
    img = cv2.imread(os.path.join(training_faces_folder, filename), flags = cv2.IMREAD_COLOR)
    RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    Gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    Gray_img = cv2.resize(Gray_img, (10, 10))
    if img is not None:
        faces_images.append(RGB_img)
        faces_images_rescaled_grayscale.append(Gray_img)

training_non_faces_folder = "../training/non_faces/"
non_faces_images = []
non_faces_images_rescaled_gray = []
non_faces_mean = np.zeros((60, 60, 3), dtype = np.float)
non_faces_size = 1000

for filename in os.listdir(training_non_faces_folder):
    img = cv2.imread(os.path.join(training_non_faces_folder, filename), flags = cv2.IMREAD_COLOR)
    RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    Gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    Gray_img = cv2.resize(Gray_img, (10, 10))
    if img is not None:
        non_faces_images.append(RGB_img)
        non_faces_images_rescaled_gray.append(Gray_img)

D = 100
K = 4
# Faces Factor Analyzer
dataset_images = []
for i in faces_images_rescaled_grayscale:
    # for i in faces_images:
    im_reshape = i.reshape((1, D))
    dataset_images.append(im_reshape)

dataset_matrix = np.vstack(tuple(dataset_images))

# Initialize mean 
mean = np.mean(dataset_matrix, axis = 0)
mean = mean.reshape((1, D))

# print mean.shape
# print mean

# Initialize Phi
np.random.seed(0)
phi = np.random.randn(D, K)

sig = np.zeros((D, D))
for i in range(0, 1000):
    x = dataset_matrix[i, :].reshape((D, 1))
    temp_mean = mean.reshape((D, 1))
    x_minus_mean = x - temp_mean
    sig += np.matmul(x_minus_mean, x_minus_mean.T)
sig = sig / 1000
sig = np.square(sig)
sig = np.sum(sig, axis = 0)
sig = sig / 1000
sig = sig.reshape((1, D))

# print sig.shape
# print sig

iterations = 0
previous_L = 1000000
precision = 0.01
while True:
    ## Expectation Step
    divide_array = np.divide(1, sig)

    inv_sig = np.diag(divide_array[0])

    E_h_i = np.zeros((1000, K))
    phi_transpose_times_sig_inv = np.matmul(phi.T, inv_sig)
    temp_inv = np.linalg.inv(np.matmul(phi_transpose_times_sig_inv, phi) + np.identity(K))
    for i in range(0, 1000):
        x = dataset_matrix[i, :].reshape((D, 1))
        temp_mean = mean.reshape((D, 1))
        x_mean = x - temp_mean
        t = np.matmul(temp_inv, phi_transpose_times_sig_inv)
        t = np.matmul(t, x_mean)
        E_h_i[i, :] = t.T

    E_h_i_h_i_transpose = np.zeros((K, K, 1000))
    for i in range(0, 1000):
        temp_E_h_i = E_h_i[i, :].reshape((K, 1))
        temp = np.matmul(temp_E_h_i, temp_E_h_i.T)
        temp = temp_inv + temp
        E_h_i_h_i_transpose[:, :, i] = temp

    ## Maximization Step
    E_h_i_h_i_transpose_sum_inv = np.zeros((K, K))
    for i in range(0, 1000):
        E_h_i_h_i_transpose_sum_inv += E_h_i_h_i_transpose[:, :, i]
    E_h_i_h_i_transpose_sum_inv = np.linalg.inv(E_h_i_h_i_transpose_sum_inv)

    # print E_h_i_h_i_transpose_sum_inv.shape
    # print E_h_i_h_i_transpose_sum_inv

    # Update new phi value
    phi_new = np.zeros((D, K))
    for i in range(0, 1000):
        x = dataset_matrix[i, :].reshape((D, 1))
        temp_mean = mean.reshape((D, 1))
        x_mean = x - temp_mean
        temp_E_h_i = E_h_i[i, :].reshape((1, K))
        phi_new += np.matmul(x_mean, temp_E_h_i)

    phi_new = np.matmul(phi_new, E_h_i_h_i_transpose_sum_inv)
    phi = phi_new

    sig_new = np.zeros((D, D))
    for i in range(0, 1000):
        x = dataset_matrix[i, :].reshape((D, 1))
        temp_mean = mean.reshape((D, 1))
        x_mean = x - temp_mean
        prod_x_mean = np.matmul(x_mean, x_mean.T)
        temp_E_h_i = E_h_i[i, :].reshape((K, 1))
        temp_phi_E_h_i = np.matmul(phi, temp_E_h_i)
        temp_phi_E_h_i_x_mean_transpose = np.matmul(temp_phi_E_h_i, x_mean.T)
        sig_new += prod_x_mean - temp_phi_E_h_i_x_mean_transpose
    sig_new = sig_new.diagonal().reshape((1, D))
    sig_new = sig_new / 1000
    sig = sig_new

    # Diagonal Covariance
    sig_diag = np.diag(sig[0])

    ## Compute data log likelihood
    mvn_cov = np.matmul(phi, phi.T) + sig_diag
    mvn = multivariate_normal(mean[0], mvn_cov)
    pdf = mvn.pdf(dataset_matrix)

    L = 0
    for i in range(0, 1000):
        if pdf[i] != 0.0:
            L += np.log(pdf[i])
    
    iterations += 1
    print
    print "Iterations Completed: " + str(iterations)
    print previous_L
    print L
    print

    if abs(L - previous_L) < precision or iterations > 1000000:
        break
    else:
        previous_L = L