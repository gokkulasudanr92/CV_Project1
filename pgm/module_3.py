# Module 3 - t Distribution Model
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

def t_cost(nu, E_h_i, E_h_i_sum, E_log_h_i, E_log_h_i_sum):
    nu_half = nu / 2
    size = 1000
    val = size * nu_half * np.log(nu_half) - special.gammaln(nu_half)
    val = val + ((nu_half - 1) * E_log_h_i_sum)
    val = val - (nu_half * E_h_i_sum)
    return val

D = 100
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

## Faces t Distribution
dataset_images = []
for i in faces_images_rescaled_grayscale:
    # for i in faces_images:
    im_reshape = i.reshape((1, D))
    dataset_images.append(im_reshape)

dataset_matrix = np.vstack(tuple(dataset_images))

faces_nu_k = []
faces_mean_k = []
faces_sig_k = []
faces_covariance_epsilon = []

# Initialize the teta parameters
# Initialize Mean
dataset_mean = np.mean(dataset_matrix, axis = 0)
mean_k = dataset_mean

# Initialize Nu to large value
nu_k = 10

# Initialize Sigma
dataset_variance = np.zeros((D, D))
for i in range(0, 1000):
    x = dataset_matrix[i, :]
    mu = mean_k
    temp = x - mu
    temp = temp.reshape((1, D))
    dataset_variance = dataset_variance + np.matmul(temp.T, temp)

dataset_variance /= 1000
sig_k = dataset_variance

precision = 0.01
iterations = 0
previous_L = 1000000
delta_i = np.zeros((1, 1000))
while True:
    ## Expectation Step
    sig_k_inverse = np.linalg.inv(sig_k)
    for i in range(0, 1000):
        x = dataset_matrix[i, :].reshape((1, D))
        mu = mean_k.reshape((1, D))
        temp = x - mu
        temp_delta = np.matmul(temp, sig_k_inverse)
        temp_delta = np.matmul(temp_delta, temp.T)
        delta_i[0, i] = temp_delta
    
    # Compute E[h i]
    E_h_i = np.zeros((1, 1000))
    for i in range(0, 1000):
        E_h_i[0, i] = (nu_k + D) / (nu_k + delta_i[0, i])

    # print E_h_i.shape
    # print E_h_i
    # print delta_i.shape
    # print delta_i

    # Compute the E[log h i]
    E_log_h_i = np.zeros((1, 1000))
    for i in range(0, 1000):
        temp = special.psi((nu_k + D) / 2) - np.log(nu_k / 2 + delta_i[0, i] / 2)
        E_log_h_i[0, i] = temp

    # print E_log_h_i.shape
    # print E_log_h_i

    # Update new mean
    E_h_i_sum = np.sum(E_h_i, axis = 1)
    new_mean_k = np.zeros((1, D))
    for i in range(0, 1000):
        temp_xi = dataset_matrix[i, :].reshape((1, D))
        temp_mean = mean_k.reshape((1, D))
        temp = E_h_i[0, i] * temp_xi
        new_mean_k += temp
    new_mean_k /= E_h_i_sum
    mean_k = new_mean_k
    
    # Update new sigma
    new_sig_k = np.zeros((D, D))
    for i in range(0, 1000):
        temp_xi = dataset_matrix[i, :].reshape((D, 1))
        temp_mean = mean_k.reshape((D, 1))
        temp = temp_xi - temp_mean
        temp_mul = np.matmul(temp, temp.T)
        new_sig_k += E_h_i[0, i] * temp_mul
    new_sig_k /= E_h_i_sum
    sig_k = new_sig_k

    E_log_h_i_sum = np.sum(E_log_h_i, axis = 1)
    
    # Update new nu
    new_nu = optimize.fminbound(t_cost, 0, 10, args=(E_h_i, E_h_i_sum, E_log_h_i, E_log_h_i_sum))
    # for i in range(1, 100):
    #     print t_cost(i, E_h_i, E_h_i_sum, E_log_h_i, E_log_h_i_sum), 
    # print
    # break
    nu_k = new_nu

    # Compute Log likelihood
    new_delta_i = np.zeros((1, 1000))
    sig_k_inverse = np.linalg.inv(sig_k)
    for i in range(0, 1000):
        temp_xi = dataset_matrix[i, :].reshape((1, D))
        temp_mean = mean_k.reshape((1, D))
        temp_data = temp_xi - temp_mean
        temp = np.matmul(temp_data, sig_k_inverse)
        temp = np.matmul(temp, temp_data.T)
        new_delta_i[0, i] = temp
    delta_i = new_delta_i

    # Compute L
    size = 1000
    (sig_sign, sig_k_logdet) = np.linalg.slogdet(sig_k)
    sig_k_logdet_half = sig_k_logdet / 2
    gammaln_nu_D_half = special.gammaln((nu_k + D) / 2)
    log_nu_pi = (D / 2) * np.log(nu_k * math.pi)
    gammaln_nu_half = special.gammaln(nu_k / 2)

    L = size * (gammaln_nu_D_half - log_nu_pi - sig_k_logdet_half - gammaln_nu_half)
    log_delta_nu_sum = 0
    for i in range(0, 1000):
        temp = np.log(1 + (delta_i[0, i] / nu_k))
        log_delta_nu_sum += temp
    log_delta_nu_sum /= 2
    L = L - ((nu_k + D) * log_delta_nu_sum)

    iterations += 1
    print
    print "Iterations Completed: " + str(iterations)
    print

    if abs(L - previous_L) < precision or iterations > 100:
        break
    else:
        previous_L = L

faces_nu_k = nu_k
faces_mean_k = mean_k
faces_sig_k = sig_k

print faces_nu_k
print faces_mean_k.shape
print faces_mean_k
print faces_sig_k.shape
print faces_sig_k
