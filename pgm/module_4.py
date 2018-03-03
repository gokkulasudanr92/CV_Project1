# Module 4 - Mixture of t-distribution model
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

def multivariate_t(X, mean, covariance, nu, D, size):
    c = np.exp(special.gammaln((nu + D) / 2)) - special.gammaln(nu / 2)
    c = c / (((nu * math.pi) ** (D / 2)) * (math.sqrt(np.linalg.det(covariance))))

    pdf = np.zeros((1, size))
    temp_mean = mean.reshape((D, 1))
    for i in range(0, size):
        temp_data = X[i, :].reshape((D, 1))
        temp_data_minus_mean = temp_data - temp_mean
        temp_data_minus_mean_transpose_inv_sig = np.matmul(temp_data_minus_mean.T, np.linalg.inv(covariance))
        temp_data_minus_mean_transpose_inv_sig_data_minus_mean = np.matmul(temp_data_minus_mean_transpose_inv_sig, temp_data_minus_mean)
        pdf[0, i] = temp_data_minus_mean_transpose_inv_sig_data_minus_mean

    for i in range(0, size):
        pdf[0, i] = 1 + (pdf[0, i] / nu)
        pdf[0, i] = pdf[0, i] ** (-1 * (nu + D) / 2)
        pdf[0, i] = c * pdf[0, i]
    
    return pdf

def mixture_t_cost_func(nu, D, posterior_matrix, E_h_i, E_log_h_i, k, size):
    first_term = -1 * special.psi(nu / 2)
    second_term =  np.log(nu / 2)
    third_term = 1
    fourth_term = 0.0
    fourth_term_denominator = 0.0
    for i in range(0, size):
        fourth_term_denominator += posterior_matrix[i, k]
        E_log_h_i_minus_E_h_i = E_log_h_i[k, i] - E_h_i[k, i]
        fourth_term += posterior_matrix[i, k] * E_log_h_i_minus_E_h_i
    fourth_term /= fourth_term_denominator
    fifth_term = special.psi((nu + D) / 2)
    sixth_term = -1 * np.log((nu + D) / 2)
    # print first_term, second_term, third_term, fourth_term, fifth_term, sixth_term
    val = first_term + second_term + third_term + fourth_term + fifth_term + sixth_term
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

K = 5
## Mixture of Faces t Distribution
dataset_images = []
for i in faces_images_rescaled_grayscale:
    # for i in faces_images:
    im_reshape = i.reshape((1, D))
    dataset_images.append(im_reshape)

dataset_matrix = np.vstack(tuple(dataset_images))

faces_nu_k = []
faces_lambda_k = []
faces_mean_k = []
faces_sig_k = []
faces_covariance_epsilon = []
for k in range(2, K + 1):
    # Initialize the teta parameters
    # Initialize Mean
    mean_k = np.zeros((D, k))
    means_index = np.random.permutation(1000)
    for i in range(0, k):
        mean_k[:, i] = dataset_matrix[means_index[i], :].reshape(D)

    # Initialize Nu to large value
    nu_k = [10] * k

    # Initialize Sigma
    sig_k = np.zeros((D, D, k))
    for j in range(0, k):
        dataset_variance = np.zeros((D, D))
        for i in range(0, 1000):
            x = dataset_matrix[i, :].reshape((1, D))
            mu = mean_k[:, j].reshape((1, D))
            temp = x - mu
            temp = temp.reshape((1, D))
            dataset_variance = dataset_variance + np.matmul(temp.T, temp)
        dataset_variance /= 1000
        sig_k[:, :, j] = dataset_variance

    # Initialize lambda_k
    lambda_k = [1 / float(k)] * k

    precision = 0.01
    iterations = 0
    previous_L = 1000000
    print k
    while True:
        ## Expectation Step
        # Computing Delta's
        delta_i = np.zeros((k, 1000))
        for j in range(0, k):
            sig_k_inverse = np.linalg.inv(sig_k[:, :, j])
            for i in range(0, 1000):
                x = dataset_matrix[i, :].reshape((1, D))
                mu = mean_k[:, j].reshape((1, D))
                temp = x - mu
                temp_delta = np.matmul(temp, sig_k_inverse)
                temp_delta = np.matmul(temp_delta, temp.T)
                delta_i[j, i] = temp_delta

        # Compute E[h i]'s
        E_h_i = np.zeros((k, 1000))
        for j in range(0, k):
            for i in range(0, 1000):
                E_h_i[j, i] = (nu_k[j] + D) / (nu_k[j] + delta_i[j, i])

        # Compute the E[log h i]
        E_log_h_i = np.zeros((k, 1000))
        for j in range(0, k):
            for i in range(0, 1000):
                temp = special.psi((nu_k[j] + D) / 2) - np.log(nu_k[j] / 2 + delta_i[j, i] / 2)
                E_log_h_i[j, i] = temp
        
        # Compute the l_posterior matrix
        posterior_matrix = np.zeros((1000, k))
        for i in range(0, 1000):
            data_entry = dataset_matrix[i, :].reshape((1, D))
            for j in range(0, k):
                posterior_val = multivariate_t(data_entry, mean_k[:, j], sig_k[:, :, j], nu_k[j], D, 1)
                posterior_matrix[i, j] = lambda_k[j] * posterior_val[0]

        ## Maximization Step
        # Compute the new lambda_k's
        posterior_matrix_colwise_sum = np.sum(posterior_matrix, axis = 0)
        total_posterior_matrix_sum = np.sum(posterior_matrix)
        for j in range(0, k):
            lambda_k[j] = posterior_matrix_colwise_sum[j] / total_posterior_matrix_sum
        
        # Compute new mean_k's
        for j in range(0, k):
            new_mean_k = np.zeros((D, 1))
            sum_denominator = 0.0
            for i in range(0, 1000):
                temp_data = dataset_matrix[i, :].reshape((D, 1))
                temp_prod = posterior_matrix[i, j] * E_h_i[j, i]
                temp = temp_prod * temp_data
                new_mean_k += temp
                sum_denominator +=  temp_prod
            new_mean_k /= sum_denominator
            mean_k[:, j] = new_mean_k.reshape(D)

        # Compute new sig_k's
        for j in range(0, k):
            new_sig_k = np.zeros((D, D))
            sum_denominator = 0.0
            for i in range(0, 1000):
                temp_data = dataset_matrix[i, :].reshape((D, 1))
                temp_mean = mean_k[:, j].reshape((D, 1))
                temp_data_minus_mean = temp_data - temp_mean
                temp_prod = posterior_matrix[i, j] * E_h_i[j, i]
                temp = np.matmul(temp_data_minus_mean, temp_data_minus_mean.T)
                new_sig_k += temp_prod * temp
                sum_denominator += temp_prod
            new_sig_k /= sum_denominator
            sig_k[:, :, j] = new_sig_k
        
        # Compute nu_k's
        for j in range(0, k):
            temp = optimize.fminbound(mixture_t_cost_func, 0, 10, args=(D, posterior_matrix, E_h_i, E_log_h_i, j, 1000))
            # temp = optimize.fsolve(mixture_t_cost_func, 0.0, args=(D, posterior_matrix, E_h_i, E_log_h_i, j, 1000))
            nu_k[j] = temp
        
        iterations += 1
        if iterations >= 1:
            break
    faces_lambda_k.append(lambda_k)
    faces_mean_k.append(mean_k)
    faces_sig_k.append(sig_k)
    faces_nu_k.append(nu_k)

print faces_lambda_k
print faces_nu_k
print faces_mean_k
print faces_sig_k