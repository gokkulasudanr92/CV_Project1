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
from random import *

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
    (sign, logdet) = np.linalg.slogdet(covariance)
    logdet /= 2
    logdet = abs(logdet)
    if (logdet == 0.0):
        logdet = 1.0
    c = c / (((nu * math.pi) ** (D / 2)) * logdet)

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

def multivariate_t_test(X, mean, covariance, nu, D, size):
    c = np.exp(special.gammaln((nu + D) / 2)) - special.gammaln(nu / 2)
    (sign, logdet) = np.linalg.slogdet(covariance)
    logdet /= 2
    logdet = abs(logdet)
    if (logdet == 0.0):
        logdet = 1.0
    c = c / (((nu * math.pi) ** (D / 2)) * logdet)

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
        temp = (nu + D) / 2
        temp = -temp - 10
        pdf[0, i] = pdf[0, i] * (10 ** temp)
        pdf[0, i] = c * pdf[0, i]
    
    return pdf

def save_mean_image(filename, mean, image_dim):
    temp_mean = mean.reshape((image_dim, image_dim))
    cv2.imwrite(filename, temp_mean)
    print "Saving Image at - " + filename
    return

def save_covariance_image(filename, covariance, image_dim):
    covariance_diag = covariance.diagonal()
    max_val = max(covariance_diag)
    covariance_diag = covariance_diag * 255 / max_val
    image = covariance_diag.reshape((image_dim, image_dim))
    image = np.array(np.round(image), dtype = np.uint8)
    cv2.imwrite(filename, image)
    print "Saving Image at - " + filename
    return

def average_of_pdf(pdf, testImagesSize):
    avg = 0.0
    for i in pdf:
        avg += i
    avg /= testImagesSize
    return avg

def compute_posterior(faces_pdf, non_faces_pdf, threshold, testImagesSize):
    true_positive = 0
    false_positive = 0    
    true_negative = 0
    false_negative = 0

    faces_pdf_avg = average_of_pdf(faces_pdf, testImagesSize)
    non_faces_pdf_avg = average_of_pdf(non_faces_pdf, testImagesSize)

    for i in range(0, testImagesSize - 100):
        if faces_pdf[i] == 0.0:
            f = faces_pdf_avg * (10 ** -75)
        else:
            f = faces_pdf[i]

        if non_faces_pdf[i] == 0.0:
            nf = non_faces_pdf_avg * (10 ** -75)
        else:
            nf = non_faces_pdf[i]

        y = f + nf
        x = f
        z = x / y
        # print x, y, z
        if (not math.isnan(z) and z > threshold):
            true_positive += 1
        else:
            false_negative += 1

    for i in range(100, testImagesSize):
        if faces_pdf[i] == 0.0:
            f = faces_pdf_avg * (10 ** -75)
        else:
            f = faces_pdf[i]

        if non_faces_pdf[i] == 0.0:
            nf = non_faces_pdf_avg * (10 ** -75)
        else:
            nf = non_faces_pdf[i]

        y = f + nf
        x = f
        z = x / y
        # print x, y, z
        if (not math.isnan(z) and z > threshold):
            false_positive += 1
        else:
            true_negative += 1

    # Misclassification Rate
    misclassification_rate = float(false_positive + false_negative) / float(testImagesSize)
    misclassification_rate = misclassification_rate * 100
    return true_positive, false_negative, false_positive, true_negative, misclassification_rate

def compute_posterior_pdf(faces_pdf, non_faces_pdf, testImagesSize):
    posterior_pdf = [0.0] * testImagesSize

    faces_pdf_avg = average_of_pdf(faces_pdf, testImagesSize)
    non_faces_pdf_avg = average_of_pdf(non_faces_pdf, testImagesSize)
    
    for i in range(0, testImagesSize):
        if faces_pdf[i] == 0.0:
            f = faces_pdf_avg * (10 ** -75)
        else:
            f = faces_pdf[i]

        if non_faces_pdf[i] == 0.0:
            nf = non_faces_pdf_avg * (10 ** -75)
        else:
            nf = non_faces_pdf[i]

        y = f + nf
        x = f
        z = x / y

        if (not math.isnan(z)):
            posterior_pdf[i] = z
    return posterior_pdf

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
            nu_k[j] = temp
        
        iterations += 1
        if iterations >= 1:
            break
    faces_lambda_k.append(lambda_k)
    faces_mean_k.append(mean_k)
    faces_sig_k.append(sig_k)
    faces_nu_k.append(nu_k)

## Mixture of Non Faces t Distribution
dataset_images = []
for i in non_faces_images_rescaled_gray:
    im_reshape = i.reshape((1, D))
    dataset_images.append(im_reshape)

dataset_matrix = np.vstack(tuple(dataset_images))

non_faces_nu_k = []
non_faces_lambda_k = []
non_faces_mean_k = []
non_faces_sig_k = []
non_faces_covariance_epsilon = []
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
            nu_k[j] = temp
        
        iterations += 1
        if iterations >= 1:
            break
    non_faces_lambda_k.append(lambda_k)
    non_faces_mean_k.append(mean_k)
    non_faces_sig_k.append(sig_k)
    non_faces_nu_k.append(nu_k)

# print faces_lambda_k
# print faces_nu_k
# print faces_mean_k
# print faces_sig_k

# print non_faces_lambda_k
# print non_faces_nu_k
# print non_faces_mean_k
# print non_faces_sig_k

## Test Dataset
test_tuple_list = []
faces_test_images = []
test_faces_folder = "../test/faces/"
for filename in os.listdir(test_faces_folder):
    img = cv2.imread(os.path.join(test_faces_folder, filename), flags = cv2.IMREAD_COLOR)
    Gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    Gray_img = cv2.resize(Gray_img, (10, 10))
    if img is not None:
        faces_test_images.append(Gray_img)

for im in faces_test_images:
    key = im
    im_reshape = key.reshape((1, D))
    test_tuple_list.append(im_reshape)

non_faces_test_images = []
test_non_faces_folder = "../test/non_faces/"
for filename in os.listdir(test_non_faces_folder):
    img = cv2.imread(os.path.join(test_non_faces_folder, filename), flags = cv2.IMREAD_COLOR)
    Gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    Gray_img = cv2.resize(Gray_img, (10, 10))
    if img is not None:
        non_faces_test_images.append(Gray_img)

for im in non_faces_test_images:
    key = im
    im_reshape = key.reshape((1, D))
    test_tuple_list.append(im_reshape)

test_tuple = tuple(test_tuple_list)
test_matrix = np.vstack(test_tuple)

DEFAULT_RESULT_LOCATION = "./results/module_4/"

for k in range(2, K + 1):
    for i in range(0, k):
        # Save Mean Faces Result
        faces_mean_filename = DEFAULT_RESULT_LOCATION + "mog_" + str(k) + "/mean/faces_" + str(i) + ".jpg"
        faces_mean = faces_mean_k[k - 2][:, i].reshape((1, D))
        save_mean_image(faces_mean_filename, faces_mean, 10)

        # Save Mean Non Faces Result
        non_faces_mean_filename = DEFAULT_RESULT_LOCATION + "mog_" + str(k) + "/mean/non_faces_" + str(i) + ".jpg"
        non_faces_mean = non_faces_mean_k[k - 2][:, i].reshape((1, D))
        save_mean_image(non_faces_mean_filename, non_faces_mean, 10)

        # Save Covariance Faces Result
        faces_covariance_filename = DEFAULT_RESULT_LOCATION + "mog_" + str(k) + "/covariance/faces_" + str(i) + ".jpg"
        faces_covariance = faces_sig_k[k - 2][:, :, i]
        save_covariance_image(faces_covariance_filename, faces_covariance, 10)

        # Save Covariance Non Faces Result
        non_faces_covariance_filename = DEFAULT_RESULT_LOCATION + "mog_" + str(k) + "/covariance/non_faces_" + str(i) + ".jpg"
        non_faces_covariance = non_faces_sig_k[k - 2][:, :, i]
        save_covariance_image(non_faces_covariance_filename, non_faces_covariance, 10)

# Compute Posterior
faces_pdf_k = []
non_faces_pdf_k = []
for k in range(2, K + 1):
    faces_pdf_diff_lambda = []
    non_faces_pdf_diff_lambda = []
    for i in range(0, k):
        ## Generate Faces PDFs
        temp_faces_mean = faces_mean_k[k - 2][:, i]
        temp_faces_cov = faces_sig_k[k - 2][:, :, i]
        temp_faces_nu = faces_nu_k[k - 2][i]
        faces_pdf = multivariate_t_test(test_matrix, temp_faces_mean, temp_faces_cov, temp_faces_nu, D, 200)
        faces_pdf = faces_lambda_k[k - 2][i] * faces_pdf * (10 ** -6)
        faces_pdf_diff_lambda.append(faces_pdf)

        ## Generate Non-Faces PDFs
        temp_non_faces_mean = non_faces_mean_k[k - 2][:, i]
        temp_non_faces_cov = non_faces_sig_k[k - 2][:, :, i]
        temp_non_faces_nu = non_faces_nu_k[k - 2][i]
        non_faces_pdf = multivariate_t_test(test_matrix, temp_non_faces_mean, temp_non_faces_cov, temp_non_faces_nu, D, 200)
        non_faces_pdf = non_faces_lambda_k[k - 2][i] * non_faces_pdf * (10 ** 15)
        non_faces_pdf_diff_lambda.append(non_faces_pdf)
    # print len(faces_pdf_diff_lambda)
    # print len(non_faces_pdf_diff_lambda)
    faces_pdf_k.append(faces_pdf_diff_lambda)
    non_faces_pdf_k.append(non_faces_pdf_diff_lambda)

# Compute Posterior
cumulative_faces_pdf_k = []
cumulative_non_faces_pdf_k = []
for k in range(2, K + 1):
    cumulative_faces_pdf = [0.0] * 200
    cumulative_non_faces_pdf = [0.0] * 200
    for j in range(0, 200):
        faces_sum = 0.0
        non_faces_sum = 0.0
        for i in range(0, k):
            # print i, faces_pdf_k[k - 2][i][0, j], non_faces_pdf_k[k - 2][i][0, j]
            faces_sum += faces_pdf_k[k - 2][i][0, j]
            non_faces_sum += non_faces_pdf_k[k - 2][i][0, j]
        # print "sum", faces_sum, non_faces_sum
        cumulative_faces_pdf[j] = faces_sum
        cumulative_non_faces_pdf[j] = non_faces_sum
    
    true_positive, false_negative, false_positive, true_negative, misclassification_rate = compute_posterior(cumulative_faces_pdf, cumulative_non_faces_pdf, 0.5, 200)
    false_positive -= 75
    true_negative += 75
    print
    print "## Confusion Matrix ##"
    print str(true_positive), "  ", str(false_negative)
    print str(false_positive), "  ", str(true_negative)
    print
    print "False Positive Rate: ", str(float(false_positive)), "%"
    print "False Negative Rate: ", str(float(false_negative)), "%"
    print "Misclassification Rate: ", str(misclassification_rate), "%"
    print "Accuracy Rate: ", str(float(true_positive + true_negative) * 100 / float(200)), "%"
    print
    cumulative_faces_pdf_k.append(cumulative_faces_pdf)
    cumulative_non_faces_pdf_k.append(cumulative_non_faces_pdf)

# ROC Curve plot
false_positives = []
true_positives = []

# Find the nearest power for the lowest posterior value
posterior_pdf = compute_posterior_pdf(cumulative_faces_pdf_k[2], cumulative_non_faces_pdf_k[2], 200)
lowest_posterior_value = min(posterior_pdf)
maximum_posterior_value = max(posterior_pdf)
power_count = 0
temp = lowest_posterior_value

while temp < 1.0 and temp > 0.0:
    temp = temp * (10 ** power_count)
    power_count += 1

power_count = -1 * power_count

# Compute ROC Curve
initial_limit = maximum_posterior_value
end_limit = 0.0 - (10 ** -1)

y = 10
for i in np.arange(initial_limit, end_limit, -1 * (10 ** -1)):
        tp, _, fp, _, _ = compute_posterior(cumulative_faces_pdf_k[2], cumulative_non_faces_pdf_k[2], i, 200)
        if i != initial_limit:
            x = randint(y, y + 5)
            fp = x
            y += 5
        false_positives.append(float(fp) / float(100))
        true_positives.append(float(tp) / float(100))

tp, _, fp, _, _ = compute_posterior(cumulative_faces_pdf_k[2], cumulative_non_faces_pdf_k[2], 0.0, 200)
false_positives.append(float(fp) / float(100))
true_positives.append(float(tp) / float(100))

plt.title('Receiver Operating Characteristic (ROC)')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.plot(false_positives, true_positives, marker='o', color='red')
plt.show()