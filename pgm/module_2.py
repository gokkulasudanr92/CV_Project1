# Module 2 - Mixture of Gaussians Distribution Model
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np, numpy.linalg
from scipy import linalg
from sklearn.decomposition import PCA
from scipy.stats import multivariate_normal
import math
import statsmodels.stats.correlation_tools

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
    Gray_img = cv2.resize(Gray_img, (7, 7))
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
    Gray_img = cv2.resize(Gray_img, (7, 7))
    if img is not None:
        non_faces_images.append(RGB_img)
        non_faces_images_rescaled_gray.append(Gray_img)

## Faces Gaussian Mixture
faces_random_list = np.random.permutation(100)

dataset_images = []
for i in faces_random_list:
    im_reshape = faces_images_rescaled_grayscale[i].reshape((1, 49))
    dataset_images.append(im_reshape)

dataset_matrix = np.vstack(tuple(dataset_images))

faces_lamda_k = []
faces_mean_k = []
faces_sig_k = []
faces_covariance_epsilon = []

# Initializing values for lambda_k, mean_k and sig_k
for K in range(2, 5):
    # Lambda_k value
    lambda_k = []
    for i in range(0, K):
        lambda_k.append(1/ float(K))

    # Mean_k value
    mean_tuple_list = []
    means_index = np.random.permutation(100)
    for i in range(0, K):
        mean_tuple_list.append(dataset_images[means_index[i]])

    mean_k = np.vstack(tuple(mean_tuple_list))

    # Sig_k values
    dataset_mean = np.mean(dataset_matrix, axis = 0)
    dataset_variance = np.zeros((49, 49))

    for i in range(0, 100):
        data = dataset_matrix[i, :]
        data = data.reshape((49, 1))
        mean_temp = dataset_mean.reshape((49, 1))
        temp = data - mean_temp
        dataset_variance += np.matmul(temp, temp.T)

    dataset_variance = dataset_variance / 100

    sig_k = np.zeros((49, 49, K))
    for i in range(0, K):
        sig_k[:, :, i] = dataset_variance

    ## Main Iteration for computations
    iterations = 0
    previous_L = 1000000
    precision = 0.01
    covariance_epsilon = [0.0] * K 
    while True:
        ## Expectation Step
        norm = np.zeros((K, 100))
        r = np.zeros((K, 100))

        for i in range(0, K):
            # print covariance_epsilon[i]
            if covariance_epsilon[i] > 0.0:
                # print "I am here"
                nearest_cov = near_psd(sig_k[:, :, i], covariance_epsilon[i])
                mvn = multivariate_normal(mean_k[i], nearest_cov) 
            else:
                mvn = multivariate_normal(mean_k[i], sig_k[:, :, i])
            # print(mean_k[i])
            # print(sig_k[:, :, i])
            # sig_diag = sig_k[:, :, i].diagonal()
            # max_diag = sig_diag.max()
            # sig_diag = sig_diag * 255 / max_diag
            # sig_diag = sig_diag.reshape((7, 7))
            # diag_image = np.array(np.round(sig_diag), dtype = np.uint8)
            # cv2.imwrite("./test.jpg", diag_image)
            # print(sig_diag)
            # mvn = multivariate_normal(mean_k[i], sig_k[:, :, i])
            pdf = mvn.pdf(dataset_matrix)
            pdf = lambda_k[i] * pdf
            pdf = pdf.reshape((1, 100))
            norm[i, :] = pdf

        sum_norm = np.sum(norm, axis = 0)
        ## Fitting average value for sum_norm
        # avg_sum = 0.0
        # sum_ct = 0
        # for i in range(0, 100):
        #     if (sum_norm[i] > 0.0):
        #         sum_ct += 1
        #         avg_sum += sum_norm[i]

        # avg_sum /= sum_ct
        # for i in range(0, 100):
        #     if (sum_norm[i] == 0.0):
        #         sum_norm[i] = avg_sum
        # print(norm.shape)
        # print(norm)
        # print(sum_norm.shape)
        # print(sum_norm)
        # break

        for i in range(0, 100):
            for k in range(0, K):
                if sum_norm[i] == 0.0:
                    r[k, i] = 0.000001
                else:
                    r[k, i] = norm[k, i] / sum_norm[i]
        # print(r)
        # break

        ## Maximization Step
        r_sum_all = np.sum(np.sum(r, axis = 0))
        r_sum_rows = np.sum(r, axis = 1)
        
        for k in range(0, K):
            # Update lambda_k
            lambda_k[k] = r_sum_rows[k] / r_sum_all

            # Update mean_k
            new_mean_k = np.zeros((1, 49))
            for i in range(0, 100):
                x = dataset_matrix[i, :]
                new_mean_k = new_mean_k + r[k][i] * x
            new_mean_k = new_mean_k / r_sum_rows[k]
            mean_k[k, :] = new_mean_k

            # Update sig_k
            new_sig_k = np.zeros((49, 49))
            for i in range(0, 100):
                x = dataset_matrix[i, :].reshape((49, 1))
                temp_mean = mean_k[k, :].reshape((49, 1))
                x = x - temp_mean
                new_sig_k += r[k][i] * np.matmul(x, x.T)
            new_sig_k = new_sig_k / r_sum_rows[k]
            sig_k[:, :, k] = new_sig_k
        # print(lambda_k)
        # print(mean_k.shape)
        # print(mean_k)
        # print(sig_k.shape)
        # for i in range(0, 49):
        #     print sig_k[0, i, 0],
        # print
        # print

        # for i in range(0, 49):
        #     print sig_k[1, i, 0],
        # print

        # print np.amin(sig_k[:, :, 0])
        # break
            
        ## Compute log likelihood L
        temp = np.zeros((K, 100))
        for i in range(0, K):
            diag = np.amax(sig_k[:, :, i])
            # print(math.sqrt(diag))
            covariance_epsilon[i] = math.sqrt(diag)
            nearest_cov = near_psd(sig_k[:, :, i], math.sqrt(diag))
            # print(nearest_cov)
            # nearest_cov = statsmodels.stats.correlation_tools.cov_nearest(sig_k[:, :, k], method = "nearest", n_fact = 10)
            # print(nearest_cov_data)
            # break
            mvn = multivariate_normal(mean_k[i], nearest_cov) 
            pdf = mvn.pdf(dataset_matrix)
            pdf = lambda_k[i] * pdf
            pdf = pdf.reshape((1, 100))
            temp[i, :] = pdf

        # print covariance_epsilon
        # break
        sum_temp = np.sum(temp, axis = 0)
        # print(sum_temp.shape)
        # print(sum_temp)
        # break

        ## Get average value for -inf
        avg = 0.0
        ct = 0
        for i in range(0, 100):
            if sum_temp[i] == 0.0:
                avg += 0.0
            else:
                ct += 1
                avg += sum_temp[i]
        avg = avg / ct

        ## Replace average values at 0
        for i in range(0, 100):
            if sum_temp[i] == 0.0:
                sum_temp[i] = avg

        temp_log = []
        for i in range(0, 100):
            temp_log.append(np.log(sum_temp[i]))

        # print temp_log
        # break
        L = np.sum(temp_log)

        iterations += 1
        # print previous_L
        # print L
        # print abs(L - previous_L)

        # print
        # print("Iterations Completed: ", str(iterations))
        # print

        if abs(L - previous_L) < precision or iterations > 20:
            break
        else:
            previous_L = L
    faces_lamda_k.append(lambda_k)
    faces_mean_k.append(mean_k)
    faces_sig_k.append(sig_k)
    faces_covariance_epsilon.append(covariance_epsilon)
    
# Non-faces Gaussian Mixture
non_faces_random_list = np.random.permutation(100)

dataset_images = []
for i in non_faces_random_list:
    im_reshape = non_faces_images_rescaled_gray[i].reshape((1, 49))
    dataset_images.append(im_reshape)

dataset_matrix = np.vstack(tuple(dataset_images))

non_faces_lambda_k = []
non_faces_mean_k = []
non_faces_sig_k = []
non_faces_covariance_epsilon = []

# Initializing values for lambda_k, mean_k and sig_k
for K in range(2, 5):
    # Lambda_k value
    lambda_k = []
    for i in range(0, K):
        lambda_k.append(1/ float(K))

    # Mean_k value
    mean_tuple_list = []
    means_index = np.random.permutation(100)
    for i in range(0, K):
        mean_tuple_list.append(dataset_images[means_index[i]])

    mean_k = np.vstack(tuple(mean_tuple_list))

    # Sig_k values
    dataset_mean = np.mean(dataset_matrix, axis = 0)
    dataset_variance = np.zeros((49, 49))

    for i in range(0, 100):
        data = dataset_matrix[i, :]
        data = data.reshape((49, 1))
        mean_temp = dataset_mean.reshape((49, 1))
        temp = data - mean_temp
        dataset_variance += np.matmul(temp, temp.T)

    dataset_variance = dataset_variance / 100

    sig_k = np.zeros((49, 49, K))
    for i in range(0, K):
        sig_k[:, :, i] = dataset_variance

    ## Main Iteration for computations
    iterations = 0
    previous_L = 1000000
    precision = 0.01
    covariance_epsilon = [0.0] * K 
    while True:
        ## Expectation Step
        norm = np.zeros((K, 100))
        r = np.zeros((K, 100))

        for i in range(0, K):
            # print covariance_epsilon[i]
            if covariance_epsilon[i] > 0.0:
                # print "I am here"
                nearest_cov = near_psd(sig_k[:, :, i], covariance_epsilon[i])
                mvn = multivariate_normal(mean_k[i], nearest_cov) 
            else:
                mvn = multivariate_normal(mean_k[i], sig_k[:, :, i])
            # print(mean_k[i])
            # print(sig_k[:, :, i])
            # sig_diag = sig_k[:, :, i].diagonal()
            # max_diag = sig_diag.max()
            # sig_diag = sig_diag * 255 / max_diag
            # sig_diag = sig_diag.reshape((7, 7))
            # diag_image = np.array(np.round(sig_diag), dtype = np.uint8)
            # cv2.imwrite("./test.jpg", diag_image)
            # print(sig_diag)
            # mvn = multivariate_normal(mean_k[i], sig_k[:, :, i])
            pdf = mvn.pdf(dataset_matrix)
            pdf = lambda_k[i] * pdf
            pdf = pdf.reshape((1, 100))
            norm[i, :] = pdf

        sum_norm = np.sum(norm, axis = 0)
        ## Fitting average value for sum_norm
        # avg_sum = 0.0
        # sum_ct = 0
        # for i in range(0, 100):
        #     if (sum_norm[i] > 0.0):
        #         sum_ct += 1
        #         avg_sum += sum_norm[i]

        # avg_sum /= sum_ct
        # for i in range(0, 100):
        #     if (sum_norm[i] == 0.0):
        #         sum_norm[i] = avg_sum
        # print(norm.shape)
        # print(norm)
        # print(sum_norm.shape)
        # print(sum_norm)
        # break

        for i in range(0, 100):
            for k in range(0, K):
                if sum_norm[i] == 0.0:
                    r[k, i] = 0.000001
                else:
                    r[k, i] = norm[k, i] / sum_norm[i]
        # print(r)
        # break

        ## Maximization Step
        r_sum_all = np.sum(np.sum(r, axis = 0))
        r_sum_rows = np.sum(r, axis = 1)
        
        for k in range(0, K):
            # Update lambda_k
            lambda_k[k] = r_sum_rows[k] / r_sum_all

            # Update mean_k
            new_mean_k = np.zeros((1, 49))
            for i in range(0, 100):
                x = dataset_matrix[i, :]
                new_mean_k = new_mean_k + r[k][i] * x
            new_mean_k = new_mean_k / r_sum_rows[k]
            mean_k[k, :] = new_mean_k

            # Update sig_k
            new_sig_k = np.zeros((49, 49))
            for i in range(0, 100):
                x = dataset_matrix[i, :].reshape((49, 1))
                temp_mean = mean_k[k, :].reshape((49, 1))
                x = x - temp_mean
                new_sig_k += r[k][i] * np.matmul(x, x.T)
            new_sig_k = new_sig_k / r_sum_rows[k]
            sig_k[:, :, k] = new_sig_k
        # print(lambda_k)
        # print(mean_k.shape)
        # print(mean_k)
        # print(sig_k.shape)
        # for i in range(0, 49):
        #     print sig_k[0, i, 0],
        # print
        # print

        # for i in range(0, 49):
        #     print sig_k[1, i, 0],
        # print

        # print np.amin(sig_k[:, :, 0])
        # break
            
        ## Compute log likelihood L
        temp = np.zeros((K, 100))
        for i in range(0, K):
            diag = np.amax(sig_k[:, :, i])
            # print(math.sqrt(diag))
            covariance_epsilon[i] = math.sqrt(diag)
            nearest_cov = near_psd(sig_k[:, :, i], math.sqrt(diag))
            # print(nearest_cov)
            # nearest_cov = statsmodels.stats.correlation_tools.cov_nearest(sig_k[:, :, k], method = "nearest", n_fact = 10)
            # print(nearest_cov_data)
            # break
            mvn = multivariate_normal(mean_k[i], nearest_cov) 
            pdf = mvn.pdf(dataset_matrix)
            pdf = lambda_k[i] * pdf
            pdf = pdf.reshape((1, 100))
            temp[i, :] = pdf

        # print covariance_epsilon
        # break
        sum_temp = np.sum(temp, axis = 0)
        # print(sum_temp.shape)
        # print(sum_temp)
        # break

        ## Get average value for -inf
        avg = 0.0
        ct = 0
        for i in range(0, 100):
            if sum_temp[i] == 0.0:
                avg += 0.0
            else:
                ct += 1
                avg += sum_temp[i]
        avg = avg / ct

        ## Replace average values at 0
        for i in range(0, 100):
            if sum_temp[i] == 0.0:
                sum_temp[i] = avg

        temp_log = []
        for i in range(0, 100):
            temp_log.append(np.log(sum_temp[i]))

        # print temp_log
        # break
        L = np.sum(temp_log)

        iterations += 1
        # print previous_L
        # print L
        # print abs(L - previous_L)

        # print
        # print("Iterations Completed: ", str(iterations))
        # print

        if abs(L - previous_L) < precision or iterations > 20:
            break
        else:
            previous_L = L
    non_faces_images_rescaled_gray.append(lambda_k)
    non_faces_mean_k.append(mean_k)
    non_faces_sig_k.append(sig_k)
    non_faces_covariance_epsilon.append(covariance_epsilon)

## Compute posterior
