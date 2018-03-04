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
        if faces_pdf[i] < (10 ** -310):
            f = faces_pdf_avg * (10 ** -10)
        else:
            f = faces_pdf[i]

        if non_faces_pdf[i] < (10 ** -310):
            nf = non_faces_pdf_avg * (10 ** -10)
        else:
            nf = non_faces_pdf[i]

        y = f + nf
        x = f
        z = x / y
        
        if (not math.isnan(z) and z > threshold):
            true_positive += 1
        else:
            false_negative += 1

    for i in range(100, testImagesSize):
        if faces_pdf[i] < (10 ** -310):
            f = faces_pdf_avg * (10 ** -10)
        else:
            f = faces_pdf[i]

        if non_faces_pdf[i] < (10 ** -310):
            nf = non_faces_pdf_avg * (10 ** -10)
        else:
            nf = non_faces_pdf[i]

        y = f + nf
        x = f
        z = x / y

        if (not math.isnan(z) and z > threshold):
            false_positive += 1
        else:
            true_negative += 1

    # Misclassification Rate
    misclassification_rate = float(false_positive + false_negative) / float(testImagesSize)
    misclassification_rate = misclassification_rate * 100
    return true_positive, false_negative, false_positive, true_negative, misclassification_rate

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

faces_lambda_k = []
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
            if covariance_epsilon[i] > 0.0:
                nearest_cov = near_psd(sig_k[:, :, i], covariance_epsilon[i])
                mvn = multivariate_normal(mean_k[i], nearest_cov) 
            else:
                mvn = multivariate_normal(mean_k[i], sig_k[:, :, i])
            
            pdf = mvn.pdf(dataset_matrix)
            pdf = lambda_k[i] * pdf
            pdf = pdf.reshape((1, 100))
            norm[i, :] = pdf

        sum_norm = np.sum(norm, axis = 0)
        ## Fitting average value for sum_norm
        for i in range(0, 100):
            for k in range(0, K):
                if sum_norm[i] == 0.0:
                    r[k, i] = 0.000001
                else:
                    r[k, i] = norm[k, i] / sum_norm[i]

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
            
        ## Compute log likelihood L
        temp = np.zeros((K, 100))
        for i in range(0, K):
            diag = np.amax(sig_k[:, :, i])
            covariance_epsilon[i] = math.sqrt(diag)
            nearest_cov = near_psd(sig_k[:, :, i], math.sqrt(diag))
            mvn = multivariate_normal(mean_k[i], nearest_cov) 
            pdf = mvn.pdf(dataset_matrix)
            pdf = lambda_k[i] * pdf
            pdf = pdf.reshape((1, 100))
            temp[i, :] = pdf

        sum_temp = np.sum(temp, axis = 0)

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

        L = np.sum(temp_log)

        iterations += 1
        if abs(L - previous_L) < precision or iterations > 20:
            break
        else:
            previous_L = L
    faces_lambda_k.append(lambda_k)
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
            if covariance_epsilon[i] > 0.0:
                nearest_cov = near_psd(sig_k[:, :, i], covariance_epsilon[i])
                mvn = multivariate_normal(mean_k[i], nearest_cov) 
            else:
                mvn = multivariate_normal(mean_k[i], sig_k[:, :, i])
            pdf = mvn.pdf(dataset_matrix)
            pdf = lambda_k[i] * pdf
            pdf = pdf.reshape((1, 100))
            norm[i, :] = pdf

        sum_norm = np.sum(norm, axis = 0)
        ## Fitting average value for sum_norm
        for i in range(0, 100):
            for k in range(0, K):
                if sum_norm[i] == 0.0:
                    r[k, i] = 0.000001
                else:
                    r[k, i] = norm[k, i] / sum_norm[i]

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
            
        ## Compute log likelihood L
        temp = np.zeros((K, 100))
        for i in range(0, K):
            diag = np.amax(sig_k[:, :, i])
            covariance_epsilon[i] = math.sqrt(diag)
            nearest_cov = near_psd(sig_k[:, :, i], math.sqrt(diag))
            mvn = multivariate_normal(mean_k[i], nearest_cov) 
            pdf = mvn.pdf(dataset_matrix)
            pdf = lambda_k[i] * pdf
            pdf = pdf.reshape((1, 100))
            temp[i, :] = pdf

        sum_temp = np.sum(temp, axis = 0)

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

        L = np.sum(temp_log)

        iterations += 1
        if abs(L - previous_L) < precision or iterations > 20:
            break
        else:
            previous_L = L
    non_faces_lambda_k.append(lambda_k)
    non_faces_mean_k.append(mean_k)
    non_faces_sig_k.append(sig_k)
    non_faces_covariance_epsilon.append(covariance_epsilon)

## Test Dataset
test_tuple_list = []
faces_test_images = []
test_faces_folder = "../test/faces/"
for filename in os.listdir(test_faces_folder):
    img = cv2.imread(os.path.join(test_faces_folder, filename), flags = cv2.IMREAD_COLOR)
    Gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    Gray_img = cv2.resize(Gray_img, (7, 7))
    if img is not None:
        faces_test_images.append(Gray_img)

for im in faces_test_images:
    key = im
    im_reshape = key.reshape((1, 49))
    test_tuple_list.append(im_reshape)

non_faces_test_images = []
test_non_faces_folder = "../test/non_faces/"
for filename in os.listdir(test_non_faces_folder):
    img = cv2.imread(os.path.join(test_non_faces_folder, filename), flags = cv2.IMREAD_COLOR)
    Gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    Gray_img = cv2.resize(Gray_img, (7, 7))
    if img is not None:
        non_faces_test_images.append(Gray_img)

for im in non_faces_test_images:
    key = im
    im_reshape = key.reshape((1, 49))
    test_tuple_list.append(im_reshape)

test_tuple = tuple(test_tuple_list)
test_matrix = np.vstack(test_tuple)

DEFAULT_RESULT_LOCATION = "./results/module_2/"

for K in range(2, 5):
    for i in range(0, K):
        # Save Mean Faces Result
        faces_mean_filename = DEFAULT_RESULT_LOCATION + "mog_" + str(K) + "/mean/faces_" + str(i) + ".jpg"
        faces_mean = faces_mean_k[K - 2][i, :]
        save_mean_image(faces_mean_filename, faces_mean, 7)

        # Save Mean Non Faces Result
        non_faces_mean_filename = DEFAULT_RESULT_LOCATION + "mog_" + str(K) + "/mean/non_faces_" + str(i) + ".jpg"
        non_faces_mean = non_faces_mean_k[K - 2][i, :]
        save_mean_image(non_faces_mean_filename, non_faces_mean, 7)

        # Save Covariance Faces Result
        faces_covariance_filename = DEFAULT_RESULT_LOCATION + "mog_" + str(K) + "/covariance/faces_" + str(i) + ".jpg"
        faces_covariance = faces_sig_k[K - 2][:, :, i]
        save_covariance_image(faces_covariance_filename, faces_covariance, 7)

        # Save Covariance Non Faces Result
        non_faces_covariance_filename = DEFAULT_RESULT_LOCATION + "mog_" + str(K) + "/covariance/non_faces_" + str(i) + ".jpg"
        non_faces_covariance = non_faces_sig_k[K - 2][:, :, i]
        save_covariance_image(non_faces_covariance_filename, non_faces_covariance, 7)


# Compute Posterior
faces_pdf_k = []
non_faces_pdf_k = []
for K in range(2, 5):
    faces_pdf_diff_lambda = []
    non_faces_pdf_diff_lambda = []
    for i in range(0, K):
        ## Generate Faces PDFs
        faces_nearest_cov = faces_sig_k[K - 2][:, :, i]
        if faces_covariance_epsilon[K - 2][i] > 0.0:
            faces_nearest_cov = near_psd(faces_nearest_cov, faces_covariance_epsilon[K - 2][i]) 
        faces_mvn = multivariate_normal(faces_mean_k[K - 2][i, :].reshape(49), nearest_cov)
        faces_pdf = faces_mvn.pdf(test_matrix)
        faces_pdf = faces_lambda_k[K - 2][i] * faces_pdf
        for j in range(0, 200):
            if faces_pdf[j] == 0.0:
                faces_pdf[j] = 10 ** -311
        faces_pdf_diff_lambda.append(faces_pdf)

        ## Generate Non-Faces PDFs
        non_faces_nearest_cov = non_faces_sig_k[K - 2][:, :, i]
        if non_faces_covariance_epsilon[K - 2][i] > 0.0:
            non_faces_nearest_cov = near_psd(non_faces_nearest_cov, non_faces_covariance_epsilon[K - 2][i]) 
        non_faces_mvn = multivariate_normal(non_faces_mean_k[K - 2][i, :].reshape(49), non_faces_nearest_cov)
        non_faces_pdf = non_faces_mvn.pdf(test_matrix)
        non_faces_pdf = non_faces_lambda_k[K - 2][i] * non_faces_pdf
        for j in range(0, 200):
            if non_faces_pdf[j] == 0.0:
                non_faces_pdf[j] = 10 ** -311
        non_faces_pdf_diff_lambda.append(faces_pdf)
    faces_pdf_k.append(faces_pdf_diff_lambda)
    non_faces_pdf_k.append(non_faces_pdf_diff_lambda)

# Compute Posterior
cumulative_faces_pdf_k = []
cumulative_non_faces_pdf_k = []
for K in range(2, 5):
    cumulative_faces_pdf = [0.0] * 200
    cumulative_non_faces_pdf = [0.0] * 200
    for j in range(0, 200):
        faces_sum = 0.0
        non_faces_sum = 0.0
        for i in range(0, K):
            faces_sum += faces_pdf_k[K - 2][i][j]
            non_faces_sum += non_faces_pdf_k[K - 2][i][j]
        cumulative_faces_pdf[j] = faces_sum
        cumulative_non_faces_pdf[j] = non_faces_sum
    
    true_positive, false_negative, false_positive, true_negative, misclassification_rate = compute_posterior(cumulative_faces_pdf, cumulative_non_faces_pdf, 0.5, 200)
    print
    print "## Confusion Matrix ##"
    print str(true_positive), "  ", str(false_negative)
    print str(false_positive), "  ", str(true_negative)
    print
    print "False Positive Rate: ", str(float(false_positive)), "%"
    print "False Negative Rate: ", str(float(false_negative)), "%"
    print "Misclassification Rate: ", str(misclassification_rate), "%"
    print
    cumulative_faces_pdf_k.append(cumulative_faces_pdf)
    cumulative_non_faces_pdf_k.append(cumulative_non_faces_pdf)