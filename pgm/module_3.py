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

    pdf_list = []
    for i in range(0, size):
        pdf_list.append(pdf[0, i])
    return pdf_list    

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

    # Compute the E[log h i]
    E_log_h_i = np.zeros((1, 1000))
    for i in range(0, 1000):
        temp = special.psi((nu_k + D) / 2) - np.log(nu_k / 2 + delta_i[0, i] / 2)
        E_log_h_i[0, i] = temp

    ## Maximization Step
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
    nu_k = new_nu

    ## Compute Log likelihood
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
    if abs(L - previous_L) < precision or iterations > 100:
        break
    else:
        previous_L = L

faces_nu_k = nu_k
faces_mean_k = mean_k
faces_sig_k = sig_k

## Non-faces t Distribution
dataset_images = []
for i in non_faces_images_rescaled_gray:
    im_reshape = i.reshape((1, D))
    dataset_images.append(im_reshape)

dataset_matrix = np.vstack(tuple(dataset_images))

non_faces_nu_k = []
non_faces_mean_k = []
non_faces_sig_k = []
non_faces_covariance_epsilon = []

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

    # Compute the E[log h i]
    E_log_h_i = np.zeros((1, 1000))
    for i in range(0, 1000):
        temp = special.psi((nu_k + D) / 2) - np.log(nu_k / 2 + delta_i[0, i] / 2)
        E_log_h_i[0, i] = temp

    ## Maximization Step
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
    nu_k = new_nu

    ## Compute Log likelihood
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
    if abs(L - previous_L) < precision or iterations > 100:
        break
    else:
        previous_L = L

non_faces_nu_k = nu_k
non_faces_mean_k = mean_k
non_faces_sig_k = sig_k

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

DEFAULT_RESULT_LOCATION = "./results/module_3/"

# Faces t distribution PDF
faces_pdf = multivariate_t(test_matrix, faces_mean_k, faces_sig_k, faces_nu_k, D, 200)

# Non-faces t distribution PDF
non_faces_pdf = multivariate_t(test_matrix, non_faces_mean_k, non_faces_sig_k, non_faces_nu_k, D, 200)

# Save Mean Face Result
faces_mean_filename = DEFAULT_RESULT_LOCATION + "mean/faces.jpg"
save_mean_image(faces_mean_filename, faces_mean_k, 10)

# Save Mean Non-Face Result
non_faces_mean_filename = DEFAULT_RESULT_LOCATION + "mean/non_faces.jpg"
save_mean_image(non_faces_mean_filename, non_faces_mean_k, 10)

# Save Covariance Face Result
faces_covariance_filename = DEFAULT_RESULT_LOCATION + "covariance/faces.jpg"
save_covariance_image(faces_covariance_filename, faces_sig_k, 10)

# Save Covariance Non-Face Result
non_faces_covariance_filename = DEFAULT_RESULT_LOCATION + "covariance/non_faces.jpg"
save_covariance_image(non_faces_covariance_filename, non_faces_sig_k, 10)

# Compute Posterior
true_positive, false_negative, false_positive, true_negative, misclassification_rate = compute_posterior(faces_pdf, non_faces_pdf, 0.5, 200)
print
print "## Confusion Matrix ##"
print str(true_positive), "  ", str(false_negative)
print str(false_positive), "  ", str(true_negative)
print
print "False Positive Rate: ", str(float(false_positive)), "%"
print "False Negative Rate: ", str(float(false_negative)), "%"
print "Misclassification Rate: ", str(misclassification_rate), "%"
print

# ROC Curve plot
false_positives = []
true_positives = []

# Find the nearest power for the lowest posterior value
posterior_pdf = compute_posterior_pdf(faces_pdf, non_faces_pdf, 200)
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
end_limit = 0.0 - (10 ** -7)

for i in np.arange(initial_limit, end_limit, -1 * (10 ** -7)):
        tp, _, fp, _, _ = compute_posterior(faces_pdf, non_faces_pdf, i, 200)
        false_positives.append(float(fp) / float(100))
        true_positives.append(float(tp) / float(100))

tp, _, fp, _, _ = compute_posterior(faces_pdf, non_faces_pdf, 0.0, 200)
false_positives.append(float(fp) / float(100))
true_positives.append(float(tp) / float(100))

plt.title('Receiver Operating Characteristic (ROC)')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.plot(false_positives, true_positives, marker='o')
plt.show()