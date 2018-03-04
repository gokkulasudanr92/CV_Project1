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
            f = faces_pdf_avg * (10 ** -30)
        else:
            f = faces_pdf[i]

        if non_faces_pdf[i] == 0.0:
            nf = non_faces_pdf_avg * (10 ** -30)
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
            f = faces_pdf_avg * (10 ** -30)
        else:
            f = faces_pdf[i]

        if non_faces_pdf[i] == 0.0:
            nf = non_faces_pdf_avg * (10 ** -30)
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
            f = faces_pdf_avg * (10 ** -30)
        else:
            f = faces_pdf[i]

        if non_faces_pdf[i] == 0.0:
            nf = non_faces_pdf_avg * (10 ** -30)
        else:
            nf = non_faces_pdf[i]

        y = f + nf
        x = f
        z = x / y

        if (not math.isnan(z)):
            posterior_pdf[i] = z
    return posterior_pdf

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
    im_reshape = i.reshape((1, D))
    dataset_images.append(im_reshape)

dataset_matrix = np.vstack(tuple(dataset_images))

# Initialize mean 
mean = np.mean(dataset_matrix, axis = 0)
mean = mean.reshape((1, D))

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
    if abs(L - previous_L) < precision or iterations > 1000000:
        break
    else:
        previous_L = L
faces_phi = phi
faces_mean = mean
faces_sig = sig

# Non-Faces Factor Analyzer
dataset_images = []
for i in non_faces_images_rescaled_gray:
    im_reshape = i.reshape((1, D))
    dataset_images.append(im_reshape)

dataset_matrix = np.vstack(tuple(dataset_images))

# Initialize mean 
mean = np.mean(dataset_matrix, axis = 0)
mean = mean.reshape((1, D))

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
    if abs(L - previous_L) < precision or iterations > 1000000:
        break
    else:
        previous_L = L
non_faces_phi = phi
non_faces_mean = mean
non_faces_sig = sig

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

DEFAULT_RESULT_LOCATION = "./results/module_5/"

# Computing Norm's Covariance
temp_faces_phi = np.matmul(faces_phi, faces_phi.T)
temp_faces_sig = np.diag(faces_sig[0])
faces_covariance = temp_faces_phi + temp_faces_sig

temp_non_faces_phi = np.matmul(non_faces_phi, non_faces_phi.T)
temp_non_faces_sig = np.diag(non_faces_sig[0])
non_faces_covariance = temp_non_faces_phi + temp_non_faces_sig

# Save Mean Face Result
faces_mean_filename = DEFAULT_RESULT_LOCATION + "mean/faces.jpg"
save_mean_image(faces_mean_filename, faces_mean, 10)

# Save Mean Non-Face Result
non_faces_mean_filename = DEFAULT_RESULT_LOCATION + "mean/non_faces.jpg"
save_mean_image(non_faces_mean_filename, non_faces_mean, 10)

# Save Covariance Face Result
faces_covariance_filename = DEFAULT_RESULT_LOCATION + "covariance/faces.jpg"
save_covariance_image(faces_covariance_filename, faces_covariance, 10)

# Save Covariance Non-Face Result
non_faces_covariance_filename = DEFAULT_RESULT_LOCATION + "covariance/non_faces.jpg"
save_covariance_image(non_faces_covariance_filename, non_faces_covariance, 10)

# Faces MVN Builder
faces_mvn = multivariate_normal(faces_mean.reshape(D), faces_covariance)
faces_pdf = faces_mvn.pdf(test_matrix)

# Faces MVN Builder
non_faces_mvn = multivariate_normal(non_faces_mean.reshape(D), non_faces_covariance)
non_faces_pdf = non_faces_mvn.pdf(test_matrix)

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
end_limit = 0.0 - (10 ** -5)

for i in np.arange(initial_limit, end_limit, -1 * (10 ** -5)):
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