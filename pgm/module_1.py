# Module 1 - Gaussian Distribution Model
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg
from sklearn.decomposition import PCA
from scipy.stats import multivariate_normal
import math

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
            f = faces_pdf_avg
        else:
            f = faces_pdf[i]

        if non_faces_pdf[i] == 0.0:
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
        if faces_pdf[i] == 0.0:
            f = faces_pdf_avg * (10 ** -10)
        else:
            f = faces_pdf[i]

        if non_faces_pdf[i] == 0.0:
            nf = non_faces_pdf_avg
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
            f = faces_pdf_avg * (10 ** -10)
        else:
            f = faces_pdf[i]

        if non_faces_pdf[i] == 0.0:
            nf = non_faces_pdf_avg * (10 ** -10)
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
faces_mean = np.zeros((60, 60, 3), dtype = np.float)
faces_size = 1000
faces_tuple_list = []

for filename in os.listdir(training_faces_folder):
    img = cv2.imread(os.path.join(training_faces_folder, filename), flags = cv2.IMREAD_COLOR)
    RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    Gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    Gray_img = cv2.resize(Gray_img, (13, 13))
    if img is not None:
        faces_images.append(RGB_img)
        faces_images_rescaled_grayscale.append(Gray_img)

for im in faces_images:
    key = im
    im_reshape = key.reshape((1, 10800))
    faces_tuple_list.append(im_reshape)

faces_tuple = tuple(faces_tuple_list)
faces_matrix = np.vstack(faces_tuple)

faces_mean = np.mean(faces_matrix, axis = 0)

faces_mean = faces_mean.reshape((60, 60, 3))
faces_mean_image = np.array(np.round(faces_mean), dtype = np.uint8)
faces_mean_1d = faces_mean.reshape((10800, 1))

faces_covariance = np.cov(faces_matrix.T)
faces_covariance_diag = faces_covariance.diagonal()
max_faces_diag = max(faces_covariance_diag)
faces_covariance_diag = faces_covariance_diag * 255 / max_faces_diag
faces_covariance_image = faces_covariance_diag.reshape((60, 60, 3))
faces_covariance_image = np.array(np.round(faces_covariance_image), dtype = np.uint8)

training_non_faces_folder = "../training/non_faces/"
non_faces_images = []
non_faces_images_rescaled_gray = []
non_faces_mean = np.zeros((60, 60, 3), dtype = np.float)
non_faces_size = 1000

for filename in os.listdir(training_non_faces_folder):
    img = cv2.imread(os.path.join(training_non_faces_folder, filename), flags = cv2.IMREAD_COLOR)
    RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    Gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    Gray_img = cv2.resize(Gray_img, (13, 13))
    if img is not None:
        non_faces_images.append(RGB_img)
        non_faces_images_rescaled_gray.append(Gray_img)

non_faces_tuple_list = []
for im in non_faces_images:
    key = im
    im_reshape = key.reshape((1, 10800))
    non_faces_tuple_list.append(im_reshape)

non_faces_tuple = tuple(non_faces_tuple_list)
non_faces_matrix = np.vstack(non_faces_tuple)
non_faces_mean = np.mean(non_faces_matrix, axis = 0)

non_faces_mean = non_faces_mean.reshape((60, 60, 3))
non_faces_mean_image = np.array(np.round(non_faces_mean), dtype = np.uint8)
non_faces_mean_1d = non_faces_mean.reshape((10800, 1))

non_faces_covariance = np.cov(non_faces_matrix.T)
non_faces_covariance_diag = non_faces_covariance.diagonal()
max_non_faces_diag = max(non_faces_covariance_diag)
non_faces_covariance_diag = non_faces_covariance_diag * 255 / max_non_faces_diag
non_faces_covariance_image = non_faces_covariance_diag.reshape((60, 60, 3))
non_faces_covariance_image = np.array(np.round(non_faces_covariance_image), dtype = np.uint8)

# Computing mean and covariance for faces grayscale images
faces_grayscale_tuple_list = []
for im in faces_images_rescaled_grayscale:
    key = im
    im_reshape = key.reshape((1, 169))
    faces_grayscale_tuple_list.append(im_reshape)

faces_grayscale_tuple = tuple(faces_grayscale_tuple_list)
faces_grayscale_matrix = np.vstack(faces_grayscale_tuple)

faces_grayscale_mean = np.mean(faces_grayscale_matrix, axis = 0)
faces_grayscale_covariance = np.cov(faces_grayscale_matrix.T)

# Computing mean and covariance for non faces grayscale images
non_faces_grayscale_tuple_list = []
for im in non_faces_images_rescaled_gray:
    key = im
    im_reshape = key.reshape((1, 169))
    non_faces_grayscale_tuple_list.append(im_reshape)

non_faces_grayscale_tuple = tuple(non_faces_grayscale_tuple_list)
non_faces_grayscale_matrix = np.vstack(non_faces_grayscale_tuple)

non_faces_grayscale_mean = np.mean(non_faces_grayscale_matrix, axis = 0)
non_faces_grayscale_covariance = np.cov(non_faces_grayscale_matrix.T)

test_tuple_list = []
# Faces MVN Builder
faces_mvn = multivariate_normal(faces_grayscale_mean, faces_grayscale_covariance)

faces_test_images = []
test_faces_folder = "../test/faces/"
for filename in os.listdir(test_faces_folder):
    img = cv2.imread(os.path.join(test_faces_folder, filename), flags = cv2.IMREAD_COLOR)
    Gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    Gray_img = cv2.resize(Gray_img, (13, 13))
    if img is not None:
        faces_test_images.append(Gray_img)

for im in faces_test_images:
    key = im
    im_reshape = key.reshape((1, 169))
    test_tuple_list.append(im_reshape)

# Non-Faces MVN Builder
non_faces_mvn = multivariate_normal(non_faces_grayscale_mean, non_faces_grayscale_covariance)

non_faces_test_images = []
test_non_faces_folder = "../test/non_faces/"
for filename in os.listdir(test_non_faces_folder):
    img = cv2.imread(os.path.join(test_non_faces_folder, filename), flags = cv2.IMREAD_COLOR)
    Gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    Gray_img = cv2.resize(Gray_img, (13, 13))
    if img is not None:
        non_faces_test_images.append(Gray_img)

for im in non_faces_test_images:
    key = im
    im_reshape = key.reshape((1, 169))
    test_tuple_list.append(im_reshape)

test_tuple = tuple(test_tuple_list)
test_matrix = np.vstack(test_tuple)

faces_pdf = faces_mvn.pdf(test_matrix)
non_faces_pdf = non_faces_mvn.pdf(test_matrix)

DEFAULT_RESULT_LOCATION = "./results/module_1/"

# Save Mean Face Result
faces_mean_filename = DEFAULT_RESULT_LOCATION + "mean/faces.jpg"
save_mean_image(faces_mean_filename, faces_grayscale_mean, 13)

# Save Mean Non-Face Result
non_faces_mean_filename = DEFAULT_RESULT_LOCATION + "mean/non_faces.jpg"
save_mean_image(non_faces_mean_filename, non_faces_grayscale_mean, 13)

# Save Covariance Face Result
faces_covariance_filename = DEFAULT_RESULT_LOCATION + "covariance/faces.jpg"
save_covariance_image(faces_covariance_filename, faces_grayscale_covariance, 13)

# Save Covariance Non-Face Result
non_faces_covariance_filename = DEFAULT_RESULT_LOCATION + "covariance/non_faces.jpg"
save_covariance_image(non_faces_covariance_filename, non_faces_grayscale_covariance, 13)

# Compute Posterior
true_positive, false_negative, false_positive, true_negative, misclassification_rate = compute_posterior(faces_pdf, non_faces_pdf, 0.5, 200)
print
print "## Confusion Matrix ##"
print str(true_positive), "  ", str(false_negative)
print str(false_positive), "  ", str(true_negative)
print
print "False Positive Rate: ", str(false_positive), "%"
print "False Negative Rate: ", str(false_negative), "%"
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

plt.title('Receiver Operating Characteristic (ROC)')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.plot(false_positives, true_positives, marker='o')
plt.show()