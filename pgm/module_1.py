# Module 1 - Gaussian Distribution Model
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg
from sklearn.decomposition import PCA
from scipy.stats import multivariate_normal
import math

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

'''
# Constructing co variance matrix for faces training
f = True
faces_file = "faces_covariance.npy"
faces_covariance = np.zeros((10800, 10800))
for im in faces_images:
    if f:
        break
    key = im
    X_i = key.reshape((10800, 1))
    X_i = X_i - faces_mean_1d
    faces_covariance += np.matmul(X_i, X_i.T)

if not f:
    faces_covariance = faces_covariance / 1000
    np.save(faces_file, faces_covariance)
else:
    faces_covariance = np.load(faces_file)
'''
#faces_covariance_sqrt = np.sqrt(faces_covariance)
faces_covariance = np.cov(faces_matrix.T)
faces_covariance_diag = faces_covariance.diagonal()
max_faces_diag = max(faces_covariance_diag)
faces_covariance_diag = faces_covariance_diag * 255 / max_faces_diag
faces_covariance_image = faces_covariance_diag.reshape((60, 60, 3))
faces_covariance_image = np.array(np.round(faces_covariance_image), dtype = np.uint8)
#faces_covariance_im = np.array(np.round(faces_covariance), dtype = np.uint8)
#(faces_sign, faces_covariance_logdet) = np.linalg.slogdet(faces_covariance_im)
#print(faces_sign, faces_covariance_logdet)

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

'''
# Constructing the covariance matrix for non-faces
f = True
non_faces_file = "non_faces_covariance.npy"
non_faces_covariance = np.zeros((10800, 10800))
for im in non_faces_images:
    if f:
        break
    key = im
    X_i = key.reshape((10800, 1))
    X_i = X_i - non_faces_mean_1d
    non_faces_covariance += np.matmul(X_i, X_i.T)

if not f:
    non_faces_covariance = non_faces_covariance / 1000
    np.save(non_faces_file, non_faces_covariance)
else:
    non_faces_covariance = np.load(non_faces_file)
'''

#non_faces_covariance_sqrt = np.sqrt(non_faces_covariance)
non_faces_covariance = np.cov(non_faces_matrix.T)
non_faces_covariance_diag = non_faces_covariance.diagonal()
max_non_faces_diag = max(non_faces_covariance_diag)
non_faces_covariance_diag = non_faces_covariance_diag * 255 / max_non_faces_diag
non_faces_covariance_image = non_faces_covariance_diag.reshape((60, 60, 3))
non_faces_covariance_image = np.array(np.round(non_faces_covariance_image), dtype = np.uint8)
#non_faces_covariance_im = np.array(np.round(non_faces_covariance), dtype = np.uint8)
#(non_faces_sign, non_faces_covariance_logdet) = np.linalg.slogdet(non_faces_covariance_im)
#print(non_faces_sign, non_faces_covariance_logdet)

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

'''
X = faces_images[0].reshape((10800, 1))

faces_mean_1d = faces_mean_image.reshape((10800, 1))
temp = (X - faces_mean_1d)
faces_covariance_inverse = np.linalg.inv(faces_covariance)
faces_covariance_inverse_max = np.amax(faces_covariance_inverse)
faces_covariance_inverse = faces_covariance_inverse * 0.00001 / faces_covariance_inverse_max
faces_exp = np.matmul(temp.T, faces_covariance_inverse)
faces_exp = np.matmul(faces_exp, temp)

non_faces_mean_1d = non_faces_mean_image.reshape((10800, 1))
temp2 = (X - non_faces_mean_1d)
non_faces_covariance_inverse = np.linalg.inv(non_faces_covariance)
non_faces_covariance_inverse_max = np.amax(non_faces_covariance_inverse)
non_faces_covariance_inverse = non_faces_covariance_inverse * 0.00001 / non_faces_covariance_inverse_max
non_faces_exp = np.matmul(temp2.T, non_faces_covariance_inverse)
non_faces_exp = np.matmul(non_faces_exp, temp2)

non_faces_exp = non_faces_exp / 2
faces_exp = faces_exp / 2
t = faces_covariance_logdet / 2 - non_faces_covariance_logdet / 2 - non_faces_exp + faces_exp

print(t)
print(faces_exp)
print(non_faces_exp)

print(faces_covariance_inverse[5400])
print(non_faces_covariance_inverse[5400])
'''

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

#print(faces_pdf.shape)
#print(non_faces_pdf.shape)

# Compute Posterior
faces_true_positive = 0
faces_false_positive = 0
faces_true_negative = 0
faces_false_negative = 0

for i in range(0, 100):
    y = faces_pdf[i] + non_faces_pdf[i]
    x = faces_pdf[i]
    z = x / y
    if (not math.isnan(z) and z >= 0.5):
        faces_true_positive += 1
    else:
        faces_false_negative += 1

for i in range(100, 200):
    y = faces_pdf[i] + non_faces_pdf[i]
    x = faces_pdf[i]
    z = x / y
    if (not math.isnan(z) and z >= 0.5):
        faces_false_positive += 1
    else:
        faces_true_negative += 1

print(faces_true_positive, faces_false_negative)
print(faces_false_positive, faces_true_negative)

#print(20000 - non_faces_exp[0][0] - faces_exp[0][0])

#for im in faces_images:
#    key = im
#    im_reshape = key.reshape((10800, 1))
#    faces_tuple_list.append(im_reshape - faces_mean_1d)

#print(len(faces_tuple))

#temp = np.vstack(faces_tuple)
#print(np.transpose(temp).shape)
#faces_covariance = np.cov(temp)

#t1 = np.array(temp)
#t2 = np.transpose(t1)
#faces_covariance = np.matmul(t1, t2)
#faces_covariance /= (faces_size - 1)
#faces_covariance_diag = faces_covariance[:,0]

#faces_covariance_image = faces_covariance_diag.reshape((60, 60, 3))
#print(faces_covariance.shape)

#plt.subplot(131)
#plt.imshow(faces_covariance_image)
#plt.subplot(132)
#plt.imshow(faces_mean)
#plt.show()