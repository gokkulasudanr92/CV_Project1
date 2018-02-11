import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

## Open the folds file from FDDB-folds for training faces
print("###############")
print("Training Faces:")
print("###############")
file_list = ["01", "02", "03", "04", "05", "06"]
image_map_coords = {}
count = 1
image_name = ""
f = 0

for i in file_list:
    file_name = "../data/FDDB-folds/FDDB-fold-" + i + "-ellipseList.txt"
    image_ellipsoid_file = open(file_name, "r");

    for line in image_ellipsoid_file:
        line = line.rstrip()
    
        if (f != 0):
            f -= 1
            continue

        if (count == 1):
            image_name = line
            image_map_coords[image_name] = []
        elif (count == 2):
            f = int(line) - 1
        elif (count == 3):
            l = [float(i) for i in line.split()]
            for i in range(0, len(l) - 1):
                image_map_coords[image_name].append(l[i])
            image_name = ""
            count = 0
        count += 1

###########################################
#### Saving faces for training dataset ####
###########################################
faces_training_src_folder = "../training/faces/faces-"
file_count = 1
image_format = ".jpg"

for key in image_map_coords.keys():
    if file_count == 1300:
        break
    
    im = key
    image_src = "../data/" + im + ".jpg"
    print("Processing ... " + image_src)
    img = cv2.imread(image_src)
    mask = np.zeros_like(img)
    rows, cols,_ = mask.shape

    major = math.ceil(image_map_coords[im][0])
    major = int(major)
    minor = math.ceil(image_map_coords[im][1])
    minor = int(minor)
    angle = image_map_coords[im][2]
    x = math.ceil(image_map_coords[im][3])
    x = int(x)
    y = math.ceil(image_map_coords[im][4])
    y = int(y)
    thickness = -1

    c = image_map_coords[im][0] ** 2 / 4.0 + image_map_coords[im][1] ** 2 / 4.0
    c = math.sqrt(c)
    c = math.ceil(c)
    c = int(c)

    ## Get co-ordinates of the rectangle to cut off
    top_left_bottom_right = []

    top_left_bottom_right.append((x - c, y + c))
    top_left_bottom_right.append((x + c, y - c))

    #mask = cv2.rectangle(mask, top_left_bottom_right[0], top_left_bottom_right[1], (255, 255, 255), thickness)
    mask = cv2.ellipse(mask, center=(x, y), axes=(minor, major), angle=angle, startAngle=0, endAngle=360, color=(255,255,255), thickness=thickness)
    #result = cv2.ellipse(img, center=(x, y), axes=(minor, major), angle=angle, startAngle=0, endAngle=360, color=(255,255,255), thickness=thickness)
    result = np.bitwise_and(img, mask)
    result[np.where(result == [0])] = [255]
    
    if (y - major > 0):
        min_y = y - major
    else:
        min_y = 0

    if (y + major >= cols):
        max_y = cols - 1
    else:
        max_y = y + major

    if (x - major > 0):    
        min_x = x - major
    else:
        min_x = 0

    if (x + major >= rows):
        max_x = rows - 1
    else:
        max_x = x + major

    result = result[min_y: max_y, min_x: max_x]
    try:
        result = cv2.resize(result, (60, 60))
    except cv2.error as e:
        print("Skipped Image: " + image_src)
        continue

    #mask_rgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    #result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    #image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    cv2.imwrite(faces_training_src_folder + str(file_count) + image_format, result)
    file_count += 1

## Open the folds file from FDDB-folds for training non-faces
print("###################")
print("Training Non-faces:")
print("###################")
file_list = ["01", "02", "03", "04", "05", "06"]
count = 1
images_list = []
image_name = ""
f = 0

for i in file_list:
    file_name = "../data/FDDB-folds/FDDB-fold-" + i + "-ellipseList.txt"
    image_ellipsoid_file = open(file_name, "r");

    for line in image_ellipsoid_file:
        line = line.rstrip()

        if (f != 0):
            f -= 1
            continue

        if (count == 1):
            image_name = line
            if (image_name not in images_list):
                images_list.append(image_name)
            count += 1
        elif (count == 2):
            f = int(line)
            image_name = ""
            count = 1

###############################################
#### Saving non-faces for training dataset ####
###############################################
non_faces_training_src_folder = "../training/non_faces/non_faces-"
non_faces_file_count = 1
non_faces_image_format = ".jpg"

for key in images_list:
    im = key
    image_src = "../data/" + im + ".jpg"
    print("Processing ... " + image_src)
    img = cv2.imread(image_src)
    cols, rows, d = img.shape

    #print(rows, cols)
    non_face_result = img[cols - 60: cols, rows - 60: rows]
    #try:
    #    non_face_result = cv2.resize(non_face_result, (60, 60))
    #except cv2.error as e:
    #    print("Skipped Image: " + image_src)
    #    continue

    #plt.subplot(131)
    #plt.imshow(img)
    #plt.subplot(132)
    #plt.imshow(non_face_result)
    #plt.show()
    cv2.imwrite(non_faces_training_src_folder + str(non_faces_file_count) + non_faces_image_format, non_face_result)
    non_faces_file_count += 1

## Open the folds file from FDDB-folds for testing faces
file_list = ["01", "02", "03", "04", "05", "06"]
count = 1
images_list = []
image_name = ""
f = 0

for i in file_list:
    file_name = "../data/FDDB-folds/FDDB-fold-" + i + "-ellipseList.txt"
    image_ellipsoid_file = open(file_name, "r");

    for line in image_ellipsoid_file:
        line = line.rstrip()

        if (f != 0):
            f -= 1
            continue

        if (count == 1):
            image_name = line
            if (image_name not in images_list):
                images_list.append(image_name)
            count += 1
        elif (count == 2):
            f = int(line)
            image_name = ""
            count = 1

print("###################")
print("Testing Faces:")
print("###################")
test_file_list = ["09", "08"]
test_image_map_coords = {}
count = 1
image_name = ""
f = 0

for i in test_file_list:
    file_name = "../data/FDDB-folds/FDDB-fold-" + i + "-ellipseList.txt"
    image_ellipsoid_file = open(file_name, "r");

    for line in image_ellipsoid_file:
        line = line.rstrip()
    
        if (f != 0):
            f -= 1
            continue

        if (count == 1):
            image_name = line
            if (image_name not in images_list):
                test_image_map_coords[image_name] = []
        elif (count == 2):
            f = int(line) - 1
        elif (count == 3):
            if (image_name not in images_list):
                l = [float(i) for i in line.split()]
                for i in range(0, len(l) - 1):
                    test_image_map_coords[image_name].append(l[i])
            image_name = ""
            count = 0
        count += 1

##########################################
#### Saving faces for testing dataset ####
##########################################
test_faces_training_src_folder = "../test/faces/faces-"
file_count = 1
image_format = ".jpg"

for key in test_image_map_coords.keys():
    if file_count == 300:
        break
    
    im = key
    image_src = "../data/" + im + ".jpg"
    print("Processing ... " + image_src)
    img = cv2.imread(image_src)
    mask = np.zeros_like(img)
    rows, cols,_ = mask.shape

    major = math.ceil(test_image_map_coords[im][0])
    major = int(major)
    minor = math.ceil(test_image_map_coords[im][1])
    minor = int(minor)
    angle = test_image_map_coords[im][2]
    x = math.ceil(test_image_map_coords[im][3])
    x = int(x)
    y = math.ceil(test_image_map_coords[im][4])
    y = int(y)
    thickness = -1

    c = test_image_map_coords[im][0] ** 2 / 4.0 + test_image_map_coords[im][1] ** 2 / 4.0
    c = math.sqrt(c)
    c = math.ceil(c)
    c = int(c)

    ## Get co-ordinates of the rectangle to cut off
    top_left_bottom_right = []

    top_left_bottom_right.append((x - c, y + c))
    top_left_bottom_right.append((x + c, y - c))

    #mask = cv2.rectangle(mask, top_left_bottom_right[0], top_left_bottom_right[1], (255, 255, 255), thickness)
    mask = cv2.ellipse(mask, center=(x, y), axes=(minor, major), angle=angle, startAngle=0, endAngle=360, color=(255,255,255), thickness=thickness)
    #result = cv2.ellipse(img, center=(x, y), axes=(minor, major), angle=angle, startAngle=0, endAngle=360, color=(255,255,255), thickness=thickness)
    result = np.bitwise_and(img, mask)
    result[np.where(result == [0])] = [255]
    
    if (y - major > 0):
        min_y = y - major
    else:
        min_y = 0

    if (y + major >= cols):
        max_y = cols - 1
    else:
        max_y = y + major

    if (x - major > 0):    
        min_x = x - major
    else:
        min_x = 0

    if (x + major >= rows):
        max_x = rows - 1
    else:
        max_x = x + major

    result = result[min_y: max_y, min_x: max_x]
    try:
        result = cv2.resize(result, (60, 60))
    except cv2.error as e:
        print("Skipped Image: " + image_src)
        continue

    #mask_rgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    #result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    #image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    cv2.imwrite(test_faces_training_src_folder + str(file_count) + image_format, result)
    file_count += 1

## Open the folds file from FDDB-folds for testing non-faces
print("###################")
print("Testing Non-faces:")
print("###################")
test_file_list = ["09", "08"]
count = 1
test_images_list = []
image_name = ""
f = 0

for i in test_file_list:
    file_name = "../data/FDDB-folds/FDDB-fold-" + i + "-ellipseList.txt"
    image_ellipsoid_file = open(file_name, "r");

    for line in image_ellipsoid_file:
        line = line.rstrip()

        if (f != 0):
            f -= 1
            continue

        if (count == 1):
            image_name = line
            if (image_name not in images_list):
                test_images_list.append(image_name)
            count += 1
        elif (count == 2):
            f = int(line)
            image_name = ""
            count = 1

##############################################
#### Saving non-faces for testing dataset ####
##############################################
test_non_faces_training_src_folder = "../test/non_faces/non_faces-"
non_faces_file_count = 1
non_faces_image_format = ".jpg"

for key in test_images_list:
    if non_faces_file_count == 200:
        break
    
    im = key
    image_src = "../data/" + im + ".jpg"
    print("Processing ... " + image_src)
    img = cv2.imread(image_src)
    cols, rows, d = img.shape

    #print(rows, cols)
    non_face_result = img[cols - 60: cols, rows - 60: rows]
    #try:
    #    non_face_result = cv2.resize(non_face_result, (60, 60))
    #except cv2.error as e:
    #    print("Skipped Image: " + image_src)
    #    continue

    #plt.subplot(131)
    #plt.imshow(img)
    #plt.subplot(132)
    #plt.imshow(non_face_result)
    #plt.show()
    cv2.imwrite(test_non_faces_training_src_folder + str(non_faces_file_count) + non_faces_image_format, non_face_result)
    non_faces_file_count += 1