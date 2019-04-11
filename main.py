from skimage import measure
import matplotlib.pyplot as plt
import os
import shutil
import glob
import cv2
import numpy as np
import images_compare

original = cv2.imread("images_to_compare/bg-clouds_4.jpg")
# contrast = cv2.imread("images_to_compare/bg-clouds_4_contrast.jpg")
# shopped = cv2.imread("images_to_compare/bg-clouds_4_photoshoped.jpg")
big = cv2.imread("images_to_compare/maxresdefault.jpg")

# difference = cv2.subtract(image1, image2)
# result = not np.any(difference)
# if result is True:
#    print ('The images are the same')
# else:
#    cv2.imwrite("result.jpg", difference)
#    print ('The images are different')

# convert the images to grayscale
# original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
# contrast = cv2.cvtColor(contrast, cv2.COLOR_BGR2GRAY)
# shopped = cv2.cvtColor(shopped, cv2.COLOR_BGR2GRAY)
# big = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)

#images = [cv2.imread(file) for file in glob.glob("images/*.jpg")]

images = []
images_names = []
for file in glob.glob("check/*.jpg"):
    images.append(cv2.imread(file))
    images_names.append(file)


SSIM_index_limit = 0.7
MSE_index_limit = 50

same_imgs = []
same_imgs_names = []
i = 0

for image in images:
    is_image_added = False
    j = 0
    print('--------> Compare image_name[%d]: "%s"\n' % (i, images_names[i]))
    #print('same_imgs.len = %d | j = %d | is_image_added = %s \n' % (len(same_imgs), j, is_image_added))
    while j < len(same_imgs) and not is_image_added:
        same_imgs_list = same_imgs[j]
        same_imgs_names_list = same_imgs_names[j]
        k = 0
        #print('same_imgs_list is list = %s \n' % hasattr(same_imgs_list, "__len__"))
        if(isinstance(same_imgs_list, list)):
            #print('same_imgs_list.len = %d \n' % len(same_imgs_list))
            while k < len(same_imgs_list) and not is_image_added:
                same_img = same_imgs_list[k]
                same_img_name = same_imgs_names_list[k]

                curr_height, curr_width = image.shape[:2]
                cmpr_height, cmpr_width = same_img.shape[:2]

                max_width = max(curr_width, cmpr_width)
                max_height = max(curr_height, cmpr_height)

                image_resized = cv2.resize(image, (max_width, max_height)) # to same size
                same_img_resized = cv2.resize(same_img, (max_width, max_height))

                image_resized_gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY) # to gray scale
                same_img_resized_gray = cv2.cvtColor(same_img_resized, cv2.COLOR_BGR2GRAY)

                MSE_index_curr = images_compare.mse(image_resized_gray, same_img_resized_gray)
                SSIM_index_curr = measure.compare_ssim(image_resized, same_img_resized, multichannel=True)
                print('with image: "%s" MSE_curr = %f | SSIM_curr = %f \n' % (same_img_name, MSE_index_curr, SSIM_index_curr))

                if (SSIM_index_curr >= SSIM_index_limit-0.2 and MSE_index_curr <= 4000) or SSIM_index_curr >= SSIM_index_limit or MSE_index_curr <= MSE_index_limit:
                    same_imgs[j] = same_imgs_list + [image]
                    same_imgs_names[j] = same_imgs_names_list + [images_names[i]]
                    is_image_added = True
                    print('-> find same image: %s \n' % same_img_name)
                k += 1
        j += 1

    if not is_image_added:
        same_imgs.append([image])
        same_imgs_names.append([images_names[i]])
        is_image_added = True

    i += 1


#------- save same images --------

project_path = os.getcwd()
same_imgs_dir = project_path + "/same_images"
best_imgs_dir = same_imgs_dir + "/best_images"

if os.path.exists(same_imgs_dir):
    #remove all files before write new
    shutil.rmtree(same_imgs_dir, ignore_errors=True)
    #r = glob.glob(same_imgs_dir)
    #for i in r:

try:
    os.mkdir(same_imgs_dir)
    os.mkdir(best_imgs_dir)

except OSError as e:
    print ("Creation of the directory %s failed" % e)
    # else:
    # print ("Successfully created the directory")


if os.path.exists(same_imgs_dir):
    print('Path "%s" exists\n' % same_imgs_dir)
    i = 0
    while i < len(same_imgs):
        same_imgs_list = same_imgs[i]
        same_imgs_names_list = same_imgs_names[i]
        same_imgs_subdir = same_imgs_dir + "/same_group_" + str(i)
        os.mkdir(same_imgs_subdir)
        j = 0
        biggest_img = None
        biggest_img_name = ''
        biggest_width = 0
        biggest_height = 0
        while j < len(same_imgs_list):
            save_img_path = same_imgs_subdir + "/" + same_imgs_names_list[j].replace('check/', '')
            save_img = same_imgs_list[j]
            #print('SAVE_IMG path = "%s"\n' % save_img_path)
            curr_height, curr_width = save_img.shape[:2]
            if(curr_width > biggest_width and curr_height > biggest_height):
                biggest_width = curr_width
                biggest_height = curr_height
                biggest_img_name = same_imgs_names_list[j].replace('check/', '')
                biggest_img = save_img
            cv2.imwrite(save_img_path, save_img)
            j += 1

        if biggest_img is not None:
            cv2.imwrite(best_imgs_dir + "/" + biggest_img_name, biggest_img)

        i += 1

        #Лабораторная выполнена и сдана
else:
    print('Path "%s" does not exists!\n' % same_imgs_dir)






































