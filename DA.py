import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
from itertools import combinations, permutations
from scipy.special import comb, perm
from PIL import Image, ImageFilter




def read_17_lines(route):
    arr = []
    with open(route, 'r', encoding='utf-8-sig') as f:
        for i in range(17):
            line = f.readline()
            content = line.strip()
            temp = content.split(",")
            arr.append(temp)
    return arr

def read_image_point(path, 
                     label_route="/data/lu_chenyang/Plvis/label_256/"):
    name = os.path.splitext(os.path.basename(path))[0]
    new_label_route = label_route + '/' + name + '.txt'  # read  labels
    arr = read_17_lines(new_label_route)
    new_arr = np.asarray(np.reshape(arr, (-1, 2,)), dtype=np.int32)
    landmark_location = [[int(new_arr[i][0]),int(new_arr[i][1])] for i in range(len(new_arr))]
    return landmark_location


class MyGaussianBlur(ImageFilter.Filter):
  name = "GaussianBlur"

  def __init__(self, radius=15, bounds=None):
    self.radius = radius
    self.bounds = bounds

  def filter(self, image):
    if self.bounds:
      clips = image.crop(self.bounds).gaussian_blur(self.radius)
      image.paste(clips, self.bounds)
      return image
    else:
      return image.gaussian_blur(self.radius)
    
for i in range(1,331):
    
    #img = cv2.imread("/data/lu_chenyang/Plvis/data_1024/train/{:0>3d}.jpg".format(i))
    #plt.imshow(img)
    landmarks=read_image_point("/data/lu_chenyang/Plvis/data_256/train/{:0>3d}.jpg".format(i))
    #print(landmarks)
    #print(comb(17, 3))
    c173 = list(combinations(landmarks, 3))
    #print(c173)
    #print(c173[0])
    #print(c173[0][0])
    #print(c173[0][0][0])

    for j in range(680):
        
        h1 = c173[j][0][0]
        w1 = c173[j][0][1]
        #h_patch1 = random.randint(30,90)
        h_patch1 = random.randint(8,23)
        w_patch1 = random.randint(8,23)
        bounds1 = (h1-int(h_patch1),w1-int(w_patch1),h1+int(h_patch1),w1+int(w_patch1))
        #print(bounds1)
        image = Image.open("/data/lu_chenyang/Plvis/data_256/train/{:0>3d}.jpg".format(i))
        image = image.filter(MyGaussianBlur(radius=3, bounds=bounds1))
    
        h2 = c173[j][1][0]
        w2 = c173[j][1][1]
        h_patch2 = random.randint(8,23)
        w_patch2 = random.randint(8,23)
        bounds2 = (h2-int(h_patch2),w2-int(w_patch2),h2+int(h_patch2),w2+int(w_patch2))
        #print(bounds2)
        image = image.filter(MyGaussianBlur(radius=3, bounds=bounds2))
    
        h3 = c173[j][2][0]
        w3 = c173[j][2][1]
        h_patch3 = random.randint(8,23)
        w_patch3 = random.randint(8,23)
        bounds3 = (h3-int(h_patch3),w3-int(w_patch3),h3+int(h_patch3),w3+int(w_patch3))
        #print(bounds3)
        image = image.filter(MyGaussianBlur(radius=3, bounds=bounds3))
        #plt.imshow(image)
        image.save("/data/lu_chenyang/Plvis/data_256_DA/{:0>3d}_{:0>3d}.jpg".format(i,j))
    print(i)