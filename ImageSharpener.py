# Copyright 2023 antillia.com Toshiyuki Arai 
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ImageSharpener.py


import os
import sys
import glob

import cv2

import numpy as np
import traceback
from PIL import Image

class ImageSharpener:
  
  def __init__(self, kernel_size=3, image_scaling=3.0):
    self.image_scaling = image_scaling
 
    self.kernel_size   = kernel_size


  def read(self, image_file, ks=3, image_scaling=3.0):
    buf = np.fromfile(image_file, np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)

    self.scaling = image_scaling
    self.kernel_size  = ks
    # PIL Image
    img = Image.fromarray(img)
    return self.apply(img, self.kernel_size, self.scaling)
  
      
  def apply(self, image, k, scaling): 
    # PIL image -> OpenCV Image
    img = np.array(image) 

    matrix = np.array([[0, -k,           0],
                        [-k, 1 + 4 * k, -k],
                        [0, -k,          0]])
 
    img = cv2.resize(img, dsize=None, fx=scaling, fy=scaling)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = cv2.filter2D(img, -1, matrix)
    img = cv2.convertScaleAbs(img)

   # OpenCV Image -> PIL Image
    img = Image.fromarray(img)
    img = img.convert('RGB')
    return img
  

if __name__ == "__main__":

  try:
    image_file = "./input_images/sample.png"
    output_dir = "./output_images"
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)

    sharpener = ImageSharpener()
    ks      = 3
    image_scaling = 3
    img = sharpener.read(image_file, ks=ks, image_scaling=image_scaling)
    img.show()
    basename = os.path.basename(image_file)
    output_file = os.path.join(output_dir, "sharpen_" + basename)

    img.save(output_file, format="PNG")
    print("--- Saved {}".format(output_file))

  except:
    traceback.print_exc()

  