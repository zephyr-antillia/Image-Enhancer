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
# ImageAdaptiveMeanThresholder.py

import os
import sys
import numpy as np
import cv2
from PIL import Image
import traceback

# See also: https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html
#
class ImageAdaptiveMeanThresholder:

  def __init__(self, max_value=255, block_size=11, c=2, image_scaling=3.0, type=cv2.THRESH_BINARY):
    self.max_value     = max_value
    self.block_size    = block_size
    self.c             = c
    self.image_scaling = image_scaling
    self.type          = type

  # Return PIL Image
  def read(self, image_file, max_value=255, block_size=11, c=2, image_scaling=3.0): 
 
    buf = np.fromfile(image_file, np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # PIL image
    img = Image.fromarray(img)

    return self.apply(img,  max_value, block_size, c, image_scaling)


  def apply(self, img, max_value=255, block_size=11, c=2, image_scaling=3.0): 
    #img = PIL image
    w, h = img.size
    rw = int(w * image_scaling)
    rh = int(h * image_scaling)

    img = img.resize((rw, rh))
    img = img.convert("L")
    # PIL image -> OpenCV Image
    img = np.array(img) 

    #Addaptive thresholding
    # cv2.ADAPTIVE_THRESH_MEAN_C:
    img = cv2.adaptiveThreshold(img, max_value, cv2.ADAPTIVE_THRESH_MEAN_C, self.type, block_size, c)

    img = Image.fromarray(img)
    img = img.convert('RGB')
    # return PIL image
    return img
  
if __name__ == "__main__":
  try:
    image_file = "./input_images/sample.png"
    output_dir = "./output_images"
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)

    thresholder = ImageAdaptiveMeanThresholder()
    max_value  = 255
    block_size = 11
    c          = 2
    image_scaling=3.0
    img = thresholder.read(image_file,  max_value=max_value, block_size=block_size, c=c, image_scaling=image_scaling)
    img.show()
    basename = os.path.basename(image_file)
    output_file = os.path.join(output_dir, "mean_thresholding_" + basename)

    img.save(output_file, format="PNG")
    print("--- Saved {}".format(output_file))
    
  except:
    traceback.print_exc()
