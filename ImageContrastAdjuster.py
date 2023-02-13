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

# ImageContrastAdjuster.py
#
# 2023/02/01


import os
import numpy as np
from  PIL import Image
import cv2
import traceback

class ImageContrastAdjuster:

  def __init__(self):
    pass

  
  # Return PIL Image
  def read(self, image_file, alpha=1.5, beta=40, image_scaling=3.0):
    print("--- ImageContrastAdjunster image_file {}".format(image_file))

    buf = np.fromfile(image_file, np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    # PIL Image
    img = Image.fromarray(img)
    return self.apply(img, alpha, beta, image_scaling)

  
  # brightness and contrast adjuster for PIL_Image
  def apply(self, image,  alpha=1.5, beta=40, image_scaling=3.0):
    # PIL image -> OpenCV Image
    img = np.array(image) 

    img  = cv2.resize(img, dsize=None, fx=image_scaling, fy=image_scaling)
    
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

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

    adjuster = ImageContrastAdjuster()
    alpha = 1.5
    beta  = 40
    image_scaling = 3.0
    img = adjuster.read(image_file,  alpha, beta, image_scaling)
    img.show()
    
    basename = os.path.basename(image_file)
    output_file = os.path.join(output_dir, "constrast_adjust_" + basename)

    img.save(output_file, format="PNG")
    print("--- Saved {}".format(output_file))
  except:
    traceback.print_exc()
