# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utility functions to display the pose detection results."""
""" VARIABLES && IMPORTS"""
import cv2
import operator
import math
import configparser
import numpy as np
from tflite_support.task import processor
from subprocess import call
from datetime import datetime
import time
import os
import re
import time

_MARGIN = 10 #Pixels
_MARGIN2 = 20  # pixels
_ROW_SIZE = 10 #Pixels
_ROW_SIZE2 = 20 # pixels
_FONT_SIZE = 1
_FONT_THICKNESS = 1
_TEXT_COLOR = (0, 0, 255)  # red

def follow(thefile):
    thefile.seek(0, os.SEEK_END)
    
    for x in range(5):
        line = thefile.readline()
        if not line:
            time.sleep(0.1)
            continue
        yield line
  

def visualize(
    image: np.ndarray,
    detection_result: processor.DetectionResult,
    centroid_dict,
    centroid_diff_dict,
    angle_from_center,
    appliance_time, 
    appliance_probabilities
) -> np.ndarray:
  """Draws bounding boxes on the input image and return it.

  Args:
    image: The input RGB image.
    detection_result: The list of all "Detection" entities to be visualize.

  Returns:
    Image with bounding boxes.
  """
  ####
  #centroid_dict = {} #dictionary of all centroids (person and appliances).
  #centroid_diff_dict = {} #dictionary of differences between person centroid and all centroids (person and appliances; person diff should be 0).
  #angle_from_center = {} #dictionary to get the differences of centroid of objects from the center of the camera.
  ####
  
  ####
  for detection in detection_result.detections:
    #print(detection_result)
    # Draw bounding_box and isolate certain objects only
    category = detection.categories[0]
    category_name = category.category_name
    # Filter out any other categories	
    my_now = datetime.now()
    for key in appliance_time.keys(): 
      diff = my_now - appliance_time[key]
      if diff.total_seconds() >= 10: 
        appliance_probabilities[key] = 0.0

    if category_name in ["microwave", "oven", "refrigerator", "person", "sink", "toaster"]:
        bbox = detection.bounding_box
        start_point = bbox.origin_x, bbox.origin_y
        end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
    #Calculate centroid of the bounding box and object
        centroid = ((bbox.width/2)+bbox.origin_x),((bbox.height/2)+bbox.origin_y)
        ccenter = (320,240)
   ####
   
   ####
   #""""Cycle through detected categories and find all their distances from human"""
   #"""Works only if a human is detected"""
        centroid_dict[category_name] = centroid
        #print(centroid_dict)
        
  ##Calculate the relative angle from the center of the camera to real life space.
        if category_name == "microwave" or category_name == "oven" or category_name == "sink":
          for x in centroid_dict.keys():
            angle_from_center[x] = ((((centroid_dict[x][0] - ccenter[0]) / ccenter[0])*(62.2/2)) , ((centroid_dict[x][1] - ccenter[1]) / ccenter[1]) * (48.8/2))
            #print(angle_from_center)
            
  ##Calculate the difference in centroids.
        if "person" in centroid_dict.keys(): 
          for key in centroid_dict.keys(): 
            centroid_diff_dict[key] = (abs(centroid_dict[key][0] - centroid_dict["person"][0]), abs(centroid_dict[key][1] - centroid_dict["person"][1]))
            #print(centroid_diff_dict)
    ####
    
    ####
    #"""If centroids are relatively close to one another, start to countdown, then give a "help" probability"""
          closeness_value = (125, 125)
          closeness_2 = (250, 250)
          #Microwave
          if centroid_diff_dict["microwave"][0] <= closeness_value[0] and centroid_diff_dict["microwave"][1] <= closeness_value[1]:
            #print("True, Microwave")
            appliance_time["microwave"] = datetime.now()
            appliance_probabilities["microwave"] = 0.2
            #print(appliance_probabilities)
          else:
            appliance_probabilities["microwave"] = 0.0
            #print(appliance_probabilities)
                
          #Sink
          if centroid_diff_dict["sink"][0] <= closeness_2[0] and centroid_diff_dict["sink"][1] <= closeness_2[1]:
            #print("True, Sink")
            logfile = open('/home/natepi/SystemsProject/odas/build/bin/tracks.txt','r')
            loglines = follow(logfile)
            for line in loglines:
                l = re.search('"activity": (.+?) }', line)
                if l:
                   #print(l)
                   found= l.group(1)
                   value = float(found)
                   print(value)
          else:
             appliance_probabilities["sink"] = 0.0
            
            #Oven
          if centroid_diff_dict["oven"][0] <= closeness_value[0] and centroid_diff_dict["oven"][1] <= closeness_value[1]:
            #print("True,Oven")
            appliance_probabilities["oven"] = 0.2
    ####
    
    ####
    #"""Output the probability of the categories and print"""
        probability = round(category.score, 2)
        result_text = category_name + ' (' + str(probability) + ')'
        text_location = (_MARGIN + bbox.origin_x,
                         _MARGIN + _ROW_SIZE + bbox.origin_y)
    ####
    
    ####
    #"""Output the bounding box"""
        text_location2 = (_MARGIN2 + bbox.origin_x,
                         _MARGIN2 + _ROW_SIZE2 + bbox.origin_y)
                         
        cv2.rectangle(image,start_point,end_point,_TEXT_COLOR,3)
        
        cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                    _FONT_SIZE, _TEXT_COLOR, _FONT_THICKNESS)
    ####
    
  return image

