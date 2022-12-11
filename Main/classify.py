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
"""Main scripts to run audio classification."""

import argparse
import sys
import time
import requests
import json
from datetime import datetime
import numpy as np
import vid
import aud
import cv2

from tflite_support.task import audio
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision

from aud import Plotter

url = "https://3i859s5gi8.execute-api.us-west-1.amazonaws.com/items"

def video(model1: str, max_results: int, score_threshold: float,
        overlapping_factor: float, num_threads1: int,
        enable_edgetpu1: bool, model2: str, camera_id: int, width: int, height: int, num_threads2: int,
        enable_edgetpu2: bool) -> None:
  """Continuously run inference on audio data acquired from the device.

  Args:
    model: Name of the TFLite audio classification model.
    max_results: Maximum number of classification results to display.
    score_threshold: The score threshold of classification results.
    overlapping_factor: Target overlapping between adjacent inferences.
    num_threads: Number of CPU threads to run the model.
    enable_edgetpu: Whether to run the model on EdgeTPU.
  """

  if (overlapping_factor <= 0) or (overlapping_factor >= 1.0):
    raise ValueError('Overlapping factor must be between 0 and 1.')

  if (score_threshold < 0) or (score_threshold > 1.0):
    raise ValueError('Score threshold must be between (inclusive) 0 and 1.')

  # Initialize the audio classification model.
  base_options = core.BaseOptions(
      file_name=model1, use_coral=enable_edgetpu1, num_threads=num_threads1)
  classification_options = processor.ClassificationOptions(
      max_results=max_results, score_threshold=score_threshold)
  options = audio.AudioClassifierOptions(
      base_options=base_options, classification_options=classification_options)
  classifier = audio.AudioClassifier.create_from_options(options)
  
  # Initialize the audio recorder and a tensor to store the audio input.
  audio_record = classifier.create_audio_record()
  tensor_audio = classifier.create_input_tensor_audio()

  # Variables to calculate FPS
  counter, fps = 0, 0
  start_time = time.time()

  # Start capturing video input from the camera
  cap = cv2.VideoCapture(camera_id)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

  # Visualization parameters
  row_size = 20  # pixels
  left_margin = 24  # pixels
  text_color = (0, 0, 255)  # red
  font_size = 1
  font_thickness = 1
  fps_avg_frame_count = 10

  # Initialize the object detection model
  base_options = core.BaseOptions(
      file_name=model2, use_coral=enable_edgetpu2, num_threads=num_threads2)
  detection_options = processor.DetectionOptions(
      max_results=5, score_threshold=0.40)
  options = vision.ObjectDetectorOptions(
      base_options=base_options, detection_options=detection_options)
  detector = vision.ObjectDetector.create_from_options(options)
  
  # We'll try to run inference every interval_between_inference seconds.
  # This is usually half of the model's input length to create an overlapping
  # between incoming audio segments to improve classification accuracy.
  input_length_in_second = float(len(
      tensor_audio.buffer)) / tensor_audio.format.sample_rate
  interval_between_inference = input_length_in_second * (1 - overlapping_factor)
  pause_time = interval_between_inference * 0.1
  last_inference_time = time.time()

  # Initialize a plotter instance to display the classification results.
  #plotter = Plotter()

  # Start audio recording in the background.
  audio_record.start_recording()

  centroid_dict = {} #dictionary of all centroids (person and appliances).
  centroid_diff_dict = {} #dictionary of differences between person centroid and all centroids (person and appliances; person diff should be 0).
  angle_from_center = {} 
  appliance_time = {}
  appliance_probabilities = {}

  blend_count = 0
  flag_blend_message_sent = False

  water_count = 0
  flag_water_message_sent = False
  
  chop_count = 0
  flag_song_on = False
  
  firealarm_count = 0
  firealarm_enable = 1
  
  fire_detected = 1
  flag_fire_alarm_on = False
  
  import subprocess
  import os
  import signal
    
  # Loop until the user close the classification results plot.
  while cap.isOpened():
    success, image = cap.read()
    # Wait until at least interval_between_inference seconds has passed since
    # the last inference.
    now = time.time()
    diff = now - last_inference_time
    if diff < interval_between_inference:
      time.sleep(pause_time)
      continue
    last_inference_time = now
    if not success:
      sys.exit(
          'ERROR: Unable to read from webcam. Please verify your webcam settings.'
      )
    counter += 1
    image = cv2.flip(image, 1)

    # Convert the image from BGR to RGB as required by the TFLite model.
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create a TensorImage object from the RGB image.
    input_tensor = vision.TensorImage.create_from_array(rgb_image)

    # Run object detection estimation using the model.
    detection_result = detector.detect(input_tensor)

    # Draw keypoints and edges on input image
    image = vid.visualize(image, detection_result, centroid_dict, centroid_diff_dict, angle_from_center, appliance_time, appliance_probabilities)
    
    # Load the input audio and run classify.
    tensor_audio.load_from_audio_record(audio_record)
    result = classifier.classify(tensor_audio)
    #print(type(result))
    #print(result)
    #exit(0)
    res = result.classifications[0].categories
    
    my_res = []
    tot_prob = 0
    for cat in res:
        tot_prob += cat.score
        if cat.category_name == "Silence" or cat.category_name == "Blender" or cat.category_name == "Chopping (food)" or cat.category_name == "Water tap, faucet" or cat.category_name == "Fire alarm" or cat.category_name == "Microwave oven":
        #or cat.category_name == "Speech":
            my_res.append(cat)
    
    #print("Total probability:", tot_prob)
    s = np.finfo(float).eps
    for cat in my_res:
        s += cat.score
    
    for cat in my_res:
        cat.score /= s

    for cat in my_res:
        if cat.category_name == "Microwave oven":
          new_microwave = cat.score + appliance_probabilities["microwave"]
          
    #Fix probabilities for events: Sink/Fauces  
        if cat.category_name == "Water tap, faucet":
          new_faucet = cat.score + appliance_probabilities["sink"]
          print(new_faucet)

    # Calculate the FPS
    if counter % fps_avg_frame_count == 0:
      end_time = time.time()
      fps = fps_avg_frame_count / (end_time - start_time)
      start_time = time.time()

    #print(my_res) 
    for cat in my_res:
        if cat.category_name == "Blender":
            if cat.score > 0.25:    
                blend_count += 1
                appliance_on(cat.category_name)
            else:
                blend_count = 0
                appliance_off(cat.category_name)

        if cat.category_name == "Water tap, faucet":
            if new_faucet > 0.25:    
                water_count += 1
                appliance_on(cat.category_name)
            else:
                water_count = 0
                appliance_off(cat.category_name)
                
        if cat.category_name == "Microwave oven":
            if new_microwave > 0.25:    
                appliance_on(cat.category_name)
                if chop_count > 10:
                    chop_count = 10
            else:
                appliance_off(cat.category_name)
                
        if cat.category_name == "Fire alarm":
            if cat.score > 0.25:    
                firealarm_count += 1
                appliance_on(cat.category_name)
            else:
                firealarm_count = 0
                appliance_off(cat.category_name)
    
    if chop_count == 10 and not flag_song_on:
        player = subprocess.Popen(["python3", "music.py"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        flag_song_on = True
        
    if chop_count == 0 and flag_song_on:
        os.kill(player.pid, signal.SIGSTOP)
        flag_song_on = False
         
    #print("blender count: ", blend_count)
    #print("chop count: ", chop_count)
    
    if blend_count > 5 and not flag_blend_message_sent:
        blender_message()
        blend_count = 0
        flag_blend_message_sent = True

    if water_count > 5 and not flag_water_message_sent:
        water_message()
        water_count = 0
        flag_water_message_sent = True
        
    #if flag_blend_message_sent:
        
    if firealarm_count > 5 and firealarm_enable:
        firealarm_message()
        firealarm_count = 0
        firealarm_enable = 0
        
    if fire_detected and not flag_fire_alarm_on:
        player = subprocess.Popen(["python3", "fire_alarm.py"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        flag_fire_alarm_on = True
        
    
        # Show the FPS
    fps_text = 'FPS = {:.1f}'.format(fps)
    text_location = (left_margin, row_size)
    cv2.putText(image, fps_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                font_size, text_color, font_thickness)
                
    # Stop the program if the ESC key is pressed.
    if cv2.waitKey(1) == 27:
      break
    cv2.imshow('object_detector', image)
  cap.release()
  cv2.destroyAllWindows()
    #exit()
    
    
    # Plot the classification results.
    #plotter.plot(result)
    
#def chopper_music():

def appliance_on(appliance):
    #appliance = "Water tap, faucet" #replace with actual appliance variable
    stat = True
    now = datetime.now()
    appliance_data = {}
    appliance_data['id'] = appliance
    appliance_data['stat'] = stat
    appliance_data['timestamp'] = now.strftime("%m/%d/%Y, %H:%M:%S")
    requests.put(url, json=appliance_data)

def appliance_off(appliance):
    #appliance = "Water tap, faucet" #replace with actual appliance variable
    stat = False
    now = datetime.now()
    appliance_data = {}
    appliance_data['id'] = appliance
    appliance_data['stat'] = stat
    appliance_data['timestamp'] = now.strftime("%m/%d/%Y, %H:%M:%S")
    requests.put(url, json=appliance_data)

def blender_message():
    from twilio.rest import Client

    # Your Account SID from twilio.com/console
    account_sid = "ACb101f148458276fc50991a3dbe95999d"
    # Your Auth Token from twilio.com/console
    auth_token  = "4f22ae8700d17f9ff3ef8d7b72db587b"

    client = Client(account_sid, auth_token)

    message = client.messages.create(
        to="+13322488127", 
        from_="+18087552531",
        body="Blender turned on from quite a long time!")

def water_message():
    from twilio.rest import Client

    # Your Account SID from twilio.com/console
    account_sid = "ACb101f148458276fc50991a3dbe95999d"
    # Your Auth Token from twilio.com/console
    auth_token  = "4f22ae8700d17f9ff3ef8d7b72db587b"

    client = Client(account_sid, auth_token)

    message = client.messages.create(
        to="+13322488127", 
        from_="+18087552531",
        body="Water Faucet on from quite a long time!")

def firealarm_message():
    from twilio.rest import Client

    # Your Account SID from twilio.com/console
    account_sid = "ACb101f148458276fc50991a3dbe95999d"
    # Your Auth Token from twilio.com/console
    auth_token  = "4f22ae8700d17f9ff3ef8d7b72db587b"

    client = Client(account_sid, auth_token)

    message = client.messages.create(
        to="+13322488127", 
        from_="+18087552531",
        body="Fire Alarm at Home!")

def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
      
  parser.add_argument(
      '--model1',
      help='Name of the audio classification model.',
      required=False,
      default='yamnet.tflite')
  parser.add_argument(
      '--maxResults',
      help='Maximum number of results to show.',
      required=False,
      default=10)
  parser.add_argument(
      '--overlappingFactor',
      help='Target overlapping between adjacent inferences. Value must be in (0, 1)',
      required=False,
      default=0.5)
  parser.add_argument(
      '--scoreThreshold',
      help='The score threshold of classification results.',
      required=False,
      default=0.0)
  parser.add_argument(
      '--numThreads1',
      help='Number of CPU threads to run the model.',
      required=False,
      default=4)
  parser.add_argument(
      '--enableEdgeTPU1',
      help='Whether to run the model on EdgeTPU.',
      action='store_true',
      required=False,
      default=False)

  parser.add_argument(
      '--model2',
      help='Path of the object detection model.',
      required=False,
      default='efficientdet_lite0_edgetpu.tflite')
  parser.add_argument(
      '--cameraId', help='Id of camera.', required=False, type=int, default=0)
  parser.add_argument(
      '--frameWidth',
      help='Width of frame to capture from camera.',
      required=False,
      type=int,
      default=640)
  parser.add_argument(
      '--frameHeight',
      help='Height of frame to capture from camera.',
      required=False,
      type=int,
      default=480)
  parser.add_argument(
      '--numThreads2',
      help='Number of CPU threads to run the model.',
      required=False,
      type=int,
      default=4)
  parser.add_argument(
      '--enableEdgeTPU2',
      help='Whether to run the model on EdgeTPU.',
      action='store_true',
      required=False,
      default=True)
      
  args = parser.parse_args()
      
  video(args.model1, int(args.maxResults), float(args.scoreThreshold),
      float(args.overlappingFactor), int(args.numThreads1),
      bool(args.enableEdgeTPU1), args.model2, int(args.cameraId), args.frameWidth, args.frameHeight,
      int(args.numThreads2), bool(args.enableEdgeTPU2))


if __name__ == '__main__':
  main()
