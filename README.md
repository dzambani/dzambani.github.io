# dzambani.github.io

Goal:

Smart homes are essential conveniences for consumers who want to make their lives just a little bit easier. At the forefront of these advancing technologies are smart assistants. This project aims to advance smart assistant functionality in noisy environments, such as a kitchen, and be able to help further ease a consumer’s life by identifying/classifying differing sounds while managing useful events for each noise identified.

Initial Aims:

Identify if an appliance is on or off.
Classify the appliances being used.
Identify how long an appliance is on for.
Create an event or sequence of events based on appliances being used.
Identify multiple appliances being used at once (noisy environment).
STRETCH GOAL: Use a camera to help identify objects inside of the scene as well. 
STRETCH GOAL: Send appliance ‘stats’ to the user. I.E: Potential Power Draw, Time On, Etc.

Demo: https://youtu.be/WgL5V4ro6xw
Website: https://dzambani.github.io/

RUNNING CODE:
	
	Equipment Needed: Raspberry Pi 4, ReSpeaker Microphone Array USB, Raspberry Pi Camera V2, Google Coral TPU

Running Classify:
	
	1) Install Tensorflow Lite From Official Website: https://www.tensorflow.org/lite/guide
	2) Download Classify file from Github Repo
	3) Install Dependencies: cv2, configparser, numpy, subprocess, re
	4) Run classify.py
	** If any dependancies are missing, install as dependencies come up in terminal**
	**All code paths are absolute paths, make sure you update paths according to your own directories accordingly**
	
Running ODAS (must be run in parallel with Classify for Spatial Filtering):
	
	1) Download and Install ODAS: https://github.com/introlab/odas/wiki
	2) Download the configuration files 
	3) Adjust Spatial Filtering based on variables from vid.py (uncomment print(angle_from_center))
	4) Adjust configuration file of your objects based on relative angles
	5) Confirm output of SSS: in tracked: is output to text file, not terminal
	5) Run ODAS in parallel with classify.py (separate terminals or refer to parallel.py)
	6) Running ODAS can be found here: https://github.com/introlab/odas/wiki
	- Running ODAS is tricky, here are our instructions: 
		Install and build 
			sudo apt-get install libfftw3-dev libconfig-dev libasound2-dev libgconf-2-4 libpulse-dev
			sudo apt-get install cmake
			cd ~/Desktop
			git clone https://github.com/introlab/odas.git
			mkdir odas/build
			cd odas/build
			cmake ..
			make

			***There might be some missing libraries, just sudo apt upgrade and update, sudo apt install them, sudo apt upgrade and update again, and build again.***

		Configuration 
			Feel free to look at this: https://github.com/introlab/odas/wiki/configuration 
			But it works just by using the file in PATH/odas/config/odaslive/respeaker_usb_4_mic_array.cfg
			Make sure you are reading from the socket, not a file:
			interface: {
            			type = "socket"; ip = "127.0.0.1"; port = 9004;
            			type = "file"; path = "postfiltered.raw"; 
        		};  

		Running ODAS 
			./odaslive -c myConfigFile.cfg

			***Where myConfigFile is respeaker_usb_4_mic_array.cfg. Notice how it is in the same directory as the odaslive file. The odaslive file will be in PATH/odas/build/bin/live. If you do not want them in the same directory just make sure the paths match up when calling odaslive.***
			
AWS integration setup: 

	1) Create a DynamoDB database 
	2) Create a Lambda function (lambda folder) 
	3) Create an HTTP API Gateway and configure routes to Lambda function 
