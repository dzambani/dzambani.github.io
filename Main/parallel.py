import subprocess
import sys
import os


#os.system(r"/home/natepi/SystemsProject/odas/build/bin/odaslive -c respeaker_usb_4_mic_array.cfg")

sys.path.append("/home/natepi/SystemsProject/bin")
executable_process = subprocess.Popen(["/home/natepi/SystemsProject/odas/build/bin/odaslive", "-c", "/home/natepi/SystemsProject/odas/build/bin/respeaker_usb_4_mic_array.cfg"])

python_script_process = subprocess.Popen(["python", "/home/natepi/SystemsProject/Combined_AudioVid/Updated/classify_7.py"])
