import requests
import json
from datetime import datetime

url = "https://3i859s5gi8.execute-api.us-west-1.amazonaws.com/items"

#when the appliance starts, we change the status to true and log the timestamp
appliance = "sink" #replace with actual appliance variable
stat = True
now = datetime.now()
appliance_data = {}
appliance_data['id'] = appliance
appliance_data['stat'] = stat
appliance_data['timestamp'] = now.strftime("%m/%d/%Y, %H:%M:%S")
requests.put(url, json=appliance_data)

#when the appliance stops, we change the status to false and log the timestamp
appliance = "sink" #replace with actual appliance variable
stat = False
now = datetime.now()
appliance_data = {}
appliance_data['id'] = appliance
appliance_data['stat'] = stat
appliance_data['timestamp'] = now.strftime("%m/%d/%Y, %H:%M:%S")
requests.put(url, json=appliance_data)
