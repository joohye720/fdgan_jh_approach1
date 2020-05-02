import numpy as np
import json
import os

cam_names =["cam1", "cam2", "cam3", "cam4", "cam5", "cam6"]
identity_list=[]

for i in range(0,533):
  cam1_array=[]
  cam2_array=[]
  cam3_array=[]
  cam4_array=[]
  cam5_array=[]
  cam6_array=[]
  for j in range(0,6):
    identity = str(i+1).rjust(4,'0')
    path="%s/%s"%(cam_names[j], identity)
    #print(path)
    try:
        flist=os.listdir(path)
        #print(len(flist))
        for k in range(1,len(flist)+1):
          identity = str(i).rjust(8,'0')
          cam_id = str(j).rjust(2,'0')
          img_idx = str(k).rjust(4,'0') 
          fname="%s_%s_%s.jpg"%(identity, cam_id, img_idx)
          if(j==0):
            cam1_array.append(fname)
          elif(j==1):
            cam2_array.append(fname)
          elif(j==2):
            cam3_array.append(fname)
          elif(j==3):
            cam4_array.append(fname)
          elif(j==4):
            cam5_array.append(fname)
          elif(j==5):
            cam6_array.append(fname)
    except:
        print("no such folder")

  identity_list.append([cam1_array, cam2_array, cam3_array, cam4_array, cam5_array, cam6_array])


data={"name" : "SYSUMM01", "shot" : "multiple", "num_cameras" : 6, "identities": identity_list}

with open("meta.json", "w") as json_file:
  json.dump(data, json_file, indent=4)
