import numpy as np
import json
import os
import random

n_identity=533
ratio=0.5

trainval=[]
query=[]
gallery=[]

for i in range(0, 533):
    if random.random()>ratio:
        trainval.append(i)
    else:
        query.append(i)
        gallery.append(i)


data={"trainval" : trainval, "query" : query, "gallery": gallery}

with open("splits.json", "w") as json_file:
  json.dump([data], json_file, indent=4)
