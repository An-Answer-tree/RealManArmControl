import os
import json
from pprint import pprint
from typing import List

qpos_list = list()

# path = "/home/user/Project/RealManArmControl/rgb_depth_saved/arm_pose"
# for basename in os.listdir(path):
#     filepath = os.path.join(path, basename)
#     with open(filepath, "r", encoding="utf-8") as f:
#         arm_pose = json.load(f)
#         qpos = arm_pose["joints"]
#         qpos_list.append(qpos)

# pprint(qpos_list)
# print(len(qpos_list))