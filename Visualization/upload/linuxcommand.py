import time
import json,os

import uuid
pvpythonpath = '/home/ubuntu/paraview/bin'
datapath = '/home/ubuntu/Data'
'''
调用pvpython 运行可视化app，调用data路径加载数据
'''

os.system("/home/ubuntu/paraview/bin/pvpython -m paraview.apps.visualizer --data /home/ubuntu/Data --port 9090")