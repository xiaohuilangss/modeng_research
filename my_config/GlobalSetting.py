# coding=utf-8

""" =========================== 将当前路径及工程的跟目录添加到路径中，必须在文件头部，否则易出错 ============================ """

import sys
import os

curPath = os.path.abspath(os.path.dirname(__file__))
if "MoDeng" in curPath:
    root_path = curPath[:curPath.find("MoDeng") + len("MoDeng")]
elif "MoDeng-master" in curPath:
    root_path = curPath[:curPath.find("MoDeng-master") + len("MoDeng-master")]
elif "Modengserver" in curPath:
    root_path = curPath[:curPath.find("Modengserver") + len("Modengserver")]
elif "modeng_research" in curPath:
    root_path = curPath[:curPath.find("modeng_research") + len("modeng_research")]
else:
    input('没有找到项目的根目录！请检查项目根文件夹的名字！')
    exit(1)
    
sys.path.append('..')
sys.path.append(root_path + '/' + 'server/futures/ctp/6.3.15_20190220/linux')

sys.path.append('..')
sys.path.append(root_path + '/' + 'server/futures/ctp/6.3.15_20190220/win64')