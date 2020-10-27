# encoding=utf-8

"""
本脚本用于存放测试用函数，主要是debug日志打印函数
"""
from my_config.AutoGenerateConfigFile import data_dir
from sdk.MyTimeOPT import get_current_date_str, get_current_datetime_str
import os

def myPrint(str_gui, str_temp, method='n', towho=''):
    """
    gn:note消息
    gm:msg消息
    
    :param gui:
    :param str_gui:
    :param method:
    :param towho:
    :return:
    """
    if method is 'n':
        print(str_temp)

    elif method is 'gm':
        str_gui['msg'] = str_gui['msg'] + str_temp + '\n\n'

    elif method is 'gn':
        str_gui['note'] = str_gui['note'] + str_temp + '\n\n'

    elif method is 'qq':
        # send_qq(towho, str_temp)
        pass
    return str_gui


def debug_print_txt(file_name, stk, value, enable=False):
    """

    :param enable:
    :param file_name:
    :param stk:
    :param value:
    :return:
    """
    if not enable:
        return
    
    file_dir = data_dir + 'Debug_log/' + get_current_date_str()
    file_url = file_dir + '/' + file_name + '_' + stk + '.txt'

    # 如果文件夹不存在，创建
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    with open(file_url, 'a+') as f:
        f.write(get_current_datetime_str() + ':\n-------------------\n' + value + '\n')




