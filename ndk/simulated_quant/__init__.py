#!\usr\bin\python
# -*- coding:utf-8 -*-

# __all__ = ["tf_sim_quant_int8_n4","tf_sim_quant_int8_n5","tf_sim_quant_int8_n6"]

import os, sys

cur_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_path)

from interf_func import get_function_tf
sys.path.remove(cur_path)