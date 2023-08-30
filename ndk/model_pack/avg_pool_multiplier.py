# -*- coding: utf-8 -*-
"""
Copyright(c) 2018 by Ningbo XiTang Information Technologies, Inc and
WuQi Technologies, Inc. ALL RIGHTS RESERVED.

This Information is proprietary to XiTang and WuQi, and MAY NOT be copied by
any method or incorporated into another program without the express written
consent of XiTang and WuQi. This Information or any portion thereof remains
the property of XiTang and WuQi. The Information contained herein is believed
to be accurate and XiTang and WuQi assumes no responsibility or liability for
its use in any way and conveys no license or title under any patent or copyright
and makes no representation or warranty that this Information is free from patent
or copyright infringement.
"""

import math
import sys
import os
modelpack_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(modelpack_dir)

class cnn_avg_pooling_multiplier_bean:
    def __init__(self):
        self.multiplier = 0
        self.shift_num = 0
        self.pooling_count_l_u = 0 
        self.pooling_count_r_u = 0 
        self.pooling_count_l_d = 0 
        self.pooling_count_r_d = 0 
        self.pooling_count_c_u = 0 
        self.pooling_count_c_d = 0 
        self.pooling_count_l_c = 0 
        self.pooling_count_r_c = 0 
        self.pooling_count_c_c = 0
        self.signed = 1

    def cnn_get_pooling_multiplier_8bits(self,number_in_kernel, repeat_to_16b = True):
    #    bean = cnn_avg_pooling_multiplier_bean()
        quotient = int(0x80000000 / number_in_kernel)
        first_one_in_quotient = 0;
        for i in range(31, -1, -1):
            if quotient & (1 << i) != 0:
                first_one_in_quotient = i;
                break
    
        frac_width_of_quotient = first_one_in_quotient - 7;
        multiplier = quotient >> frac_width_of_quotient;
        if quotient & (1 << (frac_width_of_quotient - 1)) != 0:
            multiplier += 1
            
        while True:
            if multiplier & 1 != 0:
                break
            frac_width_of_quotient += 1
            multiplier = multiplier >> 1
            
    #    bean.multiplier = (multiplier << 8) + multiplier if repeat_to_16b else multiplier
    #    bean.shift_num = 31 - frac_width_of_quotient
        self.multiplier = (multiplier << 8) + multiplier if repeat_to_16b else multiplier
        self.shift_num = 31 - frac_width_of_quotient
    #    return bean
    
    
    def cnn_get_pooling_multiplier_16bits(self,number_in_kernel):
    #    bean = cnn_avg_pooling_multiplier_bean()
        quotient = int(0x80000000 / number_in_kernel)
        first_one_in_quotient = 0
        for i in range(31, -1, -1):
            if quotient & (1 << i) != 0:
                first_one_in_quotient = i;
                break
            
        frac_width_of_quotient = first_one_in_quotient - 14
        multiplier = quotient >> frac_width_of_quotient
        if quotient & (1 << (frac_width_of_quotient - 1)) != 0:
            multiplier += 1
        
        while True:
            if multiplier & 1 != 0:
                break
            frac_width_of_quotient += 1
            multiplier = multiplier >> 1
    #    bean.multiplier = multiplier
    #    bean.shift_num = 31 - frac_width_of_quotient
        self.multiplier = multiplier
        self.shift_num = 31 - frac_width_of_quotient
    #    return bean
    
    def cnn_get_pooling_multiplier(self, bit_width, pad_l, pad_u, pad_d, k_h, k_w, pooling_count_pad, h_i, w_i):
        if bit_width == 8:
            self.signed = 0
        else:
            self.signed = 1
        edge_left = min(k_w - pad_l, w_i)
        edge_up = min(k_h - pad_u, h_i)
        edge_right = min(k_w, w_i)
        edge_down = min(k_h - pad_d, k_h, h_i)
        assert edge_up > 0
        assert edge_down > 0
        assert edge_left > 0
        assert edge_right > 0
        center_h = min(k_h, h_i)
        center_w = min(k_w, w_i)
        if pooling_count_pad != 0:
            edge_left = min(k_w, w_i)
            edge_up = min(k_h, h_i)
        print((edge_left, edge_up, edge_right, edge_down))
        min_count_edge = min(edge_left * edge_up, edge_left * edge_down, edge_right * edge_up, edge_right * edge_down)
        right_shift = int(math.ceil(math.log2(min_count_edge))) - 1
        right_shift_num = 2 ** right_shift
        weight_tv_frac = 0
        if self.signed > 0: # signed int
            self.pooling_count_l_u = int(round(2 ** bit_width / 2 * right_shift_num / edge_left / edge_up))
            self.pooling_count_r_u = int(round(2 ** bit_width / 2 * right_shift_num / edge_right / edge_up))
            self.pooling_count_l_d = int(round(2 ** bit_width / 2 * right_shift_num / edge_left / edge_down))
            self.pooling_count_r_d = int(round(2 ** bit_width / 2 * right_shift_num / edge_right / edge_down))
            self.pooling_count_c_u = int(round(2 ** bit_width / 2 * right_shift_num / center_w / edge_up))
            self.pooling_count_c_d = int(round(2 ** bit_width / 2 * right_shift_num / center_w / edge_down))
            self.pooling_count_l_c = int(round(2 ** bit_width / 2 * right_shift_num / edge_left / center_h))
            self.pooling_count_r_c = int(round(2 ** bit_width / 2 * right_shift_num / edge_right / center_h))
            self.pooling_count_c_c = int(round(2 ** bit_width / 2 * right_shift_num / center_w / center_h))
            weight_tv_frac = right_shift + bit_width - 1
        else:
            self.pooling_count_l_u = int(round(2 ** bit_width * right_shift_num / edge_left / edge_up))
            self.pooling_count_r_u = int(round(2 ** bit_width * right_shift_num / edge_right / edge_up))
            self.pooling_count_l_d = int(round(2 ** bit_width * right_shift_num / edge_left / edge_down))
            self.pooling_count_r_d = int(round(2 ** bit_width * right_shift_num / edge_right / edge_down))
            self.pooling_count_c_u = int(round(2 ** bit_width / center_w * right_shift_num / edge_up))
            self.pooling_count_c_d = int(round(2 ** bit_width / center_w * right_shift_num / edge_down))
            self.pooling_count_l_c = int(round(2 ** bit_width / edge_left * right_shift_num / center_h))
            self.pooling_count_r_c = int(round(2 ** bit_width / edge_right * right_shift_num / center_h))
            self.pooling_count_c_c = int(round(2 ** bit_width / center_w * right_shift_num / center_h))
            weight_tv_frac = right_shift + bit_width
        self.pooling_count_r_c = self.pooling_count_c_c
        self.pooling_count_r_d = self.pooling_count_c_d
        self.pooling_count_r_u = self.pooling_count_c_u
#        if bit_width == 8:
#            self.pooling_count_l_u = self.pooling_count_l_u + 
#            self.pooling_count_r_u = self.pooling_count_r_u * 257
#            self.pooling_count_l_d = self.pooling_count_l_d * 257
#            self.pooling_count_r_d = self.pooling_count_r_d * 257
#            self.pooling_count_c_u = self.pooling_count_c_u * 257
#            self.pooling_count_c_d = self.pooling_count_c_d * 257
#            self.pooling_count_l_c = self.pooling_count_l_c * 257
#            self.pooling_count_r_c = self.pooling_count_r_c * 257
#            self.pooling_count_c_c = self.pooling_count_c_c * 257
        self.shift_num = weight_tv_frac
