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
import numpy as np
import os
import sys
ndk_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(ndk_dir)
import ndk
from ndk.quant_tools.quant_func import quantized_value
import math
import copy

def normalize_distribution(distribution):   
    num_sum = sum(distribution) 
    if num_sum > 0.001:
        for i, data in enumerate(distribution): 
            distribution[i] = data /float(num_sum)
    
def exp_int_threshold(distribution ,maxes, pct0, pct1, interval):
    max_ex=int(math.floor(math.log(maxes, 2)))
    thres=[]
    exp_loc=int(pow(2, max_ex)* interval/maxes )
    sum_before=sum(distribution[:exp_loc])
    while sum_before>=pct0 and sum_before <= pct1:
        thres.append(exp_loc)
        max_ex= max_ex-1
        exp_loc= int(pow(2, max_ex)* interval/maxes )
        sum_before= sum(distribution[:exp_loc])
    if thres==[]:
        thres.append(int(len(distribution)* pct0) )
  #  print(thres)    
    return thres

def add_to_distribution(blob, distribution, interval):
    max_index = len(distribution) - 1
    indexes = np.minimum((np.abs(blob[blob!=0]) / interval).astype(np.int32), max_index)
    for index in indexes: 
        distribution[index] = distribution[index] + 1
      
def add_to_distribution_np(blob, distribution, distribution_n, interval):
    max_index = len(distribution) - 1
    indexes = np.maximum(-max_index, np.minimum((blob[blob!=0] / interval).astype(np.int32), max_index))
    for index in indexes: 
        if index>=0:
            distribution[index] = distribution[index] + 1
        else:
            distribution_n[-index] = distribution_n[-index] + 1
def compute_kl_divergence(dist_a, dist_b):
    for i in range(len(dist_a)):
        if dist_a[i]==0:
            dist_a[i]=1e-20
    for i in range(len(dist_b)):
        if dist_b[i]==0:
            dist_b[i]=1e-20
    return np.sum(dist_a * np.log(dist_a/ dist_b))

def compute_mse_minvalue(dist_a, dist_b):
    return np.sum(np.dot(dist_a-dist_b, dist_a-dist_b))

def threshold_mse_distribution(distribution, target_bin, thres):    
    min_mse_divergence = 1000
    length = len(distribution)
    quantize_distribution = np.zeros(target_bin)
    target_threshold = length-1
    for threshold in thres:
        if threshold >=length:
            t_distribution=[0 for x in range(0, threshold)]
            t_distribution[:length]=copy.deepcopy(distribution) 
            quantize_distribution = np.zeros(target_bin) 
            num_per_bin = threshold / target_bin 
            for i in range(0, target_bin): 
                start = i * num_per_bin 
                end = start + num_per_bin 
                left_upper = (int)(math.ceil(start)) 
                if(left_upper > start): 
                    left_scale = left_upper - start 
                    quantize_distribution[i] += left_scale * t_distribution[left_upper - 1]
                right_lower = (int)(math.floor(end)) 
                if (right_lower < end): 
                    right_scale = end - right_lower 
                    quantize_distribution[i] += right_scale * t_distribution[right_lower]
                for j in range(left_upper, right_lower): 
                    quantize_distribution[i] += t_distribution[j] 
            expand_distribution = np.zeros(threshold, dtype=np.float32)
            for i in range(0, target_bin): 
                start = i * num_per_bin 
                end = start + num_per_bin
                count = 0
                left_upper = (int)(math.ceil(start)) 
                left_scale = 0.0 
                if (left_upper > start): 
                    left_scale = left_upper - start 
                    if (t_distribution[left_upper - 1] != 0): 
                        count += left_scale 
                right_lower = (int)(math.floor(end)) 
                right_scale = 0.0 
                if (right_lower < end): 
                    right_scale = end - right_lower 
                    if (t_distribution[right_lower] != 0): 
                        count += right_scale
                for j in range(left_upper, right_lower): 
                    if (t_distribution[j] != 0): 
                        count = count + 1
                expand_value = quantize_distribution[i] / (1e-30+count)
                if (left_upper > start): 
                    if (t_distribution[left_upper - 1] != 0): 
                        expand_distribution[left_upper - 1] += expand_value * left_scale 
                if (right_lower < end): 
                    if (t_distribution[right_lower] != 0): 
                        expand_distribution[right_lower] += expand_value * right_scale 
                for j in range(left_upper, right_lower): 
                    if (t_distribution[j] != 0): 
                        expand_distribution[j] += expand_value 
          # ************************ quantize ************************ 
            mse_minvalue = compute_mse_minvalue(t_distribution, expand_distribution)
            if mse_minvalue < min_mse_divergence: 
                min_mse_divergence = mse_minvalue
                target_threshold = threshold            
        else:
            threshold_sum = sum(distribution[threshold:])
            t_distribution = copy.deepcopy(distribution[:threshold]) 
            t_distribution[threshold-1] = t_distribution[threshold-1] + threshold_sum
            quantize_distribution = np.zeros(target_bin) 
            num_per_bin = threshold / target_bin 
            for i in range(0, target_bin): 
                start = i * num_per_bin 
                end = start + num_per_bin 
                left_upper = (int)(math.ceil(start)) 
                if(left_upper > start): 
                    left_scale = left_upper - start 
                    quantize_distribution[i] += left_scale * distribution[left_upper - 1]
                right_lower = (int)(math.floor(end)) 
                if (right_lower < end): 
                    right_scale = end - right_lower 
                    quantize_distribution[i] += right_scale * distribution[right_lower] 
      
                for j in range(left_upper, right_lower): 
                    quantize_distribution[i] += distribution[j] 
            expand_distribution = np.zeros(threshold, dtype=np.float32)
            for i in range(0, target_bin): 
                start = i * num_per_bin 
                end = start + num_per_bin
                count = 0
                left_upper = (int)(math.ceil(start)) 
                left_scale = 0.0 
                if (left_upper > start): 
                    left_scale = left_upper - start 
                    if (distribution[left_upper - 1] != 0): 
                        count += left_scale 
                right_lower = (int)(math.floor(end)) 
                right_scale = 0.0 
                if (right_lower < end): 
                    right_scale = end - right_lower 
                    if (distribution[right_lower] != 0): 
                        count += right_scale
                for j in range(left_upper, right_lower): 
                    if (distribution[j] != 0): 
                        count = count + 1
                expand_value = quantize_distribution[i] / (1e-30+count)
                if (left_upper > start): 
                    if (distribution[left_upper - 1] != 0): 
                        expand_distribution[left_upper - 1] += expand_value * left_scale 
                if (right_lower < end): 
                    if (distribution[right_lower] != 0): 
                        expand_distribution[right_lower] += expand_value * right_scale 
                for j in range(left_upper, right_lower): 
                    if (distribution[j] != 0): 
                        expand_distribution[j] += expand_value 
          # ************************ quantize ************************ 
            mse_minvalue = compute_mse_minvalue(t_distribution, expand_distribution)
            if mse_minvalue < min_mse_divergence: 
                min_mse_divergence = mse_minvalue
                target_threshold = threshold
    return target_threshold 

def threshold_minmax_distribution(bitwidth, max_val ,signed=True):
    output_shift=0
    if signed:
        if max_val>0:
            output_shift= bitwidth- 1- math.ceil(math.log(max_val, 2))
    else:
        if max_val>0:
            output_shift= bitwidth- math.ceil(math.log(max_val, 2))
    return  output_shift
        
def threshold_kl_distribution(distribution, target_bin, thres):    
    min_kl_divergence = 1000
    length = len(distribution)
    quantize_distribution = np.zeros(target_bin)
    target_threshold = length-1
    for threshold in thres:
        if threshold >=length:
            t_distribution=[0 for x in range(0, threshold)]
            t_distribution[:length]=copy.deepcopy(distribution) 
            quantize_distribution = np.zeros(target_bin) 
            num_per_bin = threshold / target_bin 
            for i in range(0, target_bin): 
                start = i * num_per_bin 
                end = start + num_per_bin 
                left_upper = (int)(math.ceil(start)) 
                if(left_upper > start): 
                    left_scale = left_upper - start 
                    quantize_distribution[i] += left_scale * t_distribution[left_upper - 1]
                right_lower = (int)(math.floor(end)) 
                if (right_lower < end): 
                    right_scale = end - right_lower 
                    quantize_distribution[i] += right_scale * t_distribution[right_lower]
                for j in range(left_upper, right_lower): 
                    quantize_distribution[i] += t_distribution[j] 
            expand_distribution = np.zeros(threshold, dtype=np.float32)
            for i in range(0, target_bin): 
                start = i * num_per_bin 
                end = start + num_per_bin
                count = 0
                left_upper = (int)(math.ceil(start)) 
                left_scale = 0.0 
                if (left_upper > start): 
                    left_scale = left_upper - start 
                    if (t_distribution[left_upper - 1] != 0): 
                        count += left_scale 
                right_lower = (int)(math.floor(end)) 
                right_scale = 0.0 
                if (right_lower < end): 
                    right_scale = end - right_lower 
                    if (t_distribution[right_lower] != 0): 
                        count += right_scale
                for j in range(left_upper, right_lower): 
                    if (t_distribution[j] != 0): 
                        count = count + 1
                expand_value = quantize_distribution[i] / (1e-30+count)
                if (left_upper > start): 
                    if (t_distribution[left_upper - 1] != 0): 
                        expand_distribution[left_upper - 1] += expand_value * left_scale 
                if (right_lower < end): 
                    if (t_distribution[right_lower] != 0): 
                        expand_distribution[right_lower] += expand_value * right_scale 
                for j in range(left_upper, right_lower): 
                    if (t_distribution[j] != 0): 
                        expand_distribution[j] += expand_value 
          # ************************ quantize ************************ 
            kl_divergence = compute_kl_divergence(t_distribution, expand_distribution)
           # print(kl_divergence)
            if abs(kl_divergence) < min_kl_divergence: 
                min_kl_divergence = abs(kl_divergence)
                target_threshold = threshold            
        else:
            threshold_sum = sum(distribution[threshold:])
            t_distribution = copy.deepcopy(distribution[:threshold]) 
            t_distribution[threshold-1] = t_distribution[threshold-1] + threshold_sum
      # ************************ threshold  ************************ 
            quantize_distribution = np.zeros(target_bin) 
            num_per_bin = threshold / target_bin 
            for i in range(0, target_bin): 
                start = i * num_per_bin 
                end = start + num_per_bin 
                left_upper = (int)(math.ceil(start)) 
                if(left_upper > start): 
                    left_scale = left_upper - start 
                    quantize_distribution[i] += left_scale * distribution[left_upper - 1]
                right_lower = (int)(math.floor(end)) 
                if (right_lower < end): 
                    right_scale = end - right_lower 
                    quantize_distribution[i] += right_scale * distribution[right_lower] 
      
                for j in range(left_upper, right_lower): 
                    quantize_distribution[i] += distribution[j] 
            expand_distribution = np.zeros(threshold, dtype=np.float32)
            for i in range(0, target_bin): 
                start = i * num_per_bin 
                end = start + num_per_bin
                count = 0
                left_upper = (int)(math.ceil(start)) 
                left_scale = 0.0 
                if (left_upper > start): 
                    left_scale = left_upper - start 
                    if (distribution[left_upper - 1] != 0): 
                        count += left_scale 
                right_lower = (int)(math.floor(end)) 
                right_scale = 0.0 
                if (right_lower < end): 
                    right_scale = end - right_lower 
                    if (distribution[right_lower] != 0): 
                        count += right_scale
                for j in range(left_upper, right_lower): 
                    if (distribution[j] != 0): 
                        count = count + 1
                expand_value = quantize_distribution[i] / (1e-30+count)
                if (left_upper > start): 
                    if (distribution[left_upper - 1] != 0): 
                        expand_distribution[left_upper - 1] += expand_value * left_scale 
                if (right_lower < end): 
                    if (distribution[right_lower] != 0): 
                        expand_distribution[right_lower] += expand_value * right_scale 
                for j in range(left_upper, right_lower): 
                    if (distribution[j] != 0): 
                        expand_distribution[j] += expand_value 
          # ************************ quantize ************************ 
            kl_divergence = compute_kl_divergence(t_distribution, expand_distribution)
            if abs(kl_divergence) < min_kl_divergence: 
                min_kl_divergence = abs(kl_divergence )
                target_threshold = threshold
    return target_threshold 

def bias_quant_location(value, bitwidth, input_signed):
    output_quant=0
    output_dequant=0
    output_shift=0
    max_value=0
    min_value=0
    if input_signed:
        max_value=pow(2, bitwidth - 1) - 1
        min_value=-1*pow(2, bitwidth - 1)
    else:
        max_value=pow(2, bitwidth) - 1
        min_value=0
    min_error=sys.maxsize
    input_one=abs(value)
    ex=int(math.log(input_one, 2))
    start_num=bitwidth - 1 - ex - 1
    while start_num<=bitwidth - 1 - ex + 1:
        input_one=value
        input_one=input_one*pow(2,start_num)
        output_int=max(min_value, min(max_value, np.round(input_one) ) )
        output_back=output_int/float(pow(2, start_num))
        output_diff=abs(output_back - value)
        if output_diff <= min_error:
            min_error=output_diff
            min_count=start_num
            output_shift=min_count
            output_quant=max(-1 * max_value, min(max_value - 1, np.round(input_one) ) )
            output_dequant=output_quant/float(pow(2, min_count))        
        start_num=start_num+1
    return output_shift, output_quant, output_dequant    
    
def weight_quant_location(input, bitwidth, min_value_ex, max_value_ex, input_signed):
    max_value=0
    min_value=0
    if input_signed:
        max_value=pow(2, bitwidth - 1)- 1
        min_value=-1 * pow(2, bitwidth - 1)
    else:
        max_value=pow(2, bitwidth)-1
        min_value=0
    min_error=sys.maxsize
    output_shift=0
    start_num=min_value_ex
    while start_num <= max_value_ex:
        input_one=input * 1.0
        input_one=input_one*pow(2, bitwidth - 1 - start_num)
        input_two=np.around(input_one.astype(np.double), 0)
        output_int1=np.minimum(max_value, input_two) 
        output_int=np.maximum(min_value, output_int1 )
        output_back=output_int/float(pow(2, bitwidth - 1- start_num))
        output_diff=np.dot(output_back - input, output_back - input )
        if output_diff <= min_error:
            min_error=output_diff
            min_count=bitwidth - 1 - start_num        
            output_shift=min_count
        start_num=start_num + 1              
    return output_shift

def weight_bias_quant_mse(input, bitwidth, pct0, pct1, input_signed):
    max_value=0
    min_value=0
    if input_signed:
        max_value=pow(2, bitwidth - 1) - 1
        min_value=-1*pow(2, bitwidth - 1)
    else:
        max_value=pow(2, bitwidth) - 1
        min_value=0
    len_size=len(input.shape)
    if len_size==1:
        output_quant=np.zeros(input.shape)
        output_dequant=np.zeros(input.shape)
        output_shift=np.zeros(input.shape)
        tmp=input.flatten()
        bias_size=tmp.size  
        bias_count=0 
        while bias_count < bias_size:
            value=tmp[bias_count]
            if value==0:
                output_quant[bias_count]=0
                output_dequant[bias_count]=0
            else:
                output_shift[bias_count], output_quant[bias_count], output_dequant[bias_count]=bias_quant_location(value, bitwidth, input_signed)        
                if abs(output_shift[bias_count]) > bitwidth* 2:
                    output_shift[bias_count]=0
                    output_quant[bias_count]=0
                    output_dequant[bias_count]=0
            bias_count=bias_count + 1
        return output_quant, output_dequant, output_shift
    else:
        output_quant=np.zeros(input.shape)
        output_dequant=np.zeros(input.shape)
        output_shift=np.zeros(input.shape[0])
        channels=input.shape[0]
        for c in range(0, channels):
            shape_one=input[c, :, :, :].shape
            input_one=input[c, :, :, :].flatten()
            positive_input_one=np.abs(input_one)            
            all_nums=input_one.size            
            min_nums=int(all_nums * pct0)
            max_nums=int(all_nums * pct1)
            sorted_one=np.sort(positive_input_one)
            if sorted_one[all_nums - 1]==0:
                 output_shift[c]=0
                 output_quant[c, :, :, :]=0
                 output_dequant[c, :, :, :]=0
            else :                
                min_value_ex= math.floor(math.log(sorted_one[min_nums - 1], 2))
                max_value_ex= math.ceil(math.log(sorted_one[max_nums - 1], 2)) + 1
                output_shift[c]=weight_quant_location(input_one, bitwidth, min_value_ex, max_value_ex, input_signed)
                output_int_new=np.maximum(min_value, np.minimum(max_value, np.round(input_one * pow(2, output_shift[c]))))          
                output_quant[c, :, :, :]=output_int_new.reshape(shape_one)
                output_dequant[c, :, :, :]=(output_quant[c, :, :, :]/float(pow(2, output_shift[c])))
                if abs(output_shift[c]) > bitwidth* 2:
                    output_shift[c]=0
                    output_quant[c, :, :, :]=0
                    output_dequant[c, :, :, :]=0
        output_shift=output_shift
        return output_quant, output_dequant, output_shift

def weight_bias_quant_minmax(input, bitwidth, pct0, pct1, input_signed):
    max_value=0
    min_value=0
    if input_signed:
        max_value=pow(2, bitwidth - 1)- 1
        min_value=-1*pow(2, bitwidth - 1)
    else:
        max_value=pow(2, bitwidth)- 1
        min_value=0
    len_size=len(input.shape)
    if len_size==1: 
        output_quant=np.zeros(input.shape)
        output_dequant=np.zeros(input.shape)
        output_shift=np.zeros(input.shape)
        tmp=input.flatten()
        bias_size=tmp.size  
        bias_count=0 
        while bias_count < bias_size:
            value=tmp[bias_count]
            if value==0:
                output_quant[bias_count]=0
                output_dequant[bias_count]=0
            else:
                output_shift[bias_count], output_quant[bias_count], output_dequant[bias_count]=bias_quant_location(value, bitwidth,input_signed)        
                if abs(output_shift[bias_count]) > bitwidth* 2:
                    output_shift[bias_count]=0
                    output_quant[bias_count]=0
                    output_dequant[bias_count]=0
            bias_count=bias_count + 1
        return output_quant, output_dequant, output_shift
    else:
        output_quant=np.zeros(input.shape) 
        output_dequant=np.zeros(input.shape)
        output_shift=np.zeros(input.shape[0])        
        channels=input.shape[0]
        for c in range(0, channels):
            input_one=input[c, :, :, :]
            maxmin=np.max(input_one)
            minmax=np.min(input_one)
            max_val= max(abs(maxmin), abs(minmax))
            if max_val==0:
                 output_shift[c]=0
                 output_quant[c, :, :, :]=0
                 output_dequant[c, :, :, :]=0
            else :
                output_shift[c]=bitwidth- 1- math.ceil(math.log(max_val, 2))
                output_quant[c, :, :, :]=np.maximum(min_value, np.minimum(max_value, np.round(input_one * pow(2, output_shift[c]))))     
                output_dequant[c, :, :, :]=(output_quant[c, :, :, :]/float(pow(2, output_shift[c])))
                if abs(output_shift[c]) > bitwidth * 2:
                    output_shift[c]=0
                    output_quant[c, :, :, :]=0
                    output_dequant[c, :, :, :]=0
        output_shift=output_shift
        return output_quant, output_dequant, output_shift
    
def weight_bias_quant_kl(input, bitwidth, pct0, pct1, input_signed, factor_num_bin):
    max_value=0
    min_value=0
    if input_signed:
        max_value=pow(2, bitwidth - 1)-1
        min_value=-1*pow(2, bitwidth - 1)
    else:
        max_value=pow(2, bitwidth)-1
        min_value=0
    len_size=len(input.shape)
    if len_size==1: 
        output_quant=np.zeros(input.shape)
        output_dequant=np.zeros(input.shape)
        output_shift=np.zeros(input.shape)
        tmp=input.flatten()
        bias_size=tmp.size  
        bias_count=0 
        while bias_count < bias_size:
            value=tmp[bias_count]
            if value==0:
                output_quant[bias_count]=0
                output_dequant[bias_count]=0
            else:
                output_shift[bias_count], output_quant[bias_count], output_dequant[bias_count]=bias_quant_location(value, bitwidth,input_signed)        
                if abs(output_shift[bias_count]) > bitwidth*2:
                    output_shift[bias_count]=0
                    output_quant[bias_count]=0
                    output_dequant[bias_count]=0
            bias_count=bias_count + 1
        return output_quant, output_dequant, output_shift
    else:
        channels=input.shape[0]
        QUANTIZE_NUM=pow(2, bitwidth-1)    
        INTERVAL_NUM=QUANTIZE_NUM* pow(2, factor_num_bin)
        distributions=[]
        for id in range(channels):        
            distribution=[0 for x in range(0, INTERVAL_NUM)]
            distributions.append(distribution)
        output_quant=np.zeros(input.shape) 
        output_dequant=np.zeros(input.shape)
        output_shift=np.zeros(input.shape[0])        
        max_vals= [0 for x in range(0, channels)]
        for c in range(0, channels):
            input_one=input[c, :, :, :]
            maxmin=np.max(input_one)
            minmax=np.min(input_one)
            max_vals[c]= max(abs(maxmin), abs(minmax))
        for c in range(0, channels):
            input_one=input[c, :, :, :]
            add_to_distribution(input_one, distributions[c], max_vals[c]/INTERVAL_NUM)
            normalize_distribution(distributions[c])
            distributions[c] = np.array(distributions[c])
            thre_list= exp_int_threshold(distributions[c], max_vals[c], 0.8, 1.0, INTERVAL_NUM)
            threshold_bin= threshold_kl_distribution(distributions[c], QUANTIZE_NUM, thre_list) 
            threshold = (threshold_bin + 0.5)* max_vals[c]/INTERVAL_NUM
            output_shift[c] = math.floor(bitwidth - 1 - math.log(threshold, 2) )
            output_quant[c, :, :, :]=np.maximum(min_value, np.minimum(max_value, np.round(input_one * pow(2, output_shift[c]))))     
            output_dequant[c, :, :, :]=(output_quant[c, :, :, :]/float(pow(2, output_shift[c])))
            if abs(output_shift[c]) > bitwidth * 2:
                output_shift[c]=0
                output_quant[c, :, :, :]=0
                output_dequant[c, :, :, :]=0
        output_shift=output_shift 
        return output_quant, output_dequant, output_shift
            
def prelu_weight_quant(input, bitwidth, input_signed): 
    max_value=0
    min_value=0
    if input_signed:
        max_value=pow(2, bitwidth - 1)-1
        min_value=-1*pow(2, bitwidth - 1)
    else:
        max_value=pow(2, bitwidth)-1
        min_value=0
    output_quant=np.zeros(input.shape)
    output_dequant=np.zeros(input.shape)
    output_shift=np.zeros(input.shape)
    tmp=input.flatten()
    bias_size=tmp.size  
    bias_count=0  
    while bias_count < bias_size:
        input_one=input[bias_count]
        output_one=input_one*pow(2, 6)
        output_int=max(min_value,min(max_value,np.round(output_one)) )
        output_back=output_int/float(pow(2, 6))         
        output_shift[bias_count]=6
        output_quant[bias_count]=output_int
        output_dequant[bias_count]=output_back      
        bias_count=bias_count + 1
    return output_quant, output_dequant, output_shift  

    
def quantize_back_weight(input, shift, bitwidth, input_signed):
    max_value=0
    min_value=0
    if input_signed:
        max_value=pow(2, bitwidth- 1)- 1
        min_value=-1*pow(2, bitwidth- 1)
    else:
        max_value=pow(2, bitwidth)- 1
        min_value=0
    len_size=len(input.shape)
    if len_size==1: 
        shifts= np.zeros(input.shape)
        if type(shift)==int or len(shift)==1:
            shifts[shifts < input.shape[0]] = shift
        else:
            shifts = shift
            assert len(shift)== input.shape[0], 'bias_frac size should be the same with bias size, but got {}'.format(len(shift))
        output_quant= np.zeros(input.shape)
        output_dequant= np.zeros(input.shape)
        for i in range(input.shape[0]):
            output_quant[i]= np.maximum(min_value, np.minimum(max_value, np.floor(input[i]* pow(2, shifts[i])) ) )
            output_dequant[i]=output_quant[i]/float(pow(2, shifts[i]))
        return output_quant, output_dequant, shifts
    else:
        shifts=np.zeros(input.shape[0])        
        if type(shift)==int or len(shift)==1:
            shifts[shifts < input.shape[0]] = shift
        else:
            shifts = shift
            assert len(shift)==input.shape[0], 'weight_frac size should be the same with weight Cin size, but got {}'.format(len(shift))    
        output_quant=np.zeros(input.shape)
        output_dequant=np.zeros(input.shape)
        for i in range(input.shape[0]):
            output_quant[i, :, :, :]=np.maximum(min_value, np.minimum(max_value, np.floor(input[i, :, :, :]*pow(2, shifts[i])) ) )
            output_dequant[i, :, :, :]=output_quant[i, :, :, :]/float(pow(2, shifts[i]))
        return output_quant, output_dequant, shifts 
    

def quantize_weight_bias(merged_layer_list, merged_param_dict, bitwidth, factor_num_bin=0, method_dict=None, usr_param_dict=None):
    if method_dict== None:
        method_dict = {}
    assert isinstance(method_dict, dict), 'method_dict must be a dict, but get a {}'.format(type(method_dict))

    if usr_param_dict== None:
        usr_param_dict = {}
    assert isinstance(usr_param_dict, dict), 'usr_param_dict must be a dict, but get a {}'.format(type(usr_param_dict))
    quant_params= {}
    quant_params_float= {}
    for lay in  merged_layer_list:
        now_operation= lay.name
        now_type= lay.type
        if now_type in [ 'Convolution','BatchNorm' , 'Scale','InnerProduct']:
            Bias =False
            weight_params= merged_param_dict[str(now_operation)+'_weight']
            if str(now_operation)+'_bias' in merged_param_dict:
                Bias =True
                bias_params= merged_param_dict[str(now_operation)+'_bias']
            str_weight_frac= str(now_operation)+'_frac_weight'
            str_bias_frac= str(now_operation)+'_frac_bias'
            params_shift1=0
            params_dequant1=0
            params_quant1=0
            params_shift0=0
            params_dequant0=0
            params_quant0=0
            if usr_param_dict.get(str(str_weight_frac)) and usr_param_dict.get(str(str_bias_frac)):
                params_quant0, params_dequant0, params_shift0= quantize_back_weight(weight_params, usr_param_dict[str(str_weight_frac)], bitwidth, 1)
                if Bias:
                    params_quant1, params_dequant1, params_shift1= quantize_back_weight(bias_params, usr_param_dict[str(str_bias_frac)], bitwidth, 1)
            elif usr_param_dict.get(str(str_weight_frac)):
                if method_dict.get(str(now_operation)):
                    quant_type= method_dict[now_operation].upper()
                    if quant_type== 'KL':
                        params_quant0, params_dequant0, params_shift0= quantize_back_weight(weight_params, usr_param_dict[str(str_weight_frac)], bitwidth, 1)
                        if Bias:
                            params_quant1, params_dequant1, params_shift1= weight_bias_quant_kl(bias_params, bitwidth, 0.8, 1.0, 1, factor_num_bin)
                    elif quant_type== 'MINMAX':
                        params_quant0, params_dequant0, params_shift0= quantize_back_weight(weight_params, usr_param_dict[str(str_weight_frac)], bitwidth, 1)
                        if Bias:
                            params_quant1, params_dequant1, params_shift1= weight_bias_quant_minmax(bias_params, bitwidth, 0.8, 1.0, 1)
                    elif quant_type== 'MSE':
                        params_quant0, params_dequant0, params_shift0= quantize_back_weight(weight_params, usr_param_dict[str(str_weight_frac)], bitwidth, 1)
                        if Bias:
                            params_quant1, params_dequant1, params_shift1= weight_bias_quant_mse(bias_params, bitwidth, 0.8, 1.0, 1)
                else:
                    params_quant0, params_dequant0, params_shift0= quantize_back_weight(weight_params, usr_param_dict[str(str_weight_frac)], bitwidth, 1)
                    if Bias:
                        params_quant1, params_dequant1, params_shift1= weight_bias_quant_minmax(bias_params, bitwidth, 0.8, 1.0, 1)
            elif usr_param_dict.get(str(str_bias_frac)):
                if method_dict.get(str(now_operation)):
                    quant_type= method_dict[now_operation].upper()
                    if quant_type== 'KL':
                        params_quant0, params_dequant0, params_shift0= weight_bias_quant_kl(weight_params, bitwidth, 0.8, 1.0, 1, factor_num_bin)
                        if Bias:
                            params_quant1, params_dequant1, params_shift1= quantize_back_weight(bias_params, usr_param_dict[str(str_bias_frac)], bitwidth, 1)         
                    elif quant_type== 'MINMAX':
                        params_quant0, params_dequant0, params_shift0= weight_bias_quant_minmax(weight_params, bitwidth, 0.8, 1.0, 1)
                        if Bias:
                            params_quant1, params_dequant1, params_shift1= quantize_back_weight(bias_params, usr_param_dict[str(str_bias_frac)], bitwidth, 1)
                    elif quant_type== 'MSE':
                        params_quant0, params_dequant0, params_shift0= weight_bias_quant_mse(weight_params, bitwidth, 0.8, 1.0, 1)
                        if Bias:
                            params_quant1, params_dequant1, params_shift1= quantize_back_weight(bias_params, usr_param_dict[str(str_bias_frac)], bitwidth, 1)              
                else:
                    params_quant0, params_dequant0, params_shift0= weight_bias_quant_minmax(weight_params, bitwidth, 0.8, 1.0, 1)
                    if Bias:
                        params_quant1, params_dequant1, params_shift1= quantize_back_weight(bias_params, usr_param_dict[str(str_bias_frac)], bitwidth, 1)
            else:
                if method_dict.get(str(now_operation)):
                    quant_type= method_dict[now_operation].upper()
                    if quant_type== 'KL':
                        params_quant0, params_dequant0, params_shift0= weight_bias_quant_kl(weight_params, bitwidth, 0.8, 1.0, 1, factor_num_bin)
                        if Bias:
                            params_quant1, params_dequant1, params_shift1= weight_bias_quant_kl(bias_params, bitwidth, 0.8, 1.0, 1, factor_num_bin)
                    elif quant_type== 'MINMAX':
                        params_quant0, params_dequant0, params_shift0= weight_bias_quant_minmax(weight_params, bitwidth, 0.8, 1.0, 1)
                        if Bias:
                            params_quant1, params_dequant1, params_shift1= weight_bias_quant_minmax(bias_params, bitwidth, 0.8, 1.0, 1)   
                    elif quant_type== 'MSE':
                        params_quant0, params_dequant0, params_shift0= weight_bias_quant_mse(weight_params, bitwidth, 0.8, 1.0, 1)
                        if Bias:
                            params_quant1, params_dequant1, params_shift1= weight_bias_quant_mse(bias_params, bitwidth, 0.8, 1.0, 1)
                else:
                    params_quant0, params_dequant0, params_shift0= weight_bias_quant_minmax(weight_params, bitwidth, 0.8, 1.0, 1)
                    if Bias:
                        params_quant1, params_dequant1, params_shift1= weight_bias_quant_minmax(bias_params, bitwidth, 0.8, 1.0, 1)
            if Bias:
                quant_params[str(now_operation)+'_frac_bias']= params_shift1
                quant_params[str(now_operation)+'_quant_bias']= params_quant1
                quant_params_float[str(now_operation)+'_bias']= params_dequant1
            quant_params[str(now_operation)+'_quant_weight']= params_quant0
            quant_params_float[str(now_operation)+'_weight']= params_dequant0            
            quant_params[str(now_operation)+'_frac_weight']= params_shift0
        elif now_type=='Bias':
            bias_params= merged_param_dict[str(now_operation)+'_bias']
            str_bias_frac= str(now_operation)+'_frac_bias'
            if usr_param_dict.get(str(str_bias_frac)):
                params_quant1, params_dequant1, params_shift1= quantize_back_weight(bias_params, usr_param_dict[str(str_bias_frac)], bitwidth, 1)
            else:
                if method_dict.get(str(now_operation)):
                    quant_type= method_dict[now_operation].upper()
                    if quant_type== 'KL':
                        params_quant1, params_dequant1, params_shift1= weight_bias_quant_kl(bias_params, bitwidth, 0.8, 1.0, 1, factor_num_bin)
                    elif quant_type== 'MINMAX':
                        params_quant1, params_dequant1, params_shift1= weight_bias_quant_minmax(bias_params, bitwidth, 0.8, 1.0, 1)
                    elif quant_type== 'MSE':
                        params_quant1, params_dequant1, params_shift1= weight_bias_quant_mse(bias_params, bitwidth, 0.8, 1.0, 1)
                else:
                    params_quant1, params_dequant1, params_shift1= weight_bias_quant_minmax(bias_params, bitwidth, 0.8, 1.0, 1)
            quant_params[str(now_operation)+'_frac_bias']= params_shift1
            quant_params_float[str(now_operation)+'_bias']= params_dequant1
            quant_params[str(now_operation)+'_quant_bias']= params_quant1                    
        elif now_type== 'ReLU':
            negative_slope= lay.negative_slope            
            negative_slope_new= quantized_value(val= negative_slope, bitwidth= 8, frac= 6)
            assert negative_slope_new== negative_slope, 'negative_slope should be 6 frac, but got {}'.format(negative_slope)
            if negative_slope_new!= negative_slope:
                lay.negative_slope= negative_slope_new
    return merged_layer_list, quant_params, quant_params_float


if __name__=='__main__':
    print('...')