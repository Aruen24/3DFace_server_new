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
import os
import sys
ndk_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(ndk_dir)
import types

import copy
import time
import math
import numpy as np
try:
  import tensorflow as tf
  TF_FOUND = True
except Exception as e:
  TF_FOUND = False
  TF_IMPORT_EXCEPT = e
import ndk.layers as ndk_layers
import ndk.quant_tools
import ndk.quant_tools.numpy_net as qnet
import ndk.quant_tools.float_model as fquant
import ndk.quant_tools.quantize_weight as wquant
import ndk.quant_tools.quant_func as qfun
from ndk.quant_tools.quantize_net import quantize_net, set_quantization_log_path
from ndk.quant_tools import numpy_net
from ndk.utils import hist_func, KLD, JSD, print_log
if TF_FOUND:
  try:
    from ndk.simulated_quant.quant_train import auto_quant
    from ndk.simulated_quant.sim_quant_model import SimQuantModel
    QTRAININGMODEL_IMPORTED = True
  except Exception as e:
    QTRAININGMODEL_IMPORT_EXCEPT = e
    QTRAININGMODEL_IMPORTED = False

def quantize_model_v1(layer_list,
                   param_dict,
                   bitwidth,
                   data_generator,
                   usr_param_dict=None,
                   method_dict=None,
                   aggressive=True,
                   gaussian_approx=False,
                   factor_num_bin=4,
                   num_step_pre=20,
                   num_step=100):
    if method_dict==None:
        method_dict = {}
    assert isinstance(method_dict, dict), 'method_dict must be a dict, but get a {}'.format(type(method_dict))

    if usr_param_dict==None:
        usr_param_dict = {}
    assert isinstance(usr_param_dict, dict), 'param_dict must be a dict, but get a {}'.format(type(usr_param_dict))
    layer_list=ndk_layers.sort_layers(layer_list)
    ndk_layers.check_layers(layer_list, param_dict)
    print_log('quantize_model: quantization requires two steps.')
    tic = time.clock()
    # tensor_name_list = ndk_layers.get_tensor_name_list(layer_list, feature_only=False) #featuremap + weights
    feature_name_list = ndk_layers.get_tensor_name_list(layer_list, feature_only=True)
    layer_list, quant_params, quant_params_float = wquant.quantize_weight_bias(layer_list, param_dict, bitwidth, factor_num_bin, method_dict, usr_param_dict)
  #  lower_tensor_frac, upper_tensor_frac=fquant.set_lower_upper_frac(layer_list, quant_params, bitwidth)
    quant_layer_list=copy.deepcopy(layer_list)
    numpy_layers_quant= numpy_net.build_numpy_layers(layer_list, quant_params_float, quant=False, bitwidth=bitwidth)
    quant_layers = fquant.quant_merge_layers(layer_list)  # quant_group
    QUANTIZE_NUM = pow(2, bitwidth - 1)
    INTERVAL_NUM = QUANTIZE_NUM * pow(2, factor_num_bin)
    #run float model to get each layer's max value and check input signed or not
    print_log('quantize_model: step 1/2 - pre-precessing using {} batches.'.format(num_step_pre))
    max_vals, tensor_size_list, min_input=fquant.get_max_and_tensor_size(data_generator, layer_list, numpy_layers_quant, param_dict, num_step_pre)
    print_log('quantize_model: pre-processing finished. Elapsed time:{:.2f} seconds'.format(time.clock()-tic))
    print_log('quantize_model: step 2/2 - determining quantization parameters using {} batches.'.format(num_step))
    if aggressive:
        print_log('quantize_model: running in aggressive mode...')
        quant_param_dict = {}
        quant_param_dict.update(param_dict)
        quant_param_dict.update(quant_params)
        aggressive_distributions=fquant.get_float_distribution(layer_list, data_generator, num_step, max_vals, numpy_layers_quant, quant_params_float, INTERVAL_NUM, QUANTIZE_NUM)
        quant_featuremap_frac=fquant.aggressive_quant(layer_list, quant_param_dict, min_input, aggressive_distributions, quant_layers, max_vals, bitwidth, feature_name_list,
                                                      INTERVAL_NUM, QUANTIZE_NUM, usr_param_dict, method_dict)
        quant_param_dict.update(quant_featuremap_frac)
    else:
        print_log('quantize_model: running in normal mode...')
        quant_param_dict = {}
        quant_param_dict.update(param_dict)
        quant_param_dict.update(quant_params)
        tensor_mean, tensor_var, quant_featuremap_frac=fquant.general_quant(layer_list, min_input, data_generator, quant_param_dict, quant_params_float, quant_layers, max_vals, bitwidth,
                                                                            num_step, tensor_size_list, feature_name_list, usr_param_dict, method_dict, gaussian_approx,
                                                                             INTERVAL_NUM, QUANTIZE_NUM)
        quant_param_dict.update(quant_featuremap_frac)

    print_log('quantize_model: quantization parameters are determined. Elapsed time: {:.2f} seconds'.format(time.clock() - tic))
    print_log('quantize_model: model quantization completed.')
    return quant_layer_list, quant_param_dict

def quantize_model(layer_list,
                   param_dict,
                   bitwidth,
                   data_generator,
                   usr_param_dict=None,
                   method_dict=None,
                   aggressive=True,
                   gaussian_approx=False,
                   factor_num_bin=4,
                   num_step_pre=20,
                   num_step=100,
                   drop_logsoftmax=True,
                   priority_mode='fwb',
                   compensate_floor_on_bias=True,
                   log_dir=None,
                   version='1.1'):
    if version == '1.0':
        print_log('quantize_model: version 1.0 will be deprecated and moved in future ndk releases.')
        quant_layer_list, quant_param_dict = quantize_model_v1(layer_list=layer_list,
                                                               param_dict=param_dict,
                                                               bitwidth=bitwidth,
                                                               data_generator=data_generator,
                                                               usr_param_dict=usr_param_dict,
                                                               method_dict=method_dict,
                                                               aggressive=aggressive,
                                                               gaussian_approx=gaussian_approx,
                                                               factor_num_bin=factor_num_bin,
                                                               num_step_pre=num_step_pre,
                                                               num_step=num_step)
    else:
        set_quantization_log_path(log_dir)
        quant_layer_list, quant_param_dict = quantize_net( layer_list=layer_list,
                                                           param_dict=param_dict,
                                                           bitwidth=bitwidth,
                                                           data_generator=data_generator,
                                                           usr_param_dict=usr_param_dict,
                                                           method_dict=method_dict,
                                                           aggressive=aggressive,
                                                           gaussian_approx=gaussian_approx,
                                                           factor_num_bin=factor_num_bin,
                                                           num_step_pre=num_step_pre,
                                                           num_step=num_step,
                                                           drop_logsoftmax=drop_logsoftmax,
                                                           priority_mode=priority_mode,
                                                           compensate_floor_on_bias=compensate_floor_on_bias)
    return quant_layer_list, quant_param_dict

def _are_all_fracs_in_param_dict(param_dict, layer_list):
    for layer in layer_list:
        if layer.type.lower() in ['convolution', 'innerproduct','batchnorm','scale']:
            if layer.name + "_frac_weight" not in param_dict:
                return False
            if layer.name + "_frac_bias" not in param_dict:
                if layer.bias_term:
                    return False
        elif layer.type.lower() in ['bias']:
            if layer.name + "_frac_bias" not in param_dict:
                return False
        else:
            pass

        if layer.type.lower() == 'slice':
            for top in layer.top:
                if top + "_frac" not in param_dict:
                    return False
        else:
            if layer.top + '_frac'  not in param_dict:
                return False
    return True

def get_data_generator(input_fn, sess):
    data = input_fn()
    if isinstance(data, tuple):
        features, labels = data
    else:
        dataset = data
        iterator = dataset.make_one_shot_iterator()
        features, labels = iterator.get_next()
    def data_generator():
        data_batch = {}
        while True:
            run_features, run_labels = sess.run([features, labels])
            data_batch['input'] = run_features
            data_batch['output'] = run_labels
            yield data_batch
    return data_generator()

def quantize_model_with_training(layer_list,
                                 bitwidth,
                                 param_dict,
                                 data_generator=None,
                                 input_fn=None,
                                 log_dir='log',
                                 loss_fn=None,
                                 optimizer=None,
                                 num_step_train=1000,
                                 num_step_log=100,
                                 perlayer_quant_layer_names=None,
                                 layer_group=1,
                                 rnd_seed=None):
    if not TF_FOUND:
        raise ImportError(str(TF_IMPORT_EXCEPT))
    if not QTRAININGMODEL_IMPORTED:
        raise ImportError(str(QTRAININGMODEL_IMPORT_EXCEPT))

    if data_generator is None and input_fn is None:
        raise ValueError("The data_generator and input_fn arguments could not be None at the same time.")
    if data_generator is not None and input_fn is not None:
        raise ValueError("The data_generator and input_fn arguments could not be assigned at the same time.")

    if loss_fn == None:
        loss_fn = tf.losses.softmax_cross_entropy
    if optimizer == None:
        optimizer = tf.train.AdamOptimizer()

    if type(rnd_seed) != type(None):
        tf.set_random_seed(rnd_seed)

    if not _are_all_fracs_in_param_dict(layer_list=layer_list, param_dict=param_dict):
        print_log("Starting to determine the fracs. All fracs must be determined before quantization training.")

        if data_generator is not None:
            layer_list, param_dict = quantize_model(layer_list=layer_list,
                                                    param_dict=param_dict,
                                                    bitwidth=bitwidth,
                                                    data_generator=data_generator)
        else:
            sess = tf.Session()
            data_generator_by_input_fn = get_data_generator(input_fn, sess)
            layer_list, param_dict = quantize_model(layer_list=layer_list,
                                                    param_dict=param_dict,
                                                    bitwidth=bitwidth,
                                                    data_generator=data_generator_by_input_fn)
            sess.close()

    model = SimQuantModel(layer_list=layer_list,
                          bitwidth=bitwidth,
                          param_dict=param_dict,
                          trainable_layer_names={layer.name for layer in layer_list},
                          perlayer_quant_layer_names=perlayer_quant_layer_names)

    if isinstance(data_generator, types.GeneratorType) and data_generator is not None and input_fn is None:
        data_maker = data_generator
    elif isinstance(input_fn, types.FunctionType) and input_fn is not None and data_generator is None:
        data_maker = input_fn
    else:
        raise TypeError("Wrong type of data_generator or input_fn.")

    quant_layer_list, quant_param_dict = auto_quant(model=model,
                                                    loss_fn=loss_fn,
                                                    optimizer=optimizer,
                                                    data_maker=data_maker,
                                                    log_dir=log_dir,
                                                    num_step_train=num_step_train,
                                                    layer_group=layer_group,
                                                    num_step_log=num_step_log)

    return quant_layer_list, quant_param_dict

def analyze_tensor_distribution(layer_list,
                                param_dict,
                                target_tensor_list,
                                output_dirname=None,
                                data_generator=None,
                                quant=True,
                                bitwidth=8,
                                num_batch=1000,
                                hw_aligned=True,
                                num_bin=0,
                                max_abs_val=None,
                                log_on=False):
    stats_dict = {}

    # analysis the tensor names
    tensor_name_list = ndk_layers.get_tensor_name_list(layer_list, feature_only=False)
    feature_name_list = ndk_layers.get_tensor_name_list(layer_list, feature_only=True)
    for tensor_name in target_tensor_list:
        assert tensor_name in tensor_name_list, 'Target tensor {} not found in the layer_list.'.format(tensor_name)

    target_feature_list = []
    # process the weights and biases listed in target_tensor_list
    for tensor_name in target_tensor_list:
        if tensor_name in feature_name_list:
            target_feature_list.append(tensor_name)
            data = None
        else:
            data = qnet.get_layer_weight_bias(tensor_name, param_dict=param_dict, quant=quant, bitwidth=bitwidth)
        if type(data) != type(None):
            data_stats_dict = {}
            data_stats_dict[tensor_name+'_min'] = np.min(data)
            data_stats_dict[tensor_name+'_max'] = np.max(data)
            data_stats_dict[tensor_name+'_avg'] = np.mean(data)
            data_stats_dict[tensor_name+'_var'] = np.var(data)
            if num_bin > 0:
                if  data_stats_dict[tensor_name+'_min'] >= 0:
                    if type(max_abs_val) == type(None):
                        cur_max_abs_val = data_stats_dict[tensor_name + '_max']
                    else:
                        cur_max_abs_val = max_abs_val
                    cur_max_abs_val = max(cur_max_abs_val, 1e-3)
                    bin_vals = np.linspace(0, cur_max_abs_val, num_bin)
                else:
                    if type(max_abs_val) == type(None):
                        cur_max_abs_val = np.max([-data_stats_dict[tensor_name+'_min'], data_stats_dict[tensor_name+'_max']])
                    else:
                        cur_max_abs_val = max_abs_val
                    cur_max_abs_val = max(cur_max_abs_val, 1e-3)
                    bin_vals = np.linspace(-cur_max_abs_val,cur_max_abs_val,num_bin)
                data_hist = hist_func(data, bin_vals)
                data_stats_dict[tensor_name + '_pd'] = np.array([bin_vals, data_hist/sum(data_hist)])
            stats_dict.update(data_stats_dict) # update the result dict

    # process the features listed in target_tensor_list
    if len(target_feature_list) > 0:
        # build the numpy network
        sorted_layer_list = ndk_layers.sort_layers(layer_list)
        np_layers = qnet.build_numpy_layers(layer_list=sorted_layer_list, param_dict=param_dict, quant=quant, bitwidth=bitwidth)

        # create the bins if need to count histogram
        bin_dict = {}
        if num_bin > 0:
            max_abs_vals_dict = {}
            min_val_dict = {}
            if type(max_abs_val)==type(None):
                if log_on:
                    print_log('analyze_tensor_distribution: determining bins...')
                # run at least 1000 samples to get the maximum absolute value which will be used to determine the bin values
                for tensor_name in target_feature_list:
                    max_abs_vals_dict[tensor_name] = 0
                    min_val_dict[tensor_name] = np.inf
                num_samples = 0
                while num_samples < 1000:
                    data_batch = next(data_generator)
                    n,_,_,_ = data_batch['input'].shape
                    num_samples += n
                    result_dict = qnet.run_layers(layer_list=sorted_layer_list, target_feature_tensor_list=target_feature_list,
                                              input_data_batch=data_batch['input'],
                                              numpy_layers=np_layers, param_dict=None, bitwidth=bitwidth,
                                              quant=quant, hw_aligned=hw_aligned, log_on=False)

                    for tensor_name in target_feature_list:
                        cur_max_abs_val = np.max(np.abs(result_dict[tensor_name]))
                        max_abs_vals_dict[tensor_name] = np.max([max_abs_vals_dict[tensor_name], cur_max_abs_val])
                        min_val_dict[tensor_name] = np.min([ min_val_dict[tensor_name], np.min(result_dict[tensor_name])])

                if log_on:
                    print_log('analyze_tensor_distribution: bins determined using {} samples'.format(num_samples))
            else:
                assert max_abs_val > 0, 'max_abs_val must larger than zero'
                for tensor_name in target_feature_list:
                    max_abs_vals_dict[tensor_name] = max_abs_val
                    min_val_dict[tensor_name] = -np.inf

            # create the bins
            for tensor_name in target_feature_list:
                cur_max_abs_val = max(1e-3, max_abs_vals_dict[tensor_name])
                if min_val_dict[tensor_name] >= 0:
                    bin_dict[tensor_name] = np.linspace(0, cur_max_abs_val, num_bin)
                else:
                    bin_dict[tensor_name] = np.linspace(-cur_max_abs_val, cur_max_abs_val, num_bin)

        # process the target features
        cnt_dict = {}
        feat_stats_dict = {}
        for tensor_name in target_feature_list:
            cnt_dict[tensor_name] = np.zeros(num_bin, dtype=int)
            feat_stats_dict[tensor_name+'_min'] = np.inf
            feat_stats_dict[tensor_name+'_max'] = -np.inf
            feat_stats_dict[tensor_name+'_sum'] = 0
            feat_stats_dict[tensor_name+'_sqsum'] = 0
        batch_size = 1
        for bidx in range(num_batch):
            if log_on:
                print_log('analyze_tensor_distribution: processing bacth #{}/{}'.format(bidx, num_batch))
            data_batch = next(data_generator)
            batch_size,_,_,_=data_batch['input'].shape
            result_dict = qnet.run_layers(layer_list=sorted_layer_list, target_feature_tensor_list=target_feature_list,
                                          input_data_batch=data_batch['input'],
                                          numpy_layers=np_layers, param_dict=None, bitwidth=bitwidth,
                                          quant=quant, hw_aligned=hw_aligned, log_on=False)

            for tensor_name in target_feature_list:
                feat_stats_dict[tensor_name+'_min'] = np.min([feat_stats_dict[tensor_name+'_min'], np.min(result_dict[tensor_name])])
                feat_stats_dict[tensor_name+'_max'] = np.max([feat_stats_dict[tensor_name+'_max'], np.max(result_dict[tensor_name])])
                feat_stats_dict[tensor_name+'_sum'] += np.mean(result_dict[tensor_name])
                feat_stats_dict[tensor_name+'_sqsum'] += np.mean(np.square(result_dict[tensor_name]))
                if num_bin > 0:
                    cnt_dict[tensor_name] += hist_func(result_dict[tensor_name], bin_dict[tensor_name])

        # update the stats_dict
        num_sample = batch_size * num_batch
        for tensor_name in target_feature_list:
            stats_dict[tensor_name+'_min'] = feat_stats_dict[tensor_name+'_min']
            stats_dict[tensor_name+'_max'] = feat_stats_dict[tensor_name+'_max']
            stats_dict[tensor_name+'_avg'] = feat_stats_dict[tensor_name+'_sum']/num_batch
            stats_dict[tensor_name+'_var'] = feat_stats_dict[tensor_name+'_sqsum']/num_batch + stats_dict[tensor_name+'_avg']**2
            if num_bin > 0:
                stats_dict[tensor_name+'_pd'] = np.array([bin_dict[tensor_name], cnt_dict[tensor_name]/sum(cnt_dict[tensor_name])])

    # save to files
    if isinstance(output_dirname, str):

        import pandas as pd

        if output_dirname=='':
            output_dirname = '.'
        if not os.path.exists(output_dirname):
            os.makedirs(output_dirname)
        # tensor_name_list = []
        item_name_list = []
        for key in stats_dict.keys():
            item_name_list.append(key[::-1].split('_',1)[0][::-1])
            # tensor_name_list.append(key[::-1].split('_',1)[1][::-1])
        # tensor_name_list = sorted(list(set(tensor_name_list)))
        item_name_list = sorted(list(set(item_name_list)))

        # save statistics
        data = {}
        row_name = []
        for item_name in item_name_list:
            if item_name=='pd':
                continue
            row_name.append(item_name)
        for tensor_name in target_tensor_list:
            tensor_data = []
            for item_name in item_name_list:
                if item_name=='pd':
                    continue
                tensor_data.append(stats_dict[tensor_name+'_'+item_name])
            data[tensor_name] = tensor_data

        frame = pd.DataFrame(data=data, index=row_name, columns=target_tensor_list)
        frame.to_csv(os.path.join(output_dirname, 'stats.csv'), sep=',')
        print_log('analyze_tensor_distribution: statistic results saved to {}'.format(os.path.join(output_dirname, 'stats.csv')))

        # save probability distributions
        if num_bin > 0:
            data = {}
            col_name = []
            for tensor_name in target_tensor_list:
                for item_name in item_name_list:
                    if item_name=='pd':
                        col_name.append(tensor_name+'_'+'val')
                        tensor_data = stats_dict[tensor_name+'_'+item_name][0]
                        data[tensor_name+'_'+'val'] = tensor_data
                        col_name.append(tensor_name+'_'+'prob')
                        tensor_data = stats_dict[tensor_name+'_'+item_name][1]
                        data[tensor_name+'_'+'prob'] = tensor_data

            frame = pd.DataFrame(data=data, columns=col_name)
            frame.to_csv(os.path.join(output_dirname, 'prob.csv'), sep=',', index=False)
            print_log('analyze_tensor_distribution: probability distributions saved to {}'.format(os.path.join(output_dirname, 'prob.csv')))

    return stats_dict

def compare_float_and_quant(layer_list,
                            param_dict,
                            target_tensor_list,
                            bitwidth,
                            output_dirname=None,
                            data_generator=None,
                            num_batch=1000,
                            hw_aligned=True,
                            num_bin=0,
                            max_abs_val=5,
                            log_on=False):
    use_ai_framework = True
    ouput_name_list = [
        ['fmsv', 'Flt-Pnt Mean Sqr Val'],
        ['qmsv', 'Fix-Pnt Mean Sqr Val'],
        ['mae', 'Mean Abs Error'],
        ['maxae', 'Max Abs Error'],
        ['mse', 'Mean Sqr Error'],
        ['qsnr', 'Quant SNR (dB)'],
        ['top1err', 'Top1 Error Rate'],
        ['kld', 'KL Divergence'],
        ['jsd', 'JS Divergence'],
        ['nsamp', 'Sample Num'],
    ]
    output_suffix_list = [x[0] for x in ouput_name_list]
    output_name_list = [x[1] for x in ouput_name_list]

    # analyze and check the tensor names
    tensor_name_list = ndk_layers.get_tensor_name_list(layer_list, feature_only=False)
    feature_name_list = ndk_layers.get_tensor_name_list(layer_list, feature_only=True)
    for tensor_name in target_tensor_list:
        assert tensor_name in tensor_name_list, 'Target tensor {} not found in the layer_list.'.format(tensor_name)

    # create the bins for counting probability distributions
    # if num_bin > 0 and max_abs_val > 0:
    #     bin_vals = np.linspace(-max_abs_val, max_abs_val, num_bin)

    stats_dict = {} # dict that records the statistic properties of the target tensors.

    target_feature_list = []
    # process the weights and biases listed in target_tensor_list
    for tensor_name in target_tensor_list:
        if tensor_name in feature_name_list:
            # if feature tensor, leave it to later procedure
            target_feature_list.append(tensor_name)
        else:
            f_data = qnet.get_layer_weight_bias(tensor_name, param_dict=param_dict, quant=False)
            q_data = qnet.get_layer_weight_bias(tensor_name, param_dict=param_dict, quant=True, bitwidth=bitwidth)
            data_stats_dict = {}
            data_stats_dict[tensor_name+'_favg'] = np.mean(f_data) # float-point average
            data_stats_dict[tensor_name+'_fvar'] = np.var(f_data) # float-point variance
            data_stats_dict[tensor_name+'_fsqavg'] = np.mean(np.square(f_data)) # float-point average squared values
            data_stats_dict[tensor_name+'_qavg'] = np.mean(q_data) # fix-point average
            data_stats_dict[tensor_name+'_qvar'] = np.var(q_data) # fix-point variance
            data_stats_dict[tensor_name+'_qsqavg'] = np.mean(np.square(q_data)) # fix-point average squared values
            data_stats_dict[tensor_name+'_mse'] = np.mean(np.square(q_data-f_data)) # mean squared error
            data_stats_dict[tensor_name+'_maxabserr'] = np.max(np.abs(q_data-f_data)) # max error of absolute values
            data_stats_dict[tensor_name+'_avgabserr'] = np.mean(np.abs(q_data-f_data)) # average error of absolute values
            data_stats_dict[tensor_name+'_top1err'] = 1 - np.mean(np.equal(np.argmax(q_data), np.argmax(f_data))) # error rate of max value's index

            if num_bin > 0:
                if  np.min(f_data) >= 0:
                    if type(max_abs_val) == type(None):
                        cur_max_abs_val = np.max(f_data)
                    else:
                        cur_max_abs_val = max_abs_val
                    cur_max_abs_val = max(cur_max_abs_val, 1e-3)
                    data_stats_dict[tensor_name+'_bin'] = np.linspace(0, cur_max_abs_val, num_bin)
                else:
                    if type(max_abs_val) == type(None):
                        cur_max_abs_val = np.max(np.abs(f_data))
                    else:
                        cur_max_abs_val = max_abs_val
                    cur_max_abs_val = max(cur_max_abs_val, 1e-3)
                    data_stats_dict[tensor_name + '_bin'] = np.linspace(-cur_max_abs_val,cur_max_abs_val,num_bin)
                f_data_hist = hist_func(f_data, data_stats_dict[tensor_name + '_bin'])
                q_data_hist = hist_func(q_data, data_stats_dict[tensor_name + '_bin'])
                data_stats_dict[tensor_name + '_fpd'] = f_data_hist/np.sum(f_data_hist) # float-point prob distribution
                data_stats_dict[tensor_name + '_qpd'] = q_data_hist/np.sum(q_data_hist) # fix-point prob distribution
                data_stats_dict[tensor_name + '_nsamp'] = np.sum(f_data_hist) # number of samples

            stats_dict.update(data_stats_dict) # update the result dict

    # process the feature tensors listed in target_tensor_list
    if len(target_feature_list) > 0:
        # process the target features
        f_cnt_dict = {}
        q_cnt_dict = {}
        feat_stats_dict = {}

        # build the numpy network
        sorted_layer_list = ndk_layers.sort_layers(layer_list)
        f_layers = qnet.build_numpy_layers(layer_list=sorted_layer_list, param_dict=param_dict, quant=False)
        q_layers = qnet.build_numpy_layers(layer_list=sorted_layer_list, param_dict=param_dict, quant=True, bitwidth=bitwidth)

        # create the bins for counting probability distributions
        if num_bin > 0:
            max_abs_vals_dict = {}
            min_val_dict = {}
            if type(max_abs_val)==type(None):
                # run at least 1000 samples to get the maximum absolute value which will be used to determine the bin values
                for tensor_name in target_feature_list:
                    max_abs_vals_dict[tensor_name] = 0
                    min_val_dict[tensor_name] = -np.inf
                num_samples = 0
                while num_samples < 1000:
                    data_batch = next(data_generator)
                    n,_,_,_ = data_batch['input'].shape
                    num_samples += n
                    fres_dict = qnet.run_layers(layer_list=sorted_layer_list, target_feature_tensor_list=target_feature_list,
                                                  input_data_batch=data_batch['input'],
                                                  numpy_layers=f_layers, param_dict=None,
                                                  quant=False, hw_aligned=hw_aligned, log_on=False, use_ai_framework=use_ai_framework)
                    for tensor_name in target_feature_list:
                        cur_max_abs_val = np.max(np.abs(fres_dict[tensor_name]))
                        max_abs_vals_dict[tensor_name] = np.max([max_abs_vals_dict[tensor_name], cur_max_abs_val])
                        min_val_dict[tensor_name] = np.min([ min_val_dict[tensor_name], np.min(fres_dict[tensor_name])])

                if log_on:
                    print_log('compare_float_and_quant: bins determined using {} samples'.format(num_samples))
            else:
                assert max_abs_val > 0, 'max_abs_val must larger than zero'
                for tensor_name in target_feature_list:
                    max_abs_vals_dict[tensor_name] = max(max_abs_val, 1e-3)
                    min_val_dict[tensor_name] = -np.inf

            # create the bins
            for tensor_name in target_feature_list:
                cur_max_abs_val = max(1e-3, max_abs_vals_dict[tensor_name])
                if min_val_dict[tensor_name] >= 0:
                    feat_stats_dict[tensor_name+'_bin'] = np.linspace(0, cur_max_abs_val, num_bin)
                else:
                    feat_stats_dict[tensor_name+'_bin'] = np.linspace(-cur_max_abs_val, cur_max_abs_val, num_bin)

        for tensor_name in target_feature_list:
            f_cnt_dict[tensor_name] = np.zeros(num_bin, dtype=int)
            q_cnt_dict[tensor_name] = np.zeros(num_bin, dtype=int)
            feat_stats_dict[tensor_name+'_favg_bsum'] = 0  # float-point average, sum by batches
            feat_stats_dict[tensor_name+'_fsqavg_bsum'] = 0 # float-point average squared values, sum by batches
            feat_stats_dict[tensor_name+'_qavg_bsum'] = 0 # fix-point average, sum by batches
            feat_stats_dict[tensor_name+'_qsqavg_bsum'] = 0 # fix-point average squared values, sum by batches
            feat_stats_dict[tensor_name+'_mse_bsum'] = 0 #  mean squared error, sum by batches
            feat_stats_dict[tensor_name + '_maxabserr'] = 0 # max error of absolute values
            feat_stats_dict[tensor_name + '_avgabserr_bsum'] = 0 # average error of absolute values, sum by batches
            feat_stats_dict[tensor_name + '_top1aligned_cnt'] = 0 # counting of aligned max value's index
            feat_stats_dict[tensor_name + '_nsamp'] = int(0)     # number of samples in bin

        batch_size = 1
        for bidx in range(num_batch):
            if log_on:
                print_log('compare_float_and_quant: processing bacth #{}/{}'.format(bidx, num_batch))
            data_batch = next(data_generator)
            batch_size,_,_,_=data_batch['input'].shape
            fres_dict = qnet.run_layers(layer_list=sorted_layer_list, target_feature_tensor_list=target_feature_list,
                                          input_data_batch=data_batch['input'],
                                          numpy_layers=f_layers, param_dict=None,
                                          quant=False, hw_aligned=hw_aligned, log_on=False, use_ai_framework=use_ai_framework)

            qres_dict = qnet.run_layers(layer_list=sorted_layer_list, target_feature_tensor_list=target_feature_list,
                                          input_data_batch=data_batch['input'],
                                          numpy_layers=q_layers, param_dict=None, bitwidth=bitwidth,
                                          quant=True, hw_aligned=hw_aligned, log_on=False, use_ai_framework=use_ai_framework)

            for tensor_name in target_feature_list:
                feat_stats_dict[tensor_name+'_favg_bsum'] += np.mean(fres_dict[tensor_name])
                feat_stats_dict[tensor_name+'_fsqavg_bsum'] += np.mean(np.square(fres_dict[tensor_name]))
                feat_stats_dict[tensor_name+'_qavg_bsum'] += np.mean(qres_dict[tensor_name])
                feat_stats_dict[tensor_name+'_qsqavg_bsum'] += np.mean(np.square(qres_dict[tensor_name]))
                feat_stats_dict[tensor_name+'_mse_bsum'] += np.mean(np.square(qres_dict[tensor_name]-fres_dict[tensor_name]))
                feat_stats_dict[tensor_name+'_maxabserr'] = np.max([feat_stats_dict[tensor_name+'_maxabserr'], np.max(np.abs(qres_dict[tensor_name]-fres_dict[tensor_name]))])
                feat_stats_dict[tensor_name + '_avgabserr_bsum'] += np.mean(np.abs(qres_dict[tensor_name]-fres_dict[tensor_name]))
                _,tsr_c,tsr_h,tsr_w = fres_dict[tensor_name].shape
                feat_stats_dict[tensor_name + '_top1aligned_cnt'] += np.sum(np.equal(np.argmax(qres_dict[tensor_name], 1), np.argmax(fres_dict[tensor_name], 1))) / tsr_h / tsr_w
                if num_bin > 0:
                    f_cnt_dict[tensor_name] += hist_func(fres_dict[tensor_name], feat_stats_dict[tensor_name+'_bin'])
                    q_cnt_dict[tensor_name] += hist_func(qres_dict[tensor_name], feat_stats_dict[tensor_name+'_bin'])
                    feat_stats_dict[tensor_name + '_nsamp'] += batch_size * tsr_c * tsr_h * tsr_w

        # update the stats_dict
        for tensor_name in target_feature_list:
            stats_dict[tensor_name+'_favg'] = feat_stats_dict[tensor_name+'_favg_bsum']/num_batch
            stats_dict[tensor_name+'_fvar'] = feat_stats_dict[tensor_name+'_fsqavg_bsum']/num_batch + stats_dict[tensor_name+'_favg']**2
            stats_dict[tensor_name+'_fsqavg'] = feat_stats_dict[tensor_name+'_fsqavg_bsum']/num_batch
            stats_dict[tensor_name+'_qavg'] = feat_stats_dict[tensor_name+'_qavg_bsum']/num_batch
            stats_dict[tensor_name+'_qvar'] = feat_stats_dict[tensor_name+'_qsqavg_bsum']/num_batch + stats_dict[tensor_name+'_qavg']**2
            stats_dict[tensor_name+'_qsqavg'] = feat_stats_dict[tensor_name+'_qsqavg_bsum']/num_batch
            stats_dict[tensor_name+'_mse'] = feat_stats_dict[tensor_name+'_mse_bsum']/num_batch
            stats_dict[tensor_name+'_maxabserr'] = feat_stats_dict[tensor_name+'_maxabserr']
            stats_dict[tensor_name+'_avgabserr'] = feat_stats_dict[tensor_name+'_avgabserr_bsum']/num_batch
            stats_dict[tensor_name + '_top1err'] = 1 - feat_stats_dict[tensor_name + '_top1aligned_cnt']/num_batch/batch_size
            if num_bin > 0:
                stats_dict[tensor_name+'_fpd'] = f_cnt_dict[tensor_name]/sum(f_cnt_dict[tensor_name])
                stats_dict[tensor_name+'_qpd'] = q_cnt_dict[tensor_name]/sum(q_cnt_dict[tensor_name])
                stats_dict[tensor_name + '_nsamp'] = feat_stats_dict[tensor_name + '_nsamp']


    # summarize the result
    result_dict = {}
    for tensor_name in target_tensor_list:
        result_dict[tensor_name+'_fmsv'] = stats_dict[tensor_name+'_fsqavg']
        result_dict[tensor_name+'_qmsv'] = stats_dict[tensor_name+'_qsqavg']
        result_dict[tensor_name+'_maxae'] = stats_dict[tensor_name+'_maxabserr']
        result_dict[tensor_name+'_mae'] = stats_dict[tensor_name+'_avgabserr']
        result_dict[tensor_name+'_mse'] = stats_dict[tensor_name+'_mse']
        result_dict[tensor_name+'_qsnr'] = 10.0 * np.log10(1e-10+stats_dict[tensor_name+'_fsqavg']/(1e-10+stats_dict[tensor_name+'_mse']))
        result_dict[tensor_name+'_top1err'] = stats_dict[tensor_name+'_top1err']
        if num_bin > 0:
            result_dict[tensor_name+'_kld'] = KLD(stats_dict[tensor_name+'_fpd'], stats_dict[tensor_name+'_qpd'])
            result_dict[tensor_name+'_jsd'] = JSD(stats_dict[tensor_name+'_fpd'], stats_dict[tensor_name+'_qpd'])
            result_dict[tensor_name + '_nsamp'] = stats_dict[tensor_name + '_nsamp']

    # save to files
    if isinstance(output_dirname, str):

        import pandas as pd

        if output_dirname=='':
            output_dirname = '.'
        if not os.path.exists(output_dirname):
            os.makedirs(output_dirname)
        # tensor_name_list = []
        item_name_list = []
        for key in result_dict.keys():
            item_name_list.append(key[::-1].split('_',1)[0][::-1])
            # tensor_name_list.append(key[::-1].split('_',1)[1][::-1])
        # tensor_name_list = sorted(list(set(tensor_name_list)))
        item_name_list = sorted(list(set(item_name_list)))

        # save results
        data = {}
        row_name = []
        for i in range(len(output_suffix_list)):
            item_suffix = output_suffix_list[i]
            if item_suffix in item_name_list:
                row_name.append(output_name_list[i])

        for tensor_name in target_tensor_list:
            tensor_data = []
            for item_suffix in output_suffix_list:
                if item_suffix in item_name_list:
                    tensor_data.append(result_dict[tensor_name+'_'+item_suffix])
                data[tensor_name] = tensor_data

        frame = pd.DataFrame(data=data, index=row_name, columns=target_tensor_list)
        frame.to_csv(os.path.join(output_dirname, 'comp_results.csv'), sep=',')

        print_log('compare_float_and_quant: results saved to {}'.format(os.path.join(output_dirname, 'comp_results.csv')))

    return result_dict


if __name__=='__main__':    
    print('Not implemented')
