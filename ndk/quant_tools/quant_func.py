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

def float2quant(val, bitwidth, frac, floor=True):
    max_q = 2.0 ** (bitwidth - 1) - 1
    min_q = -2.0 ** (bitwidth - 1)
    # if isinstance(frac, np.ndarray) and len(frac.shape) == 1 and len(frac)==1:
    #     frac = int(frac[0])
    if isinstance(frac, np.ndarray) and len(frac.shape)==1:
        assert val.shape[0]==len(frac), 'mis-matched input channel number({}) and frac length({}).'.format(val.shape[0], len(frac))
        q = val.copy()
        for c in range(len(frac)):
            q[c] = val[c] / np.power(2.0, -frac[c])
    elif isinstance(frac, int):
        q = val / np.power(2.0, -frac)
    else:
        # q = val / np.power(2.0, -frac)
        if isinstance(frac, np.ndarray):
            raise Exception('frac must be a 1-dim numpy.ndarray, or an integer, but got shape {}.'.format(frac.shape))
        raise Exception('frac must be a 1-dim numpy.ndarray, or an integer, but got a {}.'.format(type(frac)))
    q = np.clip(q, min_q, max_q)
    if floor:
        q = np.floor(q)
    else:
        q = np.round(q)
    return q.astype(int)

def quant2float(q, bitwidth, frac, floor=True):
    max_q = 2.0 ** (bitwidth - 1) - 1
    min_q = -2.0 ** (bitwidth - 1)
    q = np.clip(q, min_q, max_q)
    if floor:
        q = np.floor(q)
    else:
        q = np.round(q)
    # if isinstance(frac, np.ndarray) and len(frac.shape) == 1 and len(frac)==1:
    #     frac = int(frac[0])
    if isinstance(frac, np.ndarray) and len(frac.shape)==1:
        assert q.shape[0]==len(frac), 'mis-matched input channel number({}) and frac length({}).'.format(q.shape[0], len(frac))
        val = q.copy()
        for c in range(len(frac)):
            val[c] = q[c] * np.power(2.0, -frac[c])
    elif isinstance(frac, int):
        val = q * np.power(2.0, -frac)
    else:
        # val = q * np.power(2.0, -frac)
        raise Exception('frac must be a 1-dim numpy.ndarry, or an integer, but got a {}.'.format(type(frac)))
    return val

def quantized_value(val, bitwidth, frac, floor=True, signed=True):
    if signed:
        return quant2float(float2quant(val, bitwidth=bitwidth, frac=frac, floor=floor), bitwidth=bitwidth, frac=frac, floor=floor)
    else:
        return quant2float(float2quant(np.clip(val, 0, np.inf), bitwidth=bitwidth+1, frac=frac, floor=floor), bitwidth=bitwidth+1, frac=frac, floor=floor)


