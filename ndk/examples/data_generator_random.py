# -*- coding:utf-8 -*- -
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

def data_generator_random_input(C, H, W, batch_size=5, min_val=0, max_val=255, one_hot_output_size=None, integer_only=True):
        while True:
            data_in = np.random.rand(batch_size, C, H, W) * (max_val - min_val) + min_val
            if integer_only:
                data_in = np.round(data_in).astype(dtype=np.int)
            if one_hot_output_size:
                data_out = np.zeros((batch_size, one_hot_output_size))
                for b in range(batch_size):
                    data_out[b, np.random.randint(one_hot_output_size)] = 1
                if integer_only:
                    data_out = np.round(data_out).astype(dtype=np.int)
                yield {'input': data_in, 'output': data_out}
            else:
               yield {'input': data_in}


if __name__ == "__main__":
    import types

    # data = 'a'
    #
    # try:
    #     tmp = np.array(data, dtype=np.float64)
    # except Exception:
    #     raise ValueError('not a number')
    #
    # print(tmp)


    C, H, W = 3, 224, 244
    batch_size = 4
    num_batch = 10

    g = data_generator_random_input(C,H,W, batch_size=batch_size, min_val=-1, max_val=1, integer_only=False)

    print('It is a generator? {}'.format(isinstance(g, types.GeneratorType)))

    for batch_idx in range(num_batch):
        batch = next(g)
        assert batch_size==batch['input'].shape[0]
        print('Batch #{}:'.format(batch_idx))
        print('  min={}, max={}'.format(np.min(batch['input']), np.max(batch['input'])))
        print('  avg={}, var={}'.format(np.mean(batch['input']), np.var(batch['input'])))
        print('  shape={}'.format(batch['input'].shape))



