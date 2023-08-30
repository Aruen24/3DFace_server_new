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
import os

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def data_generator_cifar10(cifar10_dirname, batch_size=5, random_order=True, use_test_set=False):
    data = []
    if use_test_set:
        data.append(unpickle(os.path.join(cifar10_dirname, r'test_batch')))
    else:
        data.append(unpickle(os.path.join(cifar10_dirname, r'data_batch_1')))
        data.append(unpickle(os.path.join(cifar10_dirname, r'data_batch_2')))
        data.append(unpickle(os.path.join(cifar10_dirname, r'data_batch_3')))
        data.append(unpickle(os.path.join(cifar10_dirname, r'data_batch_4')))
        data.append(unpickle(os.path.join(cifar10_dirname, r'data_batch_5')))

    batch_idx = 0
    sample_idx = -1
    while True:
        input_list = []
        output_list = []
        for _ in range(batch_size):
            if random_order:
                batch_idx = np.random.randint(len(data))
                sample_idx =np.random.randint(len(data[batch_idx][b'labels']))
            else:
                sample_idx += 1
                if sample_idx >= len(data[batch_idx][b'labels']):
                    sample_idx = 0
                    batch_idx += 1
                    if batch_idx >= len(data):
                        batch_idx = 0

            # draw the selected sample from the data set
            sample_data = data[batch_idx][b'data'][sample_idx]
            sample_label = data[batch_idx][b'labels'][sample_idx]

            # do any data pre-processing here!

            # e.g. normalize the input RGB values to 0-1 float
            net_input = np.array(sample_data.reshape(3,32,32), dtype=np.float64) / 256.0 # CHW format
            # e.g. translate label to one-hot vector as output
            net_output = np.zeros((10,1,1))
            net_output[sample_label, 0, 0] = 1.0

            input_list.append(net_input)
            output_list.append(net_output)
        yield {'input': np.array(input_list), 'output': np.array(output_list)}


if __name__=='__main__':
    import types
    import matplotlib.pyplot as plt

    label_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse',
                  8: 'ship', 9: 'truck'}

    batch_size = 50000
    random_order = False
    use_test_set = False
    num_batch = 1
    label_cnt = np.zeros(10)

    g = data_generator_cifar10(cifar10_dirname='cifar-10-batches-py',
                               batch_size=batch_size, random_order=random_order, use_test_set=use_test_set)

    print('It is a generator? {}'.format(isinstance(g, types.GeneratorType)))

    for batch_idx in range(num_batch):
        batch = next(g)
        assert batch_size==batch['input'].shape[0]
        print('Batch #{}:'.format(batch_idx))
        for i in range(batch_size):
            label_vec = batch['output'][i].reshape(-1)
            label = label_dict[np.nonzero(label_vec)[0][0]]
            label_cnt[np.nonzero(label_vec)[0][0]] += 1
            print('  image[{}]:{}'.format(i, label))
    print(label_cnt)
    # Note that 'input' and 'output' are 4-dim array in NCHW format
    print('Note that \'input\' and \'output\' are 4-dim array in NCHW format!')
    print('Input shape={}, Output shape={}'.format(batch['input'].shape, batch['output'].shape))

    # show the last batch
    # fig = plt.figure()
    # for i in range(batch_size):
    #     plt.subplot(1, batch_size, i + 1)
    #     im = batch['input'][i]
    #     label_vec = batch['output'][i].reshape(-1)
    #     label = label_dict[np.nonzero(label_vec)[0][0]]
    #     plt.title('Label: ' + label)
    #     image = np.zeros((32, 32, 3))
    #     image[:, :, 0] = im[0]
    #     image[:, :, 1] = im[1]
    #     image[:, :, 2] = im[2]
    #     plt.imshow(image.astype(np.float))
    # plt.show()

