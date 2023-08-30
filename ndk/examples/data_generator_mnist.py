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
import struct
import os

def data_generator_mnist(mnist_dirname, batch_size=5, random_order=True, use_test_set=False):
    data = []
    if use_test_set:
        with open(os.path.join(mnist_dirname, 't10k-images.idx3-ubyte'),'rb') as f:
            buf_image = f.read()
        with open(os.path.join(mnist_dirname, 't10k-labels.idx1-ubyte'),'rb') as f:
            buf_label = f.read()
    else:
        with open(os.path.join(mnist_dirname, 'train-images.idx3-ubyte'),'rb') as f:
            buf_image = f.read()
        with open(os.path.join(mnist_dirname, 'train-labels.idx1-ubyte'),'rb') as f:
            buf_label = f.read()

    image_index_offset = struct.calcsize('>IIII')
    label_index_offset = struct.calcsize('>II')
    image_sample_size = struct.calcsize('>784B')
    label_sample_size = struct.calcsize('>B')

    num_samples = int((len(buf_image) - image_index_offset)/image_sample_size)
    num_samples2 = int((len(buf_label) - label_index_offset)/label_sample_size)
    assert num_samples==num_samples2, 'something wrong with the the data set'

    sample_idx = -1
    while True:
        input_list = []
        output_list = []
        for _ in range(batch_size):
            if random_order:
                sample_idx =np.random.randint(num_samples)
            else:
                sample_idx += 1
                if sample_idx >= num_samples:
                    sample_idx = 0

            # draw the selected sample from the data set
            sample_data = np.array(struct.unpack_from('>784B', buf_image, image_index_offset + sample_idx*image_sample_size))
            sample_label = np.array(struct.unpack_from('>B', buf_label, label_index_offset + sample_idx*label_sample_size))

            # do any data pre-processing here!

            # e.g. padding the image to 32x32 and then normalizing to 0-1 float values
            net_input = np.zeros((1,32,32)) # CHW format
            net_input[:, 2:30, 2:30] = np.array(sample_data.reshape((1,28,28)), dtype=np.float64) / 256.0
            # e.g. translate label to one-hot vector as output
            net_output = np.zeros((10,1,1))
            net_output[sample_label, 0, 0] = 1.0

            input_list.append(net_input)
            output_list.append(net_output)
        yield {'input': np.array(input_list), 'output': np.array(output_list)}

if __name__ == "__main__":
    import types
    import matplotlib.pyplot as plt

    label_dict = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9'}

    batch_size = 4
    random_order = True
    use_test_set = False
    num_batch = 10

    g = data_generator_mnist(mnist_dirname='mnist',
                             batch_size=batch_size, random_order=random_order, use_test_set=use_test_set)

    print('It is a generator? {}'.format(isinstance(g, types.GeneratorType)))

    for batch_idx in range(num_batch):
        batch = next(g)
        assert batch_size==batch['input'].shape[0]
        print('Batch #{}:'.format(batch_idx))
        for i in range(batch_size):
            label_vec = batch['output'][i].reshape(-1)
            label = label_dict[np.nonzero(label_vec)[0][0]]
            print('  image[{}]:{}'.format(i, label))

    # Note that 'input' and 'output' are 4-dim array in NCHW format
    print('Note that \'input\' and \'output\' are 4-dim array in NCHW format!')
    print('Input shape={}, Output shape={}'.format(batch['input'].shape, batch['output'].shape))

    # show the last batch
    fig = plt.figure()
    for i in range(batch_size):
        plt.subplot(1, batch_size, i + 1)
        image = batch['input'][i]
        label_vec = batch['output'][i].reshape(-1)
        label = label_dict[np.nonzero(label_vec)[0][0]]
        plt.title('Label: ' + label)
        plt.imshow(image[0,:,:], cmap='gray')
    plt.show()


