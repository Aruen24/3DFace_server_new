#include "cnn_conv2d.h"
#include "cnn_pool2d.h"
#include "cnn_tensor_util.h"
/* @brief cnn_run_a_lovely_net() - We transfrom the prototxt file to C code as below, you can copy the code to complete the whole calculation of your network
/*        feature occupies 279552 bytes
 * @param feature_addr_offset: the address where input and output of all layers are put
 * @param weight_addr_offset: the address where weights and biases all layers are put
 * @param layer_end: if you want to stop computation after N layers finished, please set layer_end to N, otherwise, please set layer_end to 0
 */
void cnn_run_a_lovely_net(uint32_t feature_addr_offset, uint32_t weight_addr_offset, uint16_t layer_end) {
    #define RETURN_JUDGE { if(layer_index++ == layer_end) return; }
    uint16_t layer_index = 1;
    struct cnn_conv2D_config_bean conv2d_bean;
    struct cnn_pool2D_config_bean pool2d_bean;
    // layer index: 0, the 1st layer is of index 0
    // layer name: conv2d_1/Conv2D
    // layer type: ['convolution', 'relu6']
    // this layer is completed by using CNN engine from Round 1 to Round 1
    // input tensor names are ['Placeholder:0']
    // output tensor names are ['conv2d_1/Relu6:0']
    // input tensor occupy [21504] bytes individually
    // output tensor occupy [172032] bytes individually
    // weight occupies 704 bytes
    // you can also use the code below to complete this layer
    conv2d_bean.in_channel = 1; // how many channels of input
    conv2d_bean.in_height = 112; // height of input
    conv2d_bean.in_width = 96; // output of input
    conv2d_bean.out_height = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_width = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_channel = 32; // how many channels of output, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.group = 1; // how many groups, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.kernel_size_h = 3; // size of the convolving kernel
    conv2d_bean.kernel_size_w = 3; // size of the convolving kernel
    conv2d_bean.stride = 2;
    conv2d_bean.dilation = 1;
    conv2d_bean.bias_en = 1; // whether add bias when calculating conv
    conv2d_bean.softmax = 0; // whether calculate softmax after calculating conv and activation function
    conv2d_bean.mac_8bit = 0; // non_zero: 8bit mode; zero: 16bit mode
    conv2d_bean.pad_u = 0; // zero-pad added to up    sides of the input
    conv2d_bean.pad_d = 1; // zero-pad added to down  sides of the input
    conv2d_bean.pad_l = 0; // zero-pad added to left  sides of the input
    conv2d_bean.pad_r = 1; // zero-pad added to right sides of the input
    conv2d_bean.input_signed = 1; // whether input is signed
    conv2d_bean.weight_bias_signed = 1; // whether weight and bias are signed
    conv2d_bean.filter_lsb_channelwise = 1; // whether filter lsb differ from channels
    conv2d_bean.acc_out_shift = 0; // the right shift bits of the output of acc array, it should be input_fraction + weight_fraction - output_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.bias_shift = 0; // the left shift bits of bias when being added to the acc, it should be input_fraction + weight_fraction - bias_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.leaky_param = 0; // the multiplier of leaky relu, the LSB is 2^(-6)
    conv2d_bean.input_iram = 0; // nonzero - read input from iram, 0 - read input from ddr
    conv2d_bean.output_iram = 0; // nonzero - put output into iram, 0 - put output into ddr
    conv2d_bean.in_sep_mode = 0; // whether read input from iram as separable conv mode
    conv2d_bean.out_sep_mode = 0; // whether put output into iram as separable conv mode
    conv2d_bean.next_padding_left = 0; // left pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_up = 0; // up pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_type = 0; // only valid when out_sep_mode is not 0, if the next layer is maxpool, please set it to 1, or set it to 0
    conv2d_bean.nonlinearty = relu6; // the nonlinearty operation after conv, you can choose "none", "sigmoid", "relu", "tanh", "relu6", "leaky_relu"
    cnn_conv2D(0x0 + feature_addr_offset, 0x5400 + feature_addr_offset, 0x0 + weight_addr_offset, &conv2d_bean);
    RETURN_JUDGE

    /*
     * this layer end
     */
    // layer index: 1
    // layer name: expanded_conv/depthwise/depthwise
    // layer type: ['convolution', 'relu6']
    // this layer is completed by using CNN engine from Round 2 to Round 2
    // input tensor names are ['conv2d_1/Relu6:0']
    // output tensor names are ['expanded_conv/depthwise/Relu6:0']
    // input tensor occupy [172032] bytes individually
    // output tensor occupy [57344] bytes individually
    // weight occupies 1024 bytes
    // you can also use the code below to complete this layer
    conv2d_bean.in_channel = 32; // how many channels of input
    conv2d_bean.in_height = 56; // height of input
    conv2d_bean.in_width = 48; // output of input
    conv2d_bean.out_height = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_width = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_channel = 32; // how many channels of output, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.group = 32; // how many groups, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.kernel_size_h = 3; // size of the convolving kernel
    conv2d_bean.kernel_size_w = 3; // size of the convolving kernel
    conv2d_bean.stride = 2;
    conv2d_bean.dilation = 1;
    conv2d_bean.bias_en = 1; // whether add bias when calculating conv
    conv2d_bean.softmax = 0; // whether calculate softmax after calculating conv and activation function
    conv2d_bean.mac_8bit = 0; // non_zero: 8bit mode; zero: 16bit mode
    conv2d_bean.pad_u = 0; // zero-pad added to up    sides of the input
    conv2d_bean.pad_d = 1; // zero-pad added to down  sides of the input
    conv2d_bean.pad_l = 0; // zero-pad added to left  sides of the input
    conv2d_bean.pad_r = 1; // zero-pad added to right sides of the input
    conv2d_bean.input_signed = 1; // whether input is signed
    conv2d_bean.weight_bias_signed = 1; // whether weight and bias are signed
    conv2d_bean.filter_lsb_channelwise = 1; // whether filter lsb differ from channels
    conv2d_bean.acc_out_shift = 0; // the right shift bits of the output of acc array, it should be input_fraction + weight_fraction - output_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.bias_shift = 0; // the left shift bits of bias when being added to the acc, it should be input_fraction + weight_fraction - bias_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.leaky_param = 0; // the multiplier of leaky relu, the LSB is 2^(-6)
    conv2d_bean.input_iram = 0; // nonzero - read input from iram, 0 - read input from ddr
    conv2d_bean.output_iram = 0; // nonzero - put output into iram, 0 - put output into ddr
    conv2d_bean.in_sep_mode = 0; // whether read input from iram as separable conv mode
    conv2d_bean.out_sep_mode = 0; // whether put output into iram as separable conv mode
    conv2d_bean.next_padding_left = 0; // left pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_up = 0; // up pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_type = 0; // only valid when out_sep_mode is not 0, if the next layer is maxpool, please set it to 1, or set it to 0
    conv2d_bean.nonlinearty = relu6; // the nonlinearty operation after conv, you can choose "none", "sigmoid", "relu", "tanh", "relu6", "leaky_relu"
    cnn_conv2D(0x5400 + feature_addr_offset, 0x2f400 + feature_addr_offset, 0x300 + weight_addr_offset, &conv2d_bean);
    RETURN_JUDGE

    /*
     * this layer end
     */
    // layer index: 2
    // layer name: expanded_conv/project/Conv2D
    // layer type: ['convolution']
    // this layer is completed by using CNN engine from Round 3 to Round 3
    // input tensor names are ['expanded_conv/depthwise/Relu6:0']
    // output tensor names are ['expanded_conv/project/BatchNorm/FusedBatchNorm:0']
    // input tensor occupy [57344] bytes individually
    // output tensor occupy [28672] bytes individually
    // weight occupies 1088 bytes
    // you can also use the code below to complete this layer
    conv2d_bean.in_channel = 32; // how many channels of input
    conv2d_bean.in_height = 28; // height of input
    conv2d_bean.in_width = 24; // output of input
    conv2d_bean.out_height = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_width = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_channel = 16; // how many channels of output, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.group = 1; // how many groups, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.kernel_size_h = 1; // size of the convolving kernel
    conv2d_bean.kernel_size_w = 1; // size of the convolving kernel
    conv2d_bean.stride = 1;
    conv2d_bean.dilation = 1;
    conv2d_bean.bias_en = 1; // whether add bias when calculating conv
    conv2d_bean.softmax = 0; // whether calculate softmax after calculating conv and activation function
    conv2d_bean.mac_8bit = 0; // non_zero: 8bit mode; zero: 16bit mode
    conv2d_bean.pad_u = 0; // zero-pad added to up    sides of the input
    conv2d_bean.pad_d = 0; // zero-pad added to down  sides of the input
    conv2d_bean.pad_l = 0; // zero-pad added to left  sides of the input
    conv2d_bean.pad_r = 0; // zero-pad added to right sides of the input
    conv2d_bean.input_signed = 1; // whether input is signed
    conv2d_bean.weight_bias_signed = 1; // whether weight and bias are signed
    conv2d_bean.filter_lsb_channelwise = 1; // whether filter lsb differ from channels
    conv2d_bean.acc_out_shift = 0; // the right shift bits of the output of acc array, it should be input_fraction + weight_fraction - output_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.bias_shift = 0; // the left shift bits of bias when being added to the acc, it should be input_fraction + weight_fraction - bias_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.leaky_param = 0; // the multiplier of leaky relu, the LSB is 2^(-6)
    conv2d_bean.input_iram = 0; // nonzero - read input from iram, 0 - read input from ddr
    conv2d_bean.output_iram = 0; // nonzero - put output into iram, 0 - put output into ddr
    conv2d_bean.in_sep_mode = 0; // whether read input from iram as separable conv mode
    conv2d_bean.out_sep_mode = 0; // whether put output into iram as separable conv mode
    conv2d_bean.next_padding_left = 0; // left pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_up = 0; // up pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_type = 0; // only valid when out_sep_mode is not 0, if the next layer is maxpool, please set it to 1, or set it to 0
    conv2d_bean.nonlinearty = none; // the nonlinearty operation after conv, you can choose "none", "sigmoid", "relu", "tanh", "relu6", "leaky_relu"
    cnn_conv2D(0x2f400 + feature_addr_offset, 0x3d400 + feature_addr_offset, 0x700 + weight_addr_offset, &conv2d_bean);
    RETURN_JUDGE

    /*
     * this layer end
     */
    // layer index: 3
    // layer name: expanded_conv_1/expand/Conv2D
    // layer type: ['convolution', 'relu6']
    // this layer is completed by using CNN engine from Round 4 to Round 4
    // input tensor names are ['expanded_conv/project/BatchNorm/FusedBatchNorm:0']
    // output tensor names are ['expanded_conv_1/expand/Relu6:0']
    // input tensor occupy [28672] bytes individually
    // output tensor occupy [172032] bytes individually
    // weight occupies 3456 bytes
    // you can also use the code below to complete this layer
    conv2d_bean.in_channel = 16; // how many channels of input
    conv2d_bean.in_height = 28; // height of input
    conv2d_bean.in_width = 24; // output of input
    conv2d_bean.out_height = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_width = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_channel = 96; // how many channels of output, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.group = 1; // how many groups, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.kernel_size_h = 1; // size of the convolving kernel
    conv2d_bean.kernel_size_w = 1; // size of the convolving kernel
    conv2d_bean.stride = 1;
    conv2d_bean.dilation = 1;
    conv2d_bean.bias_en = 1; // whether add bias when calculating conv
    conv2d_bean.softmax = 0; // whether calculate softmax after calculating conv and activation function
    conv2d_bean.mac_8bit = 0; // non_zero: 8bit mode; zero: 16bit mode
    conv2d_bean.pad_u = 0; // zero-pad added to up    sides of the input
    conv2d_bean.pad_d = 0; // zero-pad added to down  sides of the input
    conv2d_bean.pad_l = 0; // zero-pad added to left  sides of the input
    conv2d_bean.pad_r = 0; // zero-pad added to right sides of the input
    conv2d_bean.input_signed = 1; // whether input is signed
    conv2d_bean.weight_bias_signed = 1; // whether weight and bias are signed
    conv2d_bean.filter_lsb_channelwise = 1; // whether filter lsb differ from channels
    conv2d_bean.acc_out_shift = 0; // the right shift bits of the output of acc array, it should be input_fraction + weight_fraction - output_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.bias_shift = 0; // the left shift bits of bias when being added to the acc, it should be input_fraction + weight_fraction - bias_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.leaky_param = 0; // the multiplier of leaky relu, the LSB is 2^(-6)
    conv2d_bean.input_iram = 0; // nonzero - read input from iram, 0 - read input from ddr
    conv2d_bean.output_iram = 0; // nonzero - put output into iram, 0 - put output into ddr
    conv2d_bean.in_sep_mode = 0; // whether read input from iram as separable conv mode
    conv2d_bean.out_sep_mode = 0; // whether put output into iram as separable conv mode
    conv2d_bean.next_padding_left = 0; // left pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_up = 0; // up pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_type = 0; // only valid when out_sep_mode is not 0, if the next layer is maxpool, please set it to 1, or set it to 0
    conv2d_bean.nonlinearty = relu6; // the nonlinearty operation after conv, you can choose "none", "sigmoid", "relu", "tanh", "relu6", "leaky_relu"
    cnn_conv2D(0x3d400 + feature_addr_offset, 0x5400 + feature_addr_offset, 0xc00 + weight_addr_offset, &conv2d_bean);
    RETURN_JUDGE

    /*
     * this layer end
     */
    // layer index: 4
    // layer name: expanded_conv_1/depthwise/depthwise
    // layer type: ['convolution', 'relu6']
    // this layer is completed by using CNN engine from Round 5 to Round 5
    // input tensor names are ['expanded_conv_1/expand/Relu6:0']
    // output tensor names are ['expanded_conv_1/depthwise/Relu6:0']
    // input tensor occupy [172032] bytes individually
    // output tensor occupy [43008] bytes individually
    // weight occupies 3072 bytes
    // you can also use the code below to complete this layer
    conv2d_bean.in_channel = 96; // how many channels of input
    conv2d_bean.in_height = 28; // height of input
    conv2d_bean.in_width = 24; // output of input
    conv2d_bean.out_height = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_width = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_channel = 96; // how many channels of output, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.group = 96; // how many groups, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.kernel_size_h = 3; // size of the convolving kernel
    conv2d_bean.kernel_size_w = 3; // size of the convolving kernel
    conv2d_bean.stride = 2;
    conv2d_bean.dilation = 1;
    conv2d_bean.bias_en = 1; // whether add bias when calculating conv
    conv2d_bean.softmax = 0; // whether calculate softmax after calculating conv and activation function
    conv2d_bean.mac_8bit = 0; // non_zero: 8bit mode; zero: 16bit mode
    conv2d_bean.pad_u = 0; // zero-pad added to up    sides of the input
    conv2d_bean.pad_d = 1; // zero-pad added to down  sides of the input
    conv2d_bean.pad_l = 0; // zero-pad added to left  sides of the input
    conv2d_bean.pad_r = 1; // zero-pad added to right sides of the input
    conv2d_bean.input_signed = 1; // whether input is signed
    conv2d_bean.weight_bias_signed = 1; // whether weight and bias are signed
    conv2d_bean.filter_lsb_channelwise = 1; // whether filter lsb differ from channels
    conv2d_bean.acc_out_shift = 0; // the right shift bits of the output of acc array, it should be input_fraction + weight_fraction - output_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.bias_shift = 0; // the left shift bits of bias when being added to the acc, it should be input_fraction + weight_fraction - bias_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.leaky_param = 0; // the multiplier of leaky relu, the LSB is 2^(-6)
    conv2d_bean.input_iram = 0; // nonzero - read input from iram, 0 - read input from ddr
    conv2d_bean.output_iram = 0; // nonzero - put output into iram, 0 - put output into ddr
    conv2d_bean.in_sep_mode = 0; // whether read input from iram as separable conv mode
    conv2d_bean.out_sep_mode = 0; // whether put output into iram as separable conv mode
    conv2d_bean.next_padding_left = 0; // left pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_up = 0; // up pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_type = 0; // only valid when out_sep_mode is not 0, if the next layer is maxpool, please set it to 1, or set it to 0
    conv2d_bean.nonlinearty = relu6; // the nonlinearty operation after conv, you can choose "none", "sigmoid", "relu", "tanh", "relu6", "leaky_relu"
    cnn_conv2D(0x5400 + feature_addr_offset, 0x2f400 + feature_addr_offset, 0x1a00 + weight_addr_offset, &conv2d_bean);
    RETURN_JUDGE

    /*
     * this layer end
     */
    // layer index: 5
    // layer name: expanded_conv_1/project/Conv2D
    // layer type: ['convolution']
    // this layer is completed by using CNN engine from Round 6 to Round 6
    // input tensor names are ['expanded_conv_1/depthwise/Relu6:0']
    // output tensor names are ['expanded_conv_1/project/BatchNorm/FusedBatchNorm:0']
    // input tensor occupy [43008] bytes individually
    // output tensor occupy [14336] bytes individually
    // weight occupies 6272 bytes
    // you can also use the code below to complete this layer
    conv2d_bean.in_channel = 96; // how many channels of input
    conv2d_bean.in_height = 14; // height of input
    conv2d_bean.in_width = 12; // output of input
    conv2d_bean.out_height = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_width = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_channel = 32; // how many channels of output, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.group = 1; // how many groups, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.kernel_size_h = 1; // size of the convolving kernel
    conv2d_bean.kernel_size_w = 1; // size of the convolving kernel
    conv2d_bean.stride = 1;
    conv2d_bean.dilation = 1;
    conv2d_bean.bias_en = 1; // whether add bias when calculating conv
    conv2d_bean.softmax = 0; // whether calculate softmax after calculating conv and activation function
    conv2d_bean.mac_8bit = 0; // non_zero: 8bit mode; zero: 16bit mode
    conv2d_bean.pad_u = 0; // zero-pad added to up    sides of the input
    conv2d_bean.pad_d = 0; // zero-pad added to down  sides of the input
    conv2d_bean.pad_l = 0; // zero-pad added to left  sides of the input
    conv2d_bean.pad_r = 0; // zero-pad added to right sides of the input
    conv2d_bean.input_signed = 1; // whether input is signed
    conv2d_bean.weight_bias_signed = 1; // whether weight and bias are signed
    conv2d_bean.filter_lsb_channelwise = 1; // whether filter lsb differ from channels
    conv2d_bean.acc_out_shift = 0; // the right shift bits of the output of acc array, it should be input_fraction + weight_fraction - output_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.bias_shift = 0; // the left shift bits of bias when being added to the acc, it should be input_fraction + weight_fraction - bias_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.leaky_param = 0; // the multiplier of leaky relu, the LSB is 2^(-6)
    conv2d_bean.input_iram = 0; // nonzero - read input from iram, 0 - read input from ddr
    conv2d_bean.output_iram = 0; // nonzero - put output into iram, 0 - put output into ddr
    conv2d_bean.in_sep_mode = 0; // whether read input from iram as separable conv mode
    conv2d_bean.out_sep_mode = 0; // whether put output into iram as separable conv mode
    conv2d_bean.next_padding_left = 0; // left pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_up = 0; // up pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_type = 0; // only valid when out_sep_mode is not 0, if the next layer is maxpool, please set it to 1, or set it to 0
    conv2d_bean.nonlinearty = none; // the nonlinearty operation after conv, you can choose "none", "sigmoid", "relu", "tanh", "relu6", "leaky_relu"
    cnn_conv2D(0x2f400 + feature_addr_offset, 0x39c00 + feature_addr_offset, 0x2600 + weight_addr_offset, &conv2d_bean);
    RETURN_JUDGE

    /*
     * this layer end
     */
    // layer index: 6
    // layer name: expanded_conv_2/expand/Conv2D
    // layer type: ['convolution', 'relu6']
    // this layer is completed by using CNN engine from Round 7 to Round 7
    // input tensor names are ['expanded_conv_1/project/BatchNorm/FusedBatchNorm:0']
    // output tensor names are ['expanded_conv_2/expand/Relu6:0']
    // input tensor occupy [14336] bytes individually
    // output tensor occupy [86016] bytes individually
    // weight occupies 13056 bytes
    // you can also use the code below to complete this layer
    conv2d_bean.in_channel = 32; // how many channels of input
    conv2d_bean.in_height = 14; // height of input
    conv2d_bean.in_width = 12; // output of input
    conv2d_bean.out_height = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_width = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_channel = 192; // how many channels of output, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.group = 1; // how many groups, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.kernel_size_h = 1; // size of the convolving kernel
    conv2d_bean.kernel_size_w = 1; // size of the convolving kernel
    conv2d_bean.stride = 1;
    conv2d_bean.dilation = 1;
    conv2d_bean.bias_en = 1; // whether add bias when calculating conv
    conv2d_bean.softmax = 0; // whether calculate softmax after calculating conv and activation function
    conv2d_bean.mac_8bit = 0; // non_zero: 8bit mode; zero: 16bit mode
    conv2d_bean.pad_u = 0; // zero-pad added to up    sides of the input
    conv2d_bean.pad_d = 0; // zero-pad added to down  sides of the input
    conv2d_bean.pad_l = 0; // zero-pad added to left  sides of the input
    conv2d_bean.pad_r = 0; // zero-pad added to right sides of the input
    conv2d_bean.input_signed = 1; // whether input is signed
    conv2d_bean.weight_bias_signed = 1; // whether weight and bias are signed
    conv2d_bean.filter_lsb_channelwise = 1; // whether filter lsb differ from channels
    conv2d_bean.acc_out_shift = 0; // the right shift bits of the output of acc array, it should be input_fraction + weight_fraction - output_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.bias_shift = 0; // the left shift bits of bias when being added to the acc, it should be input_fraction + weight_fraction - bias_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.leaky_param = 0; // the multiplier of leaky relu, the LSB is 2^(-6)
    conv2d_bean.input_iram = 0; // nonzero - read input from iram, 0 - read input from ddr
    conv2d_bean.output_iram = 0; // nonzero - put output into iram, 0 - put output into ddr
    conv2d_bean.in_sep_mode = 0; // whether read input from iram as separable conv mode
    conv2d_bean.out_sep_mode = 0; // whether put output into iram as separable conv mode
    conv2d_bean.next_padding_left = 0; // left pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_up = 0; // up pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_type = 0; // only valid when out_sep_mode is not 0, if the next layer is maxpool, please set it to 1, or set it to 0
    conv2d_bean.nonlinearty = relu6; // the nonlinearty operation after conv, you can choose "none", "sigmoid", "relu", "tanh", "relu6", "leaky_relu"
    cnn_conv2D(0x39c00 + feature_addr_offset, 0x5400 + feature_addr_offset, 0x3f00 + weight_addr_offset, &conv2d_bean);
    RETURN_JUDGE

    /*
     * this layer end
     */
    // layer index: 7
    // layer name: expanded_conv_2/depthwise/depthwise
    // layer type: ['convolution', 'relu6']
    // this layer is completed by using CNN engine from Round 8 to Round 8
    // input tensor names are ['expanded_conv_2/expand/Relu6:0']
    // output tensor names are ['expanded_conv_2/depthwise/Relu6:0']
    // input tensor occupy [86016] bytes individually
    // output tensor occupy [86016] bytes individually
    // weight occupies 6144 bytes
    // you can also use the code below to complete this layer
    conv2d_bean.in_channel = 192; // how many channels of input
    conv2d_bean.in_height = 14; // height of input
    conv2d_bean.in_width = 12; // output of input
    conv2d_bean.out_height = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_width = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_channel = 192; // how many channels of output, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.group = 192; // how many groups, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.kernel_size_h = 3; // size of the convolving kernel
    conv2d_bean.kernel_size_w = 3; // size of the convolving kernel
    conv2d_bean.stride = 1;
    conv2d_bean.dilation = 1;
    conv2d_bean.bias_en = 1; // whether add bias when calculating conv
    conv2d_bean.softmax = 0; // whether calculate softmax after calculating conv and activation function
    conv2d_bean.mac_8bit = 0; // non_zero: 8bit mode; zero: 16bit mode
    conv2d_bean.pad_u = 1; // zero-pad added to up    sides of the input
    conv2d_bean.pad_d = 1; // zero-pad added to down  sides of the input
    conv2d_bean.pad_l = 1; // zero-pad added to left  sides of the input
    conv2d_bean.pad_r = 1; // zero-pad added to right sides of the input
    conv2d_bean.input_signed = 1; // whether input is signed
    conv2d_bean.weight_bias_signed = 1; // whether weight and bias are signed
    conv2d_bean.filter_lsb_channelwise = 1; // whether filter lsb differ from channels
    conv2d_bean.acc_out_shift = 0; // the right shift bits of the output of acc array, it should be input_fraction + weight_fraction - output_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.bias_shift = 0; // the left shift bits of bias when being added to the acc, it should be input_fraction + weight_fraction - bias_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.leaky_param = 0; // the multiplier of leaky relu, the LSB is 2^(-6)
    conv2d_bean.input_iram = 0; // nonzero - read input from iram, 0 - read input from ddr
    conv2d_bean.output_iram = 0; // nonzero - put output into iram, 0 - put output into ddr
    conv2d_bean.in_sep_mode = 0; // whether read input from iram as separable conv mode
    conv2d_bean.out_sep_mode = 0; // whether put output into iram as separable conv mode
    conv2d_bean.next_padding_left = 0; // left pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_up = 0; // up pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_type = 0; // only valid when out_sep_mode is not 0, if the next layer is maxpool, please set it to 1, or set it to 0
    conv2d_bean.nonlinearty = relu6; // the nonlinearty operation after conv, you can choose "none", "sigmoid", "relu", "tanh", "relu6", "leaky_relu"
    cnn_conv2D(0x5400 + feature_addr_offset, 0x1a400 + feature_addr_offset, 0x7200 + weight_addr_offset, &conv2d_bean);
    RETURN_JUDGE

    /*
     * this layer end
     */
    // layer index: 8
    // layer name: expanded_conv_2/project/Conv2D
    // layer type: ['convolution']
    // this layer is completed by using CNN engine from Round 9 to Round 9
    // input tensor names are ['expanded_conv_2/depthwise/Relu6:0']
    // output tensor names are ['expanded_conv_2/project/BatchNorm/FusedBatchNorm:0']
    // input tensor occupy [86016] bytes individually
    // output tensor occupy [14336] bytes individually
    // weight occupies 12416 bytes
    // you can also use the code below to complete this layer
    conv2d_bean.in_channel = 192; // how many channels of input
    conv2d_bean.in_height = 14; // height of input
    conv2d_bean.in_width = 12; // output of input
    conv2d_bean.out_height = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_width = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_channel = 32; // how many channels of output, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.group = 1; // how many groups, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.kernel_size_h = 1; // size of the convolving kernel
    conv2d_bean.kernel_size_w = 1; // size of the convolving kernel
    conv2d_bean.stride = 1;
    conv2d_bean.dilation = 1;
    conv2d_bean.bias_en = 1; // whether add bias when calculating conv
    conv2d_bean.softmax = 0; // whether calculate softmax after calculating conv and activation function
    conv2d_bean.mac_8bit = 0; // non_zero: 8bit mode; zero: 16bit mode
    conv2d_bean.pad_u = 0; // zero-pad added to up    sides of the input
    conv2d_bean.pad_d = 0; // zero-pad added to down  sides of the input
    conv2d_bean.pad_l = 0; // zero-pad added to left  sides of the input
    conv2d_bean.pad_r = 0; // zero-pad added to right sides of the input
    conv2d_bean.input_signed = 1; // whether input is signed
    conv2d_bean.weight_bias_signed = 1; // whether weight and bias are signed
    conv2d_bean.filter_lsb_channelwise = 1; // whether filter lsb differ from channels
    conv2d_bean.acc_out_shift = 0; // the right shift bits of the output of acc array, it should be input_fraction + weight_fraction - output_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.bias_shift = 0; // the left shift bits of bias when being added to the acc, it should be input_fraction + weight_fraction - bias_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.leaky_param = 0; // the multiplier of leaky relu, the LSB is 2^(-6)
    conv2d_bean.input_iram = 0; // nonzero - read input from iram, 0 - read input from ddr
    conv2d_bean.output_iram = 0; // nonzero - put output into iram, 0 - put output into ddr
    conv2d_bean.in_sep_mode = 0; // whether read input from iram as separable conv mode
    conv2d_bean.out_sep_mode = 0; // whether put output into iram as separable conv mode
    conv2d_bean.next_padding_left = 0; // left pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_up = 0; // up pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_type = 0; // only valid when out_sep_mode is not 0, if the next layer is maxpool, please set it to 1, or set it to 0
    conv2d_bean.nonlinearty = none; // the nonlinearty operation after conv, you can choose "none", "sigmoid", "relu", "tanh", "relu6", "leaky_relu"
    cnn_conv2D(0x1a400 + feature_addr_offset, 0x3d400 + feature_addr_offset, 0x8a00 + weight_addr_offset, &conv2d_bean);
    RETURN_JUDGE

    /*
     * this layer end
     */
    // layer index: 9
    // layer name: expanded_conv_2/add
    // layer type: ['eltwise']
    // this layer is completed by using CNN engine from Round 10 to Round 10
    // input tensor names are ['expanded_conv_1/project/BatchNorm/FusedBatchNorm:0', 'expanded_conv_2/project/BatchNorm/FusedBatchNorm:0']
    // output tensor names are ['expanded_conv_2/add:0']
    // input tensor occupy [14336, 14336] bytes individually
    // output tensor occupy [14336] bytes individually
    // weight occupies 0 bytes
    // you can also use the code below to complete this layer
    conv2d_bean.in_channel = 2; // how many channels of input
    conv2d_bean.in_height = 448; // height of input
    conv2d_bean.in_width = 12; // output of input
    conv2d_bean.out_height = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_width = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_channel = 1; // how many channels of output, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.group = 1; // how many groups, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.kernel_size_h = 1; // size of the convolving kernel
    conv2d_bean.kernel_size_w = 1; // size of the convolving kernel
    conv2d_bean.stride = 1;
    conv2d_bean.dilation = 1;
    conv2d_bean.bias_en = 0; // whether add bias when calculating conv
    conv2d_bean.softmax = 0; // whether calculate softmax after calculating conv and activation function
    conv2d_bean.mac_8bit = 0; // non_zero: 8bit mode; zero: 16bit mode
    conv2d_bean.pad_u = 0; // zero-pad added to up    sides of the input
    conv2d_bean.pad_d = 0; // zero-pad added to down  sides of the input
    conv2d_bean.pad_l = 0; // zero-pad added to left  sides of the input
    conv2d_bean.pad_r = 0; // zero-pad added to right sides of the input
    conv2d_bean.input_signed = 1; // whether input is signed
    conv2d_bean.weight_bias_signed = 1; // whether weight and bias are signed
    conv2d_bean.filter_lsb_channelwise = 0; // whether filter lsb differ from channels
    conv2d_bean.acc_out_shift = 0; // the right shift bits of the output of acc array, it should be input_fraction + weight_fraction - output_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.bias_shift = 0; // the left shift bits of bias when being added to the acc, it should be input_fraction + weight_fraction - bias_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.leaky_param = 0; // the multiplier of leaky relu, the LSB is 2^(-6)
    conv2d_bean.input_iram = 0; // nonzero - read input from iram, 0 - read input from ddr
    conv2d_bean.output_iram = 0; // nonzero - put output into iram, 0 - put output into ddr
    conv2d_bean.in_sep_mode = 0; // whether read input from iram as separable conv mode
    conv2d_bean.out_sep_mode = 0; // whether put output into iram as separable conv mode
    conv2d_bean.next_padding_left = 0; // left pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_up = 0; // up pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_type = 0; // only valid when out_sep_mode is not 0, if the next layer is maxpool, please set it to 1, or set it to 0
    conv2d_bean.nonlinearty = none; // the nonlinearty operation after conv, you can choose "none", "sigmoid", "relu", "tanh", "relu6", "leaky_relu"
    cnn_conv2D(0x39c00 + feature_addr_offset, 0x5400 + feature_addr_offset, 0xbb00 + weight_addr_offset, &conv2d_bean);
    RETURN_JUDGE

    /*
     * this layer end
     */
    // layer index: 10
    // layer name: expanded_conv_3/expand/Conv2D
    // layer type: ['convolution', 'relu6']
    // this layer is completed by using CNN engine from Round 11 to Round 11
    // input tensor names are ['expanded_conv_2/add:0']
    // output tensor names are ['expanded_conv_3/expand/Relu6:0']
    // input tensor occupy [14336] bytes individually
    // output tensor occupy [86016] bytes individually
    // weight occupies 13056 bytes
    // you can also use the code below to complete this layer
    conv2d_bean.in_channel = 32; // how many channels of input
    conv2d_bean.in_height = 14; // height of input
    conv2d_bean.in_width = 12; // output of input
    conv2d_bean.out_height = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_width = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_channel = 192; // how many channels of output, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.group = 1; // how many groups, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.kernel_size_h = 1; // size of the convolving kernel
    conv2d_bean.kernel_size_w = 1; // size of the convolving kernel
    conv2d_bean.stride = 1;
    conv2d_bean.dilation = 1;
    conv2d_bean.bias_en = 1; // whether add bias when calculating conv
    conv2d_bean.softmax = 0; // whether calculate softmax after calculating conv and activation function
    conv2d_bean.mac_8bit = 0; // non_zero: 8bit mode; zero: 16bit mode
    conv2d_bean.pad_u = 0; // zero-pad added to up    sides of the input
    conv2d_bean.pad_d = 0; // zero-pad added to down  sides of the input
    conv2d_bean.pad_l = 0; // zero-pad added to left  sides of the input
    conv2d_bean.pad_r = 0; // zero-pad added to right sides of the input
    conv2d_bean.input_signed = 1; // whether input is signed
    conv2d_bean.weight_bias_signed = 1; // whether weight and bias are signed
    conv2d_bean.filter_lsb_channelwise = 1; // whether filter lsb differ from channels
    conv2d_bean.acc_out_shift = 0; // the right shift bits of the output of acc array, it should be input_fraction + weight_fraction - output_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.bias_shift = 0; // the left shift bits of bias when being added to the acc, it should be input_fraction + weight_fraction - bias_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.leaky_param = 0; // the multiplier of leaky relu, the LSB is 2^(-6)
    conv2d_bean.input_iram = 0; // nonzero - read input from iram, 0 - read input from ddr
    conv2d_bean.output_iram = 0; // nonzero - put output into iram, 0 - put output into ddr
    conv2d_bean.in_sep_mode = 0; // whether read input from iram as separable conv mode
    conv2d_bean.out_sep_mode = 0; // whether put output into iram as separable conv mode
    conv2d_bean.next_padding_left = 0; // left pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_up = 0; // up pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_type = 0; // only valid when out_sep_mode is not 0, if the next layer is maxpool, please set it to 1, or set it to 0
    conv2d_bean.nonlinearty = relu6; // the nonlinearty operation after conv, you can choose "none", "sigmoid", "relu", "tanh", "relu6", "leaky_relu"
    cnn_conv2D(0x5400 + feature_addr_offset, 0x8c00 + feature_addr_offset, 0xbb00 + weight_addr_offset, &conv2d_bean);
    RETURN_JUDGE

    /*
     * this layer end
     */
    // layer index: 11
    // layer name: expanded_conv_3/depthwise/depthwise
    // layer type: ['convolution', 'relu6']
    // this layer is completed by using CNN engine from Round 12 to Round 12
    // input tensor names are ['expanded_conv_3/expand/Relu6:0']
    // output tensor names are ['expanded_conv_3/depthwise/Relu6:0']
    // input tensor occupy [86016] bytes individually
    // output tensor occupy [43008] bytes individually
    // weight occupies 6144 bytes
    // you can also use the code below to complete this layer
    conv2d_bean.in_channel = 192; // how many channels of input
    conv2d_bean.in_height = 14; // height of input
    conv2d_bean.in_width = 12; // output of input
    conv2d_bean.out_height = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_width = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_channel = 192; // how many channels of output, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.group = 192; // how many groups, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.kernel_size_h = 3; // size of the convolving kernel
    conv2d_bean.kernel_size_w = 3; // size of the convolving kernel
    conv2d_bean.stride = 2;
    conv2d_bean.dilation = 1;
    conv2d_bean.bias_en = 1; // whether add bias when calculating conv
    conv2d_bean.softmax = 0; // whether calculate softmax after calculating conv and activation function
    conv2d_bean.mac_8bit = 0; // non_zero: 8bit mode; zero: 16bit mode
    conv2d_bean.pad_u = 0; // zero-pad added to up    sides of the input
    conv2d_bean.pad_d = 1; // zero-pad added to down  sides of the input
    conv2d_bean.pad_l = 0; // zero-pad added to left  sides of the input
    conv2d_bean.pad_r = 1; // zero-pad added to right sides of the input
    conv2d_bean.input_signed = 1; // whether input is signed
    conv2d_bean.weight_bias_signed = 1; // whether weight and bias are signed
    conv2d_bean.filter_lsb_channelwise = 1; // whether filter lsb differ from channels
    conv2d_bean.acc_out_shift = 0; // the right shift bits of the output of acc array, it should be input_fraction + weight_fraction - output_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.bias_shift = 0; // the left shift bits of bias when being added to the acc, it should be input_fraction + weight_fraction - bias_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.leaky_param = 0; // the multiplier of leaky relu, the LSB is 2^(-6)
    conv2d_bean.input_iram = 0; // nonzero - read input from iram, 0 - read input from ddr
    conv2d_bean.output_iram = 0; // nonzero - put output into iram, 0 - put output into ddr
    conv2d_bean.in_sep_mode = 0; // whether read input from iram as separable conv mode
    conv2d_bean.out_sep_mode = 0; // whether put output into iram as separable conv mode
    conv2d_bean.next_padding_left = 0; // left pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_up = 0; // up pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_type = 0; // only valid when out_sep_mode is not 0, if the next layer is maxpool, please set it to 1, or set it to 0
    conv2d_bean.nonlinearty = relu6; // the nonlinearty operation after conv, you can choose "none", "sigmoid", "relu", "tanh", "relu6", "leaky_relu"
    cnn_conv2D(0x8c00 + feature_addr_offset, 0x1dc00 + feature_addr_offset, 0xee00 + weight_addr_offset, &conv2d_bean);
    RETURN_JUDGE

    /*
     * this layer end
     */
    // layer index: 12
    // layer name: expanded_conv_3/project/Conv2D
    // layer type: ['convolution']
    // this layer is completed by using CNN engine from Round 13 to Round 13
    // input tensor names are ['expanded_conv_3/depthwise/Relu6:0']
    // output tensor names are ['expanded_conv_3/project/BatchNorm/FusedBatchNorm:0']
    // input tensor occupy [43008] bytes individually
    // output tensor occupy [10752] bytes individually
    // weight occupies 18624 bytes
    // you can also use the code below to complete this layer
    conv2d_bean.in_channel = 192; // how many channels of input
    conv2d_bean.in_height = 7; // height of input
    conv2d_bean.in_width = 6; // output of input
    conv2d_bean.out_height = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_width = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_channel = 48; // how many channels of output, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.group = 1; // how many groups, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.kernel_size_h = 1; // size of the convolving kernel
    conv2d_bean.kernel_size_w = 1; // size of the convolving kernel
    conv2d_bean.stride = 1;
    conv2d_bean.dilation = 1;
    conv2d_bean.bias_en = 1; // whether add bias when calculating conv
    conv2d_bean.softmax = 0; // whether calculate softmax after calculating conv and activation function
    conv2d_bean.mac_8bit = 0; // non_zero: 8bit mode; zero: 16bit mode
    conv2d_bean.pad_u = 0; // zero-pad added to up    sides of the input
    conv2d_bean.pad_d = 0; // zero-pad added to down  sides of the input
    conv2d_bean.pad_l = 0; // zero-pad added to left  sides of the input
    conv2d_bean.pad_r = 0; // zero-pad added to right sides of the input
    conv2d_bean.input_signed = 1; // whether input is signed
    conv2d_bean.weight_bias_signed = 1; // whether weight and bias are signed
    conv2d_bean.filter_lsb_channelwise = 1; // whether filter lsb differ from channels
    conv2d_bean.acc_out_shift = 0; // the right shift bits of the output of acc array, it should be input_fraction + weight_fraction - output_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.bias_shift = 0; // the left shift bits of bias when being added to the acc, it should be input_fraction + weight_fraction - bias_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.leaky_param = 0; // the multiplier of leaky relu, the LSB is 2^(-6)
    conv2d_bean.input_iram = 0; // nonzero - read input from iram, 0 - read input from ddr
    conv2d_bean.output_iram = 0; // nonzero - put output into iram, 0 - put output into ddr
    conv2d_bean.in_sep_mode = 0; // whether read input from iram as separable conv mode
    conv2d_bean.out_sep_mode = 0; // whether put output into iram as separable conv mode
    conv2d_bean.next_padding_left = 0; // left pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_up = 0; // up pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_type = 0; // only valid when out_sep_mode is not 0, if the next layer is maxpool, please set it to 1, or set it to 0
    conv2d_bean.nonlinearty = none; // the nonlinearty operation after conv, you can choose "none", "sigmoid", "relu", "tanh", "relu6", "leaky_relu"
    cnn_conv2D(0x1dc00 + feature_addr_offset, 0x28400 + feature_addr_offset, 0x10600 + weight_addr_offset, &conv2d_bean);
    RETURN_JUDGE

    /*
     * this layer end
     */
    // layer index: 13
    // layer name: expanded_conv_4/expand/Conv2D
    // layer type: ['convolution', 'relu6']
    // this layer is completed by using CNN engine from Round 14 to Round 14
    // input tensor names are ['expanded_conv_3/project/BatchNorm/FusedBatchNorm:0']
    // output tensor names are ['expanded_conv_4/expand/Relu6:0']
    // input tensor occupy [10752] bytes individually
    // output tensor occupy [64512] bytes individually
    // weight occupies 28800 bytes
    // you can also use the code below to complete this layer
    conv2d_bean.in_channel = 48; // how many channels of input
    conv2d_bean.in_height = 7; // height of input
    conv2d_bean.in_width = 6; // output of input
    conv2d_bean.out_height = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_width = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_channel = 288; // how many channels of output, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.group = 1; // how many groups, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.kernel_size_h = 1; // size of the convolving kernel
    conv2d_bean.kernel_size_w = 1; // size of the convolving kernel
    conv2d_bean.stride = 1;
    conv2d_bean.dilation = 1;
    conv2d_bean.bias_en = 1; // whether add bias when calculating conv
    conv2d_bean.softmax = 0; // whether calculate softmax after calculating conv and activation function
    conv2d_bean.mac_8bit = 0; // non_zero: 8bit mode; zero: 16bit mode
    conv2d_bean.pad_u = 0; // zero-pad added to up    sides of the input
    conv2d_bean.pad_d = 0; // zero-pad added to down  sides of the input
    conv2d_bean.pad_l = 0; // zero-pad added to left  sides of the input
    conv2d_bean.pad_r = 0; // zero-pad added to right sides of the input
    conv2d_bean.input_signed = 1; // whether input is signed
    conv2d_bean.weight_bias_signed = 1; // whether weight and bias are signed
    conv2d_bean.filter_lsb_channelwise = 1; // whether filter lsb differ from channels
    conv2d_bean.acc_out_shift = 0; // the right shift bits of the output of acc array, it should be input_fraction + weight_fraction - output_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.bias_shift = 0; // the left shift bits of bias when being added to the acc, it should be input_fraction + weight_fraction - bias_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.leaky_param = 0; // the multiplier of leaky relu, the LSB is 2^(-6)
    conv2d_bean.input_iram = 0; // nonzero - read input from iram, 0 - read input from ddr
    conv2d_bean.output_iram = 0; // nonzero - put output into iram, 0 - put output into ddr
    conv2d_bean.in_sep_mode = 0; // whether read input from iram as separable conv mode
    conv2d_bean.out_sep_mode = 0; // whether put output into iram as separable conv mode
    conv2d_bean.next_padding_left = 0; // left pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_up = 0; // up pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_type = 0; // only valid when out_sep_mode is not 0, if the next layer is maxpool, please set it to 1, or set it to 0
    conv2d_bean.nonlinearty = relu6; // the nonlinearty operation after conv, you can choose "none", "sigmoid", "relu", "tanh", "relu6", "leaky_relu"
    cnn_conv2D(0x28400 + feature_addr_offset, 0x5400 + feature_addr_offset, 0x14f00 + weight_addr_offset, &conv2d_bean);
    RETURN_JUDGE

    /*
     * this layer end
     */
    // layer index: 14
    // layer name: expanded_conv_4/depthwise/depthwise
    // layer type: ['convolution', 'relu6']
    // this layer is completed by using CNN engine from Round 15 to Round 15
    // input tensor names are ['expanded_conv_4/expand/Relu6:0']
    // output tensor names are ['expanded_conv_4/depthwise/Relu6:0']
    // input tensor occupy [64512] bytes individually
    // output tensor occupy [64512] bytes individually
    // weight occupies 9216 bytes
    // you can also use the code below to complete this layer
    conv2d_bean.in_channel = 288; // how many channels of input
    conv2d_bean.in_height = 7; // height of input
    conv2d_bean.in_width = 6; // output of input
    conv2d_bean.out_height = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_width = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_channel = 288; // how many channels of output, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.group = 288; // how many groups, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.kernel_size_h = 3; // size of the convolving kernel
    conv2d_bean.kernel_size_w = 3; // size of the convolving kernel
    conv2d_bean.stride = 1;
    conv2d_bean.dilation = 1;
    conv2d_bean.bias_en = 1; // whether add bias when calculating conv
    conv2d_bean.softmax = 0; // whether calculate softmax after calculating conv and activation function
    conv2d_bean.mac_8bit = 0; // non_zero: 8bit mode; zero: 16bit mode
    conv2d_bean.pad_u = 1; // zero-pad added to up    sides of the input
    conv2d_bean.pad_d = 1; // zero-pad added to down  sides of the input
    conv2d_bean.pad_l = 1; // zero-pad added to left  sides of the input
    conv2d_bean.pad_r = 1; // zero-pad added to right sides of the input
    conv2d_bean.input_signed = 1; // whether input is signed
    conv2d_bean.weight_bias_signed = 1; // whether weight and bias are signed
    conv2d_bean.filter_lsb_channelwise = 1; // whether filter lsb differ from channels
    conv2d_bean.acc_out_shift = 0; // the right shift bits of the output of acc array, it should be input_fraction + weight_fraction - output_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.bias_shift = 0; // the left shift bits of bias when being added to the acc, it should be input_fraction + weight_fraction - bias_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.leaky_param = 0; // the multiplier of leaky relu, the LSB is 2^(-6)
    conv2d_bean.input_iram = 0; // nonzero - read input from iram, 0 - read input from ddr
    conv2d_bean.output_iram = 0; // nonzero - put output into iram, 0 - put output into ddr
    conv2d_bean.in_sep_mode = 0; // whether read input from iram as separable conv mode
    conv2d_bean.out_sep_mode = 0; // whether put output into iram as separable conv mode
    conv2d_bean.next_padding_left = 0; // left pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_up = 0; // up pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_type = 0; // only valid when out_sep_mode is not 0, if the next layer is maxpool, please set it to 1, or set it to 0
    conv2d_bean.nonlinearty = relu6; // the nonlinearty operation after conv, you can choose "none", "sigmoid", "relu", "tanh", "relu6", "leaky_relu"
    cnn_conv2D(0x5400 + feature_addr_offset, 0x15000 + feature_addr_offset, 0x1c000 + weight_addr_offset, &conv2d_bean);
    RETURN_JUDGE

    /*
     * this layer end
     */
    // layer index: 15
    // layer name: expanded_conv_4/project/Conv2D
    // layer type: ['convolution']
    // this layer is completed by using CNN engine from Round 16 to Round 16
    // input tensor names are ['expanded_conv_4/depthwise/Relu6:0']
    // output tensor names are ['expanded_conv_4/project/BatchNorm/FusedBatchNorm:0']
    // input tensor occupy [64512] bytes individually
    // output tensor occupy [10752] bytes individually
    // weight occupies 27840 bytes
    // you can also use the code below to complete this layer
    conv2d_bean.in_channel = 288; // how many channels of input
    conv2d_bean.in_height = 7; // height of input
    conv2d_bean.in_width = 6; // output of input
    conv2d_bean.out_height = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_width = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_channel = 48; // how many channels of output, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.group = 1; // how many groups, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.kernel_size_h = 1; // size of the convolving kernel
    conv2d_bean.kernel_size_w = 1; // size of the convolving kernel
    conv2d_bean.stride = 1;
    conv2d_bean.dilation = 1;
    conv2d_bean.bias_en = 1; // whether add bias when calculating conv
    conv2d_bean.softmax = 0; // whether calculate softmax after calculating conv and activation function
    conv2d_bean.mac_8bit = 0; // non_zero: 8bit mode; zero: 16bit mode
    conv2d_bean.pad_u = 0; // zero-pad added to up    sides of the input
    conv2d_bean.pad_d = 0; // zero-pad added to down  sides of the input
    conv2d_bean.pad_l = 0; // zero-pad added to left  sides of the input
    conv2d_bean.pad_r = 0; // zero-pad added to right sides of the input
    conv2d_bean.input_signed = 1; // whether input is signed
    conv2d_bean.weight_bias_signed = 1; // whether weight and bias are signed
    conv2d_bean.filter_lsb_channelwise = 1; // whether filter lsb differ from channels
    conv2d_bean.acc_out_shift = 0; // the right shift bits of the output of acc array, it should be input_fraction + weight_fraction - output_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.bias_shift = 0; // the left shift bits of bias when being added to the acc, it should be input_fraction + weight_fraction - bias_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.leaky_param = 0; // the multiplier of leaky relu, the LSB is 2^(-6)
    conv2d_bean.input_iram = 0; // nonzero - read input from iram, 0 - read input from ddr
    conv2d_bean.output_iram = 0; // nonzero - put output into iram, 0 - put output into ddr
    conv2d_bean.in_sep_mode = 0; // whether read input from iram as separable conv mode
    conv2d_bean.out_sep_mode = 0; // whether put output into iram as separable conv mode
    conv2d_bean.next_padding_left = 0; // left pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_up = 0; // up pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_type = 0; // only valid when out_sep_mode is not 0, if the next layer is maxpool, please set it to 1, or set it to 0
    conv2d_bean.nonlinearty = none; // the nonlinearty operation after conv, you can choose "none", "sigmoid", "relu", "tanh", "relu6", "leaky_relu"
    cnn_conv2D(0x15000 + feature_addr_offset, 0x2ae00 + feature_addr_offset, 0x1e400 + weight_addr_offset, &conv2d_bean);
    RETURN_JUDGE

    /*
     * this layer end
     */
    // layer index: 16
    // layer name: expanded_conv_4/add
    // layer type: ['eltwise']
    // this layer is completed by using CNN engine from Round 17 to Round 17
    // input tensor names are ['expanded_conv_3/project/BatchNorm/FusedBatchNorm:0', 'expanded_conv_4/project/BatchNorm/FusedBatchNorm:0']
    // output tensor names are ['expanded_conv_4/add:0']
    // input tensor occupy [10752, 10752] bytes individually
    // output tensor occupy [10752] bytes individually
    // weight occupies 0 bytes
    // you can also use the code below to complete this layer
    conv2d_bean.in_channel = 2; // how many channels of input
    conv2d_bean.in_height = 336; // height of input
    conv2d_bean.in_width = 6; // output of input
    conv2d_bean.out_height = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_width = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_channel = 1; // how many channels of output, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.group = 1; // how many groups, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.kernel_size_h = 1; // size of the convolving kernel
    conv2d_bean.kernel_size_w = 1; // size of the convolving kernel
    conv2d_bean.stride = 1;
    conv2d_bean.dilation = 1;
    conv2d_bean.bias_en = 0; // whether add bias when calculating conv
    conv2d_bean.softmax = 0; // whether calculate softmax after calculating conv and activation function
    conv2d_bean.mac_8bit = 0; // non_zero: 8bit mode; zero: 16bit mode
    conv2d_bean.pad_u = 0; // zero-pad added to up    sides of the input
    conv2d_bean.pad_d = 0; // zero-pad added to down  sides of the input
    conv2d_bean.pad_l = 0; // zero-pad added to left  sides of the input
    conv2d_bean.pad_r = 0; // zero-pad added to right sides of the input
    conv2d_bean.input_signed = 1; // whether input is signed
    conv2d_bean.weight_bias_signed = 1; // whether weight and bias are signed
    conv2d_bean.filter_lsb_channelwise = 0; // whether filter lsb differ from channels
    conv2d_bean.acc_out_shift = 1; // the right shift bits of the output of acc array, it should be input_fraction + weight_fraction - output_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.bias_shift = 0; // the left shift bits of bias when being added to the acc, it should be input_fraction + weight_fraction - bias_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.leaky_param = 0; // the multiplier of leaky relu, the LSB is 2^(-6)
    conv2d_bean.input_iram = 0; // nonzero - read input from iram, 0 - read input from ddr
    conv2d_bean.output_iram = 0; // nonzero - put output into iram, 0 - put output into ddr
    conv2d_bean.in_sep_mode = 0; // whether read input from iram as separable conv mode
    conv2d_bean.out_sep_mode = 0; // whether put output into iram as separable conv mode
    conv2d_bean.next_padding_left = 0; // left pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_up = 0; // up pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_type = 0; // only valid when out_sep_mode is not 0, if the next layer is maxpool, please set it to 1, or set it to 0
    conv2d_bean.nonlinearty = none; // the nonlinearty operation after conv, you can choose "none", "sigmoid", "relu", "tanh", "relu6", "leaky_relu"
    cnn_conv2D(0x28400 + feature_addr_offset, 0x5400 + feature_addr_offset, 0x25100 + weight_addr_offset, &conv2d_bean);
    RETURN_JUDGE

    /*
     * this layer end
     */
    // layer index: 17
    // layer name: expanded_conv_5/expand/Conv2D
    // layer type: ['convolution', 'relu6']
    // this layer is completed by using CNN engine from Round 18 to Round 18
    // input tensor names are ['expanded_conv_4/add:0']
    // output tensor names are ['expanded_conv_5/expand/Relu6:0']
    // input tensor occupy [10752] bytes individually
    // output tensor occupy [64512] bytes individually
    // weight occupies 28800 bytes
    // you can also use the code below to complete this layer
    conv2d_bean.in_channel = 48; // how many channels of input
    conv2d_bean.in_height = 7; // height of input
    conv2d_bean.in_width = 6; // output of input
    conv2d_bean.out_height = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_width = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_channel = 288; // how many channels of output, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.group = 1; // how many groups, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.kernel_size_h = 1; // size of the convolving kernel
    conv2d_bean.kernel_size_w = 1; // size of the convolving kernel
    conv2d_bean.stride = 1;
    conv2d_bean.dilation = 1;
    conv2d_bean.bias_en = 1; // whether add bias when calculating conv
    conv2d_bean.softmax = 0; // whether calculate softmax after calculating conv and activation function
    conv2d_bean.mac_8bit = 0; // non_zero: 8bit mode; zero: 16bit mode
    conv2d_bean.pad_u = 0; // zero-pad added to up    sides of the input
    conv2d_bean.pad_d = 0; // zero-pad added to down  sides of the input
    conv2d_bean.pad_l = 0; // zero-pad added to left  sides of the input
    conv2d_bean.pad_r = 0; // zero-pad added to right sides of the input
    conv2d_bean.input_signed = 1; // whether input is signed
    conv2d_bean.weight_bias_signed = 1; // whether weight and bias are signed
    conv2d_bean.filter_lsb_channelwise = 1; // whether filter lsb differ from channels
    conv2d_bean.acc_out_shift = 0; // the right shift bits of the output of acc array, it should be input_fraction + weight_fraction - output_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.bias_shift = 0; // the left shift bits of bias when being added to the acc, it should be input_fraction + weight_fraction - bias_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.leaky_param = 0; // the multiplier of leaky relu, the LSB is 2^(-6)
    conv2d_bean.input_iram = 0; // nonzero - read input from iram, 0 - read input from ddr
    conv2d_bean.output_iram = 0; // nonzero - put output into iram, 0 - put output into ddr
    conv2d_bean.in_sep_mode = 0; // whether read input from iram as separable conv mode
    conv2d_bean.out_sep_mode = 0; // whether put output into iram as separable conv mode
    conv2d_bean.next_padding_left = 0; // left pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_up = 0; // up pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_type = 0; // only valid when out_sep_mode is not 0, if the next layer is maxpool, please set it to 1, or set it to 0
    conv2d_bean.nonlinearty = relu6; // the nonlinearty operation after conv, you can choose "none", "sigmoid", "relu", "tanh", "relu6", "leaky_relu"
    cnn_conv2D(0x5400 + feature_addr_offset, 0xa800 + feature_addr_offset, 0x25100 + weight_addr_offset, &conv2d_bean);
    RETURN_JUDGE

    /*
     * this layer end
     */
    // layer index: 18
    // layer name: expanded_conv_5/depthwise/depthwise
    // layer type: ['convolution', 'relu6']
    // this layer is completed by using CNN engine from Round 19 to Round 19
    // input tensor names are ['expanded_conv_5/expand/Relu6:0']
    // output tensor names are ['expanded_conv_5/depthwise/Relu6:0']
    // input tensor occupy [64512] bytes individually
    // output tensor occupy [64512] bytes individually
    // weight occupies 9216 bytes
    // you can also use the code below to complete this layer
    conv2d_bean.in_channel = 288; // how many channels of input
    conv2d_bean.in_height = 7; // height of input
    conv2d_bean.in_width = 6; // output of input
    conv2d_bean.out_height = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_width = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_channel = 288; // how many channels of output, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.group = 288; // how many groups, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.kernel_size_h = 3; // size of the convolving kernel
    conv2d_bean.kernel_size_w = 3; // size of the convolving kernel
    conv2d_bean.stride = 1;
    conv2d_bean.dilation = 1;
    conv2d_bean.bias_en = 1; // whether add bias when calculating conv
    conv2d_bean.softmax = 0; // whether calculate softmax after calculating conv and activation function
    conv2d_bean.mac_8bit = 0; // non_zero: 8bit mode; zero: 16bit mode
    conv2d_bean.pad_u = 1; // zero-pad added to up    sides of the input
    conv2d_bean.pad_d = 1; // zero-pad added to down  sides of the input
    conv2d_bean.pad_l = 1; // zero-pad added to left  sides of the input
    conv2d_bean.pad_r = 1; // zero-pad added to right sides of the input
    conv2d_bean.input_signed = 1; // whether input is signed
    conv2d_bean.weight_bias_signed = 1; // whether weight and bias are signed
    conv2d_bean.filter_lsb_channelwise = 1; // whether filter lsb differ from channels
    conv2d_bean.acc_out_shift = 0; // the right shift bits of the output of acc array, it should be input_fraction + weight_fraction - output_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.bias_shift = 0; // the left shift bits of bias when being added to the acc, it should be input_fraction + weight_fraction - bias_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.leaky_param = 0; // the multiplier of leaky relu, the LSB is 2^(-6)
    conv2d_bean.input_iram = 0; // nonzero - read input from iram, 0 - read input from ddr
    conv2d_bean.output_iram = 0; // nonzero - put output into iram, 0 - put output into ddr
    conv2d_bean.in_sep_mode = 0; // whether read input from iram as separable conv mode
    conv2d_bean.out_sep_mode = 0; // whether put output into iram as separable conv mode
    conv2d_bean.next_padding_left = 0; // left pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_up = 0; // up pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_type = 0; // only valid when out_sep_mode is not 0, if the next layer is maxpool, please set it to 1, or set it to 0
    conv2d_bean.nonlinearty = relu6; // the nonlinearty operation after conv, you can choose "none", "sigmoid", "relu", "tanh", "relu6", "leaky_relu"
    cnn_conv2D(0xa800 + feature_addr_offset, 0x1a400 + feature_addr_offset, 0x2c200 + weight_addr_offset, &conv2d_bean);
    RETURN_JUDGE

    /*
     * this layer end
     */
    // layer index: 19
    // layer name: expanded_conv_5/project/Conv2D
    // layer type: ['convolution']
    // this layer is completed by using CNN engine from Round 20 to Round 20
    // input tensor names are ['expanded_conv_5/depthwise/Relu6:0']
    // output tensor names are ['expanded_conv_5/project/BatchNorm/FusedBatchNorm:0']
    // input tensor occupy [64512] bytes individually
    // output tensor occupy [10752] bytes individually
    // weight occupies 27840 bytes
    // you can also use the code below to complete this layer
    conv2d_bean.in_channel = 288; // how many channels of input
    conv2d_bean.in_height = 7; // height of input
    conv2d_bean.in_width = 6; // output of input
    conv2d_bean.out_height = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_width = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_channel = 48; // how many channels of output, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.group = 1; // how many groups, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.kernel_size_h = 1; // size of the convolving kernel
    conv2d_bean.kernel_size_w = 1; // size of the convolving kernel
    conv2d_bean.stride = 1;
    conv2d_bean.dilation = 1;
    conv2d_bean.bias_en = 1; // whether add bias when calculating conv
    conv2d_bean.softmax = 0; // whether calculate softmax after calculating conv and activation function
    conv2d_bean.mac_8bit = 0; // non_zero: 8bit mode; zero: 16bit mode
    conv2d_bean.pad_u = 0; // zero-pad added to up    sides of the input
    conv2d_bean.pad_d = 0; // zero-pad added to down  sides of the input
    conv2d_bean.pad_l = 0; // zero-pad added to left  sides of the input
    conv2d_bean.pad_r = 0; // zero-pad added to right sides of the input
    conv2d_bean.input_signed = 1; // whether input is signed
    conv2d_bean.weight_bias_signed = 1; // whether weight and bias are signed
    conv2d_bean.filter_lsb_channelwise = 1; // whether filter lsb differ from channels
    conv2d_bean.acc_out_shift = 0; // the right shift bits of the output of acc array, it should be input_fraction + weight_fraction - output_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.bias_shift = 0; // the left shift bits of bias when being added to the acc, it should be input_fraction + weight_fraction - bias_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.leaky_param = 0; // the multiplier of leaky relu, the LSB is 2^(-6)
    conv2d_bean.input_iram = 0; // nonzero - read input from iram, 0 - read input from ddr
    conv2d_bean.output_iram = 0; // nonzero - put output into iram, 0 - put output into ddr
    conv2d_bean.in_sep_mode = 0; // whether read input from iram as separable conv mode
    conv2d_bean.out_sep_mode = 0; // whether put output into iram as separable conv mode
    conv2d_bean.next_padding_left = 0; // left pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_up = 0; // up pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_type = 0; // only valid when out_sep_mode is not 0, if the next layer is maxpool, please set it to 1, or set it to 0
    conv2d_bean.nonlinearty = none; // the nonlinearty operation after conv, you can choose "none", "sigmoid", "relu", "tanh", "relu6", "leaky_relu"
    cnn_conv2D(0x1a400 + feature_addr_offset, 0x7e00 + feature_addr_offset, 0x2e600 + weight_addr_offset, &conv2d_bean);
    RETURN_JUDGE

    /*
     * this layer end
     */
    // layer index: 20
    // layer name: expanded_conv_5/add
    // layer type: ['eltwise']
    // this layer is completed by using CNN engine from Round 21 to Round 21
    // input tensor names are ['expanded_conv_4/add:0', 'expanded_conv_5/project/BatchNorm/FusedBatchNorm:0']
    // output tensor names are ['expanded_conv_5/add:0']
    // input tensor occupy [10752, 10752] bytes individually
    // output tensor occupy [10752] bytes individually
    // weight occupies 0 bytes
    // you can also use the code below to complete this layer
    conv2d_bean.in_channel = 2; // how many channels of input
    conv2d_bean.in_height = 336; // height of input
    conv2d_bean.in_width = 6; // output of input
    conv2d_bean.out_height = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_width = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_channel = 1; // how many channels of output, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.group = 1; // how many groups, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.kernel_size_h = 1; // size of the convolving kernel
    conv2d_bean.kernel_size_w = 1; // size of the convolving kernel
    conv2d_bean.stride = 1;
    conv2d_bean.dilation = 1;
    conv2d_bean.bias_en = 0; // whether add bias when calculating conv
    conv2d_bean.softmax = 0; // whether calculate softmax after calculating conv and activation function
    conv2d_bean.mac_8bit = 0; // non_zero: 8bit mode; zero: 16bit mode
    conv2d_bean.pad_u = 0; // zero-pad added to up    sides of the input
    conv2d_bean.pad_d = 0; // zero-pad added to down  sides of the input
    conv2d_bean.pad_l = 0; // zero-pad added to left  sides of the input
    conv2d_bean.pad_r = 0; // zero-pad added to right sides of the input
    conv2d_bean.input_signed = 1; // whether input is signed
    conv2d_bean.weight_bias_signed = 1; // whether weight and bias are signed
    conv2d_bean.filter_lsb_channelwise = 0; // whether filter lsb differ from channels
    conv2d_bean.acc_out_shift = 0; // the right shift bits of the output of acc array, it should be input_fraction + weight_fraction - output_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.bias_shift = 0; // the left shift bits of bias when being added to the acc, it should be input_fraction + weight_fraction - bias_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.leaky_param = 0; // the multiplier of leaky relu, the LSB is 2^(-6)
    conv2d_bean.input_iram = 0; // nonzero - read input from iram, 0 - read input from ddr
    conv2d_bean.output_iram = 0; // nonzero - put output into iram, 0 - put output into ddr
    conv2d_bean.in_sep_mode = 0; // whether read input from iram as separable conv mode
    conv2d_bean.out_sep_mode = 0; // whether put output into iram as separable conv mode
    conv2d_bean.next_padding_left = 0; // left pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_up = 0; // up pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_type = 0; // only valid when out_sep_mode is not 0, if the next layer is maxpool, please set it to 1, or set it to 0
    conv2d_bean.nonlinearty = none; // the nonlinearty operation after conv, you can choose "none", "sigmoid", "relu", "tanh", "relu6", "leaky_relu"
    cnn_conv2D(0x5400 + feature_addr_offset, 0xa800 + feature_addr_offset, 0x35300 + weight_addr_offset, &conv2d_bean);
    RETURN_JUDGE

    /*
     * this layer end
     */
    // layer index: 21
    // layer name: expanded_conv_6/expand/Conv2D
    // layer type: ['convolution', 'relu6']
    // this layer is completed by using CNN engine from Round 22 to Round 22
    // input tensor names are ['expanded_conv_5/add:0']
    // output tensor names are ['expanded_conv_6/expand/Relu6:0']
    // input tensor occupy [10752] bytes individually
    // output tensor occupy [64512] bytes individually
    // weight occupies 28800 bytes
    // you can also use the code below to complete this layer
    conv2d_bean.in_channel = 48; // how many channels of input
    conv2d_bean.in_height = 7; // height of input
    conv2d_bean.in_width = 6; // output of input
    conv2d_bean.out_height = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_width = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_channel = 288; // how many channels of output, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.group = 1; // how many groups, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.kernel_size_h = 1; // size of the convolving kernel
    conv2d_bean.kernel_size_w = 1; // size of the convolving kernel
    conv2d_bean.stride = 1;
    conv2d_bean.dilation = 1;
    conv2d_bean.bias_en = 1; // whether add bias when calculating conv
    conv2d_bean.softmax = 0; // whether calculate softmax after calculating conv and activation function
    conv2d_bean.mac_8bit = 0; // non_zero: 8bit mode; zero: 16bit mode
    conv2d_bean.pad_u = 0; // zero-pad added to up    sides of the input
    conv2d_bean.pad_d = 0; // zero-pad added to down  sides of the input
    conv2d_bean.pad_l = 0; // zero-pad added to left  sides of the input
    conv2d_bean.pad_r = 0; // zero-pad added to right sides of the input
    conv2d_bean.input_signed = 1; // whether input is signed
    conv2d_bean.weight_bias_signed = 1; // whether weight and bias are signed
    conv2d_bean.filter_lsb_channelwise = 1; // whether filter lsb differ from channels
    conv2d_bean.acc_out_shift = 0; // the right shift bits of the output of acc array, it should be input_fraction + weight_fraction - output_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.bias_shift = 0; // the left shift bits of bias when being added to the acc, it should be input_fraction + weight_fraction - bias_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.leaky_param = 0; // the multiplier of leaky relu, the LSB is 2^(-6)
    conv2d_bean.input_iram = 0; // nonzero - read input from iram, 0 - read input from ddr
    conv2d_bean.output_iram = 0; // nonzero - put output into iram, 0 - put output into ddr
    conv2d_bean.in_sep_mode = 0; // whether read input from iram as separable conv mode
    conv2d_bean.out_sep_mode = 0; // whether put output into iram as separable conv mode
    conv2d_bean.next_padding_left = 0; // left pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_up = 0; // up pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_type = 0; // only valid when out_sep_mode is not 0, if the next layer is maxpool, please set it to 1, or set it to 0
    conv2d_bean.nonlinearty = relu6; // the nonlinearty operation after conv, you can choose "none", "sigmoid", "relu", "tanh", "relu6", "leaky_relu"
    cnn_conv2D(0xa800 + feature_addr_offset, 0xfc00 + feature_addr_offset, 0x35300 + weight_addr_offset, &conv2d_bean);
    RETURN_JUDGE

    /*
     * this layer end
     */
    // layer index: 22
    // layer name: expanded_conv_6/depthwise/depthwise
    // layer type: ['convolution', 'relu6']
    // this layer is completed by using CNN engine from Round 23 to Round 23
    // input tensor names are ['expanded_conv_6/expand/Relu6:0']
    // output tensor names are ['expanded_conv_6/depthwise/Relu6:0']
    // input tensor occupy [64512] bytes individually
    // output tensor occupy [64512] bytes individually
    // weight occupies 9216 bytes
    // you can also use the code below to complete this layer
    conv2d_bean.in_channel = 288; // how many channels of input
    conv2d_bean.in_height = 7; // height of input
    conv2d_bean.in_width = 6; // output of input
    conv2d_bean.out_height = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_width = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_channel = 288; // how many channels of output, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.group = 288; // how many groups, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.kernel_size_h = 3; // size of the convolving kernel
    conv2d_bean.kernel_size_w = 3; // size of the convolving kernel
    conv2d_bean.stride = 1;
    conv2d_bean.dilation = 1;
    conv2d_bean.bias_en = 1; // whether add bias when calculating conv
    conv2d_bean.softmax = 0; // whether calculate softmax after calculating conv and activation function
    conv2d_bean.mac_8bit = 0; // non_zero: 8bit mode; zero: 16bit mode
    conv2d_bean.pad_u = 1; // zero-pad added to up    sides of the input
    conv2d_bean.pad_d = 1; // zero-pad added to down  sides of the input
    conv2d_bean.pad_l = 1; // zero-pad added to left  sides of the input
    conv2d_bean.pad_r = 1; // zero-pad added to right sides of the input
    conv2d_bean.input_signed = 1; // whether input is signed
    conv2d_bean.weight_bias_signed = 1; // whether weight and bias are signed
    conv2d_bean.filter_lsb_channelwise = 1; // whether filter lsb differ from channels
    conv2d_bean.acc_out_shift = 0; // the right shift bits of the output of acc array, it should be input_fraction + weight_fraction - output_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.bias_shift = 0; // the left shift bits of bias when being added to the acc, it should be input_fraction + weight_fraction - bias_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.leaky_param = 0; // the multiplier of leaky relu, the LSB is 2^(-6)
    conv2d_bean.input_iram = 0; // nonzero - read input from iram, 0 - read input from ddr
    conv2d_bean.output_iram = 0; // nonzero - put output into iram, 0 - put output into ddr
    conv2d_bean.in_sep_mode = 0; // whether read input from iram as separable conv mode
    conv2d_bean.out_sep_mode = 0; // whether put output into iram as separable conv mode
    conv2d_bean.next_padding_left = 0; // left pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_up = 0; // up pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_type = 0; // only valid when out_sep_mode is not 0, if the next layer is maxpool, please set it to 1, or set it to 0
    conv2d_bean.nonlinearty = relu6; // the nonlinearty operation after conv, you can choose "none", "sigmoid", "relu", "tanh", "relu6", "leaky_relu"
    cnn_conv2D(0xfc00 + feature_addr_offset, 0x1f800 + feature_addr_offset, 0x3c400 + weight_addr_offset, &conv2d_bean);
    RETURN_JUDGE

    /*
     * this layer end
     */
    // layer index: 23
    // layer name: expanded_conv_6/project/Conv2D
    // layer type: ['convolution']
    // this layer is completed by using CNN engine from Round 24 to Round 24
    // input tensor names are ['expanded_conv_6/depthwise/Relu6:0']
    // output tensor names are ['expanded_conv_6/project/BatchNorm/FusedBatchNorm:0']
    // input tensor occupy [64512] bytes individually
    // output tensor occupy [10752] bytes individually
    // weight occupies 27840 bytes
    // you can also use the code below to complete this layer
    conv2d_bean.in_channel = 288; // how many channels of input
    conv2d_bean.in_height = 7; // height of input
    conv2d_bean.in_width = 6; // output of input
    conv2d_bean.out_height = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_width = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_channel = 48; // how many channels of output, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.group = 1; // how many groups, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.kernel_size_h = 1; // size of the convolving kernel
    conv2d_bean.kernel_size_w = 1; // size of the convolving kernel
    conv2d_bean.stride = 1;
    conv2d_bean.dilation = 1;
    conv2d_bean.bias_en = 1; // whether add bias when calculating conv
    conv2d_bean.softmax = 0; // whether calculate softmax after calculating conv and activation function
    conv2d_bean.mac_8bit = 0; // non_zero: 8bit mode; zero: 16bit mode
    conv2d_bean.pad_u = 0; // zero-pad added to up    sides of the input
    conv2d_bean.pad_d = 0; // zero-pad added to down  sides of the input
    conv2d_bean.pad_l = 0; // zero-pad added to left  sides of the input
    conv2d_bean.pad_r = 0; // zero-pad added to right sides of the input
    conv2d_bean.input_signed = 1; // whether input is signed
    conv2d_bean.weight_bias_signed = 1; // whether weight and bias are signed
    conv2d_bean.filter_lsb_channelwise = 1; // whether filter lsb differ from channels
    conv2d_bean.acc_out_shift = 0; // the right shift bits of the output of acc array, it should be input_fraction + weight_fraction - output_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.bias_shift = 0; // the left shift bits of bias when being added to the acc, it should be input_fraction + weight_fraction - bias_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.leaky_param = 0; // the multiplier of leaky relu, the LSB is 2^(-6)
    conv2d_bean.input_iram = 0; // nonzero - read input from iram, 0 - read input from ddr
    conv2d_bean.output_iram = 0; // nonzero - put output into iram, 0 - put output into ddr
    conv2d_bean.in_sep_mode = 0; // whether read input from iram as separable conv mode
    conv2d_bean.out_sep_mode = 0; // whether put output into iram as separable conv mode
    conv2d_bean.next_padding_left = 0; // left pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_up = 0; // up pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_type = 0; // only valid when out_sep_mode is not 0, if the next layer is maxpool, please set it to 1, or set it to 0
    conv2d_bean.nonlinearty = none; // the nonlinearty operation after conv, you can choose "none", "sigmoid", "relu", "tanh", "relu6", "leaky_relu"
    cnn_conv2D(0x1f800 + feature_addr_offset, 0xd200 + feature_addr_offset, 0x3e800 + weight_addr_offset, &conv2d_bean);
    RETURN_JUDGE

    /*
     * this layer end
     */
    // layer index: 24
    // layer name: expanded_conv_6/add
    // layer type: ['eltwise']
    // this layer is completed by using CNN engine from Round 25 to Round 25
    // input tensor names are ['expanded_conv_5/add:0', 'expanded_conv_6/project/BatchNorm/FusedBatchNorm:0']
    // output tensor names are ['expanded_conv_6/add:0']
    // input tensor occupy [10752, 10752] bytes individually
    // output tensor occupy [10752] bytes individually
    // weight occupies 0 bytes
    // you can also use the code below to complete this layer
    conv2d_bean.in_channel = 2; // how many channels of input
    conv2d_bean.in_height = 336; // height of input
    conv2d_bean.in_width = 6; // output of input
    conv2d_bean.out_height = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_width = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_channel = 1; // how many channels of output, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.group = 1; // how many groups, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.kernel_size_h = 1; // size of the convolving kernel
    conv2d_bean.kernel_size_w = 1; // size of the convolving kernel
    conv2d_bean.stride = 1;
    conv2d_bean.dilation = 1;
    conv2d_bean.bias_en = 0; // whether add bias when calculating conv
    conv2d_bean.softmax = 0; // whether calculate softmax after calculating conv and activation function
    conv2d_bean.mac_8bit = 0; // non_zero: 8bit mode; zero: 16bit mode
    conv2d_bean.pad_u = 0; // zero-pad added to up    sides of the input
    conv2d_bean.pad_d = 0; // zero-pad added to down  sides of the input
    conv2d_bean.pad_l = 0; // zero-pad added to left  sides of the input
    conv2d_bean.pad_r = 0; // zero-pad added to right sides of the input
    conv2d_bean.input_signed = 1; // whether input is signed
    conv2d_bean.weight_bias_signed = 1; // whether weight and bias are signed
    conv2d_bean.filter_lsb_channelwise = 0; // whether filter lsb differ from channels
    conv2d_bean.acc_out_shift = 1; // the right shift bits of the output of acc array, it should be input_fraction + weight_fraction - output_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.bias_shift = 0; // the left shift bits of bias when being added to the acc, it should be input_fraction + weight_fraction - bias_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.leaky_param = 0; // the multiplier of leaky relu, the LSB is 2^(-6)
    conv2d_bean.input_iram = 0; // nonzero - read input from iram, 0 - read input from ddr
    conv2d_bean.output_iram = 0; // nonzero - put output into iram, 0 - put output into ddr
    conv2d_bean.in_sep_mode = 0; // whether read input from iram as separable conv mode
    conv2d_bean.out_sep_mode = 0; // whether put output into iram as separable conv mode
    conv2d_bean.next_padding_left = 0; // left pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_up = 0; // up pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_type = 0; // only valid when out_sep_mode is not 0, if the next layer is maxpool, please set it to 1, or set it to 0
    conv2d_bean.nonlinearty = none; // the nonlinearty operation after conv, you can choose "none", "sigmoid", "relu", "tanh", "relu6", "leaky_relu"
    cnn_conv2D(0xa800 + feature_addr_offset, 0x5400 + feature_addr_offset, 0x45500 + weight_addr_offset, &conv2d_bean);
    RETURN_JUDGE

    /*
     * this layer end
     */
    // layer index: 25
    // layer name: expanded_conv_7/expand/Conv2D
    // layer type: ['convolution', 'relu6']
    // this layer is completed by using CNN engine from Round 26 to Round 26
    // input tensor names are ['expanded_conv_6/add:0']
    // output tensor names are ['expanded_conv_7/expand/Relu6:0']
    // input tensor occupy [10752] bytes individually
    // output tensor occupy [64512] bytes individually
    // weight occupies 28800 bytes
    // you can also use the code below to complete this layer
    conv2d_bean.in_channel = 48; // how many channels of input
    conv2d_bean.in_height = 7; // height of input
    conv2d_bean.in_width = 6; // output of input
    conv2d_bean.out_height = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_width = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_channel = 288; // how many channels of output, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.group = 1; // how many groups, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.kernel_size_h = 1; // size of the convolving kernel
    conv2d_bean.kernel_size_w = 1; // size of the convolving kernel
    conv2d_bean.stride = 1;
    conv2d_bean.dilation = 1;
    conv2d_bean.bias_en = 1; // whether add bias when calculating conv
    conv2d_bean.softmax = 0; // whether calculate softmax after calculating conv and activation function
    conv2d_bean.mac_8bit = 0; // non_zero: 8bit mode; zero: 16bit mode
    conv2d_bean.pad_u = 0; // zero-pad added to up    sides of the input
    conv2d_bean.pad_d = 0; // zero-pad added to down  sides of the input
    conv2d_bean.pad_l = 0; // zero-pad added to left  sides of the input
    conv2d_bean.pad_r = 0; // zero-pad added to right sides of the input
    conv2d_bean.input_signed = 1; // whether input is signed
    conv2d_bean.weight_bias_signed = 1; // whether weight and bias are signed
    conv2d_bean.filter_lsb_channelwise = 1; // whether filter lsb differ from channels
    conv2d_bean.acc_out_shift = 0; // the right shift bits of the output of acc array, it should be input_fraction + weight_fraction - output_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.bias_shift = 0; // the left shift bits of bias when being added to the acc, it should be input_fraction + weight_fraction - bias_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.leaky_param = 0; // the multiplier of leaky relu, the LSB is 2^(-6)
    conv2d_bean.input_iram = 0; // nonzero - read input from iram, 0 - read input from ddr
    conv2d_bean.output_iram = 0; // nonzero - put output into iram, 0 - put output into ddr
    conv2d_bean.in_sep_mode = 0; // whether read input from iram as separable conv mode
    conv2d_bean.out_sep_mode = 0; // whether put output into iram as separable conv mode
    conv2d_bean.next_padding_left = 0; // left pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_up = 0; // up pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_type = 0; // only valid when out_sep_mode is not 0, if the next layer is maxpool, please set it to 1, or set it to 0
    conv2d_bean.nonlinearty = relu6; // the nonlinearty operation after conv, you can choose "none", "sigmoid", "relu", "tanh", "relu6", "leaky_relu"
    cnn_conv2D(0x5400 + feature_addr_offset, 0xfc00 + feature_addr_offset, 0x45500 + weight_addr_offset, &conv2d_bean);
    RETURN_JUDGE

    /*
     * this layer end
     */
    // layer index: 26
    // layer name: expanded_conv_7/depthwise/depthwise
    // layer type: ['convolution', 'relu6']
    // this layer is completed by using CNN engine from Round 27 to Round 27
    // input tensor names are ['expanded_conv_7/expand/Relu6:0']
    // output tensor names are ['expanded_conv_7/depthwise/Relu6:0']
    // input tensor occupy [64512] bytes individually
    // output tensor occupy [36864] bytes individually
    // weight occupies 9216 bytes
    // you can also use the code below to complete this layer
    conv2d_bean.in_channel = 288; // how many channels of input
    conv2d_bean.in_height = 7; // height of input
    conv2d_bean.in_width = 6; // output of input
    conv2d_bean.out_height = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_width = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_channel = 288; // how many channels of output, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.group = 288; // how many groups, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.kernel_size_h = 3; // size of the convolving kernel
    conv2d_bean.kernel_size_w = 3; // size of the convolving kernel
    conv2d_bean.stride = 2;
    conv2d_bean.dilation = 1;
    conv2d_bean.bias_en = 1; // whether add bias when calculating conv
    conv2d_bean.softmax = 0; // whether calculate softmax after calculating conv and activation function
    conv2d_bean.mac_8bit = 0; // non_zero: 8bit mode; zero: 16bit mode
    conv2d_bean.pad_u = 1; // zero-pad added to up    sides of the input
    conv2d_bean.pad_d = 1; // zero-pad added to down  sides of the input
    conv2d_bean.pad_l = 0; // zero-pad added to left  sides of the input
    conv2d_bean.pad_r = 1; // zero-pad added to right sides of the input
    conv2d_bean.input_signed = 1; // whether input is signed
    conv2d_bean.weight_bias_signed = 1; // whether weight and bias are signed
    conv2d_bean.filter_lsb_channelwise = 1; // whether filter lsb differ from channels
    conv2d_bean.acc_out_shift = 0; // the right shift bits of the output of acc array, it should be input_fraction + weight_fraction - output_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.bias_shift = 0; // the left shift bits of bias when being added to the acc, it should be input_fraction + weight_fraction - bias_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.leaky_param = 0; // the multiplier of leaky relu, the LSB is 2^(-6)
    conv2d_bean.input_iram = 0; // nonzero - read input from iram, 0 - read input from ddr
    conv2d_bean.output_iram = 0; // nonzero - put output into iram, 0 - put output into ddr
    conv2d_bean.in_sep_mode = 0; // whether read input from iram as separable conv mode
    conv2d_bean.out_sep_mode = 0; // whether put output into iram as separable conv mode
    conv2d_bean.next_padding_left = 0; // left pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_up = 0; // up pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_type = 0; // only valid when out_sep_mode is not 0, if the next layer is maxpool, please set it to 1, or set it to 0
    conv2d_bean.nonlinearty = relu6; // the nonlinearty operation after conv, you can choose "none", "sigmoid", "relu", "tanh", "relu6", "leaky_relu"
    cnn_conv2D(0xfc00 + feature_addr_offset, 0x1f800 + feature_addr_offset, 0x4c600 + weight_addr_offset, &conv2d_bean);
    RETURN_JUDGE

    /*
     * this layer end
     */
    // layer index: 27
    // layer name: expanded_conv_7/project/Conv2D
    // layer type: ['convolution']
    // this layer is completed by using CNN engine from Round 28 to Round 28
    // input tensor names are ['expanded_conv_7/depthwise/Relu6:0']
    // output tensor names are ['expanded_conv_7/project/BatchNorm/FusedBatchNorm:0']
    // input tensor occupy [36864] bytes individually
    // output tensor occupy [8192] bytes individually
    // weight occupies 37120 bytes
    // you can also use the code below to complete this layer
    conv2d_bean.in_channel = 288; // how many channels of input
    conv2d_bean.in_height = 4; // height of input
    conv2d_bean.in_width = 3; // output of input
    conv2d_bean.out_height = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_width = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_channel = 64; // how many channels of output, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.group = 1; // how many groups, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.kernel_size_h = 1; // size of the convolving kernel
    conv2d_bean.kernel_size_w = 1; // size of the convolving kernel
    conv2d_bean.stride = 1;
    conv2d_bean.dilation = 1;
    conv2d_bean.bias_en = 1; // whether add bias when calculating conv
    conv2d_bean.softmax = 0; // whether calculate softmax after calculating conv and activation function
    conv2d_bean.mac_8bit = 0; // non_zero: 8bit mode; zero: 16bit mode
    conv2d_bean.pad_u = 0; // zero-pad added to up    sides of the input
    conv2d_bean.pad_d = 0; // zero-pad added to down  sides of the input
    conv2d_bean.pad_l = 0; // zero-pad added to left  sides of the input
    conv2d_bean.pad_r = 0; // zero-pad added to right sides of the input
    conv2d_bean.input_signed = 1; // whether input is signed
    conv2d_bean.weight_bias_signed = 1; // whether weight and bias are signed
    conv2d_bean.filter_lsb_channelwise = 1; // whether filter lsb differ from channels
    conv2d_bean.acc_out_shift = 0; // the right shift bits of the output of acc array, it should be input_fraction + weight_fraction - output_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.bias_shift = 0; // the left shift bits of bias when being added to the acc, it should be input_fraction + weight_fraction - bias_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.leaky_param = 0; // the multiplier of leaky relu, the LSB is 2^(-6)
    conv2d_bean.input_iram = 0; // nonzero - read input from iram, 0 - read input from ddr
    conv2d_bean.output_iram = 0; // nonzero - put output into iram, 0 - put output into ddr
    conv2d_bean.in_sep_mode = 0; // whether read input from iram as separable conv mode
    conv2d_bean.out_sep_mode = 0; // whether put output into iram as separable conv mode
    conv2d_bean.next_padding_left = 0; // left pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_up = 0; // up pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_type = 0; // only valid when out_sep_mode is not 0, if the next layer is maxpool, please set it to 1, or set it to 0
    conv2d_bean.nonlinearty = none; // the nonlinearty operation after conv, you can choose "none", "sigmoid", "relu", "tanh", "relu6", "leaky_relu"
    cnn_conv2D(0x1f800 + feature_addr_offset, 0x5400 + feature_addr_offset, 0x4ea00 + weight_addr_offset, &conv2d_bean);
    RETURN_JUDGE

    /*
     * this layer end
     */
    // layer index: 28
    // layer name: expanded_conv_8/expand/Conv2D
    // layer type: ['convolution', 'relu6']
    // this layer is completed by using CNN engine from Round 29 to Round 29
    // input tensor names are ['expanded_conv_7/project/BatchNorm/FusedBatchNorm:0']
    // output tensor names are ['expanded_conv_8/expand/Relu6:0']
    // input tensor occupy [8192] bytes individually
    // output tensor occupy [49152] bytes individually
    // weight occupies 50688 bytes
    // you can also use the code below to complete this layer
    conv2d_bean.in_channel = 64; // how many channels of input
    conv2d_bean.in_height = 4; // height of input
    conv2d_bean.in_width = 3; // output of input
    conv2d_bean.out_height = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_width = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_channel = 384; // how many channels of output, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.group = 1; // how many groups, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.kernel_size_h = 1; // size of the convolving kernel
    conv2d_bean.kernel_size_w = 1; // size of the convolving kernel
    conv2d_bean.stride = 1;
    conv2d_bean.dilation = 1;
    conv2d_bean.bias_en = 1; // whether add bias when calculating conv
    conv2d_bean.softmax = 0; // whether calculate softmax after calculating conv and activation function
    conv2d_bean.mac_8bit = 0; // non_zero: 8bit mode; zero: 16bit mode
    conv2d_bean.pad_u = 0; // zero-pad added to up    sides of the input
    conv2d_bean.pad_d = 0; // zero-pad added to down  sides of the input
    conv2d_bean.pad_l = 0; // zero-pad added to left  sides of the input
    conv2d_bean.pad_r = 0; // zero-pad added to right sides of the input
    conv2d_bean.input_signed = 1; // whether input is signed
    conv2d_bean.weight_bias_signed = 1; // whether weight and bias are signed
    conv2d_bean.filter_lsb_channelwise = 1; // whether filter lsb differ from channels
    conv2d_bean.acc_out_shift = 0; // the right shift bits of the output of acc array, it should be input_fraction + weight_fraction - output_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.bias_shift = 0; // the left shift bits of bias when being added to the acc, it should be input_fraction + weight_fraction - bias_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.leaky_param = 0; // the multiplier of leaky relu, the LSB is 2^(-6)
    conv2d_bean.input_iram = 0; // nonzero - read input from iram, 0 - read input from ddr
    conv2d_bean.output_iram = 0; // nonzero - put output into iram, 0 - put output into ddr
    conv2d_bean.in_sep_mode = 0; // whether read input from iram as separable conv mode
    conv2d_bean.out_sep_mode = 0; // whether put output into iram as separable conv mode
    conv2d_bean.next_padding_left = 0; // left pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_up = 0; // up pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_type = 0; // only valid when out_sep_mode is not 0, if the next layer is maxpool, please set it to 1, or set it to 0
    conv2d_bean.nonlinearty = relu6; // the nonlinearty operation after conv, you can choose "none", "sigmoid", "relu", "tanh", "relu6", "leaky_relu"
    cnn_conv2D(0x5400 + feature_addr_offset, 0x9400 + feature_addr_offset, 0x57b00 + weight_addr_offset, &conv2d_bean);
    RETURN_JUDGE

    /*
     * this layer end
     */
    // layer index: 29
    // layer name: expanded_conv_8/depthwise/depthwise
    // layer type: ['convolution', 'relu6']
    // this layer is completed by using CNN engine from Round 30 to Round 30
    // input tensor names are ['expanded_conv_8/expand/Relu6:0']
    // output tensor names are ['expanded_conv_8/depthwise/Relu6:0']
    // input tensor occupy [49152] bytes individually
    // output tensor occupy [49152] bytes individually
    // weight occupies 12288 bytes
    // you can also use the code below to complete this layer
    conv2d_bean.in_channel = 384; // how many channels of input
    conv2d_bean.in_height = 4; // height of input
    conv2d_bean.in_width = 3; // output of input
    conv2d_bean.out_height = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_width = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_channel = 384; // how many channels of output, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.group = 384; // how many groups, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.kernel_size_h = 3; // size of the convolving kernel
    conv2d_bean.kernel_size_w = 3; // size of the convolving kernel
    conv2d_bean.stride = 1;
    conv2d_bean.dilation = 1;
    conv2d_bean.bias_en = 1; // whether add bias when calculating conv
    conv2d_bean.softmax = 0; // whether calculate softmax after calculating conv and activation function
    conv2d_bean.mac_8bit = 0; // non_zero: 8bit mode; zero: 16bit mode
    conv2d_bean.pad_u = 1; // zero-pad added to up    sides of the input
    conv2d_bean.pad_d = 1; // zero-pad added to down  sides of the input
    conv2d_bean.pad_l = 1; // zero-pad added to left  sides of the input
    conv2d_bean.pad_r = 1; // zero-pad added to right sides of the input
    conv2d_bean.input_signed = 1; // whether input is signed
    conv2d_bean.weight_bias_signed = 1; // whether weight and bias are signed
    conv2d_bean.filter_lsb_channelwise = 1; // whether filter lsb differ from channels
    conv2d_bean.acc_out_shift = 0; // the right shift bits of the output of acc array, it should be input_fraction + weight_fraction - output_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.bias_shift = 0; // the left shift bits of bias when being added to the acc, it should be input_fraction + weight_fraction - bias_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.leaky_param = 0; // the multiplier of leaky relu, the LSB is 2^(-6)
    conv2d_bean.input_iram = 0; // nonzero - read input from iram, 0 - read input from ddr
    conv2d_bean.output_iram = 0; // nonzero - put output into iram, 0 - put output into ddr
    conv2d_bean.in_sep_mode = 0; // whether read input from iram as separable conv mode
    conv2d_bean.out_sep_mode = 0; // whether put output into iram as separable conv mode
    conv2d_bean.next_padding_left = 0; // left pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_up = 0; // up pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_type = 0; // only valid when out_sep_mode is not 0, if the next layer is maxpool, please set it to 1, or set it to 0
    conv2d_bean.nonlinearty = relu6; // the nonlinearty operation after conv, you can choose "none", "sigmoid", "relu", "tanh", "relu6", "leaky_relu"
    cnn_conv2D(0x9400 + feature_addr_offset, 0x15400 + feature_addr_offset, 0x64100 + weight_addr_offset, &conv2d_bean);
    RETURN_JUDGE

    /*
     * this layer end
     */
    // layer index: 30
    // layer name: expanded_conv_8/project/Conv2D
    // layer type: ['convolution']
    // this layer is completed by using CNN engine from Round 31 to Round 31
    // input tensor names are ['expanded_conv_8/depthwise/Relu6:0']
    // output tensor names are ['expanded_conv_8/project/BatchNorm/FusedBatchNorm:0']
    // input tensor occupy [49152] bytes individually
    // output tensor occupy [8192] bytes individually
    // weight occupies 49408 bytes
    // you can also use the code below to complete this layer
    conv2d_bean.in_channel = 384; // how many channels of input
    conv2d_bean.in_height = 4; // height of input
    conv2d_bean.in_width = 3; // output of input
    conv2d_bean.out_height = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_width = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_channel = 64; // how many channels of output, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.group = 1; // how many groups, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.kernel_size_h = 1; // size of the convolving kernel
    conv2d_bean.kernel_size_w = 1; // size of the convolving kernel
    conv2d_bean.stride = 1;
    conv2d_bean.dilation = 1;
    conv2d_bean.bias_en = 1; // whether add bias when calculating conv
    conv2d_bean.softmax = 0; // whether calculate softmax after calculating conv and activation function
    conv2d_bean.mac_8bit = 0; // non_zero: 8bit mode; zero: 16bit mode
    conv2d_bean.pad_u = 0; // zero-pad added to up    sides of the input
    conv2d_bean.pad_d = 0; // zero-pad added to down  sides of the input
    conv2d_bean.pad_l = 0; // zero-pad added to left  sides of the input
    conv2d_bean.pad_r = 0; // zero-pad added to right sides of the input
    conv2d_bean.input_signed = 1; // whether input is signed
    conv2d_bean.weight_bias_signed = 1; // whether weight and bias are signed
    conv2d_bean.filter_lsb_channelwise = 1; // whether filter lsb differ from channels
    conv2d_bean.acc_out_shift = 0; // the right shift bits of the output of acc array, it should be input_fraction + weight_fraction - output_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.bias_shift = 0; // the left shift bits of bias when being added to the acc, it should be input_fraction + weight_fraction - bias_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.leaky_param = 0; // the multiplier of leaky relu, the LSB is 2^(-6)
    conv2d_bean.input_iram = 0; // nonzero - read input from iram, 0 - read input from ddr
    conv2d_bean.output_iram = 0; // nonzero - put output into iram, 0 - put output into ddr
    conv2d_bean.in_sep_mode = 0; // whether read input from iram as separable conv mode
    conv2d_bean.out_sep_mode = 0; // whether put output into iram as separable conv mode
    conv2d_bean.next_padding_left = 0; // left pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_up = 0; // up pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_type = 0; // only valid when out_sep_mode is not 0, if the next layer is maxpool, please set it to 1, or set it to 0
    conv2d_bean.nonlinearty = none; // the nonlinearty operation after conv, you can choose "none", "sigmoid", "relu", "tanh", "relu6", "leaky_relu"
    cnn_conv2D(0x15400 + feature_addr_offset, 0x7400 + feature_addr_offset, 0x67100 + weight_addr_offset, &conv2d_bean);
    RETURN_JUDGE

    /*
     * this layer end
     */
    // layer index: 31
    // layer name: expanded_conv_8/add
    // layer type: ['eltwise']
    // this layer is completed by using CNN engine from Round 32 to Round 32
    // input tensor names are ['expanded_conv_7/project/BatchNorm/FusedBatchNorm:0', 'expanded_conv_8/project/BatchNorm/FusedBatchNorm:0']
    // output tensor names are ['expanded_conv_8/add:0']
    // input tensor occupy [8192, 8192] bytes individually
    // output tensor occupy [8192] bytes individually
    // weight occupies 0 bytes
    // you can also use the code below to complete this layer
    conv2d_bean.in_channel = 2; // how many channels of input
    conv2d_bean.in_height = 256; // height of input
    conv2d_bean.in_width = 3; // output of input
    conv2d_bean.out_height = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_width = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_channel = 1; // how many channels of output, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.group = 1; // how many groups, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.kernel_size_h = 1; // size of the convolving kernel
    conv2d_bean.kernel_size_w = 1; // size of the convolving kernel
    conv2d_bean.stride = 1;
    conv2d_bean.dilation = 1;
    conv2d_bean.bias_en = 0; // whether add bias when calculating conv
    conv2d_bean.softmax = 0; // whether calculate softmax after calculating conv and activation function
    conv2d_bean.mac_8bit = 0; // non_zero: 8bit mode; zero: 16bit mode
    conv2d_bean.pad_u = 0; // zero-pad added to up    sides of the input
    conv2d_bean.pad_d = 0; // zero-pad added to down  sides of the input
    conv2d_bean.pad_l = 0; // zero-pad added to left  sides of the input
    conv2d_bean.pad_r = 0; // zero-pad added to right sides of the input
    conv2d_bean.input_signed = 1; // whether input is signed
    conv2d_bean.weight_bias_signed = 1; // whether weight and bias are signed
    conv2d_bean.filter_lsb_channelwise = 0; // whether filter lsb differ from channels
    conv2d_bean.acc_out_shift = 0; // the right shift bits of the output of acc array, it should be input_fraction + weight_fraction - output_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.bias_shift = 0; // the left shift bits of bias when being added to the acc, it should be input_fraction + weight_fraction - bias_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.leaky_param = 0; // the multiplier of leaky relu, the LSB is 2^(-6)
    conv2d_bean.input_iram = 0; // nonzero - read input from iram, 0 - read input from ddr
    conv2d_bean.output_iram = 0; // nonzero - put output into iram, 0 - put output into ddr
    conv2d_bean.in_sep_mode = 0; // whether read input from iram as separable conv mode
    conv2d_bean.out_sep_mode = 0; // whether put output into iram as separable conv mode
    conv2d_bean.next_padding_left = 0; // left pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_up = 0; // up pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_type = 0; // only valid when out_sep_mode is not 0, if the next layer is maxpool, please set it to 1, or set it to 0
    conv2d_bean.nonlinearty = none; // the nonlinearty operation after conv, you can choose "none", "sigmoid", "relu", "tanh", "relu6", "leaky_relu"
    cnn_conv2D(0x5400 + feature_addr_offset, 0x9400 + feature_addr_offset, 0x73200 + weight_addr_offset, &conv2d_bean);
    RETURN_JUDGE

    /*
     * this layer end
     */
    // layer index: 32
    // layer name: expanded_conv_9/expand/Conv2D
    // layer type: ['convolution', 'relu6']
    // this layer is completed by using CNN engine from Round 33 to Round 33
    // input tensor names are ['expanded_conv_8/add:0']
    // output tensor names are ['expanded_conv_9/expand/Relu6:0']
    // input tensor occupy [8192] bytes individually
    // output tensor occupy [49152] bytes individually
    // weight occupies 50688 bytes
    // you can also use the code below to complete this layer
    conv2d_bean.in_channel = 64; // how many channels of input
    conv2d_bean.in_height = 4; // height of input
    conv2d_bean.in_width = 3; // output of input
    conv2d_bean.out_height = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_width = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_channel = 384; // how many channels of output, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.group = 1; // how many groups, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.kernel_size_h = 1; // size of the convolving kernel
    conv2d_bean.kernel_size_w = 1; // size of the convolving kernel
    conv2d_bean.stride = 1;
    conv2d_bean.dilation = 1;
    conv2d_bean.bias_en = 1; // whether add bias when calculating conv
    conv2d_bean.softmax = 0; // whether calculate softmax after calculating conv and activation function
    conv2d_bean.mac_8bit = 0; // non_zero: 8bit mode; zero: 16bit mode
    conv2d_bean.pad_u = 0; // zero-pad added to up    sides of the input
    conv2d_bean.pad_d = 0; // zero-pad added to down  sides of the input
    conv2d_bean.pad_l = 0; // zero-pad added to left  sides of the input
    conv2d_bean.pad_r = 0; // zero-pad added to right sides of the input
    conv2d_bean.input_signed = 1; // whether input is signed
    conv2d_bean.weight_bias_signed = 1; // whether weight and bias are signed
    conv2d_bean.filter_lsb_channelwise = 1; // whether filter lsb differ from channels
    conv2d_bean.acc_out_shift = 0; // the right shift bits of the output of acc array, it should be input_fraction + weight_fraction - output_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.bias_shift = 0; // the left shift bits of bias when being added to the acc, it should be input_fraction + weight_fraction - bias_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.leaky_param = 0; // the multiplier of leaky relu, the LSB is 2^(-6)
    conv2d_bean.input_iram = 0; // nonzero - read input from iram, 0 - read input from ddr
    conv2d_bean.output_iram = 0; // nonzero - put output into iram, 0 - put output into ddr
    conv2d_bean.in_sep_mode = 0; // whether read input from iram as separable conv mode
    conv2d_bean.out_sep_mode = 0; // whether put output into iram as separable conv mode
    conv2d_bean.next_padding_left = 0; // left pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_up = 0; // up pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_type = 0; // only valid when out_sep_mode is not 0, if the next layer is maxpool, please set it to 1, or set it to 0
    conv2d_bean.nonlinearty = relu6; // the nonlinearty operation after conv, you can choose "none", "sigmoid", "relu", "tanh", "relu6", "leaky_relu"
    cnn_conv2D(0x9400 + feature_addr_offset, 0xd400 + feature_addr_offset, 0x73200 + weight_addr_offset, &conv2d_bean);
    RETURN_JUDGE

    /*
     * this layer end
     */
    // layer index: 33
    // layer name: expanded_conv_9/depthwise/depthwise
    // layer type: ['convolution', 'relu6']
    // this layer is completed by using CNN engine from Round 34 to Round 34
    // input tensor names are ['expanded_conv_9/expand/Relu6:0']
    // output tensor names are ['expanded_conv_9/depthwise/Relu6:0']
    // input tensor occupy [49152] bytes individually
    // output tensor occupy [49152] bytes individually
    // weight occupies 12288 bytes
    // you can also use the code below to complete this layer
    conv2d_bean.in_channel = 384; // how many channels of input
    conv2d_bean.in_height = 4; // height of input
    conv2d_bean.in_width = 3; // output of input
    conv2d_bean.out_height = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_width = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_channel = 384; // how many channels of output, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.group = 384; // how many groups, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.kernel_size_h = 3; // size of the convolving kernel
    conv2d_bean.kernel_size_w = 3; // size of the convolving kernel
    conv2d_bean.stride = 1;
    conv2d_bean.dilation = 1;
    conv2d_bean.bias_en = 1; // whether add bias when calculating conv
    conv2d_bean.softmax = 0; // whether calculate softmax after calculating conv and activation function
    conv2d_bean.mac_8bit = 0; // non_zero: 8bit mode; zero: 16bit mode
    conv2d_bean.pad_u = 1; // zero-pad added to up    sides of the input
    conv2d_bean.pad_d = 1; // zero-pad added to down  sides of the input
    conv2d_bean.pad_l = 1; // zero-pad added to left  sides of the input
    conv2d_bean.pad_r = 1; // zero-pad added to right sides of the input
    conv2d_bean.input_signed = 1; // whether input is signed
    conv2d_bean.weight_bias_signed = 1; // whether weight and bias are signed
    conv2d_bean.filter_lsb_channelwise = 1; // whether filter lsb differ from channels
    conv2d_bean.acc_out_shift = 0; // the right shift bits of the output of acc array, it should be input_fraction + weight_fraction - output_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.bias_shift = 0; // the left shift bits of bias when being added to the acc, it should be input_fraction + weight_fraction - bias_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.leaky_param = 0; // the multiplier of leaky relu, the LSB is 2^(-6)
    conv2d_bean.input_iram = 0; // nonzero - read input from iram, 0 - read input from ddr
    conv2d_bean.output_iram = 0; // nonzero - put output into iram, 0 - put output into ddr
    conv2d_bean.in_sep_mode = 0; // whether read input from iram as separable conv mode
    conv2d_bean.out_sep_mode = 0; // whether put output into iram as separable conv mode
    conv2d_bean.next_padding_left = 0; // left pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_up = 0; // up pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_type = 0; // only valid when out_sep_mode is not 0, if the next layer is maxpool, please set it to 1, or set it to 0
    conv2d_bean.nonlinearty = relu6; // the nonlinearty operation after conv, you can choose "none", "sigmoid", "relu", "tanh", "relu6", "leaky_relu"
    cnn_conv2D(0xd400 + feature_addr_offset, 0x19400 + feature_addr_offset, 0x7f800 + weight_addr_offset, &conv2d_bean);
    RETURN_JUDGE

    /*
     * this layer end
     */
    // layer index: 34
    // layer name: expanded_conv_9/project/Conv2D
    // layer type: ['convolution']
    // this layer is completed by using CNN engine from Round 35 to Round 35
    // input tensor names are ['expanded_conv_9/depthwise/Relu6:0']
    // output tensor names are ['expanded_conv_9/project/BatchNorm/FusedBatchNorm:0']
    // input tensor occupy [49152] bytes individually
    // output tensor occupy [8192] bytes individually
    // weight occupies 49408 bytes
    // you can also use the code below to complete this layer
    conv2d_bean.in_channel = 384; // how many channels of input
    conv2d_bean.in_height = 4; // height of input
    conv2d_bean.in_width = 3; // output of input
    conv2d_bean.out_height = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_width = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_channel = 64; // how many channels of output, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.group = 1; // how many groups, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.kernel_size_h = 1; // size of the convolving kernel
    conv2d_bean.kernel_size_w = 1; // size of the convolving kernel
    conv2d_bean.stride = 1;
    conv2d_bean.dilation = 1;
    conv2d_bean.bias_en = 1; // whether add bias when calculating conv
    conv2d_bean.softmax = 0; // whether calculate softmax after calculating conv and activation function
    conv2d_bean.mac_8bit = 0; // non_zero: 8bit mode; zero: 16bit mode
    conv2d_bean.pad_u = 0; // zero-pad added to up    sides of the input
    conv2d_bean.pad_d = 0; // zero-pad added to down  sides of the input
    conv2d_bean.pad_l = 0; // zero-pad added to left  sides of the input
    conv2d_bean.pad_r = 0; // zero-pad added to right sides of the input
    conv2d_bean.input_signed = 1; // whether input is signed
    conv2d_bean.weight_bias_signed = 1; // whether weight and bias are signed
    conv2d_bean.filter_lsb_channelwise = 1; // whether filter lsb differ from channels
    conv2d_bean.acc_out_shift = 0; // the right shift bits of the output of acc array, it should be input_fraction + weight_fraction - output_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.bias_shift = 0; // the left shift bits of bias when being added to the acc, it should be input_fraction + weight_fraction - bias_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.leaky_param = 0; // the multiplier of leaky relu, the LSB is 2^(-6)
    conv2d_bean.input_iram = 0; // nonzero - read input from iram, 0 - read input from ddr
    conv2d_bean.output_iram = 0; // nonzero - put output into iram, 0 - put output into ddr
    conv2d_bean.in_sep_mode = 0; // whether read input from iram as separable conv mode
    conv2d_bean.out_sep_mode = 0; // whether put output into iram as separable conv mode
    conv2d_bean.next_padding_left = 0; // left pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_up = 0; // up pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_type = 0; // only valid when out_sep_mode is not 0, if the next layer is maxpool, please set it to 1, or set it to 0
    conv2d_bean.nonlinearty = none; // the nonlinearty operation after conv, you can choose "none", "sigmoid", "relu", "tanh", "relu6", "leaky_relu"
    cnn_conv2D(0x19400 + feature_addr_offset, 0xb400 + feature_addr_offset, 0x82800 + weight_addr_offset, &conv2d_bean);
    RETURN_JUDGE

    /*
     * this layer end
     */
    // layer index: 35
    // layer name: expanded_conv_9/add
    // layer type: ['eltwise']
    // this layer is completed by using CNN engine from Round 36 to Round 36
    // input tensor names are ['expanded_conv_8/add:0', 'expanded_conv_9/project/BatchNorm/FusedBatchNorm:0']
    // output tensor names are ['expanded_conv_9/add:0']
    // input tensor occupy [8192, 8192] bytes individually
    // output tensor occupy [8192] bytes individually
    // weight occupies 0 bytes
    // you can also use the code below to complete this layer
    conv2d_bean.in_channel = 2; // how many channels of input
    conv2d_bean.in_height = 256; // height of input
    conv2d_bean.in_width = 3; // output of input
    conv2d_bean.out_height = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_width = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_channel = 1; // how many channels of output, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.group = 1; // how many groups, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.kernel_size_h = 1; // size of the convolving kernel
    conv2d_bean.kernel_size_w = 1; // size of the convolving kernel
    conv2d_bean.stride = 1;
    conv2d_bean.dilation = 1;
    conv2d_bean.bias_en = 0; // whether add bias when calculating conv
    conv2d_bean.softmax = 0; // whether calculate softmax after calculating conv and activation function
    conv2d_bean.mac_8bit = 0; // non_zero: 8bit mode; zero: 16bit mode
    conv2d_bean.pad_u = 0; // zero-pad added to up    sides of the input
    conv2d_bean.pad_d = 0; // zero-pad added to down  sides of the input
    conv2d_bean.pad_l = 0; // zero-pad added to left  sides of the input
    conv2d_bean.pad_r = 0; // zero-pad added to right sides of the input
    conv2d_bean.input_signed = 1; // whether input is signed
    conv2d_bean.weight_bias_signed = 1; // whether weight and bias are signed
    conv2d_bean.filter_lsb_channelwise = 0; // whether filter lsb differ from channels
    conv2d_bean.acc_out_shift = 0; // the right shift bits of the output of acc array, it should be input_fraction + weight_fraction - output_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.bias_shift = 0; // the left shift bits of bias when being added to the acc, it should be input_fraction + weight_fraction - bias_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.leaky_param = 0; // the multiplier of leaky relu, the LSB is 2^(-6)
    conv2d_bean.input_iram = 0; // nonzero - read input from iram, 0 - read input from ddr
    conv2d_bean.output_iram = 0; // nonzero - put output into iram, 0 - put output into ddr
    conv2d_bean.in_sep_mode = 0; // whether read input from iram as separable conv mode
    conv2d_bean.out_sep_mode = 0; // whether put output into iram as separable conv mode
    conv2d_bean.next_padding_left = 0; // left pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_up = 0; // up pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_type = 0; // only valid when out_sep_mode is not 0, if the next layer is maxpool, please set it to 1, or set it to 0
    conv2d_bean.nonlinearty = none; // the nonlinearty operation after conv, you can choose "none", "sigmoid", "relu", "tanh", "relu6", "leaky_relu"
    cnn_conv2D(0x9400 + feature_addr_offset, 0x5400 + feature_addr_offset, 0x8e900 + weight_addr_offset, &conv2d_bean);
    RETURN_JUDGE

    /*
     * this layer end
     */
    // layer index: 36
    // layer name: final_depthwise/depthwise
    // layer type: ['convolution']
    // this layer is completed by using CNN engine from Round 37 to Round 37
    // input tensor names are ['expanded_conv_9/add:0']
    // output tensor names are ['final_depthwise/depthwise:0']
    // input tensor occupy [8192] bytes individually
    // output tensor occupy [4096] bytes individually
    // weight occupies 2048 bytes
    // you can also use the code below to complete this layer
    conv2d_bean.in_channel = 64; // how many channels of input
    conv2d_bean.in_height = 4; // height of input
    conv2d_bean.in_width = 3; // output of input
    conv2d_bean.out_height = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_width = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_channel = 64; // how many channels of output, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.group = 64; // how many groups, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.kernel_size_h = 3; // size of the convolving kernel
    conv2d_bean.kernel_size_w = 3; // size of the convolving kernel
    conv2d_bean.stride = 2;
    conv2d_bean.dilation = 1;
    conv2d_bean.bias_en = 0; // whether add bias when calculating conv
    conv2d_bean.softmax = 0; // whether calculate softmax after calculating conv and activation function
    conv2d_bean.mac_8bit = 0; // non_zero: 8bit mode; zero: 16bit mode
    conv2d_bean.pad_u = 0; // zero-pad added to up    sides of the input
    conv2d_bean.pad_d = 1; // zero-pad added to down  sides of the input
    conv2d_bean.pad_l = 1; // zero-pad added to left  sides of the input
    conv2d_bean.pad_r = 1; // zero-pad added to right sides of the input
    conv2d_bean.input_signed = 1; // whether input is signed
    conv2d_bean.weight_bias_signed = 1; // whether weight and bias are signed
    conv2d_bean.filter_lsb_channelwise = 1; // whether filter lsb differ from channels
    conv2d_bean.acc_out_shift = 0; // the right shift bits of the output of acc array, it should be input_fraction + weight_fraction - output_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.bias_shift = 0; // the left shift bits of bias when being added to the acc, it should be input_fraction + weight_fraction - bias_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.leaky_param = 0; // the multiplier of leaky relu, the LSB is 2^(-6)
    conv2d_bean.input_iram = 0; // nonzero - read input from iram, 0 - read input from ddr
    conv2d_bean.output_iram = 0; // nonzero - put output into iram, 0 - put output into ddr
    conv2d_bean.in_sep_mode = 0; // whether read input from iram as separable conv mode
    conv2d_bean.out_sep_mode = 0; // whether put output into iram as separable conv mode
    conv2d_bean.next_padding_left = 0; // left pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_up = 0; // up pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_type = 0; // only valid when out_sep_mode is not 0, if the next layer is maxpool, please set it to 1, or set it to 0
    conv2d_bean.nonlinearty = none; // the nonlinearty operation after conv, you can choose "none", "sigmoid", "relu", "tanh", "relu6", "leaky_relu"
    cnn_conv2D(0x5400 + feature_addr_offset, 0x7400 + feature_addr_offset, 0x8e900 + weight_addr_offset, &conv2d_bean);
    RETURN_JUDGE

    /*
     * this layer end
     */
    // layer index: 37
    // layer name: global_avg/average_pooling2d/AvgPool_step_0
    // layer type: ['pooling']
    // this layer is completed by using CNN engine from Round 38 to Round 38
    // input tensor names are ['final_depthwise/depthwise:0']
    // output tensor names are ['global_avg/average_pooling2d/AvgPool:0']
    // input tensor occupy [4096] bytes individually
    // output tensor occupy [2048] bytes individually
    // weight occupies 0 bytes
    // you can also use the code below to complete this layer
    conv2d_bean.in_channel = 64; // how many channels of input
    conv2d_bean.in_height = 2; // height of input
    conv2d_bean.in_width = 2; // output of input
    conv2d_bean.out_height = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_width = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_channel = 64; // how many channels of output, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.group = 1; // how many groups, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.kernel_size_h = 2; // size of the convolving kernel
    conv2d_bean.kernel_size_w = 2; // size of the convolving kernel
    conv2d_bean.stride = 1;
    conv2d_bean.dilation = 1;
    conv2d_bean.bias_en = 0; // whether add bias when calculating conv
    conv2d_bean.softmax = 0; // whether calculate softmax after calculating conv and activation function
    conv2d_bean.mac_8bit = 0; // non_zero: 8bit mode; zero: 16bit mode
    conv2d_bean.pad_u = 0; // zero-pad added to up    sides of the input
    conv2d_bean.pad_d = 0; // zero-pad added to down  sides of the input
    conv2d_bean.pad_l = 0; // zero-pad added to left  sides of the input
    conv2d_bean.pad_r = 0; // zero-pad added to right sides of the input
    conv2d_bean.input_signed = 1; // whether input is signed
    conv2d_bean.weight_bias_signed = 1; // whether weight and bias are signed
    conv2d_bean.filter_lsb_channelwise = 0; // whether filter lsb differ from channels
    conv2d_bean.acc_out_shift = 15; // the right shift bits of the output of acc array, it should be input_fraction + weight_fraction - output_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.bias_shift = 0; // the left shift bits of bias when being added to the acc, it should be input_fraction + weight_fraction - bias_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.leaky_param = 0; // the multiplier of leaky relu, the LSB is 2^(-6)
    conv2d_bean.input_iram = 0; // nonzero - read input from iram, 0 - read input from ddr
    conv2d_bean.output_iram = 0; // nonzero - put output into iram, 0 - put output into ddr
    conv2d_bean.in_sep_mode = 0; // whether read input from iram as separable conv mode
    conv2d_bean.out_sep_mode = 0; // whether put output into iram as separable conv mode
    conv2d_bean.next_padding_left = 0; // left pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_up = 0; // up pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_type = 0; // only valid when out_sep_mode is not 0, if the next layer is maxpool, please set it to 1, or set it to 0
    conv2d_bean.nonlinearty = none; // the nonlinearty operation after conv, you can choose "none", "sigmoid", "relu", "tanh", "relu6", "leaky_relu"
    cnn_conv2D(0x7400 + feature_addr_offset, 0x8400 + feature_addr_offset, 0x8f100 + weight_addr_offset, &conv2d_bean);
    RETURN_JUDGE

    /*
     * this layer end
     */
    // layer index: 38
    // layer name: fully_connected/dense
    // layer type: ['innerproduct']
    // this layer is completed by using CNN engine from Round 39 to Round 39
    // input tensor names are ['global_avg/average_pooling2d/AvgPool:0']
    // output tensor names are ['Predictions:0']
    // input tensor occupy [2048] bytes individually
    // output tensor occupy [64] bytes individually
    // weight occupies 2112 bytes
    // you can also use the code below to complete this layer
    conv2d_bean.in_channel = 64; // how many channels of input
    conv2d_bean.in_height = 1; // height of input
    conv2d_bean.in_width = 1; // output of input
    conv2d_bean.out_height = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_width = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_channel = 2; // how many channels of output, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.group = 1; // how many groups, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.kernel_size_h = 1; // size of the convolving kernel
    conv2d_bean.kernel_size_w = 1; // size of the convolving kernel
    conv2d_bean.stride = 1;
    conv2d_bean.dilation = 1;
    conv2d_bean.bias_en = 1; // whether add bias when calculating conv
    conv2d_bean.softmax = 0; // whether calculate softmax after calculating conv and activation function
    conv2d_bean.mac_8bit = 0; // non_zero: 8bit mode; zero: 16bit mode
    conv2d_bean.pad_u = 0; // zero-pad added to up    sides of the input
    conv2d_bean.pad_d = 0; // zero-pad added to down  sides of the input
    conv2d_bean.pad_l = 0; // zero-pad added to left  sides of the input
    conv2d_bean.pad_r = 0; // zero-pad added to right sides of the input
    conv2d_bean.input_signed = 1; // whether input is signed
    conv2d_bean.weight_bias_signed = 1; // whether weight and bias are signed
    conv2d_bean.filter_lsb_channelwise = 1; // whether filter lsb differ from channels
    conv2d_bean.acc_out_shift = 0; // the right shift bits of the output of acc array, it should be input_fraction + weight_fraction - output_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.bias_shift = 0; // the left shift bits of bias when being added to the acc, it should be input_fraction + weight_fraction - bias_fraction. Only valid when filter_lsb_channelwise is 0
    conv2d_bean.leaky_param = 0; // the multiplier of leaky relu, the LSB is 2^(-6)
    conv2d_bean.input_iram = 0; // nonzero - read input from iram, 0 - read input from ddr
    conv2d_bean.output_iram = 0; // nonzero - put output into iram, 0 - put output into ddr
    conv2d_bean.in_sep_mode = 0; // whether read input from iram as separable conv mode
    conv2d_bean.out_sep_mode = 0; // whether put output into iram as separable conv mode
    conv2d_bean.next_padding_left = 0; // left pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_up = 0; // up pad of next layer, you need to set this number manually if you want next layer of type "padding to same", only valid when out_sep_mode is not 0
    conv2d_bean.next_padding_type = 0; // only valid when out_sep_mode is not 0, if the next layer is maxpool, please set it to 1, or set it to 0
    conv2d_bean.nonlinearty = none; // the nonlinearty operation after conv, you can choose "none", "sigmoid", "relu", "tanh", "relu6", "leaky_relu"
    cnn_conv2D(0x8400 + feature_addr_offset, 0x5400 + feature_addr_offset, 0x8f100 + weight_addr_offset, &conv2d_bean);
    /*
     * this layer end
     */
    // total_weight_size: 0x8fa00
}
