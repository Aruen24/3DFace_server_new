#include "cnn_conv2d.h"
#include "cnn_pool2d.h"
#include "cnn_tensor_util.h"
/* @brief cnn_run_a_lovely_net() - We transfrom the prototxt file to C code as below, you can copy the code to complete the whole calculation of your network
/*        feature occupies 107520 bytes
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
    // layer name: mobilenet/Conv1/Conv2D
    // layer type: ['convolution', 'relu']
    // this layer is completed by using CNN engine from Round 1 to Round 1
    // input tensor names are ['Placeholder:0']
    // output tensor names are ['mobilenet/Conv1/Relu:0']
    // input tensor occupy [21504] bytes individually
    // output tensor occupy [43008] bytes individually
    // weight occupies 352 bytes
    // you can also use the code below to complete this layer
    conv2d_bean.in_channel = 1; // how many channels of input
    conv2d_bean.in_height = 112; // height of input
    conv2d_bean.in_width = 96; // output of input
    conv2d_bean.out_height = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_width = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_channel = 8; // how many channels of output, if you want to run a depthwise conv, you do not need to set it
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
    conv2d_bean.nonlinearty = relu; // the nonlinearty operation after conv, you can choose "none", "sigmoid", "relu", "tanh", "relu6", "leaky_relu"
    cnn_conv2D(0x0 + feature_addr_offset, 0x5400 + feature_addr_offset, 0x0 + weight_addr_offset, &conv2d_bean);
    RETURN_JUDGE

    /*
     * this layer end
     */
    // layer index: 1
    // layer name: mobilenet/Conv2/Conv2D
    // layer type: ['convolution', 'relu']
    // this layer is completed by using CNN engine from Round 2 to Round 2
    // input tensor names are ['mobilenet/Conv1/Relu:0']
    // output tensor names are ['mobilenet/Conv2/Relu:0']
    // input tensor occupy [43008] bytes individually
    // output tensor occupy [28672] bytes individually
    // weight occupies 2368 bytes
    // you can also use the code below to complete this layer
    conv2d_bean.in_channel = 8; // how many channels of input
    conv2d_bean.in_height = 56; // height of input
    conv2d_bean.in_width = 48; // output of input
    conv2d_bean.out_height = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_width = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_channel = 16; // how many channels of output, if you want to run a depthwise conv, you do not need to set it
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
    conv2d_bean.nonlinearty = relu; // the nonlinearty operation after conv, you can choose "none", "sigmoid", "relu", "tanh", "relu6", "leaky_relu"
    cnn_conv2D(0x5400 + feature_addr_offset, 0xfc00 + feature_addr_offset, 0x200 + weight_addr_offset, &conv2d_bean);
    RETURN_JUDGE

    /*
     * this layer end
     */
    // layer index: 2
    // layer name: mobilenet/Conv3/Conv2D
    // layer type: ['convolution', 'relu']
    // this layer is completed by using CNN engine from Round 3 to Round 3
    // input tensor names are ['mobilenet/Conv2/Relu:0']
    // output tensor names are ['mobilenet/Conv3/Relu:0']
    // input tensor occupy [28672] bytes individually
    // output tensor occupy [14336] bytes individually
    // weight occupies 9344 bytes
    // you can also use the code below to complete this layer
    conv2d_bean.in_channel = 16; // how many channels of input
    conv2d_bean.in_height = 28; // height of input
    conv2d_bean.in_width = 24; // output of input
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
    conv2d_bean.nonlinearty = relu; // the nonlinearty operation after conv, you can choose "none", "sigmoid", "relu", "tanh", "relu6", "leaky_relu"
    cnn_conv2D(0xfc00 + feature_addr_offset, 0x16c00 + feature_addr_offset, 0xc00 + weight_addr_offset, &conv2d_bean);
    RETURN_JUDGE

    /*
     * this layer end
     */
    // layer index: 3
    // layer name: mobilenet/Conv4/Conv2D
    // layer type: ['convolution', 'relu']
    // this layer is completed by using CNN engine from Round 4 to Round 4
    // input tensor names are ['mobilenet/Conv3/Relu:0']
    // output tensor names are ['mobilenet/Conv4/Relu:0']
    // input tensor occupy [14336] bytes individually
    // output tensor occupy [7168] bytes individually
    // weight occupies 18560 bytes
    // you can also use the code below to complete this layer
    conv2d_bean.in_channel = 32; // how many channels of input
    conv2d_bean.in_height = 14; // height of input
    conv2d_bean.in_width = 12; // output of input
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
    conv2d_bean.nonlinearty = relu; // the nonlinearty operation after conv, you can choose "none", "sigmoid", "relu", "tanh", "relu6", "leaky_relu"
    cnn_conv2D(0x16c00 + feature_addr_offset, 0x5400 + feature_addr_offset, 0x3100 + weight_addr_offset, &conv2d_bean);
    RETURN_JUDGE

    /*
     * this layer end
     */
    // layer index: 4
    // layer name: mobilenet/average/depthwise
    // layer type: ['convolution', 'relu']
    // this layer is completed by using CNN engine from Round 5 to Round 5
    // input tensor names are ['mobilenet/Conv4/Relu:0']
    // output tensor names are ['mobilenet/average/Relu:0']
    // input tensor occupy [7168] bytes individually
    // output tensor occupy [1024] bytes individually
    // weight occupies 3072 bytes
    // you can also use the code below to complete this layer
    conv2d_bean.in_channel = 32; // how many channels of input
    conv2d_bean.in_height = 7; // height of input
    conv2d_bean.in_width = 6; // output of input
    conv2d_bean.out_height = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_width = 0; // do not need to be setted, this is used to save the height of output when the calculation finishes
    conv2d_bean.out_channel = 32; // how many channels of output, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.group = 32; // how many groups, if you want to run a depthwise conv, you do not need to set it
    conv2d_bean.kernel_size_h = 7; // size of the convolving kernel
    conv2d_bean.kernel_size_w = 6; // size of the convolving kernel
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
    conv2d_bean.nonlinearty = relu; // the nonlinearty operation after conv, you can choose "none", "sigmoid", "relu", "tanh", "relu6", "leaky_relu"
    cnn_conv2D(0x5400 + feature_addr_offset, 0x7000 + feature_addr_offset, 0x7a00 + weight_addr_offset, &conv2d_bean);
    RETURN_JUDGE

    /*
     * this layer end
     */
    // layer index: 5
    // layer name: mobilenet/logits/dense
    // layer type: ['innerproduct']
    // this layer is completed by using CNN engine from Round 6 to Round 6
    // input tensor names are ['mobilenet/average/Relu:0']
    // output tensor names are ['mobilenet/out:0']
    // input tensor occupy [1024] bytes individually
    // output tensor occupy [64] bytes individually
    // weight occupies 1088 bytes
    // you can also use the code below to complete this layer
    conv2d_bean.in_channel = 32; // how many channels of input
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
    cnn_conv2D(0x7000 + feature_addr_offset, 0x7400 + feature_addr_offset, 0x8600 + weight_addr_offset, &conv2d_bean);
    /*
     * this layer end
     */
    // total_weight_size: 0x8b00
}
