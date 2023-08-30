struct cnn_one_layer_debug_info{
    uint8_t  layer_index; // start from 0
    uint8_t  mem_type; // 0: in ddr, 1: in iram
    uint8_t  int_type; // 1: feature map 8bit, 0: feature map 16bit
    int8_t   output_fraction;
    uint32_t output_start_addr;
    uint32_t output_channel;
    uint32_t output_height;
    uint32_t output_valid_elem_per_row; // how many elements(8bit or 16bit) in a row of feature map
    uint32_t output_size; // how many bytes occupied by output
};
struct cnn_one_layer_debug_info debug_info[] = {
    {0, 0, 0, 14, 21504, 8, 56, 48, 43008},
    {1, 0, 0, 12, 64512, 16, 28, 24, 28672},
    {2, 0, 0, 11, 93184, 32, 14, 12, 14336},
    {3, 0, 0, 10, 21504, 32, 7, 6, 7168},
    {4, 0, 0, 9, 28672, 32, 1, 1, 1024},
    {5, 0, 0, 8, 29696, 2, 1, 1, 64}
};
uint32_t input_size = 21504;
uint32_t total_weight_size = 35584;
uint32_t features_total_size = 107520;
