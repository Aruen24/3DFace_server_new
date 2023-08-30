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
    {0, 0, 0, 12, 21504, 32, 56, 48, 172032},
    {1, 0, 0, 12, 193536, 32, 28, 24, 57344},
    {2, 0, 0, 10, 250880, 16, 28, 24, 28672},
    {3, 0, 0, 12, 21504, 96, 28, 24, 172032},
    {4, 0, 0, 12, 193536, 96, 14, 12, 43008},
    {5, 0, 0, 11, 236544, 32, 14, 12, 14336},
    {6, 0, 0, 12, 21504, 192, 14, 12, 86016},
    {7, 0, 0, 12, 107520, 192, 14, 12, 86016},
    {8, 0, 0, 11, 250880, 32, 14, 12, 14336},
    {9, 0, 0, 11, 21504, 32, 14, 12, 14336},
    {10, 0, 0, 12, 35840, 192, 14, 12, 86016},
    {11, 0, 0, 12, 121856, 192, 7, 6, 43008},
    {12, 0, 0, 11, 164864, 48, 7, 6, 10752},
    {13, 0, 0, 12, 21504, 288, 7, 6, 64512},
    {14, 0, 0, 12, 86016, 288, 7, 6, 64512},
    {15, 0, 0, 11, 175616, 48, 7, 6, 10752},
    {16, 0, 0, 11, 21504, 48, 7, 6, 10752},
    {17, 0, 0, 12, 43008, 288, 7, 6, 64512},
    {18, 0, 0, 12, 107520, 288, 7, 6, 64512},
    {19, 0, 0, 11, 32256, 48, 7, 6, 10752},
    {20, 0, 0, 11, 43008, 48, 7, 6, 10752},
    {21, 0, 0, 12, 64512, 288, 7, 6, 64512},
    {22, 0, 0, 12, 129024, 288, 7, 6, 64512},
    {23, 0, 0, 11, 53760, 48, 7, 6, 10752},
    {24, 0, 0, 10, 21504, 48, 7, 6, 10752},
    {25, 0, 0, 12, 64512, 288, 7, 6, 64512},
    {26, 0, 0, 12, 129024, 288, 4, 3, 36864},
    {27, 0, 0, 12, 21504, 64, 4, 3, 8192},
    {28, 0, 0, 12, 37888, 384, 4, 3, 49152},
    {29, 0, 0, 12, 87040, 384, 4, 3, 49152},
    {30, 0, 0, 12, 29696, 64, 4, 3, 8192},
    {31, 0, 0, 11, 37888, 64, 4, 3, 8192},
    {32, 0, 0, 12, 54272, 384, 4, 3, 49152},
    {33, 0, 0, 12, 103424, 384, 4, 3, 49152},
    {34, 0, 0, 11, 46080, 64, 4, 3, 8192},
    {35, 0, 0, 11, 21504, 64, 4, 3, 8192},
    {36, 0, 0, 10, 29696, 64, 2, 2, 4096},
    {37, 0, 0, 11, 33792, 64, 1, 1, 2048},
    {38, 0, 0, 9, 21504, 2, 1, 1, 64}
};
uint32_t input_size = 21504;
uint32_t total_weight_size = 588288;
uint32_t features_total_size = 279552;
