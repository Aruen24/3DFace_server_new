浮点网络在hdf5文件中，没有归一化，因此需要将输入图片归一化到[-1,1]。
定点网络已经将归一化包含进来，因此直接用0~255的图像。
标签集合参见../imagenet_partial/filenames_to_class_by_number.json