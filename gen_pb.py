import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile
from google.protobuf import text_format
import os
import numpy as np
from PIL import Image
import importlib
import utli



#ckpt_path = "/home/data01_disk/zjr/3DFace_server/log/faceFlat3/20210713_063705/-59400"
#pb_path = "/home/data01_disk/zjr/3DFace_server/log/faceFlat3/pb/spc_2d_0713_v2.pb"

#ckpt_path = "./log/faceFlat3/20210917_054659_0.6/-57618"   
#pb_path = "./log/faceFlat3/pb/spc_2d_0917_v2.pb"
#ckpt_path = "./log/faceFlat3/20210926_092548/-596"  
#pb_path = "./log/faceFlat3/pb/spc_2d_0926_v2.pb" 
#ckpt_path = "./log/faceFlat3/20210926_110553/-60792"
#pb_path = "./log/faceFlat3/pb/spc_2d_0926_110553_v2.pb"

#ckpt_path = "./log_ir_print/faceFlat3/20220713_210035/-106518"
#pb_path = "./log_ir_print/faceFlat3/pb/spc_liveness_best_v2.pb"
#ckpt_path = "./log_no_ir_print/faceFlat3/20220714_010230/-116899"
#ckpt_path = "./log/faceFlat3/20220714_183422/-93700"
ckpt_path = "./log/faceFlat3/20220714_221059/-132117"
#pb_path = "./log_no_ir_print/faceFlat3/pb/spc_2d_220714_no_ir_v2.pb"
#pb_path = "./log/faceFlat3/pb/spc_2d_220715_feathernet_add_v2.pb"
pb_path = "./log/faceFlat3/pb/spc_2d_220715_feathernet_add_epoch150_v2.pb"
#model_def = "models.faceFlat3"
model_def = "models.Feathernet_slim"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
image_h_size = 112
image_w_size = 96
# with tf.device('/cpu:0'):
with tf.Session() as sess:
    output_graph_txt = pb_path.replace(".pb",".pbtxt")
    img_placeholder = tf.placeholder(tf.float32, [None, image_h_size, image_w_size, 1])
    phase = False

    network = importlib.import_module(model_def)
    #logits, Predictions = network.inference(img_placeholder, phase_train=False,mode="test")
    #Predictions = network.inference(img_placeholder, phase_train=False, mode="test")
    phase_train_placeholder = False
    Predictions = network.interface(img_placeholder,phase_train_placeholder,2,depth_mult=1.0,Selayer=False,avgdown=False,scope='FeatherNet')
    graph_def = tf.get_default_graph().as_graph_def()
    print(graph_def)
    vars_to_restore = utli.get_vars_to_restore(ckpt_path)
    saver = tf.train.Saver(vars_to_restore)

    saver.restore(sess, ckpt_path)

    print('load')

    #output_graph_def = graph_util.convert_variables_to_constants(sess, graph_def, ['mobilenet/out']) 
    #output_graph_def = graph_util.convert_variables_to_constants(sess, graph_def, ['Predictions/Reshape_1'])
    output_graph_def = graph_util.convert_variables_to_constants(sess, graph_def, ['Predictions'])
    with tf.gfile.GFile(pb_path,"wb") as f:
        f.write(output_graph_def.SerializeToString())

    with gfile.FastGFile(output_graph_txt, 'wb') as f:
        f.write(text_format.MessageToString(graph_def))
