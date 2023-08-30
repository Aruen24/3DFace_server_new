import tensorflow as tf
import tensorflow.contrib.slim as slim

def Feathernet_arg_scope(is_training,dropout_rate=0.5,bn_decay=0.9):
    batch_norm_params = {
        'center': True,
        'scale': False,
        'decay': bn_decay,
        'is_training': is_training
        }
    with slim.arg_scope([slim.conv2d,slim.separable_conv2d],
        normalizer_fn=slim.batch_norm,
        activation_fn=tf.nn.relu6,
        biases_initializer=None,
        padding='SAME'),\
        slim.arg_scope([slim.dropout], keep_prob=dropout_rate,is_training=is_training),\
        slim.arg_scope([slim.batch_norm], **batch_norm_params) as s:
        #slim.arg_scope([slim.conv2d],weights_regularizer=slim.l2_regularizer(weight_decay)),\
        return s

@slim.add_arg_scope
def InvertedResidual(x,input_channel, output_channel, stride, expand_ratio, downsample,scope=None):
    with tf.variable_scope(scope, 'expanded_conv') as s, tf.name_scope(s.original_name_scope):
        hidden_dim = round(input_channel * expand_ratio)
        use_res_connect = stride == 1 and input_channel == output_channel
        net=x
        if expand_ratio == 1:
            net=slim.separable_conv2d(net,num_outputs=None,kernel_size=3,depth_multiplier=1,stride=stride,scope="depthwise")
            net = slim.conv2d(net, output_channel,kernel_size=1,stride=1, padding='valid', activation_fn=None,scope="project")
        else:
            net = slim.conv2d(net, hidden_dim, 1, 1,padding='valid', scope='expand')
            net = slim.separable_conv2d(net, num_outputs=None, kernel_size=3, depth_multiplier=1, stride=stride,scope="depthwise")
            net = slim.conv2d(net, output_channel, 1, 1, activation_fn=None, scope='project')
        if  use_res_connect:
            net=x+net
        elif downsample is not None:
            net=downsample+net
    return net

def Fully_connected(x, units, layer_name='fully_connected') :
    with tf.name_scope(layer_name) :
        return tf.layers.dense(inputs=x, use_bias=True, units=units)

def global_avg(x,s=1):
    with tf.name_scope('global_avg'):
        net=tf.layers.average_pooling2d(x, x.get_shape()[1:-1], s)
        return net

def hard_sigmoid(x,name='hard_sigmoid'):
    with tf.name_scope(name):
        h_sigmoid = tf.nn.relu6(x+3)/6
        # h_sigmoid = tf.nn.sigmoid(x)
        return h_sigmoid

def SELayer(x,input_channel,ratio):
    with tf.name_scope('SELayer'):
        net=x
        squeeze = global_avg(net)
        excitation=slim.conv2d(squeeze,input_channel / ratio, kernel_size=1, stride=1, padding='valid',activation_fn=tf.nn.relu,normalizer_fn=None)
        excitation = slim.conv2d(excitation,input_channel, kernel_size=1, stride=1, padding='valid', activation_fn=None, normalizer_fn=None)
        excitation= hard_sigmoid(excitation)
        scale = net * excitation
        # scale = tf.multiply(net,excitation)
        return scale

def interface(x,is_training,num_class,depth_mult=1.0,Selayer=True,avgdown=False,scope='FeatherNet'):
    with slim.arg_scope(Feathernet_arg_scope(is_training)):
        net=x
        block=InvertedResidual
        input_channel = 32
        last_channel = 1024
        interverted_residual_setting = [
            # t, c, n, s
            #[1, 16, 1, 2],  # 112x112  
            #[6, 32, 2, 2],  # 56x56    
            #[6, 48, 6, 2],  # 14x14    
            #[6, 64, 3, 2],  # 7x7
            
            # t, c, n, s
            [1, 16, 1, 2],  # 56x48 
            [6, 32, 2, 2],  # 28x24    
            [6, 48, 4, 2],  # 14x12    
            [6, 64, 3, 2],  # 7x6
        ]
        input_channel = int(input_channel * depth_mult)
        last_channel = int(last_channel * depth_mult) if depth_mult > 1.0 else last_channel
        net=slim.conv2d(net,input_channel,3,2,scope='conv2d_1')
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * depth_mult)
            for i in range(n):
                downsample = None
                if i == 0:
                    if avgdown:
                        downsample=slim.avg_pool2d(net,2,2)
                        downsample=slim.batch_norm(downsample)
                        downsample=slim.conv2d(downsample,output_channel, kernel_size=1,stride=1,normalizer_fn=None,activation_fn=None,biases_initializer=None)
                    net=block(net,input_channel, output_channel, s,t,downsample)
                else:
                    net = block(net, input_channel, output_channel, 1, t, downsample)
                input_channel = output_channel
            if Selayer:
                net = SELayer(net, input_channel, 8)
        net=slim.separable_conv2d(net,num_outputs=None,kernel_size=3,depth_multiplier=1,stride=2,activation_fn=None,normalizer_fn=None,biases_initializer=None,scope='final_depthwise')
        net=global_avg(net)
        net = tf.squeeze(net, [1, 2])
        #net=tf.reshape(net,[-1,1024])

        net=slim.dropout(net,keep_prob=0.5,scope='Dropout')

        logits=slim.linear(net,num_class,activation_fn=None)
        #predictions = slim.softmax(logits, scope='Predictions')
        predictions = tf.identity(logits, name='Predictions')
        return logits,predictions
        

        #return logits

