import types
import time
import os

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops

from ndk.utils import print_log
from ndk.modelpack import load_from_file, save_to_file
class DataRunner():
    def __init__(self, data_maker):
        self.data_maker = data_maker
        self.__build()

    def __build(self):
        if isinstance(self.data_maker, types.GeneratorType):
            data_batch = next(self.data_maker)
            features = data_batch['input']
            labels = data_batch['output']
            feature_shape = list(features.shape)
            label_shape = list(labels.shape)
            self.input_pl = tf.placeholder(dtype=tf.float32, shape=feature_shape)
            self.output_pl = tf.placeholder(dtype=tf.int64, shape=label_shape)

    def _get_input_output_dg(self, data_generator):
        return self.input_pl, self.output_pl

    def _get_net_input_kw_args_dg(self, data_generator):
        data_batch = next(data_generator)
        features = data_batch['input']
        return dict(feed_dict={self.input_pl: features})

    def _get_net_input_output_kw_args_dg(self, data_generator):
        data_batch = next(data_generator)
        features = data_batch['input']
        labels = data_batch['output']
        return dict(feed_dict={self.input_pl: features, self.output_pl: labels})

    def _get_input_output_input_fn(self, input_fn):
        data = input_fn()
        if isinstance(data, tuple):
            features, labels = data
        else:
            dataset = data
            iterator = dataset.make_one_shot_iterator()
            features, labels = iterator.get_next()
        return features, labels

    def _get_net_input_kw_args_input_fn(self, input_fn):
        return {}

    def _get_net_input_output_kw_args_input_fn(self, input_fn):
        return {}

    def get_input_output(self):
        if isinstance(self.data_maker, types.GeneratorType):
            return self._get_input_output_dg(self.data_maker)
        elif isinstance(self.data_maker, types.FunctionType):
            return self._get_input_output_input_fn(self.data_maker)
        else:
            raise TypeError("Unsupported input data type.")

    def get_net_input_kw_args(self):
        if isinstance(self.data_maker, types.GeneratorType):
            return self._get_net_input_kw_args_dg(self.data_maker)
        elif isinstance(self.data_maker, types.FunctionType):
            return self._get_net_input_kw_args_input_fn(self.data_maker)
        else:
            raise TypeError("Unsupported input data type.")

    def get_net_input_output_kw_args(self):
        if isinstance(self.data_maker, types.GeneratorType):
            return self._get_net_input_output_kw_args_dg(self.data_maker)
        elif isinstance(self.data_maker, types.FunctionType):
            return self._get_net_input_output_kw_args_input_fn(self.data_maker)
        else:
            raise TypeError("Unsupported input data type.")

def _run_train(data_runner, sess, train_op, loss, merged_summary):
    ops = [train_op, merged_summary, loss]
    kw_args = data_runner.get_net_input_output_kw_args()
    _, summarys, train_loss = sess.run(ops, **kw_args)
    return train_loss, summarys

def _fit(model,
         loss_fn,
         optimizer,
         train_data_maker,
         model_dir,
         num_step_train=500,
         num_step_log=100,
         num_step_save=2000):

    with ops.Graph().as_default() as g:
        data_runner = DataRunner(train_data_maker)
        print_log("start constructing graph")
        net_input, net_output = data_runner.get_input_output()
        print_log("finish constructing graph")

        logits = model.construct_network_graph(net_input)
        if len(logits.shape) == 4:
            if logits.shape[-2].value == 1 and logits.shape[-1].value == 1:
                logits = tf.layers.flatten(logits)
        labels = tf.layers.flatten(net_output)
        global_step = tf.Variable(0, dtype=tf.int64, name='global_step', trainable=False)
        loss = loss_fn(labels, logits)
        tf.summary.scalar('train_loss', loss)
        print_log("start adding train op")
        train_op = optimizer.minimize(loss, global_step=global_step)
        print_log("finish adding train op")
        merged_summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter(model_dir)

        model._generate_weight_bias_tensors_whose_dimension_in_param_dict_order()


        with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.8))) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            if model.param_dict is not None:
                if model.param_dict:
                    model.restore_weight_bias(sess)
            print_log("start adding graph")
            writer.add_graph(sess.graph)
            print_log("finish adding graph")
            writer.flush()
            tic = time.time()

            print_log('starting train!')
            i = 0
            while True:
                # print_log("start runing train")
                train_loss, summarys = _run_train(data_runner=data_runner, sess=sess,
                                                  train_op=train_op, loss=loss, merged_summary=merged_summary)
                # print_log("finish runing train")
                i = i + 1
                if i % num_step_log == 0 and i != 0:
                    toc = time.time()
                    delta_time = (toc - tic)
                    tic = time.time()
                    print_log('loss = {:.3e}, step = {}/{}, {}s/step.\n'.format(train_loss, i, num_step_train,
                                                                                round(delta_time / num_step_log, 3)))
                    writer.add_summary(summarys, global_step=i)
                if i % num_step_save == 0 and i != 0 and i != num_step_train:
                    quant_layer_list, quant_param_dict = model.export_quant_param_dict(sess=sess)
                    fname = os.path.join(model_dir, 'quant_param_dict_step_{}.npz'.format(i))
                    save_to_file(layer_list=quant_layer_list, fname_prototxt=fname, param_dict=quant_param_dict, fname_npz=fname)

                if i == num_step_train:
                    break

            quant_layer_list, quant_param_dict = model.export_quant_param_dict(sess=sess)

        return quant_param_dict

def _get_the_predicts_dict(logits, fitter_type):
    '''Compute the prdicts dict according the fitter_type.
    '''
    if fitter_type.lower() == 'classify':
        predicted_classes = tf.argmax(logits, 1)
        predictins_dict = {
            'class_ids': predicted_classes[:, tf.newaxis],
            'probabilities': tf.nn.softmax(logits),
            'logits': logits,
        }
    else:  # regression type
        predictins_dict = {
            'logits': logits,
        }

    return predictins_dict

class WeightBiasRestoreSaveHook(tf.train.SessionRunHook):
    def __init__(self,
                 model,
                 restoring_operations,
                 weight_bias_tensors,
                 save_dir,
                 every_n_iter=None,
                 every_n_secs=None):
        self.model = model
        self.restoring_operations = restoring_operations
        if ((every_n_iter is None) == (every_n_secs is None)):
            raise ValueError(
                "exactly one of every_n_iter or every_n_secs "
                "must be provided.")
        if every_n_iter is not None and every_n_iter <= 0:
            raise ValueError("invalid every_n_iter=%s." % every_n_iter)
        self.weight_bias_tensors = weight_bias_tensors
        self.save_dir = save_dir
        self._timer = tf.train.SecondOrStepTimer(every_secs=every_n_secs, every_steps=every_n_iter)

    def begin(self):
        self._timer.reset()
        self._iter_count = 0

    def after_create_session(self, session, coord):
        session.run(self.restoring_operations)

    def before_run(self, run_context):
        self._should_trigger = self._timer.should_trigger_for_step(self._iter_count)

    def after_run(self, run_context, run_values):
        if self._should_trigger:
            _trained_weight_bias = run_context.session.run(self.weight_bias_tensors)
            _quant_weight_bias = self.model._get_quant_weight_bias(weight_bias_dict=_trained_weight_bias, param_dict=self.model.param_dict)
            self.model.param_dict.update(_trained_weight_bias)
            self.model.param_dict.update(_quant_weight_bias)
            fname = os.path.join(self.save_dir, "quant_param_dict_step_{}.npz".format(self._iter_count))
            save_to_file(layer_list=self.model.layer_list, fname_prototxt=fname,
                         param_dict=self.model.param_dict, fname_npz=fname)
            self._timer.update_last_triggered_step(self._iter_count)

        self._iter_count += 1

    def end(self, session):
        _trained_weight_bias = session.run(self.weight_bias_tensors)
        _quant_weight_bias = self.model._get_quant_weight_bias(weight_bias_dict=_trained_weight_bias,
                                                           param_dict=self.model.param_dict)
        self.model.param_dict.update(_trained_weight_bias)
        self.model.param_dict.update(_quant_weight_bias)
        fname = os.path.join(self.save_dir, "quant_param_dict_step_{}.npz".format(self._iter_count))
        save_to_file(layer_list=self.model.layer_list, fname_prototxt=fname,
                     param_dict=self.model.param_dict, fname_npz=fname)

def _model_fn(features, labels, mode, params):
    model = params['model']
    loss_fn = params['loss_fn']
    optimizer = params['optimizer']
    log_every_steps = params['num_step_log']
    save_every_steps = params['num_step_save']
    model_dir = params['model_dir']
    logits = model.construct_network_graph(features)
    if len(logits.shape) == 4:
        if logits.shape[-2].value == 1 and logits.shape[-1].value == 1:
            logits = tf.layers.flatten(logits)
    labels = tf.layers.flatten(labels)

    # Compute loss.
    loss = loss_fn(labels, logits)

    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.contrib.estimator.TowerOptimizer(optimizer)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    logging_hook = tf.train.LoggingTensorHook(tensors={'loss': loss}, every_n_iter=log_every_steps, at_end=False)
    model._generate_weight_bias_tensors_whose_dimension_in_param_dict_order()
    model._construct_weight_bias_restoring_operations(from_quant=False)
    weight_bias_save_hook = WeightBiasRestoreSaveHook(model=model,
                                                      weight_bias_tensors=model._weight_bias_tensors_whose_dimension_in_param_dict_order,
                                                      restoring_operations=model.restoring_operations,
                                                      save_dir=model_dir,
                                                      every_n_iter=save_every_steps)
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, training_hooks=[logging_hook, weight_bias_save_hook])

def _estimator_fit(model,
                   loss_fn,
                   optimizer,
                   model_dir,
                   train_input_fn,
                   num_step_train=500,
                   num_step_log=100,
                   num_step_save=2000):
    tf.logging.set_verbosity(tf.logging.INFO)
    run_config = tf.estimator.RunConfig(save_summary_steps=num_step_log,
                                        save_checkpoints_secs=None,
                                        save_checkpoints_steps=None,
                                        log_step_count_steps=num_step_log)

    classifier = tf.estimator.Estimator(
        model_fn=tf.contrib.estimator.replicate_model_fn(_model_fn),
        model_dir=model_dir,
        params={'model': model,
                'loss_fn':loss_fn,
                'optimizer':optimizer,
                'num_step_log': num_step_log,
                'num_step_save': num_step_save,
                'model_dir':model_dir},
        config=run_config
    )

    classifier.train(input_fn=train_input_fn, steps=num_step_train)
    fname = os.path.join(model_dir, "quant_param_dict_step_{}.npz".format(num_step_train))
    quant_layer_list, quant_param_dict = load_from_file(fname_prototxt=fname, fname_npz=fname)
    return quant_param_dict

def train(model,
          loss_fn,
          optimizer,
          train_data_maker,
          model_dir,
          num_step_train=500,
          num_step_log=100,
          num_step_save=2000):
    if isinstance(train_data_maker, types.FunctionType):
        return _estimator_fit(model=model,
                              loss_fn=loss_fn,
                              optimizer=optimizer,
                              model_dir=model_dir,
                              train_input_fn=train_data_maker,
                              num_step_train=num_step_train,
                              num_step_log=num_step_log,
                              num_step_save=num_step_save)
    else:
        return _fit(model=model,
                    loss_fn=loss_fn,
                    optimizer=optimizer,
                    train_data_maker=train_data_maker,
                    model_dir=model_dir,
                    num_step_train=num_step_train,
                    num_step_log=num_step_log,
                    num_step_save=num_step_save)

def auto_quant(model,
               loss_fn,
               optimizer,
               data_maker,
               log_dir,
               num_step_train,
               layer_group,
               num_step_log):

    if not isinstance(layer_group, int):
        raise TypeError('The layer_group must be int!')
    if layer_group < 1:
        raise ValueError("The layer_group argument must be equal or greater than 1.")

    count = 1
    for idx in range(layer_group + 1, len(model.order_lys), layer_group):
        model_dir = os.path.join(log_dir, 'Quant{}-{}layer'.format(0, count * layer_group + 1))
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model._quant_layer_names = {layer.name for layer in model.order_lys[:idx]}
        model._trainable_layer_names = {lyr.name for lyr in model.order_lys[count * layer_group + 1 - layer_group:] if lyr.type.lower() not in ['batchnorm', 'scale']}
        is_real_trainable = [lyr.type.lower() in ['innerproduct', 'convolution', 'scale', 'bias'] for lyr in
                             model.order_lys[count * layer_group + 1 - layer_group:]]
        is_real_trainable = any(is_real_trainable)
        if not is_real_trainable:
            continue
        model.param_dict = train(model, loss_fn, optimizer, data_maker, model_dir, num_step_train, num_step_log=num_step_log)
        count = count + 1

    model_dir = os.path.join(log_dir, 'Quant{}-{}layer'.format(0, len(model.order_lys)))
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model._quant_layer_names = {layer.name for layer in model.order_lys}
    model._trainable_layer_names = {lyr.name for lyr in model.order_lys[count * layer_group + 1 - layer_group:] if lyr.type.lower() not in ['batchnorm', 'scale']}
    is_real_trainable = [lyr.type.lower() in ['innerproduct', 'convolution', 'scale', 'bias'] for lyr in
                         model.order_lys[count * layer_group + 1 - layer_group:]]
    is_real_trainable = any(is_real_trainable)
    if not is_real_trainable:
        pass
    else:
        model.param_dict = train(model, loss_fn, optimizer, data_maker, model_dir, num_step_train, num_step_log=num_step_log)

    fname = os.path.join(model_dir, 'quant_param_dict.npz')
    save_to_file(layer_list=model.order_lys, fname_prototxt=fname, param_dict=model.param_dict, fname_npz=fname)

    return model.order_lys, model.param_dict
