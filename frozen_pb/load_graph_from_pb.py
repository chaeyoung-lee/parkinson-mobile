#-*- coding: utf-8 -*-
"""
#-----------------------------------------------------------------
  filename: convert_tflite_from_pb.py
  Written by Jaewook Kang @ 2018 Sep.
#-----------------------------------------------------------------
"""

from os import getcwd
import os
from datetime import datetime
import tensorflow as tf
from tensorflow.python.platform import gfile


model_info = {\
    'input_shape': [1,128,128,3],
    'output_shape': [None,4],
    'input_node_name': 'input',
    'output_node_name': 'final_result',
    'dtype':        str(tf.float32)

}


filename = 'retrained_graph.pb'
model_dir = getcwd()
model_filename = os.path.join(model_dir,filename)

base_dir    = model_dir + '/tf_logs'
now         = datetime.utcnow().strftime("%Y%m%d%H%M%S")
logdir      = "{}/run-{}/".format(base_dir,now)
tflite_dir  = logdir + 'tflite/'
tflite_path = tflite_dir + filename.split('.')[0] + '.tflite'

if not gfile.Exists(logdir):
    gfile.MakeDirs(logdir)

if not gfile.Exists(tflite_dir):
    gfile.MakeDirs(tflite_dir)



# load TF computational graph from a pb file
tf.reset_default_graph()

with gfile.FastGFile(model_filename,'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

# Import the graph from "graph_def" into the current default graph
_ = tf.import_graph_def(graph_def=graph_def,name='')


# tf summary
summary_writer = tf.summary.FileWriter(logdir=logdir)
summary_writer.add_graph(graph=tf.get_default_graph())
summary_writer.close()


graph = tf.get_default_graph()
model_in     = graph.get_operation_by_name(model_info['input_node_name']).outputs[0]
model_out    = graph.get_operation_by_name(model_info['output_node_name']).outputs[0]

## tflite conversion
with tf.Session() as sess:
    # tflite generation


    toco = tf.contrib.lite.TocoConverter.from_session(sess=sess,
                                                      input_tensors=[model_in],
                                                      output_tensors=[model_out])

    tflite_model = toco.convert()


with tf.gfile.GFile(tflite_path, 'wb') as f:
    f.write(tflite_model)
    tf.logging.info('tflite is generated.')
