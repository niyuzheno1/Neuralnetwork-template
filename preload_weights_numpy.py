# Copyright (C) 2020 Zach (Yuzhe) Ni 
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2, or (at your option)
# any later version.
#
#
# This python file is for loading computed weights into neural models. 
# It is useful when kaggle submission does not permit multiple file submission.

# naming the weights files and export them into different text file
def get_weights_files_from_neural_model(nmodel):
  for x in nmodel.trainable_weights:
    name_layer_association[x.name] = [str(abs(hash(x.name)))[:8] + ".txt",  str(abs(hash(x.name)))[:8] + ".shape"]
    shape_file = str(abs(hash(x.name)))[:8] + ".shape"
    f_1 = open("/content/"+shape_file, "w")
    f_1.write(str(x.shape))
    f_1.close()
    np.savetxt("/content/"+str(abs(hash(x.name)))[:8] + ".txt", x.numpy().flatten())

#get weights and load them into global variable named weights

def getweightsoverall():
  global weights
  dense_23_bias_0 = [1e-02,1e-01,8e-01]
  dense_23_bias_0 = np.asarray(dense_23_bias_0)
  dense_23_bias_0 = dense_23_bias_0.reshape(3,)
  weights={ 6:  dense_23_bias_0 }

#load our weights to a specific layer by initializer
class LayerSpecifics(tf.keras.initializers.Initializer):
  def __init__(self, idx):
    self.idx = idx
    pass
  def __call__(self, shape, dtype=None, **kwargs):
    global weights
    return tf.convert_to_tensor(weights[self.idx], dtype=np.float32)


