# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 18:21:11 2019

@author: jason
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import os
import io
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
##################################################################################################################
tf.enable_eager_execution()  
# preprocess the data
data_train = 'shakespeare_train.txt'
with io.open(data_train, 'r', encoding='utf8') as f:
    text = f.read()
vocab = sorted(set(text))
char2idx = {c: i for i, c in enumerate(vocab)}  # no order
idx2char = np.array(vocab)
text_as_int = np.array([char2idx[c] for c in text], dtype=np.int32)
seq_length = 100
examples_per_epoch = len(text)//seq_length 
# Create training examples / targets
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
sequences = char_dataset.batch(seq_length+1, drop_remainder=True)
def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text
dataset = sequences.map(split_input_target)  

data_valid = 'shakespeare_valid.txt'
with io.open(data_valid, 'r', encoding='utf8') as f:
    text_valid = f.read()
vocab_valid = set(text_valid)  # collect all characters
char2idx_valid = {c: i for i, c in enumerate(vocab_valid)}  # no order
idx2char_valid = {i: u for i, u in enumerate(vocab_valid)}
int_to_vocab_valid = dict(enumerate(vocab_valid))  # reverse sequence from vocab_to_int
text_as_int_valid = np.array([char2idx_valid[c] for c in text_valid], dtype=np.int32)
# seq_length = 100
examples_per_epoch_valid = len(text_valid)//(seq_length+1)
char_dataset_valid = tf.data.Dataset.from_tensor_slices(text_as_int_valid)
sequences_valid = char_dataset_valid.batch(seq_length+1, drop_remainder=True)
dataset_valid = sequences_valid.map(split_input_target)
##################################################################################################################
# Create training batches
# Batch size
BATCH_SIZE = 64  # this would be quick than 64
steps_per_epoch = examples_per_epoch//BATCH_SIZE
steps_per_epoch_valid = examples_per_epoch_valid//BATCH_SIZE

BUFFER_SIZE = 10000
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
dataset_valid = dataset_valid.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
##################################################################################################################
# Build the Model
# Length of the vocabulary in chars
vocab_size = len(vocab)
parameter = 128  # 1024 ...
# The embedding dimension
embedding_dim = parameter
# Number of RNN units
rnn_units = parameter
if tf.test.is_gpu_available():
  rnn = tf.keras.layers.CuDNNGRU
  lstm = tf.keras.layers.CuDNNLSTM
else:
  import functools
  rnn = functools.partial(tf.keras.layers.GRU, recurrent_activation='sigmoid')
  lstm = functools.partial(tf.keras.layers.LSTM, recurrent_activation='sigmoid')

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
  model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim,
                              batch_input_shape=[batch_size, None]),
      # [batch_size, max_time_steps]
    # tf.keras.layers.GRU(rnn_units,
    #                     return_sequences=True,
    #                     stateful=True,
    #                     recurrent_initializer='glorot_uniform'),
    lstm(rnn_units,
          # return_sequences=False,
         return_sequences=True,
          stateful=True,
          recurrent_initializer='glorot_uniform'),
    tf.keras.layers.Dense(vocab_size)
  ])
  return model
model = build_model(
    vocab_size=len(vocab),
    embedding_dim=embedding_dim,
    rnn_units=rnn_units,
    batch_size=BATCH_SIZE)
model.summary()
##################################################################################################################
# compile
def loss(labels, logits):
  return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
checkpoint_dir = './training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    # monitor='val_loss',  # try this one
    filepath=checkpoint_prefix,
    save_weights_only=True)
model.compile(
    optimizer=tf.train.AdamOptimizer(),
    loss=loss,
    # loss='categorical_crossentropy',
    metrics=['sparse_categorical_crossentropy'],)
# Execute the training
EPOCHS = 15
history = model.fit(dataset.repeat(),
                    steps_per_epoch=steps_per_epoch,
                    # validation_split=0.1, not work, cuz x is a iterator
                    validation_data=dataset_valid.repeat(),
                    validation_steps=steps_per_epoch_valid,
                    epochs=EPOCHS,
                    callbacks=[checkpoint_callback],
                    )
score = model.evaluate(dataset_valid, steps=steps_per_epoch_valid, verbose=1)
##################################################################################################################
# plot loss pictures
history_dict = history.history
history_dict.keys()
loss = history_dict['loss']
epochs = range(1, len(loss) + 1)
# "bo" is for "blue dot"
# plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, loss, color='cornflowerblue', label='Training loss')
plt.title('Training loss')
# plt.xticks(np.linspace(0, epochs, max(epochs) + 1))  # no need when iterations num becomes big
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# plot cross-entropy picture
# 'loss', 'sparse_categorical_crossentropy', 'val_loss', 'val_sparse_categorical_crossentropy'
history_dict = history.history
# check "history_dict.keys()" first
plt.plot(history_dict['sparse_categorical_crossentropy'])
plt.plot(history_dict['val_sparse_categorical_crossentropy'])
plt.title('Accuracy History')
plt.ylabel('Accuracy Rate')
plt.xlabel('Epoch')
plt.legend(['train', 'validation'], loc='center right')
plt.show()
##################################################################################################################
# Generate text
# Restore the latest checkpoint
ckpt = tf.train.latest_checkpoint(checkpoint_dir)
model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)
model.load_weights(ckpt)
model.summary()
# The prediction loop
def generate_text(model, start_string):
  # 用訓練的模型生成文本 
  num_generate = 1000  # 要生成的字符個数
  # 将起始字符串转换为数字（向量化）
  input_eval = [char2idx[s] for s in start_string]
  input_eval = tf.expand_dims(input_eval, 0)
  # 空字符串用于存储结果
  text_generated = []
  temperature = 1  # 這裡的批量大小為 1  
  model.reset_states()
  for i in range(num_generate):
      predictions = model(input_eval)
      # 删除批次的维度
      predictions = tf.squeeze(predictions, 0)
      # 用分类分布预测模型返回的字符
      predictions = predictions / temperature
      predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()
      # 把预测字符和前面的隐藏状态一起传递给模型作为下一个输入
      input_eval = tf.expand_dims([predicted_id], 0)
      text_generated.append(idx2char[predicted_id])
  return (start_string + ''.join(text_generated))
print(generate_text(model, start_string=u"ROMEO: "))  


