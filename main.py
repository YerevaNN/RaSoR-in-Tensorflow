from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import cPickle as pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default='10', help='Batch size', type=int)
parser.add_argument('--dropout', default='0.1', help='Dropout in LSTMs', type=float)
parser.add_argument('--epochs', default='10', help='Number of epochs', type=int)
parser.add_argument('--test_every', default='100', help='Number of iterations before validation testing', type=int)
parser.add_argument('--lr', default='0.01', help='Learning rate', type=float)
args = parser.parse_args()


learning_rate = args.lr
epochs = args.epochs
dropout = args.dropout
batch_size = args.batch_size
test_iter = args.test_every

print("Reading train data...")
with open('data/train_data.pkl', 'rb') as fd:
    train_data = pickle.load(fd)
print("Done!")

print("Reading val data...")
with open('data/valid_data.pkl', 'rb') as fd:
    val_data = pickle.load(fd)
print("Done!")

###############################################################################################################

# train_data = [[train_data[0][0][:1000], train_data[0][1][:1000]], [train_data[1][0][:1000], train_data[1][1][:1000]]]
# val_data = [[val_data[0][0][:1000], val_data[0][1][:1000]], [val_data[1][0][:1000], val_data[1][1][:1000]]]

################################################################################################################

max_span_length = 30
n_hidden = 50
word_vec_size = 300

def data_generator(data, is_train=True, batch_size=batch_size, shuffle=False):
    n_samples = len(data[0][0])
    if shuffle:
        perm = np.random.permutation(n_samples)
    else:
        perm = np.arange(n_samples)
    for i in range(0, n_samples, batch_size):
        indices = perm[i:i+batch_size]
        bs = len(indices)
        max_plen = max([data[0][0][j].shape[0] for j in indices])
        max_qlen = max([data[0][1][j].shape[0] for j in indices])
        p_mask = np.ones( (bs, max_plen, 1), dtype=np.float32)
        q_mask = np.ones( (bs, max_qlen, 1), dtype=np.float32)
        p_s = []
        q_s = []
        for j in range(bs):
            ind = indices[j]
            l_p = data[0][0][ind].shape[0]
            l_q = data[0][1][ind].shape[0]
            p_s.append(np.lib.pad(data[0][0][ind], ((0, max_plen - l_p), (0, 0)), 'constant', constant_values=(0,0)))
            q_s.append(np.lib.pad(data[0][1][ind], ((0, max_qlen - l_q), (0, 0)), 'constant', constant_values=(0,0)))
            p_mask[j, l_p:, 0] = 0
            q_mask[j, l_q:, 0] = 0
        p = np.stack(p_s)
        q = np.stack(q_s)


        n_s = np.zeros((bs), dtype=np.int32)

        for j in range(bs):
            ind = indices[j]
            l_p = data[0][0][ind].shape[0]
            if l_p >= max_span_length:
                n_s[j] = (max_span_length + 1) * max_span_length / 2 + (l_p - max_span_length) * max_span_length
            else:
                n_s[j] = (l_p + 1) * l_p / 2
        max_n_s = n_s.max()
        y = np.zeros((bs, max_n_s))
        i_p = np.zeros((bs, max_n_s, 2), dtype=np.int32)
        i_p_mask = np.ones((bs, max_n_s, 1), dtype=np.float32)
        for j in range(bs):
            ind = indices[j]
            l_p = data[0][0][ind].shape[0]
            k = 0
            a1 = data[1][0][ind]
            a2 = data[1][1][ind]
            for m in range(l_p):
                for n in range(m, min(m+max_span_length, l_p)):
                    i_p[j, k, 0] = m
                    i_p[j, k, 1] = n
                    if is_train and m == a1 and n == a2:
                        y[j, k] = 1
                    k += 1
                    assert k <= n_s[j]
            i_p_mask[j, n_s[j]:, 0] = 0
        if is_train:
            yield ( (p, q, i_p), (p_mask, q_mask, i_p_mask), y)
        else:
            yield ( (p, q, i_p), (p_mask, q_mask, i_p_mask))
    return 

p = tf.placeholder("float", [None, None, word_vec_size])
q = tf.placeholder("float", [None, None, word_vec_size])
p_mask = tf.placeholder("float", [None, None, 1])
q_mask = tf.placeholder("float", [None, None, 1])
index_pairs = tf.placeholder("int32", [None, None, 2])
index_pairs_mask = tf.placeholder("float", [None, None, 1])
y = tf.placeholder("float", [None, None])

def softmax_with_mask(input, mask, dim=-1):
    m = tf.reduce_max(input,axis=dim, keep_dims=True)
    e = tf.exp(input - m) * mask
    s = tf.reduce_sum(e, axis=dim, keep_dims=True) 
    s = tf.clip_by_value(s, np.finfo(np.float32).eps, np.finfo(np.float32).max)
    return e / s

def FFNN(input, input_mask, name, layer_shapes = [n_hidden]):
    # A Feed Forward Neural Network

    x = input
    for i in range(len(layer_shapes)):
        s = layer_shapes[i]
        with tf.variable_scope('{}_{}'.format(name, i)):
            x = tf.layers.dense(inputs=x, units=s, activation=tf.nn.relu)
            x = x * input_mask

    return x


def BiLSTM(input, input_mask, name):
    with tf.variable_scope(name):
        lstm_fw_cell = rnn.LSTMCell(n_hidden, forget_bias=1.0)
        lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(lstm_fw_cell, state_keep_prob=1.0-dropout,
#                                                     input_keep_prob=1.0-dropout, input_size=tf.shape(input)[1:],
                                                     variational_recurrent=True, dtype=tf.float32)
        lstm_bw_cell = rnn.LSTMCell(n_hidden, forget_bias=1.0)
        lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(lstm_bw_cell, state_keep_prob=1.0-dropout,
#                                                     input_keep_prob=1.0-dropout, input_size=tf.shape(input)[1:],
                                                     variational_recurrent=True,dtype=tf.float32)
        outputs, states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, input, dtype=tf.float32)
    outputs = tf.concat(outputs, axis=-1) * input_mask
    return outputs

def q_align(p, q, p_mask, q_mask):
    p_n = FFNN(p, p_mask, 'align_p')
    q_n = FFNN(q, q_mask, 'align_q')
    s = tf.matmul(p_n, q_n, transpose_b=True)
    a = softmax_with_mask(s, p_mask)
    return tf.matmul(a, q)

def q_indep(q, q_mask):
    q_s = q
    for i in range(2):
        q_s = BiLSTM(q_s, q_mask, 'BiLSTM_q_indep_{}'.format(i))
    w_q = tf.Variable(tf.random_normal([1, n_hidden]))
    s = tf.tensordot(FFNN(q_s, q_mask, 'FFNN_q_s'), w_q, axes=[[-1],[-1]])
    a = softmax_with_mask(s,q_mask, dim=1)
    return tf.matmul(a, q_s, transpose_a=True)

def concat(p, q_a, q_i):
    p_tmp = tf.reduce_sum(p, axis=-1, keep_dims=True)
    q_i = q_i + p_tmp * 0 
    return tf.concat([p, q_a, q_i], axis=-1)

def question_focused_passage(p, q, p_mask, q_mask):
    q_a = q_align(p, q, p_mask, q_mask)
    q_i = q_indep(q, q_mask)
    h_a = concat(p, q_a, q_i)
    return h_a

p_qf = question_focused_passage(p, q, p_mask, q_mask)

for i in range(2):
    p_qf = BiLSTM(p_qf, p_mask, 'BiLSTM_p_qf_{}'.format(i))


# Getting answer span representation

start_indices = index_pairs[:, :, 0]
start_indices = tf.expand_dims(start_indices, -1)
end_indices = index_pairs[:, :, 1]
end_indices = tf.expand_dims(end_indices, -1)
symbolic_batch_size = tf.shape(index_pairs)[0]
b_s = tf.range(0, symbolic_batch_size, dtype=tf.int32)
b_s = tf.expand_dims(b_s,-1)
b_s = tf.expand_dims(b_s,-1)  # b_s.shape == (B, 1, 1)
b_s = start_indices * 0 + b_s # b_s broadcasts to shape (batch_size, n_spans, 1) == shape of start_indices

start_indices = tf.concat((b_s, start_indices), axis=-1)
end_indices = tf.concat((b_s, end_indices), axis=-1)

start_vectors = tf.gather_nd(p_qf, start_indices)
end_vectors = tf.gather_nd(p_qf, end_indices)

spans = tf.concat((start_vectors, end_vectors), axis=-1) # spans.shape == (batch_size, n_spans, 2 * n_hidden)
spans = spans * index_pairs_mask


def span_score_logits(spans, spans_mask):
    w_a = tf.Variable(tf.random_normal([n_hidden]))
    h_a = FFNN(spans, spans_mask, 'spans')
    s_a = tf.tensordot(h_a, w_a, axes=[[-1],[-1]])
    return s_a * spans_mask[:, :, 0]

logits = span_score_logits(spans, index_pairs_mask)
probs = softmax_with_mask(logits, index_pairs_mask[:, :, 0])
def cross_entropy(y_, y):
    y_ = tf.clip_by_value(y_, np.finfo(np.float32).eps, np.finfo(np.float32).max)
    return tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_), reduction_indices=[1]))


#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
cost = cross_entropy(probs, y)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(logits,-1), tf.argmax(y,-1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


init = tf.global_variables_initializer()
saver = tf.train.Saver(max_to_keep=epochs) 

def test_on_validation_set(val_data, global_iter):
    acc = 0
    loss = 0
    iter = 0
    counter = 0.0
    data_len = len(val_data[1][1])
    print("Running on Validataion Set")

    for ((batch_p, batch_q, batch_i_p), (batch_p_mask, batch_q_mask, batch_i_p_mask), batch_y) in data_generator(val_data, is_train=True, batch_size=batch_size, shuffle=False):
        f_dict={p: batch_p, q: batch_q, index_pairs: batch_i_p, p_mask: batch_p_mask, q_mask: batch_q_mask, index_pairs_mask: batch_i_p_mask, y: batch_y}            
        acc += sess.run(accuracy, feed_dict=f_dict)
        loss += sess.run(cost, feed_dict=f_dict)
        counter += len(batch_p)
        iter += 1
        print("{:.4f}%".format(counter * 100 / data_len), end='\r')

    print("\nIter: {:4d}  Val Loss: {:.4f} Val Acc: {:.4f}".format(global_iter, loss/iter, acc/iter))

with tf.Session() as sess:
    sess.run(init)
    global_iter = 0

    for e in range(epochs):
        
        for ((batch_p, batch_q, batch_i_p), (batch_p_mask, batch_q_mask, batch_i_p_mask), batch_y) in data_generator(train_data, is_train=True, batch_size=batch_size, shuffle=True):
            f_dict={p: batch_p, q: batch_q, index_pairs: batch_i_p, p_mask: batch_p_mask, q_mask: batch_q_mask, index_pairs_mask: batch_i_p_mask, y: batch_y}
            if global_iter % test_iter == 0:
                test_on_validation_set(val_data, global_iter)
            train_loss = sess.run(cost, feed_dict=f_dict)
            train_acc = sess.run(accuracy, feed_dict=f_dict)
            print("Iter: {:4d} Train Loss: {:.4f} Train Acc: {:.4f}".format(global_iter, train_loss, train_acc))
            sess.run(optimizer, feed_dict=f_dict)
            global_iter += 1


        
 
        save_path = saver.save(sess, "models/model", global_step=e)
        print("Model saved in file: {}".format(save_path))

    print("Optimization Finished!")
