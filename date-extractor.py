

import pickle

import numpy as np

import tensorflow as tf

import spacy
nlp = spacy.load('en_vectors_web_lg')
nlp.add_pipe(nlp.create_pipe('sentencizer'))

def label2text(label, text):
    buffer = []
    for c, word in enumerate(nlp(text)):
        if label[c] > .5:
            buffer.append(word.text)
    
    #print("--> " + str(' '.join(buffer)).replace(' - ', '-'))
    #print(str(' '.join(buffer)).replace(' - ', '-'))
    return str(' '.join(buffer)).replace(' - ', '-')


def text2vec(texts, nlp):
    labels = []
    seqlens = []
    for text in texts:
        vecbuff = []
        doc = nlp(text)
        if len(doc) == wide:
            seqlens.append(wide)
            for word in doc:
                vecbuff.append(word.vector)
        else: # can do raw else b/c clipped longs in pipleine.py
            seqlens.append(len(doc))
            for word in doc:
                vecbuff.append(word.vector)
            # pad rear w/ zeros
            while len(vecbuff) < wide:
                vecbuff.append(np.zeros(vec_size))
        labels.append(vecbuff)

    return labels, seqlens


def get_rand_batch(batch_size, data_x, data_y, nlp):
    indecies = list(range(len(data_x)))
    np.random.shuffle(indecies)
    batch_indexes = indecies[:batch_size] 
    raw_texts = [data_x[i] for i in batch_indexes]

    inputs, seqlens = text2vec(raw_texts, nlp)
    labels = [data_y[i] for i in batch_indexes]

    return np.nan_to_num(np.array(inputs)), np.nan_to_num(np.array(labels)), np.nan_to_num(np.array(seqlens)), raw_texts


# -------------------------- Loading Data ----------------------------------------------
train = pickle.load(open("train.dat", 'rb'))
valid = pickle.load(open("valid.dat", 'rb'))

train_x = [i[1] for i in train]
train_y = [i[0] for i in train]

test_x = [i[1] for i in valid]
test_y = [i[0] for i in valid]
print("[+] Data Imported [+]")
# ------------------------- Python Variables -------------------------------------------
wide = 56
vec_size = 300 # spacy
num_classes = wide

# Goddy: [4, 1024, 128]
num_LSTM_layers = 5
hidden_layer_size = 1024

# ------------------------- Network Architecture ---------------------------------------
_labels = tf.placeholder(tf.float32, shape=[None, num_classes], name='labels') # batch_size
_seqlens = tf.placeholder(tf.int32, shape=[None], name='seqlens') # batch_size

embed = tf.placeholder(tf.float32, shape=[None, wide, vec_size], name='embed')

# -------------- Network Specs ----------------
def make_cell(lstm_size):
    return tf.contrib.rnn.BasicLSTMCell(hidden_layer_size, forget_bias=1.0, state_is_tuple=True)

with tf.variable_scope("lstm"):
    #lstm_cell = tf.contrib.rnn.BasicLSTMCell(hidden_layer_size, forget_bias=1.0) # Basic LSTM Cell yo
    #outputs, states = tf.nn.dynamic_rnn(lstm_cell, embed, sequence_length=_seqlens, dtype=tf.float32) # original
    cell = tf.contrib.rnn.MultiRNNCell(cells=[make_cell(hidden_layer_size) for n in range(num_LSTM_layers)], state_is_tuple=True)
    outputs, states = tf.nn.dynamic_rnn(cell, embed, sequence_length=_seqlens, dtype=tf.float32)


weights =  tf.Variable(tf.truncated_normal([hidden_layer_size, num_classes], mean=0, stddev=0.01), name="weights")
biases = tf.Variable(tf.truncated_normal([num_classes], mean=0, stddev=0.01), name="biases") 

# ------------- Outputs
#final_output = tf.add(tf.matmul(states[1], weights), biases) #original
final_output = tf.add(tf.matmul(states[num_LSTM_layers-1][1], weights), biases)
pred = tf.nn.sigmoid_cross_entropy_with_logits(logits=final_output, labels=_labels)
final_output_sig = tf.sigmoid(final_output, name="final_output_sig")
cross_entropy = tf.reduce_mean(pred) # cost, loss

#-------------- Training
# --> RMS

# Original
train_step = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cross_entropy)
#train_step = tf.train.RMSPropOptimizer(0.001, 0.9, centered=True).minimize(cross_entropy)
#grads_and_vars = train_Step.compute_gradients(cross_entropy, )

# --> GradientDescent
#train_step = tf.train.GradientDescentOptimizer(.01).minimize(cross_entropy)
"""
optimizer = tf.train.GradientDescentOptimizer(.01)
gradients, variables = zip(*optimizer.compute_gradients(cross_entropy))
gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
train_step = optimizer.apply_gradients(zip(gradients, variables)) #https://stackoverflow.com/questions/36498127/how-to-apply-gradient-clipping-in-tensorflow/43486487
#train_step = tf.abs([1])
"""

# --> Adam
#train_step = tf.train.AdamOptimizer().minimize(cross_entropy)

# -------------- Accuracy
correct_prediction = tf.equal(tf.argmax(_labels, 1), tf.argmax(final_output, 1))
accuracy = (tf.reduce_mean(tf.cast(correct_prediction, tf.float32)))*100
"""
correct = tf.equal(final_output, tf.equal(_labels,1.0))
accuracy = tf.reduce_mean( tf.cast(correct, 'float') )
"""

info = {
    'test_acc': [],
    'test_step': [],

    'train_acc': [],
    'train_loss': [],
    'train_step': []
}

# ------------------------------------------- RUNNING ----------------------------------------------------
batch_size = 128
rounds = 10000
meta_name = "meta-" + str(num_LSTM_layers) + "-" + str(hidden_layer_size) + "-" + str(batch_size)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(rounds):
        x_batch, y_batch, seqlen_batch, raw_texts = get_rand_batch(batch_size, train_x, train_y, nlp)
        #print("x:", x_batch.shape, "y:", y_batch.shape, "s:", seqlen_batch.shape)
        sess.run(train_step, feed_dict={embed:x_batch, _labels:y_batch, _seqlens:seqlen_batch})

        if step % 10 == 0:
            x_test, y_test, seqlen_test, raw_texts = get_rand_batch(256, test_x, test_y, nlp)
            acc, pred = sess.run([accuracy, final_output_sig], feed_dict={embed:x_test, _labels:y_test, _seqlens:seqlen_test})
            for i in range(50):
                #print(y_test[i], raw_texts[i])
                #print("\n---------------------------------------")
                print("--------------------------------------------------------------------------------------------------------------------------------")
                print("[ Trgt:", label2text(y_test[i], raw_texts[i]), "] | ", raw_texts[i].strip())
                print("[ Pred:", label2text(pred[i], raw_texts[i]), "]")
                #print("     ---------- Pred ----------")
                #label2text(pred[i], raw_texts[i])

                """
                label2text(pred[i], raw_texts[i])
                for j in range(5):
                    print([y_test[i][j], pred[i][j]])
                """

            info['test_acc'].append(acc)
            info['test_step'].append(step)

            print(str(step) + '/' + str(rounds) + " R-init RAN: Accuracy at %d: %.5f" % (step, acc))
            print("-----------------------------------------------------------------------------------------------")
        elif step % 25 == 0:
            acc, loss = sess.run([accuracy, cross_entropy], feed_dict={embed:x_batch, _labels:y_batch, _seqlens:seqlen_batch})

            info['train_acc'].append(acc)
            info['train_loss'].append(loss)
            info['train_step'].append(step)

            print("Training Accuracy at %d: %.5f | Loss: %.5f" % (step, acc, loss))
            print("-----------------------------------------------------------------------------------------------")
            pickle.dump(info, open(meta_name + ".dat", 'wb'), -1)

        


"""
ToDo:

[OK]    get en_large
[NO]    def fuzzyMatcher(zeroVec)
[OK]    Better rule based system for month abbreviations (prepare mentally for this final push)

[OK]    Make ones_set

[OK]    Clean up FP from month abbreviations
2) Network Arch focus
    layer size
    lstm ct
    gru cell?
    try out on cnn?
[OK]    use better dict? 
            https://spacy.io/models/en#en_vectors_web_lg
[OK]    Also implement what training accuracy is to ensure its not overfitting, doubt it is but still
"""
