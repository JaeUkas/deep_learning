import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn import datasets
import numpy as np
from sklearn.preprocessing import MinMaxScaler

'''
next_batch function
'''
def next_batch(X, y, start, batch_size):
    num_examples = len(X) #X.shape[0]
    assert batch_size <= num_examples
    end = start + batch_size

    # when all trainig data have been already used, it is reorder randomly
    if end > num_examples:
        remain = batch_size - (num_examples % batch_size)
        perm = np.arange(num_examples)
        np.random.shuffle(perm)

        X = np.concatenate((X[start:], X[perm[:remain]]), axis=0)
        y = np.concatenate((y[start:],y[perm[:remain]]), axis=0)

        return X, y, 0

    return X[start:end], y[start:end], end

'''
main 
'''

# 상수 정의
n_inputs = 4 # IRIS
n_hidden1 = 100
n_hidden2 = 100
n_outputs = 3

# 학습 입력 데이터와 타겟 데이터 입력 노드 정의
inputs = tf.placeholder(tf.float32, shape=(None, n_inputs), name="inputs")
targets = tf.placeholder(tf.int32, shape=(None), name="targets")
one_hot_ecoded_targets = tf.one_hot(targets, n_outputs)

# 인공 신경망 네트워크 구성
hidden1 = tf.layers.dense(inputs, n_hidden1, activation=tf.nn.relu, name="hidden1")
hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=tf.nn.relu, name="hidden2")
outputs = tf.layers.dense (hidden2, n_outputs, name="outputs")

# 손실 값 계산 노드 정의
loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=one_hot_ecoded_targets, logits=outputs)
total_loss = tf.reduce_mean(loss, name="total_loss")

# 옵티마이저 정의
learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
training_op = optimizer.minimize(total_loss)

# 변수 초기화 노드 및 변수 저장 노드 정의
init = tf.global_variables_initializer()
saver = tf.train.Saver()

# 성능 측정 노드 정의
correct_list = tf.nn.in_top_k(outputs, targets, 1)
accuracy = tf.count_nonzero(correct_list, dtype=tf.int32)/tf.size(correct_list)
#accuracy = tf.reduce_mean(tf.cast(correct_list, tf.float32))

# 학습 데이터 읽어 오기
iris = datasets.load_iris()
x_data = iris.data
y_data = iris.target

minmax_scaler = MinMaxScaler()
minmax_scaler.fit(x_data)
x_data = minmax_scaler.transform(x_data)
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=3, stratify=y_data)

# 학습 패러미터 설정
n_epochs = 100; batch_size = 5; num_trains = len(x_train)

# 학습 실행
with tf.Session() as sess:
   init.run()
   for epoch in range(n_epochs):
       start = 0
       for iteration in range(num_trains // batch_size):
           batch_x, batch_y, start = next_batch(x_train, y_train, start, batch_size)
           batch_y = np.reshape(batch_y, (batch_size, 1))
           sess.run(training_op, feed_dict= {inputs: batch_x, targets: batch_y})
       acc_train = accuracy.eval(feed_dict= {inputs: x_train, targets: y_train})
       acc_test = accuracy.eval(feed_dict= {inputs: x_test, targets: y_test})
       print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)
   save_path = saver.save(sess, "./my_model_final.ckpt")