import tensorflow as tf

n = int(input("1부터 n까지 합 : (n을 입력하시오)"))
i = tf.Variable(1)
sum = tf.Variable(0)

summing = sum.assign_add(i)
incrementing = i.assign_add(1)

init = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init)
	for i in range(n):
		summing.eval()
		incrementing.eval()
	print(sum.eval())
