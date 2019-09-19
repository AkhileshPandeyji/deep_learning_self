import tensorflow as tf 
import numpy as np 
import os 
import time
import random
import collections
from tensorflow.contrib import rnn


# Helper Functions

def read_data(filepath):

	with open(filepath,'r') as file:
		lines = file.read().splitlines()
		lines = [line.strip() for line in lines]
		words = [word  for line in lines for word in line.split()]
		return words


words = read_data("C:\\Users\\pande\\OneDrive\\Desktop\\Workspace\\DeepLearning\\eminem_venom_lyrics.txt")

print(words)

# def convert_dic(words):
# 	count = collections.Counter(words).most_common()
# 	dictionary = dict()
# 	for word,_ in count:
# 		dictionary[word] = len(dictionary)
# 	reverse_dictionary = dict(zip(dictionary.values(),dictionary.keys()))
# 	vocab_size = len(dictionary)
# 	return dictionary,reverse_dictionary,vocab_size

def convert_dic2(words):
	words = list(set(words))
	vocab_size = len(words)
	dictionary = { i:word for i,word in enumerate(words) }
	reverse_dictionary = { word:i for i,word in enumerate(words) }
	return dictionary,reverse_dictionary,vocab_size

dic,reverse_dic,vocab_size = convert_dic2(words)

print(dic)
print(reverse_dic)
print(vocab_size)



# Initializations

num_hidden = 128
num_inputs = 1
timesteps = 3
batch_size = 10

learning_rate = 1e-3
decay_rate = 1e-5
display_step = 100
EPOCHS = 50000
num_words_gen = 100


# model placeholders and variables

X = tf.placeholder(tf.float32,[None,timesteps,num_inputs])
Y = tf.placeholder(tf.float32,[None,vocab_size])


W = tf.Variable(tf.random.normal([num_hidden,vocab_size]))
B = tf.Variable(tf.random.normal([vocab_size]))


# model
def RNN(x,w,b):

	x = tf.unstack(x,timesteps,1)
	
	lstm_cells = rnn.MultiRNNCell([rnn.BasicLSTMCell(num_hidden),rnn.BasicLSTMCell(num_hidden)])
	outputs,states = rnn.static_rnn(lstm_cells,x,dtype=tf.float32)

	return tf.matmul(outputs[-1],w)+b


Ylogits = RNN(X,W,B)
Ypred = tf.nn.softmax(Ylogits)

# saving checkpoints
checkpoint_name = "mymodels/rnn_textgen_lstm.ckpt"
saver = tf.train.Saver()


# Success Metrics = Accuracy+Loss+Optimizer

is_correct = tf.equal(tf.argmax(Ypred,1),tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(is_correct,tf.float32))

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits,labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate)
train_step = optimizer.minimize(loss)

# Training


init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)

	offset = 0
	record = offset
	end_offset = offset+timesteps
	
	steps = 0	
	for epoch in range(EPOCHS):

		steps+=1

		batch_x = []
		batch_y = []

	

		if end_offset >= (len(words)-batch_size):
			offset = 0
			end_offset = offset+timesteps

		for i in range(batch_size):
			tokens_x = words[offset+i:end_offset+i]
			vector_y = np.zeros(vocab_size)

			# print(end_offset+i)
			# print(words[end_offset+i])
			# print(reverse_dic[words[end_offset+i]])
			vector_y[reverse_dic[words[end_offset+i]]] = 1.0

			batch_x.append([reverse_dic[word] for word in tokens_x])
			batch_y.append([vector_y])


		offset+=batch_size
		end_offset+=batch_size

		batch_x = np.array(batch_x).reshape([batch_size,timesteps,num_inputs])
		batch_y = np.array(batch_y).reshape([batch_size,vocab_size])
		train_dict = {X:batch_x,Y:batch_y}
		sess.run(train_step,feed_dict=train_dict)

		if steps % display_step == 0 or steps == 1:
			a,l = sess.run([accuracy,loss],feed_dict=train_dict)
			yval = sess.run(tf.argmax(Ypred,1),feed_dict=train_dict)
			# print(yval)			
			
			print("Training : EPOCHS({}/{})====> Loss : {} ; Accuracy : {} .".format(steps,EPOCHS,l,a))
			for i in range(len(yval)):
				ylabel = dic[np.argmax(batch_y[i])]
				ypredlabel = dic[yval[i]]
				print("[{}] vs [{}]".format(ylabel,ypredlabel))

	saver.save(sess,checkpoint_name)

	while True:
		prompt = "Enter any 3 words from story:"
		words_user = str(input(prompt))

		if words_user == 'exit':
			break

		words_ = words_user.split()
		sentence = words_user

		if len(words_) != 3:
			continue

		try:
			for i in range(num_words_gen):	
				if i == 0:
					words_val= [reverse_dic[word] for word in words_]
				else :
					words_ = sentence.split()
					words_val = [reverse_dic[word] for word in words_]
					words_val = words_val[-timesteps:]

				words_val = np.array(words_val)
				words_val = words_val.reshape(-1,timesteps,num_inputs)
				pred_val = sess.run(tf.argmax(Ypred,1),feed_dict={X:words_val})			
				pred_word = dic[pred_val[0]]
				sentence += " "+pred_word
		
		except Exception as e:
			print("These Three words are not in Vocabulary dictionary !!!!! Try again")
			continue

		
		print(sentence)
		
print("Text Generator Execution Finished!!!")





 		

 
































