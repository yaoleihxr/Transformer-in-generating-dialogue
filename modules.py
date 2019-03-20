# -*- coding: utf-8 -*-
from __future__ import print_function
import tensorflow as tf
import numpy as np
import math

def normalize(inputs,
			epsilon = 1e-8,
			scope = "ln",
			reuse = None):
	'''
	Implement layer normalization

	Args:
		inputs: [Tensor], A tensor with two or more dimensions, where the first one is "batch_size"
		epsilon: [Float], A small number for preventing ZeroDivision Error
		scope: [String], Optional scope for "variable_scope"
		reuse: [Boolean], If to reuse the weights of a previous layer by the same name
	
	Returns:
		A tensor with the same shape and data type as "inputs"
	'''
	with tf.variable_scope(scope, reuse = reuse):
		# a.get_shape().as_list() --->a维度是(2,3)，那么这个返回就是 [2, 3]
		inputs_shape = inputs.get_shape()
		# params_shape就是最后的一个维度了
		params_shape = inputs_shape[-1 :]
		# input形如：[batch_size, height, width, kernels]
		# tf.nn.moments 计算返回的 mean 和 variance 作为 tf.nn.batch_normalization 参数调用。
		mean, variance = tf.nn.moments(inputs, [-1], keep_dims = True)
		beta = tf.Variable(tf.zeros(params_shape))
		gamma = tf.Variable(tf.ones(params_shape))
		normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
		# 按值相乘, 广播
		outputs = gamma * normalized + beta

	return outputs


def positional_encoding(inputs,
						vocab_size,
						num_units,
						zero_pad = True,
						scale = True,
						scope = "positional_embedding",
						reuse = None):
	'''
	Positional_Encoding for a given tensor.

	Args:
		inputs: [Tensor], A tensor contains the ids to be search from the lookup table, shape = [batch_size, 1 + len(inpt)]
		vocab_size: [Int], Vocabulary size
		num_units: [Int], Hidden size of embedding
		zero_pad: [Boolean], If True, all the values of the first row(id = 0) should be constant zero
		scale: [Boolean], If True, the output will be multiplied by sqrt num_units(check details from paper)
		scope: [String], Optional scope for 'variable_scope'
		reuse: [Boolean], If to reuse the weights of a previous layer by the same name

		Returns:
			A 'Tensor' with one more rank than inputs's, with the dimensionality should be 'num_units'
	'''

	# f = 10000.
	# position_block = np.broadcast_to(np.arange(vocab_size)[None, None, :], (inputs, num_units // 2, vocab_size)).astype('float32')
	# unit_block = np.broadcast_to(np.arange(num_units // 2)[None, :, None], (inputs, num_units // 2, vocab_size)).astype('float32')
	# rad_block = position_block / (f * 1.) ** (unit_block / (num_units // 2))
	# sin_block = np.sin(rad_block)
	# cos_block = np.cos(rad_block)

#	with tf.variable_scope(scope, reuse = reuse):

# 		input_one = tf.tile(tf.expand_dims(tf.range(tf.shape(inputs)[1]), 0), [tf.shape(inputs)[0], 1])
# 		position_block = tf.tile(tf.expand_dims(tf.range(vocab_size), 1), [1, num_units // 2])
# 		unit_block = tf.tile(tf.expand_dims(tf.range(num_units // 2), 0), [vocab_size, 1])
# 		rad_block = tf.pow(tf.div(position_block, tf.multiply(10000, 1)), tf.div(unit_block, num_units // 2))
# 		sin_block = tf.sin(tf.cast(rad_block, tf.float32))
# 		cos_block = tf.cos(tf.cast(rad_block, tf.float32))
# 		lookup_table = tf.concat([sin_block, cos_block], axis = 1)

# 		# lookup_table = tf.get_variable('lookup_table',
# 		# 								dtype = tf.float32,
# 		# 								shape = [vocab_size, num_units],
# 		# 								initializer = tf.contrib.layers.xavier_initializer())

# 		if zero_pad:

# 			lookup_table = tf.concat((tf.zeros(shape = [1, num_units]),
# 									lookup_table[1:, :]), 0)
# 		outputs = tf.nn.embedding_lookup(lookup_table, input_one)
		
# 		if scale:
# 			outputs = outputs * math.sqrt(num_units)

# 	return outputs
	"""
	    inputs (batch_size, 1+len(inputs)) 那么N就是batch_size, 然后T就是maxlen，大小为10
	    num_units 就是隐层单元的个数，维度的大小
	"""
	N, T = inputs.get_shape().as_list()
 	with tf.variable_scope(scope, reuse=reuse):
		# axes:0 重复N次, axes:1 不变
		position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1])

		# First part of the PE function: sin and cos argument
		position_enc = np.array([
		    [pos / np.power(10000, 2.*i/num_units) for i in range(num_units)]
		    for pos in range(T)])

		# Second part, apply the cosine to even columns and sin to odds.
		position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
		position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1

		# Convert to a tensor
		lookup_table = tf.convert_to_tensor(position_enc)

		if zero_pad:
		    lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
					      lookup_table[1:, :]), 0)
		outputs = tf.nn.embedding_lookup(lookup_table, position_ind)

		if scale:
		    outputs = outputs * num_units**0.5
		
	return tf.cast(outputs, tf.float32)


def embedding(inputs,
			vocab_size,
			num_units,
			zero_pad = True,
			scale = True,
			scope = "embedding",
			reuse = None):
	'''
	Embed a given tensor.

	Args:
		inputs: [Tensor], A tensor contains the ids to be search from the lookup table
		vocab_size: [Int], Vocabulary size
		num_units: [Int], Hidden size of embedding
		zero_pad: [Boolean], If True, all the values of the first row(id = 0) should be constant zero
		scale: [Boolean], If True, the output will be multiplied by sqrt num_units(check details from paper)
		scope: [String], Optional scope for 'variable_scope'
		reuse: [Boolean], If to reuse the weights of a previous layer by the same name

		Returns:
			A 'Tensor' with one more rank than inputs's, with the dimensionality should be 'num_units'
	'''
	"""
		inputs传进来就(batch_size, 10)
		lookup_table维度(vocab_size, 512)，进行了随机的初始化
	"""
	# shape = [vocabsize, 8]
	with tf.variable_scope(scope, reuse = reuse):
		lookup_table = tf.get_variable('lookup_table',
										dtype = tf.float32,
										shape = [vocab_size, num_units],
										initializer = tf.contrib.layers.xavier_initializer())

		if zero_pad:
			''' 
				tf.zeros 维度(1, 512)
				lookup_table[1:, :]的目的是抛开了<PAD>这玩意儿，赋值为0，然后进行了合并
				现在look  _table维度还是(vocab_size, 512  ) 
			'''
			lookup_table = tf.concat((tf.zeros(shape = [1, num_units]),
									lookup_table[1:, :]), 0)
		outputs = tf.nn.embedding_lookup(lookup_table, inputs)

		if scale:
			outputs = outputs * math.sqrt(num_units)

	return outputs


def multihead_attention(queries,
						keys,
						num_units = None,
						num_heads = 8,
						dropout_rate = 0,
						is_training = True,
						causality = False,
						scope = "multihead_attention",
						reuse = None):
	'''
	Implement multihead attention

	Args:
		queries: [Tensor], A 3-dimensions tensor with shape of [N, T_q, S_q]
		keys: [Tensor], A 3-dimensions tensor with shape of [N, T_k, S_k]
		num_units: [Int], Attention size
		num_heads: [Int], Number of heads
		dropout_rate: [Float], A ratio of dropout
		is_training: [Boolean], If true, controller of mechanism for dropout
		causality: [Boolean], If true, units that reference the future are masked
		scope: [String], Optional scope for "variable_scope"
		reuse: [Boolean], If to reuse the weights of a previous layer by the same name
	
	Returns:
		A 3-dimensions tensor with shape of [N, T_q, S]
	'''
	""" 
		queries = self.enc  (batch_size, 10 ,512)==[N, T_q, S] keys也是self.enc  
	    num_units =512, num_heads =10
	"""
	with tf.variable_scope(scope, reuse = reuse):
		if num_units is None:
			# length of sentence
			num_units = queries.get_shape().as_list()[-1]

		# Linear layers in Figure 2(right)
		""" 
			Linear layers in Figure 2(right) 就是Q、K、V进入scaled Dot-product Attention前的Linear的操作
		    首先是进行了全连接的线性变换
		    shape = [N, T_q, S]  (batch_size, 10 ,512)， S可以理解为512
		"""
		Q = tf.layers.dense(queries, num_units, activation = tf.nn.relu)
		# shape = [N, T_k, S]
		K = tf.layers.dense(keys, num_units, activation = tf.nn.relu)
		# shape = [N, T_k, S]
		V = tf.layers.dense(keys, num_units, activation = tf.nn.relu)

		# Split and concat
		'''
		    Q_、K_、V_就是权重WQ、WK、WV。
		    shape (batch_size*8， 10, 512/8=64)
		'''
		# shape = [N*h, T_q, S/h]
		Q_ = tf.concat(tf.split(Q, num_heads, axis = 2), axis = 0)
		# shape = [N*h, T_k, S/h]
		K_ = tf.concat(tf.split(K, num_heads, axis = 2), axis = 0)
		# shape = [N*h, T_k, S/h]
		V_ = tf.concat(tf.split(V, num_heads, axis = 2), axis = 0)

		# shape = [N*h, T_q, T_k]
		outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))

		# Scale
		outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

		# Masking
		# 这里的tf.reduce_sum进行了降维，由三维降低到了2维度，然后是取绝对值，转成0-1之间的值
		'''[N, T_k, 512]------> [N, T_k] -----》[N*h, T_k] -----》[N*h, T_q, T_k] '''
		# shape = [N, T_k]
		key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis = -1)))
		# shape = [N*h, T_k]
		key_masks = tf.tile(key_masks, [num_heads, 1])
		# shape = [N*h, T_q, T_k]
		key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])

		# If key_masks == 0 outputs = [1]*length(outputs) 为了使softmax后概率为0
		paddings = tf.ones_like(outputs) * (-math.pow(2, 32) + 1)
		# shape = [N*h, T_q, T_k]
		outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)

		if causality:
			# reduce dims : shape = [T_q, T_k]
			diag_vals = tf.ones_like(outputs[0, :, :])
			# shape = [T_q, T_k]
			# use triangular matrix to ignore the affect from future words
			# like : [[1,0,0]
			#         [1,2,0]
			#         [1,2,3]]
			tril = tf.contrib.linalg.LinearOperatorTriL(diag_vals).to_dense()
			# shape = [N*h, T_q, T_k]
			masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])

			paddings = tf.ones_like(masks) * (-math.pow(2, 32) + 1)
			# shape = [N*h, T_q, T_k]
			outputs = tf.where(tf.equal(masks, 0), paddings, outputs)

		# Output Activation
		outputs = tf.nn.softmax(outputs)

		# Query Masking
		# shape = [N, T_q]
		query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis = -1)))
		# shape = [N*h, T_q]
		query_masks = tf.tile(query_masks, [num_heads, 1])
		# shape = [N*h, T_q, T_k]
		query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])
		outputs *= query_masks
		# 对空缺的词概率置位0

		# Dropouts
		outputs = tf.layers.dropout(outputs, rate = dropout_rate, training = tf.convert_to_tensor(is_training))

		# Weighted sum
		# shape = [N*h, T_q, S/h]
		outputs = tf.matmul(outputs, V_)

		# Restore shape
		# shape = [N, T_q, S]
		outputs = tf.concat(tf.split(outputs, num_heads, axis = 0), axis = 2)

		# Residual connection
		outputs += queries

		# Normalize
		# shape = [N, T_q, S]
		outputs = normalize(outputs)

	return outputs

def feedforward(inputs,
				num_units = [2048, 512],
				scope = "multihead_attention",
				reuse = None):
	'''
	Position-wise feed forward neural network

	Args:
		inputs: [Tensor], A 3d tensor with shape [N, T, S]
		num_units: [Int], A list of convolution parameters
		scope: [String], Optional scope for "variable_scope"
		reuse: [Boolean], If to reuse the weights of a previous layer by the same name 
	
	Return:
		A tensor converted by feedforward layers from inputs
	'''

	with tf.variable_scope(scope, reuse = reuse):
		# params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1, \
				  # "activation": tf.nn.relu, "use_bias": True}
		# outputs = tf.layers.conv1d(inputs = inputs, filters = num_units[0], kernel_size = 1, activation = tf.nn.relu, use_bias = True)
		# outputs = tf.layers.conv1d(**params)
		params = {"inputs": inputs, "num_outputs": num_units[0],"activation_fn": tf.nn.relu}
		outputs = tf.contrib.layers.fully_connected(**params)

		# params = {"inputs": inputs, "filters": num_units[1], "kernel_size": 1, \
		# 		  "activation": None, "use_bias": True}
		params = {"inputs": inputs, "num_outputs": num_units[1], "activation_fn": None}
		# outputs = tf.layers.conv1d(inputs = inputs, filters = num_units[1], kernel_size = 1, activation = None, use_bias = True)
		# outputs = tf.layers.conv1d(**params)
		outputs = tf.contrib.layers.fully_connected(**params)

		# residual connection
		outputs += inputs

		outputs = normalize(outputs)

	return outputs

def label_smoothing(inputs, epsilon = 0.1):
	'''
	Implement label smoothing

	Args:
		inputs: [Tensor], A 3d tensor with shape of [N, T, V]
		epsilon: [Float], Smoothing rate

	Return:
		A tensor after smoothing
	'''
	''' 
		inputs的维度应该是(batch_size, sentense_length, vector dimension)
	    N就是batch_size, T就是句子的长度，V就是向量的维度大小
	'''
	K = inputs.get_shape().as_list()[-1]
	return ((1 - epsilon) * inputs) + (epsilon / K)
