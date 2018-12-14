import numpy as np
import random

class RatioDataGenerator(object):
	def __init__(self, batch_size, type, te_max_iter=100, verbose=1):
		assert type in ['train', 'test']
		self._batch_size_ = batch_size
		self._type_ = type
		self._te_max_iter_ = te_max_iter
		self._verbose_ = verbose



	def generate(self, xs, ys):
		batch_size = self._batch_size_
		x = xs[0]
		y = ys[0]
		(n_samples, n_labs) = y.shape
		# (samples, 527) -> (527) 
		# where each element contain the number of samples that are this class
		n_samples_list = np.sum(y, axis=0)
		
		lb_list = list(range(n_labs))

		if self._verbose_ == 1:
			print("n_samples_list: %s" % (n_samples_list,))

		index_list = []
		for i in range(n_labs):
			index_list.append(np.where(y[:, i] == 1)[0])

		for i in range(n_labs):
			np.random.shuffle(index_list[i])

		queue = []
		pointer_list = [0] * n_labs
		len_list = [len(e) for e in index_list]
		iter = 0
		while True:
			#if(self._type_) == 'test' and (iter == self._te_max_iter_):
			#	break
			#iter += 1
			batch_x = []
			batch_y = []
			while len(queue) < batch_size:
				np.random.shuffle(lb_list)
				queue += lb_list

			batch_idx = queue[0 : batch_size]
			queue[0 : batch_size] = []

			n_per_class_list = [batch_idx.count(idx) for idx in range(n_labs)]

			for i in range(n_labs):
				if (pointer_list[i] + n_per_class_list[i]) >= len_list[i]:
					pointer_list[i] = 0
					np.random.shuffle(index_list[i])

				per_class_batch_idx = index_list[i][pointer_list[i] : pointer_list[i] + n_per_class_list[i]]
				batch_x.append(x[per_class_batch_idx])
				batch_y.append(y[per_class_batch_idx])
				pointer_list[i] += n_per_class_list[i]

			batch_x = np.concatenate(batch_x, axis=0)
			batch_y = np.concatenate(batch_y, axis=0)
			yield (batch_x, batch_y)























