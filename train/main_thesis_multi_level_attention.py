## Title: Training AudioSet using multi-level attention model
## Author: Zhicun Xu (zhicun.xu@qq.com)
## Date: 30.Nov.2018
## Code: This is a modified version for keras. Reference is https://github.com/qiuqiangkong/ICASSP2018_audioset
## Refereces:	
##     Kong, Q., Xu, Y., Wang, W. and Plumbley, M.D., 2018, April. Audio set classification with attention model: A probabilistic perspective. In 2018 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (pp. 316-320). IEEE.   
##     Yu, C., Barsim, K.S., Kong, Q. and Yang, B., 2018. Multi-level Attention Model for Weakly Supervised Audio Classification. arXiv preprint arXiv:1803.02353.
import argparse
import os
import time
import sys
import numpy as np
import logging
import h5py
import _pickle as cPickle
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, LSTM, Dropout, TimeDistributed, Input, Multiply, Lambda, Concatenate, BatchNormalization
from keras.optimizers import Adam 
from keras import backend as K
from sklearn import metrics
#from beautifultable import BeautifulTable
import pandas as pd
# self build
import prepare_data as pp_data
from data_generator import RatioDataGenerator
####MODELS
## 3-layer DNN model
def get_dnn():
#		output: a keras model
    n_hid = [600, 600, 600]
    n_out = 527
    #######
    #1st Hidden Layer with Batch Normalization and Dropout
    IN = Input(shape=(10, 128))
    h_0 = BatchNormalization()(IN)
    h_1 = Dense(n_hid[0])(h_0)
    h_1 = BatchNormalization()(h_1)
    h_1 = Activation('relu')(h_1)
    h_1 = Dropout(rate=0.4)(h_1)
    #2nd Hidden Layer with Batch Normalization and Dropout
    h_2 = Dense(n_hid[1])(h_1)
    h_2 = BatchNormalization()(h_2)
    h_2 = Activation('relu')(h_2)
    h_2 = Dropout(rate=0.4)(h_2)
    #3rd Hidden Layer with Batch Normalization and Dropout
    h_3 = Dense(n_hid[2])(h_2)
    h_3 = BatchNormalization()(h_3)
    h_3 = Activation('relu')(h_3)
    h_3 = Dropout(rate=0.4)(h_3)
    out = Dense(n_out, name='output')(h_3)
    out = BatchNormalization()(out)
    out = Activation('sigmoid')(out)
    out = GlobalAveragePooling1D()(out)
    model = Model(inputs=IN, outputs=out)
    return model
## LSTM model
#		output: a keras model
def get_lstm():
    IN = Input(shape=(10, 128))
    h = BatchNormalization()(IN)
    h = LSTM(600, dropout=0.4)(h)
    h = Dense(527)(h)
    h = BatchNormalization()(h)
    out = Activation('sigmoid')(h)
    model = Model(inputs=IN, outputs=out)
    return model
## Multi-level attention model
#		input: list of numbers representing which levels are used. e.g. levels = [1,3] means level 1 and level 3 are used.
#		output: a keras model
def get_ml_attention(levels):
	n_hid = [600, 600, 600]
	n_out = 527
#######
	#1st Hidden Layer with Batch Normalization and Dropout
	IN = Input(shape=(10, 128))
	h_0 = BatchNormalization()(IN)
	h_1 = Dense(n_hid[0])(h_0)
	h_1 = BatchNormalization()(h_1)
	h_1 = Activation('relu')(h_1)
	h_1 = Dropout(rate=0.4)(h_1)
    #2nd Hidden Layer with Batch Normalization and Dropout
	h_2 = Dense(n_hid[1])(h_1)
	h_2 = BatchNormalization()(h_2)
	h_2 = Activation('relu')(h_2)
	h_2 = Dropout(rate=0.4)(h_2)
	#3rd Hidden Layer with Batch Normalization and Dropout
	h_3 = Dense(n_hid[2])(h_2)
	h_3 = BatchNormalization()(h_3)
	h_3 = Activation('relu')(h_3)
	h_3 = Dropout(rate=0.4)(h_3)
    # 1-A Attention
	cla_1 = Dense(n_out, name='cla_1_A')(h_1)
	cla_1 = BatchNormalization()(cla_1)
	cla_1 = Activation('sigmoid')(cla_1)
	att_1 = Dense(n_out, name='att_1_A')(h_1)
	att_1 = BatchNormalization()(att_1)
	att_1 = Activation('softmax')(att_1)
	out_1_A = Lambda(_attention, output_shape=_att_output_shape)([cla_1, att_1])
	# 2-A Attention
	cla_2 = Dense(n_out, name='cla_2_A')(h_2)
	cla_2 = BatchNormalization()(cla_2)
	cla_2 = Activation('sigmoid')(cla_2)
	att_2 = Dense(n_out, name='att_2_A')(h_2)
	att_2 = BatchNormalization()(att_2)
	att_2 = Activation('softmax')(att_2)
	out_2_A = Lambda(_attention, output_shape=_att_output_shape)([cla_2, att_2])
	# 3-A Attention
	cla_3 = Dense(n_out, name='cla_3_A')(h_3)
	cla_3 = BatchNormalization()(cla_3)
	cla_3 = Activation('sigmoid')(cla_3)
	att_3 = Dense(n_out, name='att_3_A')(h_3)
	att_3 = BatchNormalization()(att_3)
	att_3 = Activation('softmax')(att_3)
	out_3_A = Lambda(_attention, output_shape=_att_output_shape)([cla_3, att_3])
	# Concatenate two Attention Layers
	concatenate_out = []	
	if 1 in levels: 
		print('#level 1#')
		concatenate_out.append(out_1_A)
	if 2 in levels: 
		print('#level 2#')
		concatenate_out.append(out_2_A)
	if 3 in levels: 
		print('#level 3#')
		concatenate_out.append(out_3_A)
	if len(concatenate_out) == 1: 
		out = concatenate_out[0]
	else: out = Concatenate(axis=1)(concatenate_out) 
	#out = Concatenate(axis=1)([out_3_A, out_2_A]) 
	# Add DNN for classification
	out = Dense(n_out, name='output')(out)
	out = BatchNormalization()(out)

	out = Activation('sigmoid')(out)
	model = Model(inputs=IN, outputs=out)
	return model

## attention mechanism, weighted sum
def _attention(inputs):
	[cla, att] = inputs
	_eps = 1e-7
	att = K.clip(att, _eps, 1. - _eps)
	norm_att = att / K.sum(att, axis=1)[:, None, :]
	return K.sum(cla * norm_att, axis=1)

def _att_output_shape(input_shape):
    #print("shape of input_shape:", input_shape)
    return tuple([input_shape[0][0], input_shape[0][2]])
## evalution. MAP is returned
#		probabilties are saved in 'out_dir'
def eval(model, x, y, out_dir, out_probs_dir, md_iter):
	pp_data.create_folder(out_dir)

	# Predict
	t1 = time.time()
	(n_clips, n_time_, n_freq) = x.shape
	(x, y) = pp_data.transform_data(x, y)
	prob = model.predict(x)
	prob = prob.astype(np.float32)
	print("The %d time into evalution." %md_iter)
	if out_probs_dir:
		pp_data.create_folder(out_probs_dir)
		out_prob_path = os.path.join(out_probs_dir, "prob_%d_iters.p" % md_iter)
		#cPickle.dump(prob, open(out_prob_path, 'wb'))
	# Dump predicted probabilities for future average
	n_out = y.shape[1]
	stats =[]
	t1 = time.time()
	for k in range(n_out):
		(precisions, recalls, thresholds) = metrics.precision_recall_curve(y[:, k], prob[:, k])
		avg_precision = metrics.average_precision_score(y[:, k], prob[:, k], average=None)
		(fpr, tpr, thresholds) = metrics.roc_curve(y[:, k], prob[:, k])
		auc = metrics.roc_auc_score(y[:, k], prob[:, k], average=None)
		eer = pp_data.eer(prob[:, k], y[:, k])
		skip = 1000
		dict = {'precisions': precisions[0::skip], 'recalls': recalls[0::skip], 'AP': avg_precision,
				'fpr': fpr[0::skip], 'fnr': 1. - tpr[0::skip], 'auc': auc}
		stats.append(dict)

	logging.info("Callback time: %s" %(time.time() - t1,))
	dump_path = os.path.join(out_dir, "model_%d_iters.p" % (md_iter,))
	cPickle.dump(stats, open(dump_path, 'wb'))
	mAP = np.mean([e['AP'] for e in stats])	
	logging.info("mAP of %d iteration: %f" % (md_iter, mAP))
	return mAP
## training the model with parameters
def train(args):
	EVAL_MAP = -1000.
	PATIENCE = 0
	data_dir = args.data_dir
	workspace = args.workspace
	tag = args.tag
	levels = args.levels
	# Path for the hdf5 dara
	bal_train_path = os.path.join(data_dir, "bal_train.h5")
	unbal_train_path = os.path.join(data_dir, "unbal_train.h5")
	eval_path = os.path.join(data_dir, "eval.h5")

	# Load data
	t1 = time.time()
	(tr_x1, tr_y1, tr_id_list1) = pp_data.load_data(bal_train_path)
	(tr_x2, tr_y2, tr_id_list2) = pp_data.load_data(unbal_train_path)
	(eval_x, eval_y, eval_id_list) = pp_data.load_data(eval_path)
	#tr_x = tr_x1
	#tr_y = tr_y1
	#tr_id_list = tr_id_list1
	tr_x = np.concatenate((tr_x1, tr_x2))
	tr_y = np.concatenate((tr_y1, tr_y2))
	tr_id_list = tr_id_list1 + tr_id_list2

	logging.info("Loading dat time: %s s" % (time.time() - t1))
	logging.info(tr_x1.shape, tr_x2.shape)
	logging.info("tr_x.shape: %s" % (tr_x.shape,))

	(_, n_time, n_freq) = tr_x.shape

	# Build Model
	
	model = get_ml_attention(levels)
	logging.info(model.to_json())
	# Optimization method
	optimizer = Adam(lr=args.lr)
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
	#logging.info(model.summary())
	# 
	batch_size = 500
	tr_gen = RatioDataGenerator(batch_size=batch_size, type='train')
	# Save Model every call_freq iterations
	model_iter = 0
	call_freq = 1000
	dump_fd = os.path.join(workspace, "models", pp_data.get_filename(__file__))
	pp_data.create_folder(dump_fd)


	# Train
	stat_dir = os.path.join(workspace, "stats", pp_data.get_filename(__file__))
	pp_data.create_folder(stat_dir)
	prob_dir = os.path.join(workspace, "probs", pp_data.get_filename(__file__))
	pp_data.create_folder(prob_dir)

	tr_time = time.time()
	
	for (tr_batch_x, tr_batch_y) in tr_gen.generate(xs=[tr_x], ys=[tr_y]):
		# Computes stats every several iterations
		print(model_iter)
		if model_iter % call_freq == 0: # every 1000 iterations
			# Stats of evaluation dataset
			t1 = time.time()
			eval_MAP = eval(model=model, x=eval_x, y=eval_y,
							out_dir=os.path.join(stat_dir, "eval"),
							out_probs_dir=os.path.join(prob_dir, "eval"),
							md_iter=model_iter)
			
			logging.info("Evaluate evaluation-set time: %s" % (time.time() - t1,))
			if eval_MAP >= EVAL_MAP:
				#md_name = "/scratch/work/xuz2/model_" + tag + "_.h5"
				md_name = tag + "_.h5"
				model.save(md_name)
				EVAL_MAP = eval_MAP
				PATIENCE = 0
			else: 
				PATIENCE += 1
				logging.info("Patience now: %d" % (PATIENCE,))
				if PATIENCE >= 10:
					break					
			#	print("Training stop at %s iterations" % (model_iter,))				
			#	break
			# Stats of training dataset
			#t1 =time.time()
			#tr_bal_err = eval(model=model, x=tr_x1, y=tr_y1,

			#				  out_dir=os.path.join(stat_dir, "train_bal"),
			##				  out_probs_dir=None,
			#				  md_iter=model_iter)
			#logging.info("Evaluate tr_bal time: %s" % (time.time() - t1,))

			# Save Model
			#if eval_MAP > 0.342:
			#	md_name = "/scratch/work/xuz2/model_" + str(model_iter) + "_.h5"
			#	model.save(md_name)
			


		# Update params
		(tr_batch_x, tr_batch_y) = pp_data.transform_data(tr_batch_x, tr_batch_y)
		model.train_on_batch(tr_batch_x, tr_batch_y)

		model_iter += 1

			# Stop training when maximum iteration achieves
		if model_iter == call_freq*151:
			break

def get_avg_stats(args, file_name, bgn_iter, fin_iter, interval_iter):
	workspace = args.workspace	
	stats_dir = os.path.join(workspace, "stats", file_name, "eval")	
	iters = range(bgn_iter, fin_iter, interval_iter)
	
	mAP = []
	mAUC = []
	d_prime = []
	for iter in iters:
		pickle_path = os.path.join(stats_dir, "model_%d_iters.p" % iter)
		with open(pickle_path, "rb") as f:
			stats = cPickle.load(f)
		APs = [stats[i]['AP'] for i in range(527)]
		AUCs = [stats[i]['auc'] for i in range(527)]
		mAP.append(sum(APs)/527.0)
		mAUC.append(sum(AUCs)/527.0)
		d_prime.append(pp_data.d_prime(sum(AUCs)/527.0))
	raw_data = {"iterations": iters, "----mAP----": mAP, "----AUC----": mAUC, "--d-prime--": d_prime}
	df = pd.DataFrame(raw_data, columns =["iterations", "----mAP----", "----AUC----", "--d-prime--"])
	print(df.to_string())
	#table = BeautifulTable()	
	#table.column_headers = ["iterations/1k", "----mAP----", "----AUC----", "--d-prime--"]	

	#for iter in iters:
	#	pickle_path = os.path.join(stats_dir, "model_%d_iters.p" % iter)
	#	with open(pickle_path, "rb") as f:
	#		stats = cPickle.load(f)
	#	APs = [stats[i]['AP'] for i in range(527)]
	#	AUCs = [stats[i]['auc'] for i in range(527)]
	#	mAP = sum(APs)/527.0
	#	mAUC = sum(AUCs)/527.0
	#	table.append_row([iter/1000, mAP, mAUC, pp_data.d_prime(mAUC)])
		#logging.info("Begin, End, Step Iterations: %d, %d, %d" % (bgn_iter, fin_iter, interval_iter))
		#logging.info("The %d iterations has \n mAP: %f and \n AUC: %f \n d-prime: %f" % (iter, mAP, mAUC, pp_data.d_prime(mAUC)))
	
	#logging.info("{}{}".format("\n\n", table))
		
if __name__ == '__main__':

	# Add Arguments
	parser = argparse.ArgumentParser()
	subparsers = parser.add_subparsers(dest='mode')

	# Subparser 1: train
	parser_train = subparsers.add_parser('train')
		# Add arguments to subparser 1
	parser_train.add_argument('--lr', type=float, default=1e-3)
	parser_train.add_argument('--data_dir', type=str)
	parser_train.add_argument('--workspace', type=str) # workspace/logs/main
	parser_train.add_argument('--levels', nargs='*', type =int)
	parser_train.add_argument('--tag', type=str)
	# Subparser 2: get average stats
	parser_get_avg_stats = subparsers.add_parser('get_avg_stats')
		# Add arguments to subparser 2
	#parser_get_avg_stats.add_argument('--data_dir', type=str)
	parser_get_avg_stats.add_argument('--workspace', type=str)	
	parser_get_avg_stats.add_argument("--begin", type=int)
	parser_get_avg_stats.add_argument("--step", type=int)
	parser_get_avg_stats.add_argument("--end", type=int)	
	args = parser.parse_args()

	# Logs

	logs_dir = os.path.join(args.workspace, "logs", pp_data.get_filename(__file__))
	pp_data.create_folder(logs_dir)
	logging = pp_data.create_logging(logs_dir, filemode='w')
	logging.info(os.path.abspath(__file__))
	logging.info(sys.argv)

	if args.mode == 'train':
		train(args)
	elif args.mode == 'get_avg_stats':
		file_name=pp_data.get_filename(__file__)
		bgn_iter, fin_iter, interval_iter = args.begin, args.end, args.step
		get_avg_stats(args, file_name, bgn_iter, fin_iter, interval_iter)
	else:
		raise Exception("Error!")
