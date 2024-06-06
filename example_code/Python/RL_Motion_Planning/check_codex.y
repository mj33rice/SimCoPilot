import argparse
import datetime
import json
import os
import pickle
import sys
import time
from abc import ABC
from argparse import Namespace
from collections import OrderedDict
from typing import Dict, Union, List
from typing import Tuple

import logging
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import wandb
from keras.layers import Dense
from tf_agents.replay_buffers.table import Table
from tqdm import tqdm

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = os.path.join('./logging', 'BC', 'run' + current_time)
if not os.path.exists(log_dir):
	os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(filename=os.path.join(log_dir, 'logs.txt'), filemode='w',
					format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
					datefmt='%m/%d/%Y %H:%M:%S',
					level=logging.INFO)
logger = logging.getLogger(__name__)


def get_buffer_shape(args) -> Dict[str, Tuple[int, ...]]:
	buffer_shape = {
		'prev_goals': (args.horizon, args.ag_dim),
		'prev_skills': (args.horizon, args.c_dim),
		'states': (args.horizon + 1, args.s_dim),
		'env_goals': (args.horizon + 1, args.g_dim),
		'curr_goals': (args.horizon, args.ag_dim),
		'curr_skills': (args.horizon, args.c_dim),
		'states_2': (args.horizon, args.s_dim),
		'actions': (args.horizon, args.a_dim),
		'successes': (args.horizon,),
		'distances': (args.horizon,),
		'has_gt_skill': (args.horizon,),
	}
	return buffer_shape


def state_to_goal(num_objs: int):
	"""
	Converts state to goal. (Achieved Goal Space)
	If obj_identifiers is not None, then it further filters the achieved goals based on the object/skill id.
	"""
	
	@tf.function(experimental_relax_shapes=True)  # Imp otherwise code will be very slow
	def get_goal(states: tf.Tensor, obj_identifiers: tf.Tensor = None):
		# Get achieved goals
		goals = tf.map_fn(lambda x: x[3: 3 + num_objs * 3], states, fn_output_signature=tf.float32)
		return goals
	
	return get_goal


def repurpose_skill_seq(args, skill_seq):
	"""
	Repurpose the skill sequence to be used for training the policy. Use value of wrap_skill_id
	= "0": no change
	= "1": wrap pick/grab/drop:obj_id to pick/grab/drop
	= "2": wrap pick:obj_id to pick/grab/drop:obj_id to obj_id
	:param skill_seq: one-hot skill sequence of shape (n_trajs, horizon, c_dim)
	:return: tensor of shape (n_trajs, horizon, c_dim) and type same as skill_seq
	"""
	if args.env_name != 'OpenAIPickandPlace':
		tf.print("Wrapping skill sequence is currently only supported for PnP tasks!")
		sys.exit(-1)
	
	if args.wrap_level == "0":
		return skill_seq
	elif args.wrap_level == "1":
		# wrap by i = j % 3 where i is the new position of skill originally at j. Dim changes from c_dim to 3
		skill_seq = tf.argmax(skill_seq, axis=-1)
		skill_seq = skill_seq % 3
		# Convert back to one-hot
		skill_seq = tf.one_hot(skill_seq, depth=3)
		return skill_seq
	elif args.wrap_level == "2":
		# wrap such that 0/1/2 -> 0, 3/4/5 -> 1, 6/7/8 -> 2 ... Dim changes from c_dim to self.args.num_objs
		skill_seq = tf.argmax(skill_seq, axis=-1)
		skill_seq = skill_seq // 3
		# Convert back to one-hot
		skill_seq = tf.one_hot(skill_seq, depth=args.num_objs)
		return skill_seq
	else:
		raise NotImplementedError("Invalid value for wrap_skill_id: {}".format(args.wrap_level))


def orthogonal_regularization(model, reg_coef=1e-4):
	"""Orthogonal regularization v2.
		See equation (3) in https://arxiv.org/abs/1809.11096.
		Rβ(W) = β∥W⊤W ⊙ (1 − I)∥2F, where ⊙ is the Hadamard product.
		Args:
		  model: A keras model to apply regularization for.
		  reg_coef: Orthogonal regularization coefficient. Don't change this value.
		Returns:
		  A regularization loss term.
	"""
	reg = 0
	for layer in model.layers:
		if isinstance(layer, tf.keras.layers.Dense):
			prod = tf.matmul(tf.transpose(layer.kernel), layer.kernel)