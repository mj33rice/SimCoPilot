import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Suppress oneDNN warning
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import datetime
import json
import os
import pickle
import sys
import time
import pickle
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
import random

# Set the seed
SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
current_time = datetime.datetime(2024, 1, 1, 0, 0, 0).strftime("%Y%m%d-%H%M%S")

# Ensure TensorFlow doesn't try to use GPU if it's not available
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' if tf.test.is_gpu_available() else ''

# Suppress other warnings
tf.get_logger().setLevel('ERROR')

# Get the absolute path of the script module
script_path = os.path.realpath(__file__)
script_dir = os.path.dirname(script_path)

log_dir = os.path.join(script_dir, './logging', 'BC', 'run' + current_time)
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
			reg += tf.reduce_sum(tf.math.square(prod * (1 - tf.eye(prod.shape[0]))))
	
	print("Orthogonal Regularization: {}".format(reg * reg_coef))
	return reg * reg_coef


def sample_transitions(sample_style: str, state_to_goal=None, num_options: int = None):
	def sample_random_transitions(episodic_data, batch_size_in_transitions=None):
		"""
		Sample random transitions without HER.
		Functionality: Sample random time-steps from each episode: (g_t-1, c_t-1, s_t, g_t, c_t, a_t) for all episodes.
		"""
		
		batch_size = batch_size_in_transitions  # Number of transitions to sample
		T = episodic_data['actions'].shape[1]
		successes = episodic_data['successes']
		
		# Get index at which episode terminated
		terminate_idxes = tf.math.argmax(successes, axis=-1)
		# If no success, set to last index
		mask_no_success = tf.math.equal(terminate_idxes, 0)
		terminate_idxes += tf.multiply((T - 1) * tf.ones_like(terminate_idxes),
									   tf.cast(mask_no_success, terminate_idxes.dtype))
		
		# Get episode idx for each transition to sample: more likely to sample from episodes which didn't end in success
		p = (terminate_idxes + 1) / tf.reduce_sum(terminate_idxes + 1)
		episode_idxs = tfp.distributions.Categorical(probs=p).sample(sample_shape=(batch_size,))
		episode_idxs = tf.cast(episode_idxs, dtype=terminate_idxes.dtype)
		# Get terminate index for the selected episodes
		terminate_idxes = tf.gather(terminate_idxes, episode_idxs)
		print("terminate_idxes: ", terminate_idxes)
		
		# ------------------------------------------------------------------------------------------------------------
		# --------------------------------- 2) Select which time steps + goals to use --------------------------------
		# Get the current time step
		t_samples_frac = tf.experimental.numpy.random.random(size=(batch_size,))
		t_samples = t_samples_frac * tf.cast(terminate_idxes, dtype=t_samples_frac.dtype)
		t_samples = tf.cast(tf.round(t_samples), dtype=terminate_idxes.dtype)
		
		# Get random init time step (before t_samples)
		rdm_past_offset_frac = tf.zeros_like(t_samples_frac)
		t_samples_init = rdm_past_offset_frac * tf.cast(t_samples, dtype=rdm_past_offset_frac.dtype)
		t_samples_init = tf.cast(tf.floor(t_samples_init), dtype=t_samples.dtype)
		print("t_samples_init: ", t_samples_init)
		
		# Get the future time step
		rdm_future_offset_frac = tf.experimental.numpy.random.random(size=(batch_size,))
		future_offset = rdm_future_offset_frac * tf.cast((terminate_idxes - t_samples), rdm_future_offset_frac.dtype)
		future_offset = tf.cast(future_offset, terminate_idxes.dtype)
		t_samples_future = t_samples + future_offset
		print("t_samples_future: ", t_samples_future)
		
		# ------------------------------------------------------------------------------------------------------------
		# ----------------- 3) Select the batch of transitions corresponding to the current time steps ---------------
		curr_indices = tf.stack((episode_idxs, t_samples), axis=-1)
		transitions = {}
		for key in episodic_data.keys():
			transitions[key] = tf.gather_nd(episodic_data[key], indices=curr_indices)
		
		transitions['achieved_goals'] = state_to_goal(
			states=tf.gather_nd(episodic_data['states'], indices=curr_indices),
			obj_identifiers=None)
		
		# --------------- 4) Select the batch of transitions corresponding to the future time steps ------------
		future_indices = tf.stack((episode_idxs, t_samples_future), axis=-1)
		transitions['her_goals'] = state_to_goal(states=tf.gather_nd(episodic_data['states'], indices=future_indices),
												 obj_identifiers=None)  # Object ids are not used for unsegmented HER
		
		# --------------- 5) Select the batch of transitions corresponding to the initial time steps ------------
		init_indices = tf.stack((episode_idxs, t_samples_init), axis=-1)
		transitions['init_states'] = tf.gather_nd(episodic_data['states'], indices=init_indices)
		print("transitions: ", transitions)
		return transitions
	
	if sample_style == 'random_unsegmented':
		return sample_random_transitions
	else:
		raise NotImplementedError


class ReplayBufferTf:
	def __init__(self, buffer_shapes: Dict[str, Tuple[int, ...]], size_in_transitions, T, transition_fn=None):
		"""Creates a replay buffer.

		Args:
			buffer_shapes (dict of ints): the shape for all buffers that are used in the replay
				buffer
			size_in_transitions (int): the size of the buffer, measured in transitions
			T (int): the time horizon for episodes
			transition_fn (function): a function that samples from the replay buffer
		"""
		self.T = tf.constant(T, dtype=tf.int32)
		self.buffer_size = tf.constant(size_in_transitions // T, dtype=tf.int32)
		
		self.current_size = tf.Variable(0, dtype=tf.int32)  # Size of buffer in terms of no. of episodes
		self.n_transitions_stored = tf.Variable(0, dtype=tf.int32)  # Size of buffer in terms of no. of transitions
		
		self.transition_fn = transition_fn
		self.buffer_keys: List[str] = [key for key in buffer_shapes.keys()]
		tensor_spec = [tf.TensorSpec(buffer_shapes[key], tf.float32, key) for key in self.buffer_keys]
		self.table = Table(tensor_spec, capacity=self.buffer_size)
	
	@tf.function  # Make sure batch_size passed here is a tf.constant to avoid retracing
	def sample_transitions(self, batch_size):
		
		buffered_data = {}
		_data = self.table.read(rows=tf.range(self.current_size))
		for index, key in enumerate(self.buffer_keys):
			buffered_data[key] = _data[index]
		
		transitions = self.transition_fn(buffered_data, batch_size)
		print("transitions: ", transitions)
		return transitions
	
	@tf.function
	def sample_episodes(self, ep_start: int = None, ep_end: int = None, num_episodes: int = None):
		
		if ep_start is None or ep_end is None:
			if num_episodes:
				num_episodes = tf.math.minimum(tf.cast(num_episodes, dtype=self.current_size.dtype), self.current_size)
			else:
				num_episodes = self.current_size
			ep_range = tf.range(num_episodes)
		else:
			ep_range = tf.range(ep_start, ep_end)
		
		buffered_data = {}
		_data = self.table.read(rows=ep_range)
		for index, key in enumerate(self.buffer_keys):
			buffered_data[key] = _data[index]
		print("buffered_data: ", buffered_data)
		return buffered_data
	
	@tf.function
	def store_episode(self, episode_batch):
		"""
			Store each episode into replay buffer
			episode_batch: {"": array(1 x (T or T+1) x dim)}
		"""
		idxs = self._get_storage_idxs(num_to_ins=tf.constant(1, dtype=tf.int32))
		values = [episode_batch[key] for key in self.buffer_keys if key in episode_batch.keys()]
		self.table.write(rows=idxs, values=values)
		self.n_transitions_stored.assign(self.n_transitions_stored + self.T)
	
	def store_episodes(self, episodes_batch):
		for ep_idx in tf.range(tf.shape(episodes_batch['actions'])[0]):
			episode_batch = {}
			for key in self.buffer_keys:
				episode_batch[key] = tf.gather(episodes_batch[key], ep_idx)
			self.store_episode(episode_batch)
	
	def _get_storage_idxs(self, num_to_ins=None):
		if num_to_ins is None:
			num_to_ins = tf.cast(1, dtype=tf.int32)
		
		# consecutively insert until you hit the end of the buffer, and then insert randomly.
		if self.current_size + num_to_ins <= self.buffer_size:
			idxs = tf.range(self.current_size, self.current_size + num_to_ins)
		elif self.current_size < self.buffer_size:
			overflow = num_to_ins - (self.buffer_size - self.current_size)
			idx_a = tf.range(self.current_size, self.buffer_size)
			idx_b = tf.experimental.numpy.random.randint(0, self.current_size, size=(overflow,), dtype=tf.int32)
			idxs = tf.concat([idx_a, idx_b], axis=0)
		else:
			idxs = tf.experimental.numpy.random.randint(0, self.buffer_size, size=(num_to_ins,), dtype=tf.int32)
		
		# update buffer size
		self.current_size.assign(tf.math.minimum(self.buffer_size, self.current_size + num_to_ins))
		print("idxs: ", idxs)
		return idxs
	
	def get_current_size_ep(self):
		return self.current_size
	
	def get_current_size_trans(self):
		return self.current_size * self.T
	
	def clear_buffer(self):
		self.current_size.assign(0)
	
	@property
	def full(self):
		return self.current_size == self.buffer_size
	
	def __len__(self):
		return self.current_size
	
	def save_buffer_data(self, path):
		buffered_data = {}
		_data = self.table.read(rows=tf.range(self.current_size))
		for index, key in enumerate(self.buffer_keys):
			buffered_data[key] = _data[index]
		
		with open(path, 'wb') as handle:
			pickle.dump(buffered_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
	
	def load_data_into_buffer(self, buffered_data=None, clear_buffer=True, num_demos_to_load=None):
		
		if buffered_data is None:
			raise ValueError("No buffered_data provided")
		
		if clear_buffer:
			self.clear_buffer()
		
		if num_demos_to_load is not None:
			
			# Randomly sample idxs to load
			idxs = np.random.choice(len(buffered_data['actions']), size=num_demos_to_load, replace=False).tolist()
			
			for key in buffered_data.keys():
				buffered_data[key] = tf.gather(buffered_data[key], idxs)
		
		# Check if all tensors are present in loaded data
		data_sizes = [len(buffered_data[key]) for key in self.buffer_keys]
		assert np.all(np.array(data_sizes) == data_sizes[0])
		
		idxs = self._get_storage_idxs(num_to_ins=data_sizes[0])
		values = [buffered_data[key] for key in self.buffer_keys]
		
		self.table.write(rows=idxs, values=values)
		self.n_transitions_stored.assign(self.n_transitions_stored + len(idxs) * self.T)


class Actor(tf.keras.Model):
	def __init__(self, action_dim):
		super(Actor, self).__init__()
		
		# Rewrite the base weights to initialise using Xavier(gain=1.0) and bias=0.0
		self.base = tf.keras.Sequential([
			Dense(units=256, activation=tf.nn.relu, kernel_initializer='glorot_uniform', bias_initializer='zeros'),
			Dense(units=256, activation=tf.nn.relu, kernel_initializer='glorot_uniform', bias_initializer='zeros'),
			Dense(units=128, activation=tf.nn.relu, kernel_initializer='glorot_uniform', bias_initializer='zeros'),
			Dense(units=action_dim, kernel_initializer='glorot_uniform', bias_initializer='zeros')
		])
		
		self.MEAN_MIN, self.MEAN_MAX = -7, 7
		self.eps = np.finfo(np.float32).eps
		self.pi = tf.constant(np.pi)
		self.FIXED_STD = 0.05
		
		self.train = True
	
	def get_log_prob(self, states, actions):
		"""Evaluate log probs for actions conditioned on states.
		Args:
		  states: A batch of states.
		  actions: A batch of actions to evaluate log probs on.
		Returns:
		  Log probabilities of actions.
		"""
		mu = self.base(states)
		mu = tf.nn.tanh(mu)
		mu = tf.clip_by_value(mu, self.MEAN_MIN, self.MEAN_MAX)
		
		std = tf.ones_like(mu) * self.FIXED_STD
		
		actions = tf.clip_by_value(actions, -1 + self.eps, 1 - self.eps)
		
		# Get log probs from Gaussian distribution
		log_probs = -0.5 * tf.square((actions - mu) / std) - 0.5 * tf.math.log(2 * self.pi) - tf.math.log(std)
		log_probs = tf.reduce_sum(log_probs, axis=1, keepdims=False)
		print("log_probs: ", log_probs)
		return log_probs
	
	def call(self, states, training=None, mask=None):
		"""Computes actions for given inputs.
		Args:
		  states: A batch of states.
		  training: Ignored
		  mask: Ignored.
		Returns:
		  A mode action, a sampled action and log probability of the sampled action.
		"""
		mu = self.base(states)
		mu = tf.nn.tanh(mu)
		mu = tf.clip_by_value(mu, self.MEAN_MIN, self.MEAN_MAX)
		
		if self.train:
			# Sample actions from the distribution
			actions = tf.random.normal(shape=mu.shape, mean=mu, stddev=self.FIXED_STD)
		else:
			actions = mu
		
		# Compute log probs
		log_probs = self.get_log_prob(states, actions)
		log_probs = tf.expand_dims(log_probs, -1)  # To avoid broadcasting
		
		actions = tf.clip_by_value(actions, -1 + self.eps, 1 - self.eps)
		print("mu: ", mu)
		print("actions: ", actions)
		print("log_probs: ", log_probs)
		return mu, actions, log_probs


class BC(tf.keras.Model, ABC):
	def __init__(self, args: Namespace):
		super(BC, self).__init__()
		self.args = args
		
		# Declare Policy Network and Optimiser
		self.actor = Actor(args.a_dim)
		self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=args.actor_lr)
		
		# Build Model
		self.build_model()
		
		# For HER
		self.use_her = False
		logger.info('[[[ Using HER ? ]]]: {}'.format(self.use_her))
	
	@tf.function(experimental_relax_shapes=True)
	def train(self, data_exp, data_rb):
		with tf.GradientTape(watch_accessed_variables=False, persistent=True) as tape:
			tape.watch(self.actor.variables)
			
			actions_mu, _, _ = self.actor(tf.concat([data_rb['states'], data_rb['goals']], axis=1))
			pi_loss = tf.reduce_sum(tf.math.squared_difference(data_rb['actions'], actions_mu), axis=-1)
			pi_loss = tf.reduce_mean(pi_loss)
			penalty = orthogonal_regularization(self.actor.base)
			pi_loss_w_penalty = pi_loss + penalty
		
		grads = tape.gradient(pi_loss_w_penalty, self.actor.trainable_variables)
		self.actor_optimizer.apply_gradients(zip(grads, self.actor.trainable_variables))
		print("loss/pi: ", pi_loss)
		print("penalty/pi_ortho_penalty: ", penalty)
		return {
			'loss/pi': pi_loss,
			'penalty/pi_ortho_penalty': penalty,
		}
	
	def act(self, state, env_goal, prev_goal, prev_skill, epsilon, stddev):
		state = tf.clip_by_value(state, -self.args.clip_obs, self.args.clip_obs)
		env_goal = tf.clip_by_value(env_goal, -self.args.clip_obs, self.args.clip_obs)
		prev_goal = tf.clip_by_value(prev_goal, -self.args.clip_obs, self.args.clip_obs)
		
		# ###################################### Current Goal ####################################### #
		curr_goal = env_goal
		
		# ###################################### Current Skill ###################################### #
		curr_skill = prev_skill  # Not used in this implementation
		
		# ########################################## Action ######################################### #
		# Explore
		if tf.random.uniform(()) < epsilon:
			action = tf.random.uniform((1, self.args.a_dim), -self.args.action_max, self.args.action_max)
		# Exploit
		else:
			action_mu, _, _ = self.actor(tf.concat([state, curr_goal], axis=1))  # a_t = mu(s_t, g_t)
			action_dev = tf.random.normal(action_mu.shape, mean=0.0, stddev=stddev)
			action = action_mu + action_dev  # Add noise to action
			action = tf.clip_by_value(action, -self.args.action_max, self.args.action_max)
		
		# Safety check for action, should not be nan or inf
		has_nan = tf.math.reduce_any(tf.math.is_nan(action))
		has_inf = tf.math.reduce_any(tf.math.is_inf(action))
		if has_nan or has_inf:
			logger.warning('Action has nan or inf. Setting action to zero. Action: {}'.format(action))
			action = tf.zeros_like(action)
		
		return curr_goal, curr_skill, action
	
	def get_init_skill(self):
		"""
		demoDICE does not use skills. Use this function to return a dummy skill of dimension (1, c_dim)
		"""
		skill = tf.zeros((1, self.args.c_dim))
		return skill
	
	@staticmethod
	def get_init_goal(init_state, g_env):
		return g_env
	
	def build_model(self):
		# a_t <- f(s_t) for each skill
		_ = self.actor(tf.concat([np.ones([1, self.args.s_dim]), np.ones([1, self.args.g_dim])], 1))
	
	def save_(self, dir_param):
		self.actor.save_weights(dir_param + "/policy.h5")
	
	def load_(self, dir_param):
		self.actor.load_weights(dir_param + "/policy.h5")
	
	def change_training_mode(self, training_mode: bool):
		pass
	
	def update_target_networks(self):
		pass


class AgentBase(object):
	def __init__(
			self,
			args,
			model,
			algo: str,
			expert_buffer: ReplayBufferTf,
			offline_buffer: ReplayBufferTf
	):
		
		self.args = args
		self.model = model
		
		# Define the Buffers
		self.expert_buffer = expert_buffer
		self.offline_buffer = offline_buffer
		
		self.offline_gt_prev_skill = None
		self.offline_gt_curr_skill = None
		
		# Define Tensorboard for logging Losses and Other Metrics
		if not os.path.exists(args.dir_summary):
			os.makedirs(args.dir_summary)
		
		if not os.path.exists(args.dir_plot):
			os.makedirs(args.dir_plot)
		self.summary_writer = tf.summary.create_file_writer(args.dir_summary)
		
		# Define wandb logging
		if self.args.log_wandb:
			self.wandb_logger = wandb.init(
				project=args.wandb_project,
				config=vars(args),
				id='{}_{}'.format(algo, current_time),
				reinit=True,  # Allow multiple wandb.init() calls in the same process.
			)
			# Clear tensorflow graph and cache
			tf.keras.backend.clear_session()
			tf.compat.v1.reset_default_graph()
	
	def preprocess_in_state_space(self, item):
		item = tf.clip_by_value(item, -self.args.clip_obs, self.args.clip_obs)
		return item
	
	def save_model(self, dir_param):
		if not os.path.exists(dir_param):
			os.makedirs(dir_param)
		self.model.save_(dir_param)
	
	def load_model(self, dir_param):
		self.model.load_(dir_param)
	
	def process_data(self, transitions, expert=False, is_supervised=False):
		
		trans = transitions.copy()
		
		# Process the states and goals
		trans['states'] = self.preprocess_in_state_space(trans['states'])
		trans['states_2'] = self.preprocess_in_state_space(trans['states_2'])
		trans['env_goals'] = self.preprocess_in_state_space(trans['env_goals'])
		trans['init_states'] = self.preprocess_in_state_space(trans['init_states'])
		trans['her_goals'] = self.preprocess_in_state_space(trans['her_goals'])
		trans['achieved_goals'] = self.preprocess_in_state_space(trans['achieved_goals'])
		
		if self.model.use_her:
			trans['goals'] = trans['her_goals']
		else:
			trans['goals'] = trans['env_goals']
		
		# Define if the transitions are from expert or not/are supervised or not
		trans['is_demo'] = tf.cast(expert, dtype=tf.int32) * tf.ones_like(trans['successes'], dtype=tf.int32)
		trans['is_sup'] = tf.cast(is_supervised, dtype=tf.int32) * tf.ones_like(trans['successes'], dtype=tf.int32)
		
		# Compute terminate skills i.e. if prev_skill != curr_skill then terminate_skill = 1 else 0
		trans['terminate_skills'] = tf.cast(tf.not_equal(tf.argmax(trans['prev_skills'], axis=-1),
														 tf.argmax(trans['curr_skills'], axis=-1)),
											dtype=tf.int32)
		# reshape the terminate_skills to be of shape (batch_size, 1)
		trans['terminate_skills'] = tf.reshape(trans['terminate_skills'], shape=(-1, 1))
		
		# Make sure the data is of type tf.float32
		for key in trans.keys():
			trans[key] = tf.cast(trans[key], dtype=tf.float32)
		print("trans :", trans)
		return trans
	
	def sample_data(self, buffer, batch_size):
		
		# Sample Transitions
		transitions: Union[Dict[int, dict], dict] = buffer.sample_transitions(batch_size)
		
		# Process the transitions
		keys = None
		if all(isinstance(v, dict) for v in transitions.values()):
			for skill in transitions.keys():
				
				# For skills whose transition data is not None
				if transitions[skill] is not None:
					transitions[skill] = self.process_data(
						transitions[skill], tf.constant(True, dtype=tf.bool), tf.constant(True, dtype=tf.bool)
					)
					
					keys = transitions[skill].keys()
			
			# If keys is None, No transitions were sampled
			if keys is None:
				raise ValueError("No transitions were sampled")
			
			# Concatenate the transitions from different skills
			combined_transitions = {key: [] for key in keys}
			
			for skill in transitions.keys():
				
				if transitions[skill] is not None:
					for key in keys:
						combined_transitions[key].append(transitions[skill][key])
			
			for key in keys:
				combined_transitions[key] = tf.concat(combined_transitions[key], axis=0)
			
			transitions = combined_transitions
		
		elif isinstance(transitions, dict):
			transitions = self.process_data(
				transitions, tf.constant(True, dtype=tf.bool), tf.constant(True, dtype=tf.bool)
			)
		
		else:
			raise ValueError("Invalid type of transitions")
		print("transitions: ", transitions)
		return transitions
	
	@tf.function
	def train(self):
		
		self.model.change_training_mode(training_mode=True)
		
		data_expert = self.sample_data(self.expert_buffer, self.args.batch_size)
		data_policy = self.sample_data(self.offline_buffer, self.args.batch_size)
		loss_dict = self.model.train(data_expert, data_policy)
		
		# Average the losses
		avg_loss_dict = {}
		for key in loss_dict.keys():
			if key not in avg_loss_dict.keys():
				avg_loss_dict[key] = []
			avg_loss_dict[key].append(loss_dict[key])
		for key in avg_loss_dict.keys():
			avg_loss_dict[key] = tf.reduce_mean(avg_loss_dict[key])
		print("avg_loss_dict: ", avg_loss_dict)
		return avg_loss_dict
	
	def learn(self):
		# This is a base class method, inherited classes must implement this method
		raise NotImplementedError


class Agent(AgentBase):
	def __init__(self, args,
				 expert_buffer: ReplayBufferTf = None,
				 offline_buffer: ReplayBufferTf = None):
		
		super(Agent, self).__init__(args, BC(args), 'BC', expert_buffer, offline_buffer)
	
	def load_actor(self, dir_param):
		self.model.actor.load_weights(dir_param + "/policy.h5")
	
	def learn(self):
		args = self.args
		
		# Tracker for wandb logging
		log_step = 0
		
		# [Update] Load the expert data into the expert buffer, expert data and offline data into the offline buffer
		data_exp = self.expert_buffer.sample_episodes()
		data_off = self.offline_buffer.sample_episodes()
		self.expert_buffer.load_data_into_buffer(buffered_data=data_exp, clear_buffer=True)
		self.offline_buffer.load_data_into_buffer(buffered_data=data_exp, clear_buffer=True)
		self.offline_buffer.load_data_into_buffer(buffered_data=data_off, clear_buffer=False)
		
		with tqdm(total=args.max_time_steps, leave=False) as pbar:
			for curr_t in range(0, args.max_time_steps):
				
				# Update the reference actors and directors using polyak averaging
				if curr_t % args.update_target_interval == 0:
					tf.print("Updating the target actors and critics at train step {}".format(curr_t))
					self.model.update_target_networks()
				
				# Train the policy
				pbar.set_description('Training')
				avg_loss_dict = self.train()
				for key in avg_loss_dict.keys():
					avg_loss_dict[key] = avg_loss_dict[key].numpy().item()
				
				# Log
				if self.args.log_wandb:
					self.wandb_logger.log(avg_loss_dict, step=log_step)
					self.wandb_logger.log({
						'policy_buffer_size': self.offline_buffer.get_current_size_trans(),
						'expert_buffer_size': self.expert_buffer.get_current_size_trans(),
					}, step=log_step)
				
				# Update
				pbar.update(1)
				log_step += 1
		
		# Save the model
		self.save_model(args.dir_param)


def get_config_env(args, ag_in_env_goal):
	"""
	:param args: Namespace object
	:param ag_in_env_goal: If True, then achieved goal is in the same space as env goal
	"""
	
	args.g_dim = 3
	args.s_dim = 10
	args.a_dim = 4
	
	# Specify the expert's latent skill dimension [Default]
	# Define number of skills, this could be different from agent's practiced skill dimension
	assert hasattr(args, 'num_objs')
	args.c_dim = 3 * args.num_objs
	
	if ag_in_env_goal:
		args.ag_dim = args.g_dim  # Achieved Goal in the same space as Env Goal
	else:
		args.ag_dim = 3  # Goal/Object position in the 3D space
	print("args: ", args)
	return args


def get_config(db=False):
	# Construct the absolute path of the data directory
	data_dir = os.path.join(script_dir, 'pnp_data')

	parser = argparse.ArgumentParser()
	
	parser.add_argument('--log_wandb', type=bool, default=False)
	parser.add_argument('--wandb_project', type=str, default='offlineILPnPOne',
						choices=['offlineILPnPOne', 'offlineILPnPOneExp', 'offlineILPnPTwoExp'])
	
	parser.add_argument('--expert_demos', type=int, default=25)
	parser.add_argument('--offline_demos', type=int, default=75)
	parser.add_argument('--eval_demos', type=int, default=1 if db else 10,
						help='Use 10 (num of demos to evaluate trained pol)')
	parser.add_argument('--test_demos', type=int, default=0, help='For Visualisation')
	parser.add_argument('--perc_train', type=int, default=1.0)
	
	# Specify Environment Configuration
	parser.add_argument('--env_name', type=str, default='OpenAIPickandPlace')
	parser.add_argument('--num_objs', type=int, default=1)
	parser.add_argument('--horizon', type=int, default=100,
						help='Set 100 for one_obj, 150 for two_obj and 200 for three_obj')
	parser.add_argument('--stacking', type=bool, default=False)
	parser.add_argument('--expert_behaviour', type=str, default='0', choices=['0', '1'],
						help='Expert behaviour in two_object env')
	parser.add_argument('--full_space_as_goal', type=bool, default=False)
	parser.add_argument('--fix_goal', type=bool, default=False,
						help='[Debugging] Fix the goal position for one object task')
	parser.add_argument('--fix_object', type=bool, default=False,
						help='[Debugging] Fix the object position for one object task')
	
	# Specify Data Collection Configuration
	parser.add_argument('--buffer_size', type=int, default=int(2e5),
						help='Number of transitions to store in buffer (max_time_steps)')
	
	# Specify Training configuration
	parser.add_argument('--max_pretrain_time_steps', type=int, default=0 if not db else 0,
						help='No. of time steps to run pretraining - actor, director on expert data. Set to 0 to skip')
	parser.add_argument('--max_time_steps', type=int, default=10000 if not db else 1,
						help='No. of time steps to run. Recommended 5k for one_obj, 10k for two_obj')
	parser.add_argument('--batch_size', type=int, default=1,
						help='No. of trans to sample from buffer for each update')
	parser.add_argument('--trans_style', type=str, default='random_unsegmented',
						choices=['random_unsegmented', 'random_segmented'],
						help='How to sample transitions from expert buffer')
	
	# Viterbi configuration
	parser.add_argument('--skill_supervision', type=str, default='none',
						choices=['full', 'semi:0.10', 'semi:0.25', 'none'],
						help='Type of supervision for latent skills. '
							 'full: Use ground truth skills for offline data.'
							 'semi:x: Use Viterbi to update latent skills for offline data.'
							 'none: Use Viterbi to update latent skills for expert and offline data.')
	parser.add_argument('--num_skills', type=int, default=None,
						help='Number of skills to use for agent, if provided, will override expert skill set. '
							 'Use when skill supervision is "none"')
	parser.add_argument('--wrap_level', type=str, default='1', choices=['0', '1', '2'],
						help='consumed by multi-object expert to determine how to wrap effective skills of expert')
	
	# Polyak
	parser.add_argument('--update_target_interval', type=int, default=20,
						help='Number of time steps after which target networks will be updated using polyak averaging')
	parser.add_argument('--actor_polyak', type=float, default=0.95,
						help='Polyak averaging coefficient for actor.')
	parser.add_argument('--director_polyak', type=float, default=0.95,
						help='Polyak averaging coefficient for director.')
	parser.add_argument('--critic_polyak', type=float, default=0.95,
						help='Polyak averaging coefficient for critic.')
	
	# Evaluation
	parser.add_argument('--eval_interval', type=int, default=100)
	parser.add_argument('--visualise_test', type=bool, default=False, help='Visualise test episodes?')
	
	# Parameters
	parser.add_argument('--discount', type=float, default=0.99, help='Discount used for returns.')
	parser.add_argument('--replay_regularization', type=float, default=0.05,
						help='Replay Regularization Coefficient. Used by both ValueDICE (0.1) and DemoDICE (0.05)')
	parser.add_argument('--nu_grad_penalty_coeff', type=float, default=1e-4,
						help='Nu Net Gradient Penalty Coefficient. ValueDICE uses 10.0, DemoDICE uses 1e-4')
	parser.add_argument('--cost_grad_penalty_coeff', type=float, default=10,
						help='Cost Net Gradient Penalty Coefficient')
	parser.add_argument('--actor_lr', type=float, default=3e-3)
	parser.add_argument('--critic_lr', type=float, default=3e-4)
	parser.add_argument('--disc_lr', type=float, default=3e-4)
	parser.add_argument('--clip_obs', type=float, default=200.0,
						help='Un-normalised i.e. raw Observed Values (State and Goals) are clipped to this value')
	
	# Specify Path Configurations
	parser.add_argument('--dir_data', type=str, default=data_dir)
	parser.add_argument('--dir_root_log', type=str, default=log_dir)
	parser.add_argument('--dir_summary', type=str, default=os.path.join(log_dir, 'summary'))
	parser.add_argument('--dir_plot', type=str, default=os.path.join(log_dir, 'plots'))
	parser.add_argument('--dir_param', type=str, default=os.path.join(log_dir, 'models'))
	parser.add_argument('--dir_post', type=str, default='./finetuned_models',
						help='Provide the <path_to_models>')
	parser.add_argument('--dir_pre', type=str, default='./pretrained_models',
						help='Provide the <path_to_models>')
	
	args = parser.parse_args()
	
	# Load the environment config
	args = get_config_env(args, ag_in_env_goal=True)
	
	# Other Configurations
	args.train_demos = int(args.expert_demos * args.perc_train)
	args.val_demos = args.expert_demos - args.train_demos
	
	# Set number of skills [For unsupervised skill learning]
	if args.num_skills is not None and args.skill_supervision == 'none':
		print('Overriding c_dim with specified %d skills' % args.num_skills)
		args.c_dim = args.num_skills
	
	# Set number of skills [For full or semi-supervised skill learning]
	if args.env_name == 'OpenAIPickandPlace' and args.wrap_level != '0' and args.skill_supervision != 'none':
		print('Overriding c_dim based on Wrap Level %s' % args.wrap_level)
		if args.wrap_level == '1':
			args.c_dim = 3
		elif args.wrap_level == '2':
			args.c_dim = args.num_objs
		else:
			raise NotImplementedError('Wrap level %s not implemented' % args.wrap_level)
	
	return args


def run(db: bool, algo: str):
	
	if db:
		print("Running in Debug Mode. (db=True)")
	
	tf.config.run_functions_eagerly(db)
	
	logger.info("# ################# Working on Model: \"{}\" ################# #".format(algo))
	
	args = get_config(db=db)
	args.algo = algo
	args.log_dir = log_dir
	
	logger.info("---------------------------------------------------------------------------------------------")
	config: dict = vars(args)
	config = {key: str(value) for key, value in config.items()}
	config = OrderedDict(sorted(config.items()))
	logger.info(json.dumps(config, indent=4))
	
	# Clear tensorflow graph and cache
	tf.keras.backend.clear_session()
	tf.compat.v1.reset_default_graph()
	
	# ######################################################################################################## #
	# ############################################# DATA LOADING ############################################# #
	# ######################################################################################################## #
	# Load Buffer to store expert data
	n_objs = args.num_objs
	buffer_shape: Dict[str, Tuple[int, ...]] = get_buffer_shape(args)
	
	expert_buffer = ReplayBufferTf(
		buffer_shape, args.buffer_size, args.horizon,
		sample_transitions(args.trans_style, state_to_goal=state_to_goal(n_objs), num_options=args.c_dim),
	)
	offline_buffer = ReplayBufferTf(
		buffer_shape, args.buffer_size, args.horizon,
		sample_transitions(args.trans_style, state_to_goal=state_to_goal(n_objs), num_options=args.c_dim)
	)
	if n_objs == 3:
		expert_data_file = 'three_obj_{}_train.pkl'.format(args.expert_behaviour)
		offline_data_file = 'three_obj_{}_offline.pkl'.format(args.expert_behaviour)
	elif n_objs == 2:
		expert_data_file = 'two_obj_{}_train.pkl'.format(args.expert_behaviour)
		offline_data_file = 'two_obj_{}_offline.pkl'.format(args.expert_behaviour)
	elif n_objs == 1:
		expert_data_file = 'single_obj_train.pkl'
		offline_data_file = 'single_obj_offline.pkl'
	else:
		raise NotImplementedError
	expert_data_path = os.path.join(args.dir_data, expert_data_file)
	offline_data_path = os.path.join(args.dir_data, offline_data_file)
	
	if not os.path.exists(expert_data_path):
		logger.error(
			"Expert data not found at {}. Please run the data generation script first.".format(expert_data_path))
		sys.exit(-1)
	
	if not os.path.exists(offline_data_path):
		logger.error(
			"Offline data not found at {}. Please run the data generation script first.".format(offline_data_path))
		sys.exit(-1)
	
	# Store the expert data in the expert buffer -> D_E
	logger.info("Loading Expert Demos from {} into Expert Buffer for training.".format(expert_data_path))
	with open(expert_data_path, 'rb') as handle:
		buffered_data = pickle.load(handle)
	
	# [Optional] Reformat the G.T. skill sequences
	curr_skills = repurpose_skill_seq(args, buffered_data['curr_skills'])
	prev_skills = repurpose_skill_seq(args, buffered_data['prev_skills'])
	buffered_data['curr_skills'] = curr_skills
	buffered_data['prev_skills'] = prev_skills
	# Add a new key "has_gt_skill" indicating that the skill is G.T.
	buffered_data['has_gt_skill'] = tf.ones_like(buffered_data['successes'], dtype=tf.float32)
	expert_buffer.load_data_into_buffer(buffered_data=buffered_data, num_demos_to_load=args.expert_demos)
	
	# Store the offline data in the policy buffer for DemoDICE -> D_O
	logger.info("Loading Offline Demos from {} into Offline Buffer for training.".format(offline_data_path))
	with open(offline_data_path, 'rb') as handle:
		buffered_data = pickle.load(handle)
	
	# [Optional] Reformat the G.T. skill sequences
	curr_skills = repurpose_skill_seq(args, buffered_data['curr_skills'])
	prev_skills = repurpose_skill_seq(args, buffered_data['prev_skills'])
	buffered_data['curr_skills'] = curr_skills
	buffered_data['prev_skills'] = prev_skills
	# Add a new key "has_gt_skill" indicating that the skill is G.T.
	buffered_data['has_gt_skill'] = tf.ones_like(buffered_data['successes'], dtype=tf.float32)
	offline_buffer.load_data_into_buffer(buffered_data=buffered_data, num_demos_to_load=args.offline_demos)
	# ########################################################################################################### #
	# ############################################# TRAINING #################################################### #
	# ########################################################################################################### #
	start = time.time()
	
	agent = Agent(args, expert_buffer, offline_buffer)
	
	logger.info("Training .......")
	agent.learn()


if __name__ == "__main__":
	num_runs = 1
	for i in range(num_runs):
		run(db=True, algo='BC')