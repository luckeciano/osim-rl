"""
Code for DAgger
Example usage:
    python dagger.py --num_expert_rollouts=5

Author of this modified script: luckeciano@gmail.com
"""


import os
import numpy as np
from osim.env import L2M2019Env
from osim.control.osim_loco_reflex_song2019 import OsimReflexCtrl
import tensorflow as tf
from tqdm import tqdm
import collections

INIT_POSE = np.array([
1.699999999999999956e+00, # forward speed
.5, # rightward speed
9.023245653983965608e-01, # pelvis height
2.012303881285582852e-01, # trunk lean
0*np.pi/180, # [right] hip adduct
-6.952390849304798115e-01, # hip flex
-3.231075259785813891e-01, # knee extend
1.709011708233401095e-01, # ankle flex
0*np.pi/180, # [left] hip adduct
-5.282323914341899296e-02, # hip flex
-8.041966456860847323e-01, # knee extend
-1.745329251994329478e-01]) # ankle flex

dict_keys = ['v_tgt_field', 'pelvis', 'r_leg', 'l_leg']

#---------------------Wrapper-------------------------------#
def ravel_tuple_observation(observation):
        obs = []
        for item in observation:
            obs.append(_ravel(item))
        return np.concatenate(obs)

def _ravel(space):
        if isinstance(space, dict):
            return ravel_dict_observation(space, space.keys())
        elif isinstance(space, tuple):
            return ravel_tuple_observation(space)
        else:
            return np.array(space).ravel()

def ravel_dict_observation(observation, dict_keys):
        assert isinstance(observation, dict)
        obs = []
        for key in dict_keys:
            obs.append(_ravel(observation[key]))
        return np.concatenate(obs)
#------------------------------------------------------------#

#------------------------tf function-------------------------#
def function(inputs, outputs, updates=None, givens=None):
    if isinstance(outputs, list):
        return _Function(inputs, outputs, updates, givens=givens)
    elif isinstance(outputs, (dict, collections.OrderedDict)):
        f = _Function(inputs, outputs.values(), updates, givens=givens)
        return lambda *inputs : type(outputs)(zip(outputs.keys(), f(*inputs)))
    else:
        f = _Function(inputs, [outputs], updates, givens=givens)
        return lambda *inputs : f(*inputs)[0]

class _Function(object):
    def __init__(self, inputs, outputs, updates, givens, check_nan=False):
        assert all(len(i.op.inputs)==0 for i in inputs), "inputs should all be placeholders"
        self.inputs = inputs
        updates = updates or []
        self.update_group = tf.group(*updates)
        self.outputs_update = list(outputs) + [self.update_group]
        self.givens = {} if givens is None else givens
        self.check_nan = check_nan
    def __call__(self, *inputvals):
        assert len(inputvals) == len(self.inputs)
        feed_dict = dict(zip(self.inputs, inputvals))
        feed_dict.update(self.givens)
        results = tf.get_default_session().run(self.outputs_update, feed_dict=feed_dict)[:-1]
        if self.check_nan:
            if any(np.isnan(r).any() for r in results):
                raise RuntimeError("Nan detected")
        return results

#------------------------------------------------------------#

class DAgger():

	def build_mlp_policy(self, n_hidden_layers = 2, hidden_layer_size = 64, ob_shape=None, action_shape=None):
		ob = tf.placeholder(name='ob', dtype=tf.float32, shape=(None, ob_shape[1]))

		x = ob
		for i in range(n_hidden_layers):
			x = tf.nn.tanh(tf.layers.dense(x, hidden_layer_size, name='fc%i'%(i+1)))
		actor = tf.layers.dense(x, action_shape[1], name = 'actions')

		ac = tf.placeholder(name='expected_actions', dtype=tf.float32, shape=(None,action_shape[1]))

		#actor = tf.reshape(actor, shape=np.array([-1, action_shape[1], action_shape[2]]))
		error = tf.reduce_mean(0.5 * tf.square(actor - ac))
		opt = tf.train.AdamOptimizer(learning_rate=3e-4).minimize(error)
		sess = tf.get_default_session()
		sess.run(tf.global_variables_initializer())

		return [ob, ac, opt, error, actor]

	def train(self, policy, S, A, epochs, batch_size):

		ob, ac, opt, error, actor = policy

		sess = tf.get_default_session()

		number_of_batches = S.shape[0]//batch_size
		sample_index = np.arange(S.shape[0])
		print("Training...")
		for i in range(epochs):
			np.random.shuffle(sample_index)
			#pbar = tqdm(range(number_of_batches))
			pbar = range(number_of_batches)
			for k in pbar:
				batch_index = sample_index[batch_size*k:batch_size*(k+1)]
				s_batch = S[batch_index,:]
				a_batch = A[batch_index,:]
				_, mse_run = sess.run([opt, error], feed_dict={ob: s_batch, ac: a_batch})
				#pbar.set_description("Loss %s" % str(mse_run))


		return function([ob], actor)

	def collect_rollouts(self, env, max_steps, num_rollouts, policy_fn, controller=False):
		returns = []
		observations = []
		observations_json = []
		actions = []
		for i in range(num_rollouts):
			print('iter', i)
			obs = env.reset(project=True, seed=None, obs_as_dict=True, init_pose=INIT_POSE)
			done = False
			totalr = 0.
			steps = 0
			while not done:

				if controller:
					action = policy_fn.update(obs)
				else:
					x = ravel_dict_observation(obs, dict_keys)
					action = policy_fn(x[None,:])[0]

				observations.append(ravel_dict_observation(obs, dict_keys))
				observations_json.append(obs)
				actions.append(action)
				obs, r, done, _ = env.step(action, project=True, obs_as_dict=True)
				totalr += r
				if done:
					print(totalr)
				steps += 1
			returns.append(totalr)
		rollouts_returns = np.array(returns)

		policy_data = {'observations': np.array(observations),
		                   'actions': np.array(actions),
		                   'observations_json': np.array(observations_json)}
		return policy_data, rollouts_returns

	def run_dagger(self, env, load_data, load_policy,  max_steps, num_expert_rollouts, num_dagger_updates, rollouts_per_update, epochs, batch_size, expert_policy_fn, n_hidden_layers=2, hidden_layer_size=64, eval_running_policy=True):

		aggregated_data = {}
		if load_data:
			aggregated_data['observations'] = np.loadtxt('dagger_observations')
			aggregated_data['actions'] = np.loadtxt('dagger_actions')

		print(aggregated_data['observations'].shape, aggregated_data['actions'].shape)

		
		#1. collect data from expert policy data
		if num_expert_rollouts > 0:
			expert_data, _ = self.collect_rollouts(env, max_steps, num_expert_rollouts, expert_policy_fn, controller=True)

			aggregated_data['observations'] = np.concatenate([aggregated_data['observations'], expert_data['observations']])
			aggregated_data['actions'] = np.concatenate([aggregated_data['actions'], expert_data['actions']])

		
		#save rollouts before starting dagger updates
		np.savetxt('dagger_observations', aggregated_data['observations'])
		np.savetxt('dagger_actions', aggregated_data['actions'])

		policy = self.build_mlp_policy(n_hidden_layers, hidden_layer_size, aggregated_data['observations'].shape, aggregated_data['actions'].shape)

		saver = tf.train.Saver()
		os.makedirs(os.path.dirname('./trained_policy'), exist_ok=True)

		if load_policy:
			saver.restore(tf.get_default_session(), 'trained_policy')

		all_returns = []
		#2. train from aggregated data
		for i in range(num_dagger_updates):
			print(aggregated_data['observations'].shape, aggregated_data['actions'].shape)
			policy_fn = self.train(policy, aggregated_data['observations'], aggregated_data['actions'], epochs, batch_size)
			if eval_running_policy:
				_, curr_return = self.collect_rollouts(env, max_steps, 1, policy_fn)
				print ("Current Evaluation: " + str(np.mean(curr_return)) + " " + str(np.std(curr_return)))
				all_returns.append(curr_return)

			new_data, _ = self.collect_rollouts(env, max_steps, rollouts_per_update, policy_fn)
			new_data['actions'] = np.array(list(map(lambda x: np.array(expert_policy_fn.update(x)), new_data['observations_json'])))
			
			new_data['actions'] = new_data['actions'].reshape((-1, aggregated_data['actions'].shape[1]))	
			aggregated_data['observations'] = np.concatenate([aggregated_data['observations'], new_data['observations']])
			aggregated_data['actions'] = np.concatenate([aggregated_data['actions'], new_data['actions']])
			saver.save(tf.get_default_session(), 'trained_policy')



		np.savetxt('dagger_observations', aggregated_data['observations'])
		np.savetxt('dagger_actions', aggregated_data['actions'])
		print(all_returns)
		for el in all_returns:
			print(np.mean(el), np.std(el))
		return policy_fn



def evaluate_policy(env, max_timesteps, num_rollouts, policy_fn):

	returns = []
	observations = []
	actions = []
	for i in range(num_rollouts):
	    print('iter', i)
	    obs = env.reset(project=True, seed=None, obs_as_dict=True)
	    done = False
	    totalr = 0.
	    steps = 0
	    while not done:
	        obs = ravel_dict_observation(obs, dict_keys)
	        action = policy_fn(obs[None,:])
	        observations.append(obs)
	        actions.append(action)
	        obs, r, done, _ = env.step(action, project=True, obs_as_dict=True)
	        totalr += r
	        steps += 1
	    returns.append(totalr)

	print('returns', returns)
	print('mean return', np.mean(returns))
	print('std of return', np.std(returns))

	expert_data = {'observations': np.array(observations),
	                   'actions': np.array(actions)}
	return expert_data


ALREADY_INITIALIZED = set()
def initialize():
    new_variables = set(tf.all_variables()) - ALREADY_INITIALIZED
    tf.get_default_session().run(tf.initialize_variables(new_variables))
    ALREADY_INITIALIZED.update(new_variables)

def main():
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--visualize', action='store_true')
	parser.add_argument('--load_data', action='store_true')
	parser.add_argument('--load_policy', action='store_true')
	parser.add_argument("--max_timesteps", type=int)
	parser.add_argument('--num_expert_rollouts', type=int, default=1,
	                    help='Number of expert roll outs')
	parser.add_argument('--num_dagger_updates', type=int, default=20,
						help='Number of dagger iterations')
	parser.add_argument('--rollouts_per_update', type=int, default=5,
						help='Number of rollouts collected per dagger iteration')
	parser.add_argument('--epochs', type=int, default=1000)
	parser.add_argument('--batch_size', type=int, default=32)

	mode = '2D'
	difficulty = 2
	visualize=False
	seed=None
	sim_dt = 0.01
	sim_t = 10
	timstep_limit = int(round(sim_t/sim_dt))

	if mode is '2D':
		params = np.loadtxt('./osim/control/params_2D.txt')
	elif mode is '3D':
		params = np.loadtxt('./osim/control/params_3D.txt')
	
	args = parser.parse_args()

	locoCtrl = OsimReflexCtrl(mode=mode, dt=sim_dt)
	locoCtrl.set_control_params(params)
	env = L2M2019Env(visualize=args.visualize, seed=seed, difficulty=difficulty)
	env.change_model(model=mode, difficulty=difficulty, seed=seed)

	env.spec.timestep_limit = timstep_limit

	max_steps = args.max_timesteps or env.spec.timestep_limit

	with tf.Session():
		initialize()
		dagger_policy_fn = DAgger().run_dagger(env, args.load_data, args.load_policy, max_steps, args.num_expert_rollouts,
											args.num_dagger_updates, args.rollouts_per_update, 
											args.epochs, args.batch_size, locoCtrl)


if __name__ == '__main__':
    main()

