from osim.env import L2M2019Env
from osim.control.osim_loco_reflex_song2019 import OsimReflexCtrl
from joblib import Parallel, delayed

import sys
import numpy as np

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--difficulty', type=int, required=True)
parser.add_argument('--std', type=float, required=True)

args = parser.parse_args()

trial_name = 'trial_opt_from_2d' + str(args.difficulty) + '_' + str(args.std)

params_2d = np.loadtxt('../osim/control/params_2D.txt')
params_opt = np.ones(9)

#params = np.loadtxt('./optim_data/cma/trial_181029_walk_3D_noStand_8_best.txt')
N_POP = 12 # 8 = 4 + floor(3*log(9))
N_PROC = 5
TIMEOUT = 10*60
      
init_pose = np.array([1.5, .9, 10*np.pi/180, # forward speed, pelvis height, trunk lean
        -3*np.pi/180, -30*np.pi/180, -10*np.pi/180, 10*np.pi/180, # [right] hip abduct, hip extend, knee extend, ankle extend
        -3*np.pi/180, 5*np.pi/180, -40*np.pi/180, -0*np.pi/180]) # [left] hip abduct, hip extend, knee extend, ankle extend
  
def f_ind(n_gen, i_worker, params):
    flag_model = '3D'
    flag_ctrl_mode = '3D' # use 2D
    seed = None
    difficulty = args.difficulty
    sim_dt = 0.01
    sim_t = 10
    timstep_limit = int(round(sim_t/sim_dt))

    init_error = True
    error_count = 0
    while init_error:
        try:
            locoCtrl = OsimReflexCtrl(mode=flag_ctrl_mode, dt=sim_dt)
            env = L2M2019Env(seed=seed, difficulty=difficulty, visualize=False)
            env.change_model(model=flag_model, difficulty=difficulty, seed=seed)
            obs_dict = env.reset(project=True, seed=seed, obs_as_dict=True)
            init_error = False
        except Exception as e_msg:
            error_count += 1
            print('\ninitialization error (x{})!!!'.format(error_count))
            #print(e_msg)
            #import pdb; pdb.set_trace()
    env.spec.timestep_limit = timstep_limit+100

    total_reward = 0
    error_sim = 0;
    t = 0
    for i in range(timstep_limit+100):
        t += sim_dt

        final_params = np.concatenate([params_2d, params])
        locoCtrl.set_control_params(final_params)
        action = locoCtrl.update(obs_dict)
        obs_dict, reward, done, info = env.step(action, project=True, obs_as_dict=True)
        total_reward += reward

        if done:
            break

    print('\n    gen#={} sim#={}: score={} time={}sec #step={}'.format(n_gen, i_worker, total_reward, t, env.footstep['n']))

    return total_reward  # minimization


class CMATrainPar(object):
    def __init__(self, ):
        self.n_gen = 0
        self.best_total_reward = -np.inf

    def f(self, v_params):
        self.n_gen += 1
        timeout_error = True
        error_count = 0
        while timeout_error:
            try:
                v_total_reward = Parallel(n_jobs=N_PROC, timeout=TIMEOUT)\
                (delayed(f_ind)(self.n_gen, i, p) for i, p in enumerate(v_params))
                timeout_error = False
            except Exception as e_msg:
                error_count += 1
                print('\ntimeout error (x{})!!!'.format(error_count))
                print(e_msg)

        mean_reward = np.mean(v_total_reward)

        if self.best_total_reward  < mean_reward:
            filename = "./optim_data/cma/" + trial_name + "best_w.txt"
            print("\n")
            print("----")
            print("update the best score!!!!")
            print("\tprev = %.8f" % self.best_total_reward )
            print("\tcurr = %.8f" % mean_reward)
            print("\tsave to [%s]" % filename)
            print("----")
            print("")
            self.best_total_reward  = mean_reward
            np.savetxt(filename, v_params)

        return [-r for r in v_total_reward]

if __name__ == '__main__':
    prob = CMATrainPar()

    from cmaes.solver_cma import CMASolverPar
    solver = CMASolverPar(prob)

    solver.options.set("popsize", N_POP)
    solver.options.set("maxiter", 800)
    solver.options.set("verb_filenameprefix", 'optim_data/cma/' + trial_name)
    solver.set_verbose(True)

    x0 = params_opt
    sigma = args.std

    res = solver.solve(x0, sigma)
    print(res)
