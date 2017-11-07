import os
import time
from collections import deque
import pickle

from baselines.ddpg.ddpg import DDPG
import baselines.common.tf_util as U

from baselines import logger
import numpy as np
import tensorflow as tf
from mpi4py import MPI


def train(env, nb_episodes, render_eval, reward_scale, render, param_noise, actor, critic,
          normalize_returns, normalize_observations, critic_l2_reg, actor_lr, critic_lr, action_noise,
          popart, gamma, clip_norm, nb_rollout_steps, nb_eval_steps, nb_train_steps, train_freq, batch_size, memory,
          tau=0.01, eval_env=None, param_noise_adaption_interval=50):

    assert (np.abs(env.action_space.low) == env.action_space.high).all()  # we assume symmetric actions.
    max_action = env.action_space.high
    logger.info('scaling actions by {} before executing in env'.format(max_action))
    agent = DDPG(actor, critic, memory, memory.state_shape, memory.action_shape,
                 gamma=gamma, tau=tau, normalize_returns=normalize_returns,
                 normalize_observations=normalize_observations,
                 batch_size=batch_size, action_noise=action_noise, param_noise=param_noise, critic_l2_reg=critic_l2_reg,
                 actor_lr=actor_lr, critic_lr=critic_lr, enable_popart=popart, clip_norm=clip_norm,
                 reward_scale=reward_scale)
    logger.info('Using agent with the following configuration:')
    logger.info(str(agent.__dict__.items()))

    # Set up logging stuff only for a single worker.
    # if rank == 0:
    #     saver = tf.train.Saver()
    # else:
    #     saver = None

    eval_episode_rewards_history = deque(maxlen=100)
    episode_rewards_history = deque(maxlen=100)
    with tf.Session() as sess:

        # Prepare everything.
        agent.initialize(sess)
        agent.reset()
        if eval_env is not None:
            eval_obs = eval_env.reset()

        total_steps = 0
        start_time = time.time()
        episode_rewards = []
        episode_steps = []
        episode_actions = []
        episode_qs = []
        successes = 0

        for episode in range(nb_episodes):

            # Perform rollouts.
            episode_step = 0
            episode_reward = 0.
            agent.reset()
            obs = env.reset()

            # Selects a goal for the current episode
            goal_found = False
            while not goal_found:
                goal_episode = agent.memory.env_wrapper.sample_goal()
                state_init = agent.memory.env_wrapper.process_state(obs, goal_episode)
                goal_found = agent.memory.env_wrapper.evaluate_goal(state_init)

            for iter in range(nb_rollout_steps):

                # Compute state from observation and internal observation (goal)
                state0 = agent.memory.env_wrapper.process_state(obs, goal_episode)

                # Predict next action.
                action, q = agent.pi(state0, apply_noise=True, compute_Q=True)

                if render:
                    env.render()

                # Perform one step in the environment
                new_obs, r, done, info = env.step(max_action * action)  # scale for execution in env

                # Compute next state from next observation and internal observation (goal)
                state1 = agent.memory.env_wrapper.process_state(new_obs, goal_episode)

                # Compute reward and terminal condition from environment wrapper instead of from environment
                r, terminal1 = agent.memory.env_wrapper.evaluate_transition(state0, action, state1)

                if render:
                    env.render()

                # Book-keeping.
                episode_actions.append(action)
                episode_qs.append(q)
                episode_reward += r
                episode_step += 1
                total_steps += 1
                if terminal1:
                    successes += 1

                # Add to experience replay : TODO make it automatic from memory
                buffer_item = {'state0': state0,
                               'action': action,
                               'reward': r,
                               'state1': state1,
                               'terminal1': terminal1}
                agent.store_transition(buffer_item)

                # Update current observation
                obs = new_obs

                if terminal1:
                    break

                # Train.
                if episode % train_freq == 0:
                    episode_actor_losses = []
                    episode_critic_losses = []
                    for t_train in range(1):
                        if memory.nb_entries >= batch_size * 3:
                            cl, al = agent.train()
                            episode_critic_losses.append(cl)
                            episode_actor_losses.append(al)
                            agent.update_target_net()

            episode_rewards.append(episode_reward)
            episode_steps.append(episode_step)

            # # Evaluate.
            # eval_episode_rewards = []
            # eval_qs = []
            # if eval_env is not None:
            #     eval_episode_reward = 0.
            #     for t_rollout in range(nb_eval_steps):
            #         eval_action, eval_q = agent.pi(eval_obs, apply_noise=False, compute_Q=True)
            #         eval_obs, eval_r, eval_done, eval_info = eval_env.step(
            #             max_action * eval_action)  # scale for execution in env (as far as DDPG is concerned, every action is in [-1, 1])
            #         if render_eval:
            #             eval_env.render()
            #         eval_episode_reward += eval_r
            #
            #         eval_qs.append(eval_q)
            #         if eval_done:
            #             eval_obs = eval_env.reset()
            #             eval_episode_rewards.append(eval_episode_reward)
            #             eval_episode_rewards_history.append(eval_episode_reward)
            #             eval_episode_reward = 0.

            # Log stats
            if episode % 1 == 0:
                print('| Reward: {:d} | Episode: {:d}'.format(int(episode_reward), episode))
                duration = time.time() - start_time
                stats = agent.get_stats()
                combined_stats = {}
                for key in sorted(stats.keys()):
                    combined_stats[key] = np.mean(stats[key])

                # Rollout statistics.
                combined_stats['rollout/return'] = np.mean(episode_rewards)
                combined_stats['rollout/episode_steps'] = np.mean(episode_steps)
                combined_stats['rollout/actions_mean'] = np.mean(episode_actions)
                combined_stats['rollout/actions_std'] = np.std(episode_actions)
                combined_stats['rollout/Q_mean'] = np.mean(episode_qs)

                # Train statistics.
                combined_stats['train/loss_actor'] = np.mean(episode_actor_losses)
                combined_stats['train/loss_critic'] = np.mean(episode_critic_losses)

                # # Evaluation statistics.
                # if eval_env is not None:
                #     combined_stats['eval/return'] = mpi_mean(eval_episode_rewards)
                #     combined_stats['eval/return_history'] = mpi_mean(np.mean(eval_episode_rewards_history))
                #     combined_stats['eval/Q'] = mpi_mean(eval_qs)
                #     combined_stats['eval/episodes'] = mpi_mean(len(eval_episode_rewards))

                # Total statistics.
                combined_stats['total/duration'] = duration
                combined_stats['total/successes'] = successes
                combined_stats['total/steps'] = total_steps
                combined_stats['total/steps_per_second'] = float(total_steps) / float(duration)

                for key in sorted(combined_stats.keys()):
                    logger.record_tabular(key, combined_stats[key])
                logger.dump_tabular()
                logger.info('')
                logdir = logger.get_dir()
                # if rank == 0 and logdir:
                if logdir:
                    if hasattr(env, 'get_state'):
                        with open(os.path.join(logdir, 'env_state.pkl'), 'wb') as f:
                            pickle.dump(env.get_state(), f)
                    if eval_env and hasattr(eval_env, 'get_state'):
                        with open(os.path.join(logdir, 'eval_env_state.pkl'), 'wb') as f:
                            pickle.dump(eval_env.get_state(), f)
