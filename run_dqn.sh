python dqn.py \
    --seed 6 \
    --name BREAKOUT_DQN_$1 \
    --env_id Breakout \
    --number_epochs 2000 \
    --snapshot_gap 100 \
    --steps_per_batch 100 \
    --steps_per_epoch 100 \
    --buffer_size 1000000 \
    --n_workers 20 \
    --video_fps 15 \
    --qf "DiscreteCNNQFunction(
			env_spec=env.spec,
			hidden_channels=(32, 64, 64),
			kernel_sizes=(8, 4, 3),
			strides=(4, 2, 1),
			hidden_w_init=(
				lambda x: torch.nn.init.orthogonal_(x, gain=np.sqrt(2))),
			hidden_sizes=(512,),
			is_image=True)" \
    --policy "DiscreteQFArgmaxPolicy(env_spec=env.spec, qf=qf)" \
    --exploration_policy "EpsilonGreedyPolicy(
        env_spec=env.spec,
        policy=policy,
        total_timesteps=num_timesteps,
        max_epsilon=1.0,
        min_epsilon=0.01,
        decay_ratio=0.05)" \
    --algo "CustomDQN(env_spec=env.spec,
               snapshot_dir=snapshot_dir,
               eval_sampler=eval_sampler,
               render_freq=100,
               policy=policy,
               qf=qf,
               exploration_policy=exploration_policy,
               replay_buffer=replay_buffer,
               sampler=sampler,
               steps_per_epoch=steps_per_epoch,
               qf_lr=0.00025,
               clip_rewards=1,
               clip_gradient=10,
               discount=0.99,
               min_buffer_size=50000,
               num_eval_episodes=20,
               n_train_steps=100,
               target_update_freq=100,
               buffer_batch_size=32)" \
    --ray \
    --use_gpu \
    --gpu_id $1
