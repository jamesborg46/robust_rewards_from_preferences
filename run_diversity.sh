python diversity.py \
    --name WANDB_LOGGING_TEST \
    --env_id Safexp-PointIRLGoalThree-v0 \
    --n_workers 2 \
    --steps_per_epoch 4000 \
    --number_skills 4 \
    --render_freq 2 \
    --alpha 0.5 \
    --number_epochs 1001 \
    --seed 12 \
    --max_episode_length 1000 \
    --policy "TanhGaussianMLPPolicy(
                env_spec=env.spec,
                hidden_sizes=[256, 256],
                hidden_nonlinearity=nn.ReLU,
                output_nonlinearity=None,
                min_std=np.exp(-2),
                max_std=np.exp(2))" \
    --qf "ContinuousMLPQFunction(
            env_spec=env.spec,
            hidden_sizes=[256, 256],
            hidden_nonlinearity=F.relu)" \
    --skill_discriminator "SkillDiscriminator(
                                env_spec=env.spec,
                                num_skills=number_skills,
                                learning_rate=0.001,
                                hidden_sizes=[256, 256],
                                hidden_nonlinearity=nn.ReLU,
                                output_nonlinearity=None)" \
    --replay_buffer "PathBuffer(capacity_in_transitions=int(1e6))" \
    --diayn "DIAYN(env_spec=env.spec,
                   sampler=sampler,
                   log_sampler=log_sampler,
                   snapshot_dir=snapshot_dir,
                   policy=policy,
                   qf1=qf1,
                   qf2=qf2,
                   discriminator=skill_discriminator,
                   replay_buffer=replay_buffer,
                   number_skills=number_skills,
                   render_freq=render_freq,
                   gradient_steps_per_itr=20,
                   fixed_alpha=alpha,
                   buffer_batch_size=128,
                   )" \
    --ray

