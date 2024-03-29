python robust_rewards.py \
    --seed 6 \
    --name FIXED_IRL_$1 \
    --env_id 'Safexp-PointIRLGoalThree-v0' \
    --number_epochs 1000 \
    --snapshot_gap 200 \
    --steps_per_epoch 10000 \
    --max_episode_length 1000 \
    --n_workers 10 \
    --policy "GaussianMLPPolicy(env.spec,
                                hidden_sizes=[32, 32],
                                hidden_nonlinearity=torch.tanh,
                                output_nonlinearity=None)" \
    --value_function "GaussianMLPValueFunction(env.spec,
                                               hidden_sizes=[32, 32],
                                               hidden_nonlinearity=torch.tanh,
                                               output_nonlinearity=None)" \
    --vf_optimizer "OptimizerWrapper((torch.optim.Adam, dict(lr=1e-3)),
                                     value_function,
                                     max_optimization_epochs=100,
                                     minibatch_size=256)" \
    --label_scheduler "LabelAnnealer(number_epochs=kwargs['number_epochs'],
                                     final_labels=1000,
                                     pretrain_labels=100)" \
    --data_collector "SyntheticPreferenceCollector(env.spec,
                                                   label_scheduler,
                                                   segment_length=1,
                                                   max_capacity=20000)" \
    --reward_predictor "PrefMLP(env.spec,
                                hidden_sizes=[64, 64],
                                preference_collector=data_collector,
                                pretrain_epochs=50,
                                iterations_per_epoch=100,
                                minibatch_size=256)" \
    --algo "IrlTRPO(env.spec,
                    reward_predictor=reward_predictor,
                    snapshot_dir=snapshot_dir,
                    log_sampler=log_sampler,
                    sampler=sampler,
                    policy=policy,
                    value_function=value_function,
                    vf_optimizer=vf_optimizer,
                    render_freq=200,
                    entropy_method='regularized',
                    policy_ent_coeff=0.1,
                    discount=1,
                    gae_lambda=1,)" \
    --ray
