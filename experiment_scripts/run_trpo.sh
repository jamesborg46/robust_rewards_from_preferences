python trpo.py \
    --seed $1 \
    --name TRPO_INITIAL_EXP_$1 \
    --env_id 'Safexp-PointIRLGoalThree-v0' \
    --number_epochs 1000 \
    --snapshot_gap 200 \
    --steps_per_epoch 8000 \
    --max_episode_length 1000 \
    --n_workers 8 \
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
                                     max_optimization_epochs=20,
                                     minibatch_size=64)" \
    --algo "TRPOWithVideos(render_freq=100,
                           env_spec=env.spec,
                           policy=policy,
                           value_function=value_function,
                           vf_optimizer=vf_optimizer,)" \
    --ray
