python robust_rewards.py \
    --seed 6 \
    --name TEST \
    --env_id Safexp-PointIRLGoalThree-v0 \
    --number_epochs 1000 \
    --snapshot_gap 200 \
    --steps_per_epoch 1000 \
    --max_episode_length 1000 \
    --n_workers 2 \
    --policy "GaussianMLPPolicy(env.spec, hidden_sizes=[32, 32], hidden_nonlinearity=torch.tanh, output_nonlinearity=None)" \
    --value_function "GaussianMLPValueFunction(env.spec, hidden_sizes=[32, 32], hidden_nonlinearity=torch.tanh, output_nonlinearity=None)" \
    --label_scheduler "LabelAnnealer(number_epochs=kwargs['number_epochs'], final_labels=1000, pretrain_labels=200)" \
    --data_collector "SyntheticPreferenceCollector(env.spec, label_scheduler, segment_length=1, max_capacity=20000)" \
    --reward_predictor "PrefMLP(env.spec, preference_collector=data_collector, pretrain_epochs=10)" \
    --algo "IrlTRPO(env.spec, reward_predictor=reward_predictor, policy=policy, value_function=value_function)"
    
