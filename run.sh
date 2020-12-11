python robust_rewards.py \
    --seed 6 \
    --name PRECOLLECTED_DIAYN_TRAJS_2 \
    --env_id Safexp-PointGoalThree0-v0 \
    --precollected_trajectories ./data/local/experiment/diversity_is_all_you_need_seed=10_name=fixed_eval_determinism_12042020_003752_number_skills=50_number_epochs=500_alpha=0.5_env_id=Safexp-PointGoalThree0-v0/comparison_collector.pkl
python robust_rewards.py \
    --seed 6 \
    --name RAW_TRAJS_2 \
    --env_id Safexp-PointGoalThree0-v0 \
