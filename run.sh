python robust_rewards.py \
    --seed 6 \
    --name PRECOLLECTED_3 \
    --env_id Safexp-PointGoalThree0-v0 \
    --discount 0.99 \
    --val_opt_its 200 \
    --val_opt_lr 1e-3 \
    --reward_opt_its 100 \
    --reward_opt_lr 1e-3 \
    --center_adv \
    --precollected_trajectories ./data/local/experiment/diversity_is_all_you_need_seed=10_name=fixed_eval_determinism_12042020_003752_number_skills=50_number_epochs=500_alpha=0.5_env_id=Safexp-PointGoalThree0-v0/comparison_collector.pkl
python robust_rewards.py \
    --seed 6 \
    --name RAW_3 \
    --env_id Safexp-PointGoalThree0-v0 \
    --discount 0.99 \
    --val_opt_its 200 \
    --val_opt_lr 1e-3 \
    --reward_opt_its 100 \
    --reward_opt_lr 1e-3 \
    --center_adv \
python robust_rewards.py \
    --seed 6 \
    --name PRECOLLECTED_4 \
    --env_id Safexp-PointGoalThree0-v0 \
    --discount 0.99 \
    --val_opt_its 200 \
    --val_opt_lr 1e-4 \
    --reward_opt_its 100 \
    --reward_opt_lr 1e-3 \
    --center_adv \
    --precollected_trajectories ./data/local/experiment/diversity_is_all_you_need_seed=10_name=fixed_eval_determinism_12042020_003752_number_skills=50_number_epochs=500_alpha=0.5_env_id=Safexp-PointGoalThree0-v0/comparison_collector.pkl
python robust_rewards.py \
    --seed 6 \
    --name RAW_4 \
    --env_id Safexp-PointGoalThree0-v0 \
    --discount 0.99 \
    --val_opt_its 200 \
    --val_opt_lr 1e-4 \
    --reward_opt_its 100 \
    --reward_opt_lr 1e-3 \
    --center_adv \
