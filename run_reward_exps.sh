python reward_experiments.py \
    --seed 8 \
    --name PRECOLLECT_3_SEGS \
    --exp PRECOLLECTED_3_12122020_010311 \
    --comparisons comparisons_1000.pkl \
    --test_exp diversity_is_all_you_need_seed=10_name=fixed_eval_determinism_12042020_003752_number_skills=50_number_epochs=500_alpha=0.5_env_id=Safexp-PointGoalThree0-v0 \
    --test_comparisons comparison_collector.pkl \
    --epochs 1000
