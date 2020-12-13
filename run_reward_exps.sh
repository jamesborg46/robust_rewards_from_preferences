python reward_experiments.py \
    --seed 8 \
    --name PRECOLLECT_5_SEGS_LENGTH_5 \
    --exp PRECOLLECTED_5_SEG_LENGTH_5_TOTALLY_ORDERED_12122020_192221 \
    --comparisons comparisons_700.pkl \
    --test_exp diversity_is_all_you_need_seed=10_name=fixed_eval_determinism_12042020_003752_number_skills=50_number_epochs=500_alpha=0.5_env_id=Safexp-PointGoalThree0-v0 \
    --test_comparisons comparison_collector.pkl \
    --epochs 1000 \
    --use_total_ordering
