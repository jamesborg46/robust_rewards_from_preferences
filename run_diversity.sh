python diversity.py \
    --name TEST_OTHER_BRANCH \
    --env_id Safexp-PointIRLGoalThree-v0 \
    --n_workers 8 \
    --batch_size 16000 \
    --number_skills 24 \
    --number_epochs 1001 \
    --seed 12 \
    --max_episode_length 1000 \
    --render_freq 200 \
    --alpha 0.5 \
    --ray 
    # --use_gpu \
    # --gpu_id $1
# python diversity.py \
#     --name DIVERSITY_ALPHA_01 \
#     --env_id Safexp-PointIRLGoalThree-v0 \
#     --n_workers 10 \
#     --batch_size 10000 \
#     --number_skills 20 \
#     --number_epochs 1001 \
#     --seed 12 \
#     --max_episode_length 1000 \
#     --alpha 0.1 \
#     --ray \
#     --use_gpu
# python diversity.py \
#     --name DIVERSITY_ALPHA_10 \
#     --env_id Safexp-PointIRLGoalThree-v0 \
#     --n_workers 10 \
#     --batch_size 10000 \
#     --number_skills 20 \
#     --number_epochs 1001 \
#     --seed 12 \
#     --max_episode_length 1000 \
#     --alpha 1 \
#     --ray \
#     --use_gpu
