python robust_rewards.py --name COMPS_WITH_COST_THREE --seed 10 --number_epochs 151 --pre_train_labels 200 --final_labels 1000  --env_id Safexp-PointGoalThree0-v0  --local
python robust_rewards.py --name COMPS_WITH_COST_BEHIND --seed 10 --number_epochs 151 --pre_train_labels 200 --final_labels 1000  --env_id Safexp-PointGoalBehind0-v0  --local
python robust_rewards.py --name COMPS_WITH_COST_FRONT --seed 10 --number_epochs 151 --pre_train_labels 200 --final_labels 1000  --env_id Safexp-PointGoalCustom0-v0  --local
python robust_rewards.py --name GT_WITH_COST_THREE --seed 10 --number_epochs 151 --env_id Safexp-PointGoalThree0-v0  --local --use_gt_rewards
python robust_rewards.py --name GT_WITH_COST_BEHIND --seed 10 --number_epochs 151 --env_id Safexp-PointGoalBehind0-v0  --local --use_gt_rewards
python robust_rewards.py --name GT_WITH_COST_FRONT --seed 10 --number_epochs 151 --env_id Safexp-PointGoalCustom0-v0  --local --use_gt_rewards
python robust_rewards.py --name COMPS_WITH_COST_THREE_2 --seed 10 --number_epochs 151 --pre_train_labels 200 --final_labels 1000  --env_id Safexp-PointGoalThree0-v0  --local
python robust_rewards.py --name COMPS_WITH_COST_BEHIND_2 --seed 10 --number_epochs 151 --pre_train_labels 200 --final_labels 1000  --env_id Safexp-PointGoalBehind0-v0  --local
python robust_rewards.py --name COMPS_WITH_COST_FRONT_2 --seed 10 --number_epochs 151 --pre_train_labels 200 --final_labels 1000  --env_id Safexp-PointGoalCustom0-v0  --local
python robust_rewards.py --name GT_WITH_COST_THREE_2 --seed 10 --number_epochs 151 --env_id Safexp-PointGoalThree0-v0  --local --use_gt_rewards
python robust_rewards.py --name GT_WITH_COST_BEHIND_2 --seed 10 --number_epochs 151 --env_id Safexp-PointGoalBehind0-v0  --local --use_gt_rewards
python robust_rewards.py --name GT_WITH_COST_FRONT_2 --seed 10 --number_epochs 151 --env_id Safexp-PointGoalCustom0-v0  --local --use_gt_rewards
