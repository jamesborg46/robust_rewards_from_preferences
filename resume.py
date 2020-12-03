import garage
from garage import wrap_experiment
from garage.trainer import Trainer
from garage.experiment.deterministic import set_seed
from garage.experiment import Snapshotter

from garage import log_performance, obtain_evaluation_episodes
from garage import EpisodeBatch

import pickle


@wrap_experiment
def pre_trained(ctxt=None,
                snapshot_dir=None,
                seed=1):
    set_seed(seed)
    trainer = Trainer(snapshot_config=ctxt)
    trainer.restore(snapshot_dir)
    # algo = trainer._algo
    # trainer.resume(n_epochs=1002, batch_size=1000)
    # print(trainer._algo._evaluate_policy(trainer))

    eval_episodes = trainer._algo.get_eval_episodes(200)

    with open(trainer._snapshotter.snapshot_dir +
              '/post_skills.pkl', 'wb') as f:
        pickle.dump(eval_episodes, f)

    # eval_episodes = obtain_evaluation_episodes(
    #     algo.policy,
    #     algo._eval_env,
    #     algo._max_episode_length_eval,
    #     num_eps=100,
    #     deterministic=algo._use_deterministic_evaluation)

    # eval_episodes = EpisodeBatch.from_list(
    #     algo._eval_env.spec,
    #     algo.update_diversity_rewards(eval_episodes.to_list())
    # )


d = "/home/mil/james/safety_experiments/robust_rewards_from_preferences/data/local/experiment/diversity_is_all_you_need_seed=10_name=30_alpha_20_skills_12012020_130647_number_skills=20_number_epochs=500_alpha=3.0_env_id=Safexp-PointGoalThree0-v0"
name = "30_alpha_20_skills_resumed"

pre_trained(snapshot_dir=d)
