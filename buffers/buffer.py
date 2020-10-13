from garage.replay_buffer import PathBuffer
import numpy as np
from garage import StepType


class ComparisonBuffer(PathBuffer):
    """A buffer that extends PathBuffer with added functionality for sampling
    segments and collecting segment comparisons

    Args:
        capacity_in_transitions (int): Total memory allocated for the buffer.
        env_spec (EnvSpec): Environment specification.
    """

    def __init__(self,
                 capacity_in_transitions,
                 label_scheduler,
                 segment_length=1):

        super().__init__(capacity_in_transitions)
        self.label_scheduler = label_scheduler
        self.segment_length = segment_length
        self._comparisons = []

    def add_episode_batch(self, step_itr, episodes):
        """Add a EpisodeBatch to the buffer.
        Args:
            episodes (EpisodeBatch): Episodes to add.
        """
        env_spec = episodes.env_spec
        obs_space = env_spec.observation_space
        for eps in episodes.split():
            terminals = np.array([
                step_type == StepType.TERMINAL for step_type in eps.step_types
            ],
                                 dtype=bool)
            path = {
                'observations': obs_space.flatten_n(eps.observations),
                'next_observations':
                obs_space.flatten_n(eps.next_observations),
                'actions': env_spec.action_space.flatten_n(eps.actions),
                'gt_rewards': eps.env_infos['gt_reward'].reshape(-1, 1),
                'terminals': terminals.reshape(-1, 1),
            }
            self.add_path(path)


    def requires_more_comparisons(self, step_itr):
        num_labeled_comps = len(self.labeled_comparisons)
        num_desired_labels = self.label_scheduler.n_desired_labels(step_itr)
        return num_labeled_comps < num_desired_labels

    def sample_segment(self):
        i = 0
        length = 0

        # Try sample a path that has a length of at least segment_length at
        # most 5 times, otherwise fail
        while (i < 5) and (length < self.segment_length):
            path = self.sample_path()
            length = self._get_path_length(path)
            i += 1

        if length < self.segment_length:
            raise Exception('Unable to sample path greater than segment length'
                            'within 5 attempts')

        start = np.random.randint(length - self.segment_length)
        end = start + self.segment_length

        segment = {key: value[start:end] for key, value in path.items()}
        return segment

    def sample_comparison(self):
        left = self.sample_segment()
        right = self.sample_segment()
        self.add_segment_pair(left, right)

    def add_segment_pair(self, left_seg, right_seg):
        """Add a new unlabeled comparison from a segment pair"""
        comparison = {
            "left": left_seg,
            "right": right_seg,
            "label": None
        }
        self._comparisons.append(comparison)

    @property
    def num_comparisons(self):
        return len(self._comparisons)

    @property
    def labeled_comparisons(self):
        return [comp for comp in self._comparisons if comp['label'] is not None]

    @property
    def labeled_decisive_comparisons(self):
        return [comp for comp in self._comparisons if comp['label'] in [0, 1]]

    @property
    def unlabeled_comparisons(self):
        return [comp for comp in self._comparisons if comp['label'] is None]

    def label_unlabeled_comparisons(self):
        for comp in self.unlabeled_comparisons:
            self._add_synthetic_label(comp)

    @staticmethod
    def _add_synthetic_label(comparison):
        left_seg = comparison['left']
        right_seg = comparison['right']
        left_has_more_rew = (
            np.sum(left_seg["gt_rewards"]) > np.sum(right_seg["gt_rewards"])
        )
        # Mutate the comparison and give it the new label
        comparison['label'] = 0 if left_has_more_rew else 1


class LabelAnnealer(object):
    """Keeps track of how many labels we want to collect"""

    def __init__(self, final_timesteps, final_labels, pretrain_labels):
        self._final_timesteps = final_timesteps
        self._final_labels = final_labels
        self.pretrain_labels = pretrain_labels

    def n_desired_labels(self, itr):
        """Return the number of labels desired at this point in training. """

        # Decay from 1 to 0
        exp_decay_frac = 0.01 ** (itr / self._final_timesteps)
        pretrain_frac = self.pretrain_labels / self._final_labels
        # Start with 0.25 and anneal to 0.99
        desired_frac = (
            pretrain_frac + (1 - pretrain_frac) * (1 - exp_decay_frac)
        )
        return desired_frac * self._final_labels


# class ConstantLabelSchedule(object):
#     def __init__(self, pretrain_labels, seconds_between_labels=3.0):
#         self._started_at = None  # Don't initialize until we call n_desired_labels
#         self._seconds_between_labels = seconds_between_labels
#         self._pretrain_labels = pretrain_labels

#     @property
#     def n_desired_labels(self):
#         if self._started_at is None:
#             self._started_at = time()
#         return self._pretrain_labels + (time() - self._started_at) / self._seconds_between_labels
