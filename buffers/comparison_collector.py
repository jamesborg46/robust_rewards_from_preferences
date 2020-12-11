from garage.replay_buffer import PathBuffer
import os
import numpy as np
import multiprocessing
from garage import StepType
import uuid
import abc
import collections

from dowel import logger
import time

from utils.video import write_segment_to_video, upload_to_gcs


def always_true(i):
    return True


# class ComparisonCollector(PathBuffer, abc.ABC):
class ComparisonCollector(abc.ABC):
    """A buffer that extends PathBuffer with added functionality for sampling
    segments and collecting segment comparisons

    Args:
        capacity_in_transitions (int): Total memory allocated for the buffer.
        env_spec (EnvSpec): Environment specification.
    """

    def __init__(self,
                 capacity_in_transitions,
                 label_scheduler,
                 collect_callable=None,
                 segment_length=1,):

        # super().__init__(capacity_in_transitions)
        self._capacity = capacity_in_transitions
        self.label_scheduler = label_scheduler
        self.segment_length = segment_length
        if collect_callable is None:
            self.collect_callable = always_true
        else:
            self.collect_callable = collect_callable

        self._comparisons = []
        self._paths = collections.deque()
        self.n_transitions_stored = 0

    def flush(self):
        self._comparisons = []
        self.n_transitions_stored = 0

    @staticmethod
    def _get_path_length(path):
        """Get path length.
        Args:
            path (dict): Path.
        Returns:
            length: Path length.
        Raises:
            ValueError: If path is empty or has inconsistent lengths.
        """
        length_key = None
        length = None
        for key, value in path.items():
            if length is None:
                length = len(value)
                length_key = key
            elif len(value) != length:
                raise ValueError('path has inconsistent lengths between '
                                 '{!r} and {!r}.'.format(length_key, key))
        if not length:
            raise ValueError('Nothing in path')
        return length

    def add_path(self, path):
        l = self._get_path_length(path)
        while (l + self.n_transitions_stored) > self._capacity:
            removed = self._paths.popleft()
            self.n_transitions_stored -= self._get_path_length(removed)
        self._paths.append(path)
        self.n_transitions_stored += l

    def sample_path(self):
        path_idx = np.random.randint(len(self._paths))
        return self._paths[path_idx]

    # def collect(self, step_itr, eps):
    def collect(self, step_itr, env_spec, paths):
        if self.collect_callable(step_itr):
            # self.add_episode_batch(step_itr, eps)
            self.add_episode_batch(step_itr, env_spec, paths)

        while self.requires_more_comparisons(step_itr):
            self.sample_comparison()

        self.label_unlabeled_comparisons()
        while self.requires_more_labels(step_itr):
            logger.log('Collecting more labels...')
            time.sleep(10)
            self.label_unlabeled_comparisons()

    # def step_paths(self):
    #     for i in range(len(self._path_segments)):
    #         first_seg, second_seg = self._path_segments[i]
    #         first_seg_indices = np.arange(first_seg.start, first_seg.stop)
    #         second_seg_indices = np.arange(second_seg.start, second_seg.stop)
    #         indices = np.concatenate([first_seg_indices, second_seg_indices])
    #         path = {key: buf_arr[indices] for
    #                 key, buf_arr in self._buffer.items()}
    #         yield path

    # def add_episode_batch(self, step_itr, episodes):
    #     """Add a EpisodeBatch to the buffer.
    #     Args:
    #         episodes (EpisodeBatch): Episodes to add.
    #     """
    #     env_spec = episodes.env_spec
    #     obs_space = env_spec.observation_space
    #     for eps in episodes.split():
    #         terminals = np.array([
    #             step_type == StepType.TERMINAL for step_type in eps.step_types
    #         ], dtype=bool)
    #         path = {
    #             'observations': obs_space.flatten_n(eps.observations),
    #             'next_observations':
    #             obs_space.flatten_n(eps.next_observations),
    #             'actions': env_spec.action_space.flatten_n(eps.actions),
    #             'gt_rewards': eps.env_infos['gt_reward'].reshape(-1, 1),
    #             'states': eps.env_infos['state'],
    #             'terminals': terminals.reshape(-1, 1),
    #             'model_xml': eps.env_infos['model_xml'].reshape(-1, 1),
    #         }
    #         self.add_path(path)

    def add_episode_batch(self, step_itr, env_spec, paths):
        """Add a EpisodeBatch to the buffer.
        Args:
            episodes (EpisodeBatch): Episodes to add.
        """
        obs_space = env_spec.observation_space
        for path in paths:
            terminals = np.array([
                step_type == StepType.TERMINAL for step_type
                in path['step_types']
            ], dtype=bool)

            if terminals.flatten()[:-1].any():
                breakpoint()

            if 'predicted_rewards' in path.keys():
                original_predicted_rewards = path['predicted_rewards'].reshape(-1, 1)
                original_returns = path['env_infos']['original_returns'].reshape(-1, 1)
                original_baselines = path['env_infos']['original_baselines'].reshape(-1, 1)
            else:
                original_predicted_rewards = np.empty(
                    path['observations'].shape[0]
                ).reshape(-1, 1)
                original_predicted_rewards[:] = np.NaN

                original_returns = np.empty(
                    path['observations'].shape[0]
                ).reshape(-1, 1)
                original_predicted_rewards[:] = np.NaN

                original_baselines = np.empty(
                    path['observations'].shape[0]
                ).reshape(-1, 1)
                original_predicted_rewards[:] = np.NaN

            path = {
                'observations': obs_space.flatten_n(path['observations']),
                'next_observations':
                obs_space.flatten_n(path['next_observations']),
                'actions': env_spec.action_space.flatten_n(path['actions']),
                'gt_rewards': path['env_infos']['gt_reward'].reshape(-1, 1),
                'original_predicted_rewards': original_predicted_rewards,
                'original_returns': original_returns,
                'original_baselines': original_baselines,
                'states': path['env_infos']['state'],
                'terminals': terminals.reshape(-1, 1),
                'model_xml': path['env_infos']['model_xml'].reshape(-1, 1),
            }
            self.add_path(path)

    def requires_more_comparisons(self, step_itr):
        num_desired_labels = self.label_scheduler.n_desired_labels(step_itr)
        return self.num_comparisons < num_desired_labels

    def requires_more_labels(self, step_itr):
        num_labeled_comps = len(self.labeled_comparisons)
        num_desired_labels = self.label_scheduler.n_desired_labels(step_itr) * 0.8
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

        segment = {key: value[start:end] for key, value in path.items()
                   if key != 'model_xml'}

        segment['model_xml'] = path['model_xml'][0, 0]
        return segment

    def sample_comparison(self):
        left = self.sample_segment()
        right = self.sample_segment()
        self.add_segment_pair(left, right)

    @abc.abstractmethod
    def add_segment_pair(self, left_seg, right_sef):
        pass

    @abc.abstractmethod
    def label_unlabeled_comparisons(self):
        pass

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


class SyntheticComparisonCollector(ComparisonCollector):

    def add_segment_pair(self, left_seg, right_seg):
        """Add a new unlabeled comparison from a segment pair"""
        comparison = {
            "left": left_seg,
            "right": right_seg,
            "label": None
        }
        self._comparisons.append(comparison)

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


def _write_and_upload_video(env, gcs_path, local_path, segment):
    write_segment_to_video(segment, fname=local_path, env=env)
    upload_to_gcs(local_path, gcs_path)


class HumanComparisonCollector(ComparisonCollector):

    def __init__(self,
                 capacity_in_transitions,
                 label_scheduler,
                 env,
                 experiment_name,
                 segment_length=1):
        from human_feedback_api import Comparison

        super().__init__(capacity_in_transitions,
                         label_scheduler,
                         segment_length=segment_length)

        self.env = env
        self.experiment_name = experiment_name
        # self._upload_workers = multiprocessing.Pool(4)

        if (Comparison.objects
                      .filter(experiment_name=experiment_name)
                      .count() > 0):
            raise EnvironmentError("Existing experiment named %s!"
                                   " Pick a new experiment name."
                                   % experiment_name)

    def add_segment_pair(self, left_seg, right_seg):
        """Add a new unlabeled comparison from a segment pair"""

        comparison_id = self._create_comparison_in_webapp(left_seg, right_seg)
        comparison = {
            "left": left_seg,
            "right": right_seg,
            "id": comparison_id,
            "label": None
        }

        self._comparisons.append(comparison)

    def label_unlabeled_comparisons(self):
        from human_feedback_api import Comparison

        for comparison in self.unlabeled_comparisons:
            db_comp = Comparison.objects.get(pk=comparison['id'])
            if db_comp.response == 'left':
                comparison['label'] = 0
            elif db_comp.response == 'right':
                comparison['label'] = 1
            elif db_comp.response == 'tie' or db_comp.response == 'abstain':
                comparison['label'] = 'equal'
                # If we did not match, then there is
                # no response yet, so we just wait

    def convert_segment_to_media_url(self, comparison_uuid, side, segment):
        tmp_media_dir = '/tmp/rl_teacher_media'
        media_id = "%s-%s.mp4" % (comparison_uuid, side)
        local_path = os.path.join(tmp_media_dir, media_id)
        gcs_bucket = os.environ.get('RL_TEACHER_GCS_BUCKET')
        gcs_path = os.path.join(gcs_bucket, media_id)
        _write_and_upload_video(self.env, gcs_path, local_path, segment)
        # upload_workers = multiprocessing.Pool(4)
        # upload_workers.apply_async(_write_and_upload_video,
        #                                  (self.env,
        #                                   gcs_path,
        #                                   local_path,
        #                                   segment)
                                         # )

        media_url = "https://storage.googleapis.com/%s/%s" % (gcs_bucket.lstrip("gs://"), media_id)
        return media_url

    def _create_comparison_in_webapp(self, left_seg, right_seg):
        """Creates a comparison DB object. Returns the db_id of the comparison"""
        from human_feedback_api import Comparison

        comparison_uuid = str(uuid.uuid4())
        comparison = Comparison(
            experiment_name=self.experiment_name,
            media_url_1=self.convert_segment_to_media_url(comparison_uuid, 'left', left_seg),
            media_url_2=self.convert_segment_to_media_url(comparison_uuid, 'right', right_seg),
            response_kind='left_or_right',
            priority=1.
        )
        comparison.full_clean()
        comparison.save()
        return comparison.id


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
