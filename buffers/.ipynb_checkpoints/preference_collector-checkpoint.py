import abc
from garage import EpisodeBatch, TimeStepBatch
from garage.np import slice_nested_dict
import numpy as np
import os
from utils.video import write_segment_to_video, upload_to_gcs


class Segment(TimeStepBatch):

    @property
    def total_reward(self):
        return sum(self.rewards)

    @classmethod
    def slice_from_episode(cls, episode, start, end):
        sliced = cls(
            env_spec=episode.env_spec,
            observations=episode.observations[start:end],
            actions=episode.actions[start:end],
            rewards=episode.rewards[start:end],
            next_observations=episode.observations[start+1:end+1],
            env_infos=slice_nested_dict(episode.env_infos, start, end),
            agent_infos=slice_nested_dict(episode.agent_infos, start, end),
            step_types=episode.step_types[start:end],
        )
        return sliced


# class PreferenceCollector():
class PreferenceCollector(abc.ABC):

    def __init__(self,
                 env_spec,
                 label_scheduler,
                 segment_length=1,
                 max_capacity=100000):

        self.env_spec = env_spec
        self.label_scheduler = label_scheduler
        self.max_capacity = max_capacity
        self.episodes = None
        self._total_steps = 0
        self._comparisons = []
        self._segment_length = segment_length

    def collect(self, paths):
        """

        Args:
            paths: as returned from EpisodeBatch.to_list()
        """

        episodes = EpisodeBatch.from_list(self.env_spec, paths)
        self._total_steps += len(episodes.observations)
        self._extend(episodes)
        assert self._total_steps == len(self.episodes.observations)

        if self._total_steps > self.max_capacity:
            self._reduce()

    def _extend(self, episodes):
        if self.episodes is None:
            self.episodes = episodes
        else:
            self.episodes = EpisodeBatch.concatenate(
                self.episodes,
                episodes
            )

    def _reduce(self):
        end = len(self.episodes.observations)
        popped_length = 0
        i = 0

        while self._total_steps - popped_length > self.max_capacity:
            popped_length += self.episodes.lengths[i]
            i += 1

        popped = EpisodeBatch(
            env_spec=self.episodes.env_spec,
            observations=self.episodes.observations[:popped_length],
            last_observations=self.episodes.last_observations[:i],
            actions=self.episodes.actions[:popped_length],
            rewards=self.episodes.rewards[:popped_length],
            env_infos=slice_nested_dict(self.episodes.env_infos, 0, popped_length),
            agent_infos=slice_nested_dict(self.episodes.agent_infos, 0, popped_length),
            step_types=self.episodes.step_types[:popped_length],
            lengths=self.episodes.lengths[:i]
        )

        self.episodes = EpisodeBatch(
            env_spec=self.episodes.env_spec,
            observations=self.episodes.observations[popped_length:],
            last_observations=self.episodes.last_observations[i:],
            actions=self.episodes.actions[popped_length:],
            rewards=self.episodes.rewards[popped_length:],
            env_infos=slice_nested_dict(self.episodes.env_infos, popped_length, end),
            agent_infos=slice_nested_dict(self.episodes.agent_infos, popped_length, end),
            step_types=self.episodes.step_types[popped_length:],
            lengths=self.episodes.lengths[i:],
        )

        self._total_steps -= popped_length
        assert self._total_steps == len(self.episodes.observations)

        return popped

    def requires_more_comparisons(self, itr):
        num_required_comps = self.label_scheduler.n_desired_labels(itr)
        return self.num_comparisons < num_required_comps

    def sample_comparisons(self, itr):
        while self.requires_more_comparisons(itr):
            self._sample_comparison()

    def _sample_comparison(self):
        left = self._sample_segment(self._segment_length)
        right = self._sample_segment(self._segment_length)
        self.add_segment_pair(left, right)

    def _sample_episode(self, lengths_weighted=False):
        # Selecting an episode weighted by its length
        num_episodes = len(self.episodes.lengths)
        if lengths_weighted:
            weights = self.episodes.lengths / self.episodes.lengths.sum()
            idx = np.random.choice(num_episodes, p=weights)
            return self.episodes.split()[idx]

        # Selecting an episode at random
        else:
            idx = np.random.choice(num_episodes)
            return self.episodes.split()[idx]

    def _sample_segment(self, segment_length):
        episode = self._sample_episode(lengths_weighted=True)
        length = episode.lengths[0]
        if length < segment_length:
            raise Exception('Sampled episode shorter than required segment'
                            'length')
        start = np.random.randint(length - segment_length - 1)
        end = start + segment_length
        return Segment.slice_from_episode(episode, start, end)

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
        return [comp for comp in self._comparisons
                if comp['label'] is not None]

    @property
    def labeled_decisive_comparisons(self):
        return [comp for comp in self._comparisons if comp['label'] in [0, 1]]

    @property
    def unlabeled_comparisons(self):
        return [comp for comp in self._comparisons if comp['label'] is None]


class SyntheticPreferenceCollector(PreferenceCollector):

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
            left_seg.total_reward > right_seg.total_reward
        )
        # Mutate the comparison and give it the new label
        comparison['label'] = 0 if left_has_more_rew else 1


def _write_and_upload_video(env, gcs_path, local_path, segment):
    write_segment_to_video(segment, fname=local_path, env=env)
    upload_to_gcs(local_path, gcs_path)


class HumanPreferenceCollector(PreferenceCollector):

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

        media_url = "https://storage.googleapis.com/{}/{}".format(
            gcs_bucket.lstrip("gs://"),
            media_id)
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
        if itr == 0:
            return self.pretrain_labels

        # Decay from 1 to 0
        exp_decay_frac = 0.01 ** (itr / self._final_timesteps)
        pretrain_frac = self.pretrain_labels / self._final_labels
        # Start with 0.25 and anneal to 0.99
        desired_frac = (
            pretrain_frac + (1 - pretrain_frac) * (1 - exp_decay_frac)
        )
        return desired_frac * self._final_labels


