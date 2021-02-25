from dowel import tabular
import numpy as np
import torch
import torch.nn.functional as F

from garage import StepType
from garage.torch.algos import SAC
from garage.torch import dict_np_to_torch, global_device
from garage.sampler.env_update import EnvUpdate

import pickle
import os
import os.path as osp
from utils import split_flattened, one_hot_to_int, update_remote_agent_device


class DIAYN(SAC):
    """Docstring for DIAYN. """

    def __init__(self,
                 number_skills,
                 discriminator,
                 snapshot_dir,
                 log_sampler,
                 render_freq=200,
                 **kwargs
                 ):
        """TODO: to be defined.

        Parameters
        ----------
        number_skills : TODO
        discriminator : TODO
        snapshot_dir : TODO
        log_sampler : TODO
        render_freq : TODO, optional


        """

        super().__init__(self, **kwargs)

        self._number_skills = number_skills
        self._discriminator = discriminator
        self._snapshot_dir = snapshot_dir
        self._log_sampler = log_sampler
        self._render_freq = render_freq

