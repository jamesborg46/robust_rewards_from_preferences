from garage.torch.modules import MLPModule

class SkillDiscriminator(MLPModule):
    """
    SkillDiscriminator is a Module to discriminate amongst skills in the 
    diversity is all you need algorithm
    """

    def __init__(self,
                 env_spec,
                 num_skills,
                 optimizer,
                 **kwargs):
        """TODO: to be defined.

        Parameters
        ----------
        env_spec : TODO
        num_skills : TODO
        optimizer : TODO


        """
        MLPModule.__init__(
            self,
            env_spec=env_spec,
            input_dim=env_spec.observation_space['state'].flat_dim,
            output_dim=num_skills,
            **kwargs,
            )

        self._env_spec = env_spec
        self._num_skills = num_skills
        self._optimizer = optimizer

