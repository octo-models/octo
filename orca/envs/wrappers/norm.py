import gym


class UnnormalizeActionProprio(gym.Wrapper):
    def __init__(
        self, env: gym.Env, action_proprio_metadata: dict, normalization_type: str
    ):
        self.action_proprio_metadata = action_proprio_metadata
        self.normalization_type = normalization_type
        super().__init__(env)

    def step(self, action, *args, **kwargs):
        """
        Un-normalizes the action and proprio
        """
        if self.normalization_type == "normal":
            action = (
                action * self.action_proprio_metadata["action"]["std"]
            ) + self.action_proprio_metadata["action"]["mean"]
            obs, reward, done, trunc, info = self.env.step(action, *args, **kwargs)
            obs["proprio"] = (
                obs["proprio"] * self.action_proprio_metadata["proprio"]["std"]
            ) + self.action_proprio_metadata["proprio"]["mean"]
        elif self.normalization_type == "bounds":
            action = (
                action
                * (
                    self.action_proprio_metadata["action"]["max"]
                    - self.action_proprio_metadata["action"]["min"]
                )
            ) + self.action_proprio_metadata["action"]["min"]
            obs, reward, done, trunc, info = self.env.step(action, *args, **kwargs)
            obs["proprio"] = (
                obs["proprio"]
                * (
                    self.action_proprio_metadata["proprio"]["max"]
                    - self.action_proprio_metadata["proprio"]["min"]
                )
            ) + self.action_proprio_metadata["proprio"]["min"]
        else:
            raise ValueError

        return obs, reward, done, trunc, info
