from stable_baselines3.common.callbacks import BaseCallback

class StopTrainingAfterNStepsCallback(BaseCallback):
    def __init__(self, n_steps, verbose=0):
        super(StopTrainingAfterNStepsCallback, self).__init__(verbose)
        self.n_steps = n_steps

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.n_steps:
            #self.model.logger.info(f"Training terminated after {self.n_steps} steps.")
            return False
        return True
