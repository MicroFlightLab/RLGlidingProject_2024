import logging
import os
from stable_baselines3 import DDPG
from stable_baselines3 import A2C
from stable_baselines3 import SAC
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import wandb
from RL import training_utils
from wandb.sdk.lib import telemetry as wb_telemetry
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
import gym

logger = logging.getLogger(__name__)


class WandbCallback(BaseCallback):
    """Log SB3 experiments to Weights and Biases
        - Added model tracking and uploading
        - Added complete hyperparameters recording
        - Added gradient logging
        - Note that `wandb.init(...)` must be called before the WandbCallback can be used

    Args:
        verbose: The verbosity of sb3 output
        model_save_path: Path to the folder where the model will be saved, The default value is `None` so the model is not logged
        model_save_freq: Frequency to save the model
        gradient_save_freq: Frequency to log gradient. The default value is 0 so the gradients are not logged

        ADDED params_functions: dictionary of param to function for each param to log (function gets model)
        ADDED param_calc_freq: frequency of calculating parameters
        ADDED animation_save_freq: frequency of plotting
        ADDED plot_env: the environment for plotting
        ADDED episodes_for_param: when calculating param the model run on the env episodes_for_param simulations

    Yoav Flato Appendix:
    copied and changed the code.
    changes:
    1. save model in different names for each timestamp (zip files).
    2. added plot of the flight for each couple of rounds.
    3. added option to calculate params.
    """

    def __init__(
            self,
            verbose: int = 0,
            model_save_path: str = None,
            model_save_freq: int = 0,
            video_save_each_n_animation_save: int = 1,
            gradient_save_freq: int = 0,
            param_calc_freq: int = 1000,
            params_functions: dict = {},
            animation_save_freq: int = 0,
            episodes_for_param: int = 15,
            hyper_params_dict: dict = {},
            plot_env=None
    ):
        super().__init__(verbose)
        if wandb.run is None:
            raise wandb.Error("You must call wandb.init() before WandbCallback()")
        # with wb_telemetry.context() as tel:
        #     tel.feature.sb3 = True
        self.model_save_freq = model_save_freq
        self.video_save_each_n_animation_save = video_save_each_n_animation_save
        self.model_save_path = model_save_path
        self.param_calc_freq = param_calc_freq
        self.gradient_save_freq = gradient_save_freq
        self.params_functions = params_functions
        self.animation_save_freq = animation_save_freq
        self.plot_env = plot_env
        self.episodes_for_param = episodes_for_param
        self.hyper_params_dict = hyper_params_dict
        # Create folder if needed
        if self.model_save_path is not None:
            os.makedirs(self.model_save_path, exist_ok=True)
            self.path = os.path.join(self.model_save_path, "model.zip")
        else:
            assert (
                    self.model_save_freq == 0
            ), "to use the `model_save_freq` you have to set the `model_save_path` parameter"

    def _init_callback(self) -> None:
        d = {}
        if "algo" not in d:
            d["algo"] = type(self.model).__name__
        for key in self.model.__dict__:
            if key in wandb.config:
                continue
            if type(self.model.__dict__[key]) in [float, int, str]:
                d[key] = self.model.__dict__[key]
            else:
                d[key] = str(self.model.__dict__[key])
        # if self.gradient_save_freq > 0:
        #     wandb.watch(self.model.policy, log_freq=self.gradient_save_freq, log="all")
        wandb.config.setdefaults(d)

    def _on_step(self) -> bool:
        log_dict = {}
        # calculate params for log to wandb
        if self.n_calls % self.param_calc_freq == 0:
            info_df = self.get_info_df(episodes=self.episodes_for_param)

            # add the params to dictionary for function calculating
            params_for_param_calculations = dict()
            params_for_param_calculations["hyper_params_dict"] = self.hyper_params_dict
            params_for_param_calculations["model"] = self.model
            params_for_param_calculations["env"] = self.plot_env
            params_for_param_calculations["info_df"] = info_df
            params_for_param_calculations["df_num_episodes"] = self.episodes_for_param
            for param in self.params_functions.keys():
                log_dict[param] = self.params_functions[param](params_for_param_calculations)
                # wandb.log({param: self.params_functions[param](self.model)}, step=self.n_calls)

        # plots animation
        if self.plot_env is not None and self.n_calls % self.animation_save_freq == 0:
            print("plotting")
            figs = self.get_env_plot(return_both=True)
            animation_plot = figs["animation"]
            animation_html_file = os.path.join(self.model_save_path, "animation_tmp.html")
            animation_plot.write_html(animation_html_file, full_html=False)
            with open(animation_html_file, "r") as f:
                animation_html = "\n".join(f.readlines())

            if self.n_calls % (self.video_save_each_n_animation_save * self.animation_save_freq) == 0:
                log_dict["agent_video"] = wandb.Html(animation_html, inject=False)
            log_dict["agent_figure"] = figs["figure"]
            log_dict["agent_controls"] = figs["controls"]
            # wandb.log({"agent_video": plot}, step=self.n_calls)

        if len(log_dict) != 0:
            log_dict["timestamp"] = self.n_calls
            wandb.log(log_dict)

        # save model
        if self.model_save_freq > 0:
            if self.model_save_path is not None:
                if self.n_calls % self.model_save_freq == 0:
                    self.save_model()
        return True

    def get_env_plot(self, animation=False, return_both=True):
        """
        :return: figure of the environment with the current model
        """
        obs = self.plot_env.reset()
        done = False
        while not done:
            action, _states = self.model.predict(obs)
            obs, rewards, done, info = self.plot_env.step(action)
        return self.plot_env.render(animation=animation, return_fig=True, return_both=return_both)

    def _on_training_end(self) -> None:
        if self.model_save_path is not None:
            self.save_model()

    def save_model(self) -> None:
        path = os.path.join(self.model_save_path, f"model_{type(self.model).__name__}_{self.n_calls}.zip")
        self.model.save(path)
        wandb.save(path, base_path=self.model_save_path)
        if self.verbose > 1:
            print("Saving model checkpoint to " + path)

    def get_info_df(self, episodes=15):
        """
        return the information df of a model
        :return:
        """
        return training_utils.get_df_for_analysis_by_model_env(self.model, self.plot_env, episodes=episodes)
