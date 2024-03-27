import argparse
import os
import sys

from RL.hyperparameters.hyperparameters import hyper_parameters_dict as hyper_parameters_base
from RL.hyperparameters.runs.hyperparameters_1 import hyper_parameters_dict as hyper_parameters_1
from RL.hyperparameters.runs.hyperparameters_2 import hyper_parameters_dict as hyper_parameters_2
from RL.hyperparameters.runs.hyperparameters_3 import hyper_parameters_dict as hyper_parameters_3
from RL.hyperparameters.runs.hyperparameters_4 import hyper_parameters_dict as hyper_parameters_4
from RL.hyperparameters.hyperparameters import sweep_config as sweep_config_base
from RL.hyperparameters.runs.hyperparameters_1 import sweep_config as sweep_config_1
from RL.hyperparameters.runs.hyperparameters_2 import sweep_config as sweep_config_2
from RL.hyperparameters.runs.hyperparameters_3 import sweep_config as sweep_config_3
from RL.hyperparameters.runs.hyperparameters_4 import sweep_config as sweep_config_4
from RL.hyperparameters.hyperparameters import option_runs_hyper_parameters as option_runs_hyper_parameters_base
from RL.hyperparameters.runs.hyperparameters_1 import option_runs_hyper_parameters as option_runs_hyper_parameters_1
from RL.hyperparameters.runs.hyperparameters_2 import option_runs_hyper_parameters as option_runs_hyper_parameters_2
from RL.hyperparameters.runs.hyperparameters_3 import option_runs_hyper_parameters as option_runs_hyper_parameters_3
from RL.hyperparameters.runs.hyperparameters_4 import option_runs_hyper_parameters as option_runs_hyper_parameters_4
from RL.wandb_training import wandb_train_env, wandb_sweep


def get_hyper_parameter_by_parameters(hyper_parameter_choice, log=True):
    """
    return dictionary of dictionaries of the right choice
    :param log:
    :param hyper_parameter_choice:
    :return:
    """
    hyper_parameter_choice = str(hyper_parameter_choice)  # case of not string
    chosen_hyper_param_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "RL",
                                           "hyperparameters", "runs",
                                           f"hyperparameters_{hyper_parameter_choice}.py")

    if log:
        print(f"chosen_hyper_param_dict: {hyper_parameter_choice}")
        sys.stdout.flush()
    # take the right dictionaries by file
    if hyper_parameter_choice == "1":
        chosen_sweep_config = sweep_config_1
        chosen_hyper_parameters_dict = hyper_parameters_1
        chosen_option_runs = option_runs_hyper_parameters_1
    elif hyper_parameter_choice == "2":
        chosen_sweep_config = sweep_config_2
        chosen_hyper_parameters_dict = hyper_parameters_2
        chosen_option_runs = option_runs_hyper_parameters_2
    elif hyper_parameter_choice == "3":
        chosen_sweep_config = sweep_config_3
        chosen_hyper_parameters_dict = hyper_parameters_3
        chosen_option_runs = option_runs_hyper_parameters_3
    elif hyper_parameter_choice == "4":
        chosen_sweep_config = sweep_config_4
        chosen_hyper_parameters_dict = hyper_parameters_4
        chosen_option_runs = option_runs_hyper_parameters_4
    else:
        chosen_sweep_config = sweep_config_base
        chosen_hyper_parameters_dict = hyper_parameters_base
        chosen_option_runs = option_runs_hyper_parameters_base
        chosen_hyper_param_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "RL",
                                               "hyperparameters", "hyperparameters.py")

    chosen_params_dict = {
        "hyper_parameters_dict": chosen_hyper_parameters_dict,
        "sweep_config": chosen_sweep_config,
        "option_runs": chosen_option_runs,
        "hyper_param_path": chosen_hyper_param_path
    }

    return chosen_params_dict


if __name__ == "__main__":
    # add parser to parse arguments
    parser = argparse.ArgumentParser()
    # Required positional argument
    parser.add_argument('alg_name', type=str, help='algorithm name')
    parser.add_argument('short_desc', type=str, help='short description of the run')
    parser.add_argument('--timestamps', type=int, help='total timestamps in run')
    parser.add_argument('--horizontal_wind', type=float, help='horizontal wind in x axis')
    parser.add_argument('--time_back', type=int, help='the state time back')
    parser.add_argument('--velocity', type=float, help='initial velocity og the glider')
    parser.add_argument('--thermal_mode', type=str, help='the mode of the thermal')
    parser.add_argument('--count_sweep', type=int, help='the count for the sweep')
    parser.add_argument('--sweep_id', type=str, help='sweep id to start from')
    parser.add_argument('--hyperparameter', type=str, help='hyperparameter name for another files')

    parser.add_argument('--delete_folder', action='store_true')
    parser.add_argument('--no-delete_folder', dest='delete_folder', action='store_false')
    parser.set_defaults(delete_folder=False)

    args = parser.parse_args()

    params_dict = get_hyper_parameter_by_parameters(args.hyperparameter)

    hyper_parameters_dict = params_dict["hyper_parameters_dict"]
    sweep_config = params_dict["sweep_config"]
    option_runs = params_dict["option_runs"]
    hyper_param_path = params_dict["hyper_param_path"]

    hyper_parameters_dict["algorithm_name"] = args.alg_name
    hyper_parameters_dict["short_description"] = args.short_desc
    if args.timestamps is not None:
        hyper_parameters_dict["total_timestamps"] = args.timestamps

    if args.horizontal_wind is not None:
        hyper_parameters_dict["horizontal_wind"][0] = args.horizontal_wind

    if args.time_back is not None:
        hyper_parameters_dict["time_back"] = args.time_back

    if args.thermal_mode is not None:
        hyper_parameters_dict["mode"] = args.thermal_mode

    if args.velocity is not None:
        hyper_parameters_dict["velocity"] = args.velocity

    # for deleting the wandb logs folders
    if args.delete_folder is not None:
        hyper_parameters_dict["is_delete_folder"] = args.delete_folder

    # for count sweep
    count_sweep = 1
    if args.count_sweep is not None:
        count_sweep = args.count_sweep

    # for sweep id
    sweep_id = args.sweep_id

    # for saving the hyperparameters.py file
    hyper_parameters_dict["hyper_param_path"] = hyper_param_path

    if args.alg_name == "sweep":
        wandb_sweep(hyper_parameters_dict, sweep_config, count_sweep=count_sweep, old_sweep_id=sweep_id)
    else:
        wandb_train_env(hyper_parameters_dict)
