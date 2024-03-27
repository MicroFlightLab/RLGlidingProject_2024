import argparse
import os
import main
from RL.wandb_training import wandb_train_env
from RL import training_utils

if __name__ == "__main__":
    # add parser to parse arguments
    parser = argparse.ArgumentParser()
    # Required positional argument
    parser.add_argument('option', type=int, help='the option number')
    parser.add_argument('--hyperparameter', type=str, help='hyperparameter name for another files')
    parser.add_argument('--delete_folder', action='store_true')
    parser.add_argument('--no-delete_folder', dest='delete_folder', action='store_false')
    parser.set_defaults(delete_folder=False)

    args = parser.parse_args()

    params_dict = main.get_hyper_parameter_by_parameters(args.hyperparameter)
    hyper_parameters_dict = params_dict["hyper_parameters_dict"]
    sweep_config = params_dict["sweep_config"]
    option_runs = params_dict["option_runs"]
    hyper_param_path = params_dict["hyper_param_path"]

    run_hyper_dict = option_runs.get(args.option)

    if run_hyper_dict is None:
        print("Option number is not valid")

    # convert the run hyper dict to the new hyper-parameters dict
    hyper_parameters_dict = training_utils.update(hyper_parameters_dict, run_hyper_dict)

    # for deleting the wandb logs folders
    is_delete_folder = False
    if args.delete_folder is not None:
        is_delete_folder = args.delete_folder
    hyper_parameters_dict["is_delete_folder"] = is_delete_folder

    # for saving the hyperparameters.py file
    hyper_parameters_dict["hyper_param_path"] = hyper_param_path

    wandb_train_env(hyper_parameters_dict)
