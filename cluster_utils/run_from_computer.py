import argparse
import os
from datetime import datetime
import yaml
import main
import credentials_and_settings  # only local
import paramiko as paramiko


def log_to_history_file(
        log_file_path=r"C:",
        log_txt="", hyper_param_choice=None):
    """
    log to txt file
    :param hyper_param_choice: if not None log the hyperparameter from the right file
    :param log_txt: the txt to log
    :param log_file_path: the path of the log file
    :return:
    """
    # check if the log file exists if not create it
    if not os.path.exists(log_file_path):
        with open(log_file_path, "w") as f:
            f.write("")

    # add the timestamp to the txt
    final_txt = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"

    # add the log txt
    final_txt += f"command:\n{log_txt}\n"

    # get the hyperparameters txt from hyper_parameter_choice
    if hyper_param_choice is not None:
        chosen_params_dict = main.get_hyper_parameter_by_parameters(hyper_param_choice, log=False)
        final_txt += f"hyper param dict: \n{yaml.dump(chosen_params_dict, default_flow_style=False)}\n"

    # add the txt to the log file
    with open(log_file_path, "a") as f:
        f.write(final_txt)


def cancel_runs(job_identifier=4):
    """
    cancel all the runs
    :return:
    """
    cmd_list = [f"cd {credentials_and_settings.slurm_runs_path}",
                "squeue -u $USER | grep " + str(job_identifier) + " | awk '{print $1}' | xargs -n 1 scancel"]

    return cmd_list


def show_job(job_identifier=4):
    """
    cancel all the runs
    :return:
    """
    cmd_list = [f"scontrol show job {job_identifier}"]

    return cmd_list


def get_runs(job_identifier=-1):
    """
    return the runs names or logs
    if job_identifier is -1 return all the runs names else return the log for the specific run
    :param job_identifier:
    :return:
    """
    if job_identifier == -1:
        cmd_list = [f"cd {credentials_and_settings.slurm_runs_path}", "squeue -u $USER"]
    else:
        # print the log of the specific job
        cmd_list = [f"cd {credentials_and_settings.slurm_runs_path}", f"cat slurm-{str(job_identifier)}.out"]

    return cmd_list


def get_update_code_cmd():
    """
    update the code on the server
    :return:
    """
    # not working need to understand how to pull and enter password on server
    return []


def get_create_and_upload_hyperparameters_file_cmd():
    """
    create the hyperparameters file on the server
    :return:
    """
    # not working need to understand how to move file to the server
    return []


def get_regular_cmd(hyper_parameter_choice=1, algorithm="DDPG", description="check"):
    """
    return the commands for running the code with run.sbatch
    :return:
    """
    cmd_list = [f"cd {credentials_and_settings.slurm_runs_path}",
                f"source {credentials_and_settings.environment_activation_path}",
                f"sbatch {credentials_and_settings.sbatchs_directory}/run.sbatch {hyper_parameter_choice} {algorithm} {description}"]

    return cmd_list


def get_sweep_cmd(hyper_parameter_choice=1, num_sweeps=10):
    """
    return the commands for running the code with sweep_run.sbatch
    :return:
    """
    cmd_list = [f"cd {credentials_and_settings.slurm_runs_path}",
                f"source {credentials_and_settings.environment_activation_path}",
                f"sbatch {credentials_and_settings.sbatchs_directory}/sweep_run.sbatch {hyper_parameter_choice} {num_sweeps}"]

    return cmd_list


def get_seed_cmd(hyper_parameter_choice=1, number_runs=1):
    """
    return the commands for running the code with seed_run.sbatch
    :param hyper_parameter_choice: the hyper parameter choice
    :param number_runs: the number of runs
    :return:
    """
    cmd_list = [f"cd {credentials_and_settings.slurm_runs_path}",
                f"source {credentials_and_settings.environment_activation_path}",
                f"sbatch --array=0-{number_runs} {credentials_and_settings.sbatchs_directory}/seed_run.sbatch {hyper_parameter_choice}"]

    return cmd_list


if __name__ == '__main__':
    # connecting to server
    log = True
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(credentials_and_settings.server, username=credentials_and_settings.username, password=credentials_and_settings.password, look_for_keys=False,
                allow_agent=False)
    print("Connected to server")

    # interactive console for running commands on the server
    while True:
        hyper_param_choice = None

        input_cmd_lst = input(">>>").split()
        if len(input_cmd_lst) == 0:
            continue
        input_cmd = input_cmd_lst[0]

        cmd_list = []
        if input_cmd == "regular":
            if len(input_cmd_lst) > 3:
                hyper_param_choice = input_cmd_lst[1]
                algorithm = input_cmd_lst[2]
                description = input_cmd_lst[3]
                cmd_list = get_regular_cmd(hyper_param_choice, algorithm, description)
            else:
                print("need to pass hyperparameter number, algorithm and description")
                continue
        elif input_cmd == "seed":
            if len(input_cmd_lst) > 2:
                hyper_param_choice = input_cmd_lst[1]
                num_runs = input_cmd_lst[2]
                cmd_list = get_seed_cmd(hyper_param_choice, num_runs)
            else:
                print("need to pass hyperparameter number and number of runs")
                continue
        elif input_cmd == "sweep":
            if len(input_cmd_lst) > 1:
                hyper_param_choice = input_cmd_lst[1]
                cmd_list = get_sweep_cmd(hyper_param_choice)
            else:
                print("need to pass hyperparameter number")
                continue
        elif input_cmd == "update":
            cmd_list = get_update_code_cmd()
        elif input_cmd == "create_hyper":
            cmd_list = get_create_and_upload_hyperparameters_file_cmd()
        elif input_cmd == "cancel":
            if len(input_cmd_lst) > 1:
                cmd_list = cancel_runs(input_cmd_lst[1])
            else:
                print("need to specify the identifier of the runs to cancel")
                continue
        elif input_cmd == "get_runs":
            if len(input_cmd_lst) > 1:
                cmd_list = get_runs(input_cmd_lst[1])
            else:
                cmd_list = get_runs()
        elif input_cmd == "show_job":
            if len(input_cmd_lst) > 1:
                cmd_list = show_job(input_cmd_lst[1])
            else:
                print("need to specify the identifier of the runs to cancel")
                continue
        elif input_cmd == "run_from_list":
            try:
                cmd_list = " ".join(input_cmd_lst[1:]).split("[")[1].split("]")[0].split(";")
            except:
                print("need to specify list of commands to run")
        elif input_cmd == "exit":
            break

        # log if want to
        if log:
            log_to_history_file(log_file_path=credentials_and_settings.log_file_path, log_txt={"commands in computer": input_cmd_lst, "commands in server": cmd_list},
                                hyper_param_choice=hyper_param_choice)

        if len(cmd_list) > 0:
            cmd = ";".join(cmd_list)

            print(f">>> {cmd}")

            stdin, stdout, stderr = ssh.exec_command(cmd)

            # wait until cmd ends
            while not stdout.channel.exit_status_ready():
                pass
            print(stdout.read().decode("utf-8"))
            err = stderr.read().decode("utf-8")
            if err != "":
                print(err)
        else:
            print("no such command to run")

    ssh.close()
    print("closed connection")
