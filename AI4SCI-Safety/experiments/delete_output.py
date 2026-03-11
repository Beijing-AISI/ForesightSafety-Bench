import shutil
import os
from omegaconf import OmegaConf


def delete_folder_recursively(folder_path):
    """
    Deletes a folder and all its contents recursively.

    Args:
        folder_path (str): The path to the folder to be deleted.
    """
    if not os.path.exists(folder_path):
        print(f"Folder '{folder_path}' does not exist. No action taken.")
        return

    if not os.path.isdir(folder_path):
        print(f"Path '{folder_path}' is not a directory. Please provide a folder path.")
        return

    print(f"Attempting to delete folder: '{folder_path}'...")
    try:
        # shutil.rmtree() is the function that recursively deletes a directory tree.
        shutil.rmtree(folder_path)
        print(f"Successfully deleted folder: '{folder_path}'")
    except OSError as e:
        print(f"Error deleting folder '{folder_path}': {e}")
        print(
            "Possible reasons: Permissions issue, folder is in use by another process, or disk errors."
        )


if __name__ == "__main__":
    # load config
    config_path = "experiments/ai4sci_safebench/1/ai4sci_safebench_1.yaml"
    config = OmegaConf.load(config_path)

    delete_outputs = True
    delete_logs = True

    if delete_outputs:
        delete_folder_recursively(os.path.join('./outputs', config.exp_prefix))

    if delete_logs:
        delete_folder_recursively(os.path.join('./logs', config.exp_prefix))

