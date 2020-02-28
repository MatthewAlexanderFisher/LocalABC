from pathlib import Path
import pickle
import dill
import os

### Save output


def save_output(out, name, dir_name):
    root = Path(".")
    new_name = name + ".pkl"
    new_path = root / dir_name / new_name
    with open(new_path, "wb") as handle:
        dill.dump(out, handle, protocol=pickle.HIGHEST_PROTOCOL)


### Creates output directory


def create_out_dir(dir_name):
    try:
        os.mkdir(dir_name)
        print("Output directory", dir_name, "created")
    except FileExistsError:
        print("Output directory", dir_name, "already exists \nSaving output to", dir_name)


### Remove file


def del_file(name, dir_name):
    root = Path(".")
    new_path = root / dir_name / name
    if os.path.exists(new_path):
        os.remove(new_path)
    else:
        print("File not found error")