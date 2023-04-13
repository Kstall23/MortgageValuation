import pandas as pd
import git
from git import Repo
import os

MY_ROOT_PATH = '/Users/laurenhelbling/Documents'
# ---------------------------------------------------------------
# Check that we are in the MortgageValuation directory and change if not
def checkDirectory():
    directory_path = find_directory('MortgageValuation', MY_ROOT_PATH)
    if directory_path is not None:
        os.chdir(directory_path)
    else:
        exit("ERROR ::::: Could not find MortgageValuation directory!!")

def find_directory(directory_name, starting_directory):
    """
    Recursively search for a directory by name within a starting directory and its subdirectories.
    """
    for filename in os.listdir(starting_directory):
        full_path = os.path.join(starting_directory, filename)
        if os.path.isdir(full_path):
            if filename == directory_name:
                return os.path.abspath(full_path)
            else:
                subdirectory = find_directory(directory_name, full_path)
                if subdirectory is not None:
                    return subdirectory
    return None

def returnToFront():
    directory_path = find_directory('frontend', '/Users/laurenhelbling/Documents')
    if directory_path is not None:
        os.chdir(directory_path)

# ---------------------------------------------------------------

# ---------------------------------------------------------------
# Check for a conneciton to the dataset repository, should be in same folder as this repo, but not in this repo
# ==CREATE== a connection if there isn't one
# ===PULL=== current remote repository status 
def getRepo():
    os.chdir('..')
    repo_dir = os.getcwd() + "/Brightvine_model_files"  # path of data repo
    try:
        repo = Repo(repo_dir)  # works if it exists
    except:  # if error, enters this chunk to initialize it
        print("Repo does not yet exist locally")
        print(".....Creating and Cloning now.....")
        # initiate new Git repo
        git.Git(repo_dir).clone("https://github.com/jjbrown23/Brightvine_model_files.git")
        repo = Repo(repo_dir)
    finally:
        assert not repo.bare

    repo.remotes.origin.pull()  # pull most recent repo down locally

    return repo, repo_dir


# ---------------------------------------------------------------

# ---------------------------------------------------------------
# ==LOAD== some data into a DataFrame
def readData(repo_dir, input_file_name, columns):
    data = pd.read_csv(os.path.join(repo_dir, input_file_name), sep=',', names=columns, header=0)
    print("...Reading an input file from remote repo...")
    return data


# ---------------------------------------------------------------

# ---------------------------------------------------------------
# ==PUSH== new changes back to the remote repo, 
# added a bunch of prints to show repo status throughout
def pushRepo(repo, output_file_name, commit_msg):
    print("\n...file edited")
    print(repo.git.status())

    if type(output_file_name) == str:  # if only one file, will show up as a string
        add_files = [output_file_name]  # and it needs to be thrown in a list to be read by git.add
    if type(output_file_name) == list:  # if multiple files, will show up as a list
        add_files = output_file_name  # and it's already in list form for git.add

    repo.index.add(add_files)
    print("\n...file added")
    print(repo.git.status())

    repo.index.commit(commit_msg)
    print("\n...file committed")
    print(repo.git.status())

    repo.remotes.origin.push()
    print("\n...file pushed")
    print(repo.git.status())
# ---------------------------------------------------------------
