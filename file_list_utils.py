import glob
import random

# Write a list of .pkl files inside "pickle_dir" (including subdirectories) 
# to "file_list_file" in a fixed order. Also returns the list of files.
def produce_shuffled_file_list(pickle_dir, file_list_file, num_files=None):
    file_list = glob.glob(pickle_dir + '/**/*.pkl', recursive=True)
    random.shuffle(file_list)
    if num_files is not None:
        file_list = file_list[0:num_files]
    
    # Write to list to file
    with open(file_list_file, 'w') as f:
        for file_path in file_list:
            f.write("%s\n" % file_path)
    return file_list


# Read file paths from a pre-existing "file list" file
def read_file_list(file_list_file, num_files=None):
    with open(file_list_file, 'r') as f:
        file_list = f.readlines()
        if num_files is not None:
            file_list = file_list[0:num_files]
        return file_list


