import os

def retrieve_save_path(save_path, default_file_name):
    print(save_path)
    file_name = default_file_name
    # splits path into directories without file name and file name only
    file_path_tuple = os.path.split(save_path)
    print(file_path_tuple)
    if file_path_tuple[1] != "":
        file_name = file_path_tuple[1]
    if not os.path.exists(file_path_tuple[0]):
        os.mkdir(file_path_tuple[0])
    save_name = os.path.join(file_path_tuple[0], file_name)
    return save_name