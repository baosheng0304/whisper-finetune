
import numpy as np
import os
import ntpath

dataset_root = 'C:/zRHM/dataset'

def get_filepaths(directory):
    """
    This function will generate the file names in a directory
    tree by walking the tree either top-down or bottom-up. For each
    directory in the tree rooted at directory top (including top itself),
    it yields a 3-tuple (dirpath, dirnames, filenames).
    """
    file_paths = []  # List which will store all of the full filepaths.

    # Walk the tree.
    for root, directories, files in os.walk(directory):
        for filename in files:
            # Join the two strings in order to form the full filepath.
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)  # Add it to the list.

    return file_paths  # Self-explanatory.


if __name__ == '__main__':
    audio_file_paths = get_filepaths(os.path.join(dataset_root, 'audio'))
    audio_file_paths.sort()

    f = open(dataset_root+'/audio_paths', 'w')

    for src_path in audio_file_paths:
        head, fname = ntpath.split(src_path)
        id = fname.split('.')[0]
        info = f'{id} {src_path}\n'
        f.write(info)

    f.close()

    text_file_paths = get_filepaths(os.path.join(dataset_root, 'sentence'))
    text_file_paths.sort()

    f = open(dataset_root + '/text', 'w', encoding='utf-8')

    for src_path in text_file_paths:
        head, fname = ntpath.split(src_path)
        id = fname.split('.')[0]

        with open(src_path, encoding='utf-8') as f_r:
            lines = f_r.readlines()

        info = f'{id} {lines[0]}\n'
        f.write(info)

    f.close()
