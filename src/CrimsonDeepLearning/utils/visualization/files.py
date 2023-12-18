import os
import zipfile

def zip_folder(folder_path, output_path):
    """
    Zips the contents of a folder.

    :param folder_path: Path to the folder to be zipped.
    :param output_path: Path where the output zip file will be saved.
    """
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), os.path.join(folder_path, '..')))

