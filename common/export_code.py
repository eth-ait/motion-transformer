import os
import zipfile


def export_code(file_list, output_file):
    """Stores files in a zip."""
    if not output_file.endswith('.zip'):
        output_file += '.zip'
    ofile = output_file
    counter = 0
    while os.path.exists(ofile):
        counter += 1
        ofile = output_file.replace('.zip', '_{}.zip'.format(counter))
    zipf = zipfile.ZipFile(ofile, mode="w", compression=zipfile.ZIP_DEFLATED)
    for f in file_list:
        zipf.write(f)
    zipf.close()