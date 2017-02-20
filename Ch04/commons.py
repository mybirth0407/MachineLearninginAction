"""
Created on Sep 16, 2010
unzip and remove directory method

@author: Yedarm Seong <mybirth0407@gmail.com>
"""

def unzip(source_file):
    import zipfile
    with zipfile.ZipFile(source_file, 'r') as zf:
        zipInfo = zf.infolist()

        for member in zipInfo:
            zf.extract(member)
    zf.close()


def removeDirectory(directory):
    import shutil
    shutil.rmtree(directory)
