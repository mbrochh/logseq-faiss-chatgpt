import os
import subprocess

from .local_settings import LOGSEQ_FOLDER

def find_notes():
    """
    Finds notes in the LOGSEQ_FOLDER that contain `second-brain:: true`

    Uses `find` and `grep` to get the list of files.

    """
    os.chdir(LOGSEQ_FOLDER)
    command = 'find . -name "*.md" -exec grep -l "second-brain:: true" {} +'
    files = subprocess.run(command, shell=True, capture_output=True, text=True)
    files = files.stdout.split('\n')
    files = [f for f in files if f]
    return files

if __name__ == '__main__':
    files = find_notes()
    print(files)