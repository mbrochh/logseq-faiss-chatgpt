import sys
import re

from .local_settings import LOGSEQ_FOLDER

def extract_notes(filename=None):
    file_path = f'{LOGSEQ_FOLDER}/{filename}'

    with open(file_path, 'r') as f:
        lines = f.readlines()

    notes = []

    for line in lines:
        if not line.strip():
            continue

        indent = len(re.findall(r"(    |\t)", line))
        line = line.strip().replace("- ", '').strip()

        if indent >= 1:
            notes.append(line)

    return notes

if __name__ == '__main__':
    filename = sys.argv[1]
    notes = extract_notes(filename=filename)
    print(notes)