from .database import create_tables, store_embedding
from .find_notes import find_notes
from .logseq_parser import extract_notes
from .embeddings import get_embedding

def gather_data():
    create_tables()
    filenames = find_notes()
    total_files = len(filenames)
    files_count = 0

    for filename in filenames:
        print(f'Parsing file {filename} ({files_count} of {total_files})')
        notes = extract_notes(filename=filename)
        total_notes = len(notes)
        notes_count = 0
        for note in notes:
            print(f'Parsing sentence {notes_count} of {total_notes}')
            embedding = get_embedding(content=note)
            store_embedding(
                filename=filename,
                content=note,
                embedding=embedding
            )
            notes_count += 1
        files_count += 1

if __name__ == '__main__':
    gather_data()