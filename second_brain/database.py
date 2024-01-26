import sqlite3
import json

from .local_settings import SQLITE_DB_PATH
from .embeddings import get_embedding

def get_rows_by_id(row_ids):
    conn = sqlite3.connect(SQLITE_DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT id, filename, content 
        FROM embeddings
        WHERE id IN ({seq})
        """.format(seq=','.join(['?']*len(row_ids))), 
        row_ids
    )
    records = cursor.fetchall()
    return records

def get_all_embeddings():
    conn = sqlite3.connect(SQLITE_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, embedding FROM embeddings")
    records = cursor.fetchall()
    return records

def create_tables():
    conn = sqlite3.connect(SQLITE_DB_PATH)
    cursor = conn.cursor()

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS embeddings (
        id INTEGER PRIMARY KEY,
        filename TEXT,
        content TEXT,
        embedding TEXT,
        UNIQUE (filename, content)
    )
    ''')
    conn.commit()

def store_embedding(
        filename=None,
        content=None, 
        embedding=None
):
    embedding_str = json.dumps(embedding)

    conn = sqlite3.connect(SQLITE_DB_PATH)
    cursor = conn.cursor()

    try:
        cursor.execute(
            "INSERT INTO embeddings (filename, content, embedding) VALUES (?, ?, ?)",
            (filename or '', content or '', embedding_str)
        )
    except sqlite3.IntegrityError:
        print(f'Embedding for {filename} already exists')

    conn.commit()

if __name__ == '__main__':
    create_tables()

    embedding = get_embedding(content='This is a test.')

    store_embedding(
        filename='test.txt',
        content='This is a test.',
        embedding=embedding,
    )

    all_embeddings = get_all_embeddings()
    embeddings_count = len(all_embeddings)
    print(f'Found {embeddings_count} embeddings')

    rows = get_rows_by_id([1])
    print(rows)