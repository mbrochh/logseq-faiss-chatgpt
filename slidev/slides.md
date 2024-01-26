---
# try also 'default' to start simple
theme: default
title: Talk to your notes with Logseq and ChatGPT! 
background: images/cover.jpg
# apply any windi css classes to the current slide
class: 'text-center'
highlighter: shiki
# show line numbers in code blocks
lineNumbers: true
drawings:
  persist: true
# page transition
transition: fade-out
# use UnoCSS
css: unocss
---

<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<div class="text-left relative">
    <div class="z-90 text-lg">Talk to your notes with Logseq and ChatGPT!</div>
    <div class="text-sm mt-0">by <a href="https://twitter.com/mbrochh" target="_blank">Martin Brochhaus</a></div>
    <div class="text-sm mt-1">slides: <a href="https://bit.ly/pugs-logseq" target="_blank">https://bit.ly/pugs-logseq</a></div>
</div>

---
clicks: 4
---

# About PUGS

<ul>
    <li v-click="1">
        PUGS stands for <span class="text-red">Python User Group Singapore</span>
        <ul>
            <li>registered non-profit society, run by volunteers</li>
            <li>was created to organize <a class="text-red" href="https://pycon.sg" target="_blank">PyCon Singapore</a></li>
        </ul>
    </li>
    <li v-click="2">
        Visit <a class="text-red" href="https://pugs.org.sg/membership/" target="_blank">https://pugs.org.sg/membership/</a> to become a member
    </li>
    <li v-click="3">
        Monthly meetups at <a class="text-red" href="https://www.meetup.com/singapore-python-user-group/" target="_blank">https://www.meetup.com/singapore-python-user-group/</a>
    </li>
    <li v-click="4">
        Join our Telegram group 
    </li>
</ul>

<div class="grid grid-cols-4 gap-4 mt-4 max-h-[220px] overflow-hidden">
  <div v-click="1"><img src="/images/pycon.png" /></div>
  <div v-click="2"><img src="/images/membership.png" /></div>
  <div v-click="3"><img src="/images/meetup.png" /></div>
  <div v-click="4"><img src="/images/telegram.jpeg" /></div>
</div>

---

# About me

- Martin Brochhaus
- CTO of [theartling.com](https://theartling.com/en/)
- Committee member of PUGS
- Twitter: [@mbrochh](https://twitter.com/mbrochh)

<div class="grid grid-cols-2 gap-4 mt-4 max-h-[300px] overflow-hidden">
  <div><img src="/images/theartling.png" class="max-h-[300px] mx-auto" /></div>
  <div><img src="/images/twitter.png" class="max-h-[300px] mx-auto" /></div>
</div>

---

# Part 1: Summarising Youtube Videos With Whisper & ChatGPT

- Repo for the first talk: [here](https://github.com/mbrochh/whisper-youtube-transcribe)
- Step 1: **Download** the **audio** of a Youtube video
- Step 2: **Transcribe** the audio into a textfile with **Whisper**
- Step 3: Turn the text into **chunks** of no more than ~14000 words
- Step 4: Send each chunk to **OpenAI's API** and request a **summary**

<img src="/images/part1-cover.jpg" class="max-h-[200px] mx-auto mt-4" />

---

# Part 2: Summarising Kindle Books With PyTesseract & ChatGPT

- Repo for the second talk: [here](https://github.com/mbrochh/kindle-scrape-summarise)
- Step 1: Take **screenshot** the book while reading it 
- Step 2: Use **pyTesseract** to turn the screenshots into text files
- Step 3: Turn the text into **chunks** of no more than ~14000 words
- Step 4: Send each chunk to **OpenAI's API** and request a **summary**

<img src="/images/part2-cover.png" class="max-h-[200px] mx-auto mt-4" />

---

# Today's Project: Chatting With Logseq Notes Using SQLite3, FAISS & ChatGPT

- Step 1: **Find relevant notes** in Logseq
- Step 2: **Convert** each note into a vector
- Step 3: **Ask a question** and find all relevant notes via FAISS
- Step 4: **Send prompt** to ChatGPT to answer the question based on the found notes

<img src="/images/cover.png" class="max-h-[200px] mx-auto mt-4" />

---

# Intro: What is Logseq?

- [Logseq](https://logseq.com/) is a privacy first notes taking app
- Similar to Roam Research or Obsidian
- Fully open source
- All files are just local Markdown files on the local computer

<img src="/images/logseq.png" class="max-h-[300px] mx-auto mt-4" />

---

# Intro: What are Embeddings?

- Embeddings are a way to represent words as numbers
- Learn more from [OpenAI's blog post](https://openai.com/blog/introducing-text-and-code-embeddings/)
- See [OpenAI's Embeddings API](https://platform.openai.com/docs/guides/embeddings/what-are-embeddings)

<div class="grid grid-cols-2 gap-4 mt-4 max-h-[300px] overflow-hidden">
  <div><img src="/images/embeddings1.png" class="max-h-[300px] mx-auto" /></div>
  <div><img src="/images/embeddings2.png" class="max-h-[300px] mx-auto" /></div>
</div>

---

# Intro: What is FAISS?

- FAISS is a library for efficient similarity search and clustering of dense vectors
- Learn more from [Facebook's FAISS Github repo](https://github.com/facebookresearch/faiss)
- We will create a FAISS index that has all our embedding vectors
- We will turn our question into a vector as well
- We will then use FAISS to find the most similar vectors

---

# Intro: The ChatGPT Prompt

<img src="/images/prompt.png" class="max-h-[300px] mx-auto mt-4" />

---

# Intro: The Process

- Let's see if DALL-E has learned to draw proper diagrams since our last talk...

<img src="/images/dalle-prompt.png" class="max-h-[300px] mx-auto mt-4" />

---


<p class="text-center pt-[20%]">DRUMROLL....</p>

---

# WTF DALL-E?!

<img src="/images/dalle.png" class="max-h-[400px] mx-auto mt-4" />

---

# Intro: Retrieval Augmented Generation (RAG)

- What we are doing here is actually called RAG
- For now, it seems to be the best way to give LLMs some sort of up-to-date long-term memory
- There are hundreds of tutorials out there to accomplish this
- Here is one [with a nice diagram](https://towardsdatascience.com/retrieval-augmented-generation-rag-from-theory-to-langchain-implementation-4e9bd5f6a4f2)

---

# Let's Get Started!

- :-)

---

# Creating a project folder

- Whenever you start a new Python project, you will want to create a new folder for it

```bash
mkdir -p ~/Projects/second_brain/second_brain

# Let's `cd` into the newly created folder
cd ~/Projects/second_brain
```

- NOTE: the `mkdir` command stands for "make directory"
- NOTE: the `cd` command stands for "change directory"

---

# Creating the project files

```bash
cd ~/Projects/second_brain

# Now create a few files that we will need later:
touch .gitignore
touch requirements.txt
touch second_brain/__init__.py
touch second_brain/ask_gpt.py
touch second_brain/database.py
touch second_brain/embeddings.py
touch second_brain/faiss_utils.py
touch second_brain/find_notes.py
touch second_brain/gather_data.py
touch second_brain/local_settings.py
touch second_brain/logseq_parser.py
```

- NOTE: the `touch` command creates an empty file

---

# Double-checking

- When you run `tree . -a`, your file structure should look like this:

```bash
.
├── .gitignore
├── requirements.txt
└── second_brain
    ├── __init__.py
    ├── ask_gpt.py
    ├── database.py
    ├── embeddings.py
    ├── faiss_utils.py
    ├── find_notes.py
    ├── gather_data.py
    ├── local_settings.py
    └── logseq_parser.py

1 directory, 10 files
```

---

# The `.gitignore` file

- Put the following code into the `.gitignore` file:

```bash
local_settings.py
*.sqlite
__pycache__/
.DS_Store
```

- The `local_settings.py` file will contain secret information
- Therefore we don't want to commit it to Git or allow Github Copilot to see it
- The `*.sqlite` file will contain all our notes and vectors
- We also never want to expose that file to the outside world

---

# The `local_settings.py` file

- Put the following code into the `local_settings.py` file:

```python
OPENAI_API_KEY = "YOUR OPENAI API KEY HERE"
SQLITE_DB_PATH = '/Users/YOUR USERNAME HERE/Projects/second-brain/embeddings_db.sqlite'
LOGSEQ_FOLDER = '/Users/YOUR USERNAME HERE/Library/Mobile Documents/iCloud~com~logseq~logseq/Documents/pages'
```

- You can get your API key [here](https://platform.openai.com/account/api-keys)
- This is considered secret information, which is why we have this file in `.gitignore`
- Never share this key with anyone or they can use your OpenAI credits
- Consider storing this key in a password manager because you won't be able to see it again once it has been created

---

# The `requirements.txt` file

- We will need a few Python libraries throughout these slides
- To make things easier, we will just install them all at once
- Put the following code into the `requirements.txt` file:

```bash
openai==0.27.9
faiss-cpu==1.7.4
tiktoken==0.5.1
numpy==1.26.3
```

---

# The Virtual Environment

- Make sure that you have [pyenv](https://github.com/pyenv/pyenv) installed
- Make sure that you have the [pyenv-virtualenv plugin](https://github.com/pyenv/pyenv-virtualenv) installed
- Now you can `pip install` all the modules in the `requirements.txt` file

```bash
cd ~/Projects/second_brain
pyenv virtualenv second_brain
pyenv activate secon_brain
pip install -r requirements.txt
```

---

# The `find_notes.py` File

- We can use the `subprocess` module to run a shell command to find all files
- Try it out: `python -m second_brain.find_notes`


```python {1-22}{maxHeight:'350px'}
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
```

---

# The `logseq_parser.py` File

- Any line with indentation >= 1 is considered a note
- Try it out: `python -m second_brain.logseq_parser "<filename>"`

```python {1-29}{maxHeight:'350px'}
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
```

---

# The `embeddings.py` File

- We use [OpenAI's Embeddings API](https://platform.openai.com/docs/guides/embeddings) to turn a sentence into a vector
- Try it out: `python -m second_brain.embeddings`

```python {1-29}{maxHeight:'350px'}
import openai 

from .local_settings import OPENAI_API_KEY

openai.api_key = OPENAI_API_KEY

def get_embedding(
      content=None, 
      model="text-embedding-ada-002"
):
   embedding = openai.Embedding.create(
      input=[content], 
      model=model,
   )
   return embedding['data'][0]['embedding']

if __name__ == '__main__':
   content = 'This is a test.'
   embedding = get_embedding(content=content)
   print(embedding)
```

---

# The `database.py` File

- We use SQLite3 to store the embeddings
- Try it out: `python -m second_brain.database`

```python {1-94}{maxHeight:'350px'}
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
```

---

# The `faiss_utils.py` File

- We query all vectors from the SQLite3 DB and build a FAISS index from them
- Try it out: `python -m second_brain.faiss_utils`

```python {1-32}{maxHeight:'350px'}
import faiss
import numpy as np
import json

from .database import get_all_embeddings

# Build a Faiss index for similarity search
def build_faiss_index():
    records = get_all_embeddings()
    
    dim = len(json.loads(records[0][1]))

    # Create a matrix for the embeddings
    embedding_matrix = np.zeros((len(records), dim), dtype=np.float32)

    id_map = {}  # to map index in matrix to original id
    for i, record in enumerate(records):
        idx, embedding_str = record
        embedding = np.array(json.loads(embedding_str), dtype=np.float32)
        embedding_matrix[i] = embedding
        id_map[i] = idx

    # Build the faiss index
    index = faiss.IndexFlatL2(dim)
    index.add(embedding_matrix)
    
    return index, id_map

if __name__ == '__main__':
    faiss_index, id_map = build_faiss_index()
    print(f'FAISS index has {faiss_index.ntotal} items')
    print(f'ID map: {id_map}')

```

---

# The `gather_data.py` File

- We use the functions we have created so far to gather all the data
- Try it out: `python -m second_brain.gather_data`

```python {1-29}{maxHeight:'350px'}
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
```

---

# The `ask_gpt.py` File

- Finally, we can vectorize our question, find similar vectors, construct a prompt and get an answer from ChatGPT
- Try it out: `python -m second_brain.ask_gpt "What is the meaning of life?"`

```python {1-101}{maxHeight:'340px'}
import sys
import numpy as np
import openai
import tiktoken


from .database import get_rows_by_id
from .embeddings import get_embedding
from .faiss_utils import build_faiss_index

from .local_settings import OPENAI_API_KEY


openai.api_key = OPENAI_API_KEY

# see https://platform.openai.com/docs/models/gpt-4-and-gpt-4-turbo
MODEL = "gpt-4-1106-preview"
ENCODING = "cl100k_base"

MODEL_MAX_TOKENS = 30000
RESPONSE_TOKENS = 1000

PROMPT = """
"Answer the following question as truthfully as possible based on the provided context. 
Each part of the context starts with `SOURCE: ...` and ends with a blank line.
If the answer to the question cannot be found in any of the sources, answer with 'Insufficient context.'."
At the end of your answer, list the sources that you quoted in your answer.

QUESTION:
{question}

CONTEXT:
{context}
"""

def count_tokens(text):
    # see https://github.com/openai/tiktoken
    enc = tiktoken.encoding_for_model(MODEL)
    tokens = enc.encode(text)
    token_count = len(tokens)
    return token_count


def ask_gpt(query=None, send_to_openai=False):
    faiss_index, id_map = build_faiss_index()

    query_vector = get_embedding(content=query)
    query_vector = np.array(query_vector, dtype=np.float32)
    query_vector = query_vector.reshape(1, -1)

    # get up to 100 nearest neighbors
    _, I = faiss_index.search(query_vector, 100)
    # only consider neighbours that are less than 0.4 cosine distance
    similar_items = [id_map[idx] for idx, sim in zip(I[0], _[0]) if idx >= 0 and sim < 0.4]

    rows = get_rows_by_id(similar_items)

    prompt_context = ''
    sources = {}
    for row in rows:
        source = f'SOURCE: {row[1]}'

        if source not in sources:
            sources[source] = []

        sources[source].append(row[2])

    for source in sources:
        context = f'{source}\n'
        for note in sources[source]:
            context += f'{note}\n'
        context += '\n'
        tmp_context = prompt_context + context
        prompt_tokens = count_tokens(PROMPT.format(context=tmp_context, question=query))
        if prompt_tokens < MODEL_MAX_TOKENS - RESPONSE_TOKENS:
            prompt_context += context
        else:
            break

    question = query
    prompt = PROMPT.format(context=prompt_context, question=question)

    print('FULL PROMPT:')
    print(prompt)

    stream = openai.ChatCompletion.create(
        model=MODEL,
        stream=True,
        messages=[
            {"role": "user", "content": prompt},
        ]
    )

    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            print(chunk.choices[0].delta.content, end="")

if __name__ == '__main__':
    query = sys.argv[1]
    ask_gpt(query=query)
```

---

# BONUS: Mixtral

- Run `pip install ollama`
- Run `ollama run mixtral` - this will download the model
- Replace lines 86 - 96 in `ask_gpt.py` with the following code:
- Try it out: `python -m second_brain.ask_gpt "What is the meaning of life?"`

```python {1-101}{maxHeight:'340px'}
    import ollama
    stream = ollama.chat(
        model='mixtral',
        messages=[{'role': 'user', 'content': prompt}],
        stream=True,
    )

    for chunk in stream:
        print(chunk['message']['content'], end='', flush=True)
```

--- 

# Thank you for your attention!

- Join the Python User Group: [https://pugs.org.sg/membership](https://pugs.org.sg/membership)
- Follow me on Twitter: [https://twitter.com/@mbrochh](https://twitter.com/mbrochh)
- Find the slides at [https://bit.ly/pugs-logseq](https://bit.ly/pugs-logseq)

<img src="/images/telegram.jpeg" class="max-h-[250px] mx-auto mt-8" />

---
layout: end
---