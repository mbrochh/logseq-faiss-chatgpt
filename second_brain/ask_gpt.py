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
    # only consider neighbours that are less than 1.5 cosine distance
    similar_items = [id_map[idx] for idx, sim in zip(I[0], _[0]) if idx >= 0 and sim < 1]

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
        try:
            if chunk.choices[0].delta.content is not None:
                print(chunk.choices[0].delta.content, end="")
        except:
            pass

if __name__ == '__main__':
    query = sys.argv[1]
    ask_gpt(query=query)