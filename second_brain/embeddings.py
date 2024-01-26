import openai 

from .local_settings import OPENAI_API_KEY

openai.api_key = OPENAI_API_KEY

def get_embedding(
      content=None, 
      model="text-embedding-3-large"
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