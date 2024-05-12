# Licenciado-sob-a-Licenca-Apache-Versao-2.0-Gemini-API-Incorporacoes-de-inicio-rapido
!pip install -q -U google-generativeai  # Instala silenciosamente a biblioteca google-generativeai

# Importa a biblioteca google-generativeai
import google.generativeai as genai

# Importa o método get de userdata do Google Colab (substitua 'PASTE YOUR KEY HERE' pela sua chave de API)
from google.colab import userdata
GOOGLE_API_KEY=userdata.get('PASTE YOUR KEY HERE')

# Configura a API Key do Google Generative AI
genai.configure(api_key=GOOGLE_API_KEY)

# Define o texto de exemplo
text = "Hello world"

# Gera o embedding para o texto usando o modelo "models/text-embedding-004"
result = genai.embed_content(model="models/text-embedding-004", content=text)

# Imprime apenas uma parte do embedding para manter a saída controlável
print(str(result['embedding'])[:50], '... (resultado truncado)')
print(len(result['embedding'])) # O embedding tem 768 dimensões

# Gera embeddings para múltiplos textos
result = genai.embed_content(
    model="models/text-embedding-004",
    content=[
      'Qual é o sentido da vida?',
      'Quanto mato um marmota cortaria se um marmota pudesse cortar mato?',
      'Como o cérebro funciona?'])

# Imprime parte de cada embedding individual
for embedding in result['embedding']:
  print(str(embedding)[:50], '... (resultado truncado)')

# Gera embedding completo para "Hello world"
result1 = genai.embed_content(
    model="models/text-embedding-004",
    content="Hello world")


# Gera embedding truncado para "Hello world" com dimensionalidade de saída definida como 10
result2 = genai.embed_content(
    model="models/text-embedding-004",
    content="Hello world",
    output_dimensionality=10)


# Exibe o tamanho dos embeddings (deverão ser diferentes)
(len(result1['embedding']), len(result2['embedding']))

# Gera embeddings para "Hello world" com diferentes tipos de tarefas
result1 = genai.embed_content(
    model="models/text-embedding-004",
    content="Hello world")

result2 = genai.embed_content(
    model="models/text-embedding-004",
    content="Hello world",
    task_type="document")

print(str(result1['embedding'])[:50], '... (resultado truncado)')
print(str(result2['embedding'])[:50], '... (resultado truncado)')
