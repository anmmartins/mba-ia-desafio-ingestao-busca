import os
from dotenv import load_dotenv

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_postgres import PGVector

load_dotenv()

PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""

OPENROUTER_CHAT_MODEL = os.getenv("OPENROUTER_CHAT_MODEL")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL")
PG_VECTOR_COLLECTION_NAME = os.getenv("PG_VECTOR_COLLECTION_NAME")
DATABASE_URL = os.getenv("DATABASE_URL")

def validate_env_variables():
    for env_value in (OPENROUTER_CHAT_MODEL, OPENROUTER_API_KEY, OPENROUTER_BASE_URL, PG_VECTOR_COLLECTION_NAME, DATABASE_URL):
        if not env_value:
            raise RuntimeError(f"Environment variable {env_value} is not set")

def search_prompt(question: str):
    validate_env_variables()

    embeddings = OpenAIEmbeddings(
        model=os.getenv("OPENROUTER_EMBEDDING_MODEL"),
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url=os.getenv("OPENROUTER_BASE_URL"),
        check_embedding_ctx_length=False,        
        model_kwargs={"encoding_format": "float"},
        chunk_size=128,
    )
    store = PGVector(
        embeddings=embeddings,
        collection_name=os.getenv("PG_VECTOR_COLLECTION_NAME"),
        connection=os.getenv("DATABASE_URL"),
        use_jsonb=True,
    )
    documents = store.similarity_search_with_score(question, k=30)
    context = "\n\n".join([f"Documento: {doc.page_content}\nScore: {score}" for doc, score in documents])
    prompt = PromptTemplate(
        input_variables=["contexto", "pergunta"],
        template=PROMPT_TEMPLATE,
    )
    model = ChatOpenAI(
        model=os.getenv("OPENROUTER_CHAT_MODEL"),
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url=os.getenv("OPENROUTER_BASE_URL"),
        temperature=0.3,
    )
    # chain = prompt | model | StrOutputParser()
    # response = chain.invoke({
    #     "contexto": context,
    #     "pergunta": question,
    # })
    # return response
    formatted_prompt = prompt.format(
        contexto=context,
        pergunta=question,
    )
    print(f"Formatted Prompt:\n{formatted_prompt}\n")
    response = model.invoke(formatted_prompt)
    return response.content
    