# Ingestão e busca semântica de PDFs

Projeto de Retrieval-Augmented Generation (RAG) que ingere um PDF, armazena seus
embeddings em PostgreSQL com pgvector e permite fazer perguntas sobre o conteúdo
por meio de um chat no terminal.

## Tecnologias

- Python e LangChain
- PostgreSQL 17 com a extensão pgvector, executado via Docker Compose
- API compatível com OpenAI acessada via OpenRouter para embeddings e chat

## Pré-requisitos

- Python 3 e `venv`
- Docker com Docker Compose
- Uma chave de API do OpenRouter e os identificadores de modelos de embedding e chat

## Configuração

Na pasta do projeto, crie um ambiente virtual, ative-o e instale as dependências:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

No Windows, ative o ambiente com `venv\Scripts\activate`.

Copie `.env.example` para `.env` e preencha as configurações:

```env
OPENROUTER_API_KEY=sua-chave
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_CHAT_MODEL=identificador-do-modelo-de-chat
OPENROUTER_EMBEDDING_MODEL=identificador-do-modelo-de-embedding
DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/rag
PG_VECTOR_COLLECTION_NAME=documentos
PDF_PATH=document.pdf
```

Escolha modelos de embedding e de chat disponíveis no OpenRouter. O arquivo
`document.pdf` é o documento incluído no projeto; para usar outro PDF, atualize
`PDF_PATH` com o caminho para esse arquivo.

## Execução

Execute os comandos a partir da pasta do projeto, com o ambiente virtual ativado:

1. Inicie o PostgreSQL e aguarde o serviço ficar saudável:

   ```bash
   docker compose up -d
   ```

2. Ingira o PDF e gere os embeddings:

   ```bash
   python src/ingest.py
   ```

3. Inicie o chat no terminal:

   ```bash
   python src/chat.py
   ```

Digite uma pergunta no prompt `PERGUNTA:`. Para encerrar o chat, envie uma
pergunta vazia.

## Como funciona

1. `src/ingest.py` carrega o PDF, divide o conteúdo em trechos de até 1000
   caracteres com sobreposição de 150 e salva os embeddings no pgvector.
2. `src/chat.py` recebe perguntas e chama `src/search.py`.
3. `src/search.py` busca os 30 trechos mais similares e envia o contexto junto
   com a pergunta ao modelo de chat. O prompt orienta o modelo a responder
   somente com base no contexto e a informar quando não houver dados suficientes.

Se trocar o modelo de embeddings por outro com dimensão diferente, remova a
collection de vetores existente (ou o volume `postgres_data`) e faça a ingestão
novamente.

## Estrutura

```text
.
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── src/
│   ├── ingest.py
│   ├── search.py
│   └── chat.py
├── document.pdf
├── README.md
└── REQUISITOS.md
```

`REQUISITOS.md` preserva a especificação do desafio; este README descreve a
implementação e como executá-la.
