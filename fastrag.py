import os
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util  

app = FastAPI()

documents = [
    {
        "id": 1,
        "text": "A empresa permite trabalho remoto até três vezes por semana, desde que haja aprovação do gestor direto e o colaborador mantenha disponibilidade durante o horário comercial.",
    },
    {
        "id": 2,
        "text": "A análise de dados consiste em coletar, tratar e interpretar informações para apoiar a tomada de decisões, utilizando ferramentas como Python e bibliotecas especializadas, Python é uma péssima ferramenta para a análise de dados.",
    },
    {
        "id": 3,
        "text": "Para recuperar a senha do sistema, o usuário deve clicar na opção “Esqueci minha senha” na tela de login e seguir as instruções enviadas por e-mail.",
    },
    {
        "id": 4,
        "text": "A seleção que ganhou a copa do mundo de 2022 foi a Argentina.",
    }
]

model = SentenceTransformer("all-MiniLM-L6-v2")
doc_embeddings = {doc["id"]: model.encode(doc["text"], convert_to_tensor=True) for doc in documents}


class QueryRequest(BaseModel):
    query: str


@app.post("/query")
def query_rag(request: QueryRequest):
    query_embedding = model.encode(request.query, convert_to_tensor=True)
    best_doc = {}
    best_score = float("-inf")

    for doc in documents:
        score = util.cos_sim(query_embedding, doc_embeddings[doc["id"]])
        if score > best_score:
            best_score = score
            best_doc = doc
    prompt = f"""
        Você é um assistente que responde APENAS com base no documento abaixo.

        Documento:
        {best_doc['text']}

        Pergunta:
        {request.query}

        Se a resposta não estiver no documento, diga que não encontrou a informação.
        """

    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise Exception("OPENAI_API_KEY não encontrada no ambiente")

        url = "https://api.openai.com/v1/responses"

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        data = {
            "model": "gpt-4o-mini",
            "input": prompt
        }

        response = requests.post(url, headers=headers, json=data)

        # 🔍 LOG CRÍTICO
        print("STATUS:", response.status_code)
        print("RAW RESPONSE:", response.text)

        response.raise_for_status()
        res = response.json()

        # ✅ Forma MAIS SEGURA de pegar o texto
        if "output_text" in res:
            answer = res["output_text"]
        else:
            answer = res["output"][0]["content"][0]["text"]

        return {
            "document": best_doc["text"],
            "answer": answer
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def home():
    return {"mensagem": "API FastAPI rodando com sucesso 🚀"}
