from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

def openaiIA(modelo, StatusOficial, Noticias):
    client = OpenAI(api_key=os.getenv("OPENAI_API"))

    try:
        response = client.chat.completions.create(
            model=modelo,
            messages=[
                {
                    "role": "system",
                    "content": """
                    Você é especialista em analise de dados, com foco em seguranca de usuarios.
                    Seu papel é ler e interpretar dados recentes sobre a seguranca dos usuarios, incluindo:
                    anonimizar os dados, criando dados sinteticos com o tema da dipol e vamos clusterizar e anonimizar para quando fizer uma busca só aponte que sei la tem 110 pessoas com o nome de Juan no banco

                    Com base nas informações recebidas, gere um resumo informativo e natural, descrevendo a situação atual dos dados.
                    voce vai aplicar técnicas de Machine Learning, especialmente clusterização, para anonimizar dados pessoais. Em vez de exibir 
                    informações individuais, o sistema agrupa registros semelhantes e retorna apenas resultados agregados — por exemplo, “existem 110 pessoas com o nome Juan”.
                    
                    Regras:
                    1. Seja neutro e objetivo — evite opiniões pessoais.

        
                    """
                },
                {
                    "role": "user",
                    "content": f"""
                    metro de São Paulo oficial: {StatusOficial}

                    Notícias Recentes: {Noticias}
                    """
                }
            ],
            temperature=0.4,
        )

        message = response.choices[0].message.content.strip()

        print("\n===== RESUMO =====\n")
        print(message)
        print("\n==========================\n")

        return message

    except Exception as e:
        print("Erro ao gerar conteúdo com a OpenAI API:", e)
        return None


