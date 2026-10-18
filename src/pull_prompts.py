"""
Script para fazer pull de prompts do LangSmith Prompt Hub.

Este script:
1. Conecta ao LangSmith usando credenciais do .env
2. Faz pull do prompt semente do desafio
3. Salva localmente em prompts/bug_to_user_story_v1.yml

O parâmetro `dangerously_pull_public_prompt=True` é obrigatório sempre que o
identificador tem dono explícito ("owner/nome"). O LangSmith bloqueia esse pull
por padrão porque um prompt do Hub é um objeto LangChain serializado, que pode
vir de terceiros. Aqui o prompt é o do desafio, então o risco é conhecido.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from langsmith import Client
from utils import save_yaml, check_env_vars, print_section_header

load_dotenv()

# Prompt semente (baixa qualidade) publicado no LangSmith Hub pelo desafio.
SEED_PROMPT = "leonanluppi/bug_to_user_story_v1"

# Onde o conteúdo baixado é salvo.
OUTPUT_PATH = "prompts/bug_to_user_story_v1.yml"
LOCAL_KEY = "bug_to_user_story_v1"


def _message_role(message) -> str:
    """Descobre o papel (system/user/assistant) de uma mensagem do template."""
    name = type(message).__name__.lower()
    if "system" in name:
        return "system"
    if "human" in name or "user" in name:
        return "user"
    if "ai" in name or "assistant" in name:
        return "assistant"
    return "user"


def _message_content(message) -> str:
    """Extrai o texto de uma mensagem do template, seja ela qual for."""
    inner = getattr(message, "prompt", None)
    if inner is not None and hasattr(inner, "template"):
        return inner.template
    if hasattr(message, "content"):
        return message.content
    return str(message)


def pull_prompts_from_langsmith():
    """
    Baixa o prompt semente do LangSmith Hub e salva em prompts/bug_to_user_story_v1.yml.

    Returns:
        True se sucesso, False caso contrário.
    """
    print(f"🔎 Baixando prompt semente do Hub: {SEED_PROMPT}")

    try:
        client = Client()
        prompt = client.pull_prompt(
            SEED_PROMPT,
            dangerously_pull_public_prompt=True,
        )
    except Exception as e:
        print(f"\n❌ Erro ao baixar o prompt '{SEED_PROMPT}': {e}\n")
        print("Verifique:")
        print("- LANGSMITH_API_KEY está configurada corretamente no .env")
        print("- Sua conexão com a internet está funcionando")
        print("- O prompt 'leonanluppi/bug_to_user_story_v1' ainda existe no Hub")
        return False

    system_parts = []
    user_parts = []

    for message in prompt.messages:
        role = _message_role(message)
        content = _message_content(message).rstrip()
        if role == "system":
            system_parts.append(content)
        elif role == "user":
            user_parts.append(content)

    prompt_data = {
        LOCAL_KEY: {
            "description": (
                "Prompt para converter relatos de bugs em User Stories "
                "(prompt semente do desafio, baixa qualidade)."
            ),
            "system_prompt": "\n\n".join(system_parts).strip() + "\n",
            "user_prompt": "\n\n".join(user_parts).strip(),
            "version": "v1",
            "tags": ["bug-analysis", "user-story", "product-management"],
            "source": SEED_PROMPT,
        }
    }

    if not save_yaml(prompt_data, OUTPUT_PATH):
        print(f"\n❌ Falha ao salvar o prompt em '{OUTPUT_PATH}'")
        return False

    print(f"✅ Prompt salvo com sucesso em '{OUTPUT_PATH}'")
    return True


def main():
    """Função principal"""
    print_section_header("PULL DE PROMPTS DO LANGSMITH HUB")

    if not check_env_vars(["LANGSMITH_API_KEY"]):
        return 1

    success = pull_prompts_from_langsmith()

    if success:
        print("\nPróximos passos:")
        print(f"1. Analise o prompt baixado em {OUTPUT_PATH}")
        print("2. Crie prompts/bug_to_user_story_v2.yml com a versão otimizada")
        print("3. Faça o push: python src/push_prompts.py")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
