"""
Script para fazer push de prompts otimizados ao LangSmith Prompt Hub.

Este script:
1. Lê os prompts otimizados de prompts/bug_to_user_story_v2.yml
2. Valida os prompts
3. Faz push PÚBLICO para o LangSmith Hub
4. Adiciona metadados (tags, descrição, técnicas utilizadas)

`username` vem de USERNAME_LANGSMITH_HUB no .env e precisa ser o seu handle do
Hub. A variável do template precisa ser {bug_report}, que é a chave de entrada
usada no dataset de avaliação.
"""

import os
import sys
from dotenv import load_dotenv
from langsmith import Client
from langchain_core.prompts import ChatPromptTemplate
from utils import load_yaml, check_env_vars, print_section_header

load_dotenv()

# Arquivo com o prompt otimizado e a chave que o identifica dentro do YAML.
INPUT_PATH = "prompts/bug_to_user_story_v2.yml"
LOCAL_KEY = "bug_to_user_story_v2"
PROMPT_NAME = "bug_to_user_story_v2"


def push_prompt_to_langsmith(prompt_name: str, prompt_data: dict) -> bool:
    """
    Faz push do prompt otimizado para o LangSmith Hub (PÚBLICO).

    Args:
        prompt_name: Nome do prompt (sem o handle do usuário)
        prompt_data: Dados do prompt

    Returns:
        True se sucesso, False caso contrário
    """
    username = os.getenv("USERNAME_LANGSMITH_HUB", "").strip()
    if not username:
        print("❌ USERNAME_LANGSMITH_HUB não definido no .env.")
        print("   Crie seu handle público no LangSmith Hub (veja o README).")
        return False

    full_name = f"{username}/{prompt_name}"

    system_prompt = prompt_data.get("system_prompt", "")
    user_prompt = prompt_data.get("user_prompt", "{bug_report}")

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("user", user_prompt),
        ]
    )

    # Garante que a variável esperada pelo dataset de avaliação existe.
    if "bug_report" not in prompt.input_variables:
        print(
            "❌ O prompt não expõe a variável {bug_report}. "
            f"Variáveis encontradas: {prompt.input_variables}"
        )
        return False

    tags = list(prompt_data.get("tags", []))
    for technique in prompt_data.get("techniques_applied", []):
        tag = technique.lower().split("(")[0].strip().replace(" ", "-")
        if tag and tag not in tags:
            tags.append(tag)

    description = prompt_data.get("description", "Prompt otimizado")

    print(f"⬆️  Publicando prompt público: {full_name}")

    try:
        client = Client()
        url = client.push_prompt(
            full_name,
            object=prompt,
            is_public=True,
            description=description,
            tags=tags,
        )
    except Exception as e:
        print(f"\n❌ Erro ao publicar o prompt '{full_name}': {e}\n")
        print("Verifique:")
        print("- LANGSMITH_API_KEY está configurada corretamente no .env")
        print("- USERNAME_LANGSMITH_HUB é o seu handle já criado no Hub")
        print("- Você tem permissão para publicar prompts no workspace")
        return False

    print(f"✅ Prompt publicado com sucesso!")
    print(f"🔗 {url}")
    return True


def validate_prompt(prompt_data: dict) -> tuple[bool, list]:
    """
    Valida estrutura básica de um prompt (versão simplificada).

    Args:
        prompt_data: Dados do prompt

    Returns:
        (is_valid, errors) - Tupla com status e lista de erros
    """
    errors = []

    if not isinstance(prompt_data, dict):
        return False, ["prompt_data não é um dicionário"]

    required_fields = ["description", "system_prompt", "user_prompt", "version"]
    for field in required_fields:
        if not str(prompt_data.get(field, "")).strip():
            errors.append(f"Campo obrigatório ausente ou vazio: {field}")

    full_text = "\n".join(
        str(prompt_data.get(field, ""))
        for field in ("description", "system_prompt", "user_prompt")
    )

    if "[TODO]" in full_text or "TODO" in full_text:
        errors.append("O prompt ainda contém marcadores TODO")

    if "{bug_report}" not in full_text:
        errors.append("O prompt precisa usar a variável {bug_report}")

    techniques = prompt_data.get("techniques_applied", [])
    if not isinstance(techniques, list) or len(techniques) < 2:
        errors.append(
            "Mínimo de 2 técnicas requeridas em 'techniques_applied', "
            f"encontradas: {len(techniques) if isinstance(techniques, list) else 0}"
        )

    return (len(errors) == 0, errors)


def main():
    """Função principal"""
    print_section_header("PUSH DE PROMPTS OTIMIZADOS PARA O LANGSMITH")

    if not check_env_vars(["LANGSMITH_API_KEY", "USERNAME_LANGSMITH_HUB"]):
        return 1

    data = load_yaml(INPUT_PATH)
    if data is None:
        print(f"❌ Não foi possível carregar '{INPUT_PATH}'")
        return 1

    prompt_data = data.get(LOCAL_KEY)
    if prompt_data is None:
        print(f"❌ Chave '{LOCAL_KEY}' não encontrada em '{INPUT_PATH}'")
        return 1

    is_valid, errors = validate_prompt(prompt_data)
    if not is_valid:
        print("❌ Prompt inválido:")
        for error in errors:
            print(f"   - {error}")
        return 1

    print("✅ Prompt validado com sucesso")
    print(f"   Técnicas: {', '.join(prompt_data.get('techniques_applied', []))}\n")

    success = push_prompt_to_langsmith(PROMPT_NAME, prompt_data)

    if success:
        username = os.getenv("USERNAME_LANGSMITH_HUB", "").strip()
        print("\nPróximos passos:")
        print("1. Confira o prompt publicado no dashboard do LangSmith")
        print("2. Rode a avaliação: python src/evaluate.py")
        print(f"\n   python src/evaluate.py  # avalia {username}/{PROMPT_NAME}")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
