"""
Testes automatizados para validação do prompt otimizado (v2).

Valide com:
    pytest tests/test_prompts.py
"""
import re
import sys
from pathlib import Path

import pytest
import yaml

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils import validate_prompt_structure

PROMPT_FILE = Path(__file__).parent.parent / "prompts" / "bug_to_user_story_v2.yml"
PROMPT_KEY = "bug_to_user_story_v2"

# Palavras que indicam uma persona/role explícita no prompt.
ROLE_MARKERS = [
    "você é",
    "product manager",
    "product owner",
    "engenheiro de produto",
    "persona",
]


def load_prompts(file_path: str = PROMPT_FILE):
    """Carrega prompts do arquivo YAML."""
    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def prompt_data() -> dict:
    """Retorna o dicionário do prompt otimizado, sem o wrapper do YAML."""
    data = load_prompts()
    assert isinstance(data, dict), "O YAML deve conter um dicionário na raiz"
    assert PROMPT_KEY in data, f"Chave '{PROMPT_KEY}' não encontrada no YAML"
    return data[PROMPT_KEY]


def _full_text(prompt_data: dict) -> str:
    """Concatena todos os textos relevantes do prompt (para buscas)."""
    parts = [
        str(prompt_data.get("description", "")),
        str(prompt_data.get("system_prompt", "")),
        str(prompt_data.get("user_prompt", "")),
    ]
    parts.append(" ".join(map(str, prompt_data.get("techniques_applied", []))))
    return "\n".join(parts)


class TestPrompts:
    def test_prompt_has_system_prompt(self, prompt_data):
        """Verifica se o campo 'system_prompt' existe e não está vazio."""
        assert "system_prompt" in prompt_data, "Campo 'system_prompt' ausente"
        system_prompt = prompt_data.get("system_prompt", "")
        assert isinstance(system_prompt, str), "'system_prompt' deve ser texto"
        assert system_prompt.strip(), "'system_prompt' está vazio"

    def test_prompt_has_role_definition(self, prompt_data):
        """Verifica se o prompt define uma persona (ex: "Você é um Product Manager")."""
        system_prompt = prompt_data.get("system_prompt", "").lower()
        assert any(
            marker in system_prompt for marker in ROLE_MARKERS
        ), "O prompt não define uma persona/role explícita"

    def test_prompt_mentions_format(self, prompt_data):
        """Verifica se o prompt exige formato Markdown ou User Story padrão."""
        text = _full_text(prompt_data).lower()
        has_user_story_format = "como um" in text and (
            "critérios de aceitação" in text or "eu quero" in text
        )
        has_markdown = "markdown" in text
        assert (
            has_user_story_format or has_markdown
        ), "O prompt não exige formato User Story nem Markdown"

    def test_prompt_has_few_shot_examples(self, prompt_data):
        """Verifica se o prompt contém exemplos de entrada/saída (técnica Few-shot)."""
        text = _full_text(prompt_data)
        text_lower = text.lower()

        assert "few-shot" in text_lower, "A técnica Few-shot não está declarada no prompt"

        example_count = len(re.findall(r"exemplo\s+\d", text_lower))
        has_examples = example_count >= 2
        has_io_pairs = "entrada" in text_lower and "saída" in text_lower

        assert has_examples, "São necessários pelo menos 2 exemplos (Few-shot)"
        assert has_io_pairs, "Os exemplos devem conter seções de Entrada e Saída"

    def test_prompt_no_todos(self, prompt_data):
        """Garante que você não esqueceu nenhum `[TODO]` no texto."""
        text = _full_text(prompt_data)
        assert "[TODO]" not in text, "O prompt ainda contém marcadores [TODO]"
        assert not re.search(r"\bTODO\b", text), "O prompt ainda contém a palavra TODO"

    def test_minimum_techniques(self, prompt_data):
        """Verifica (através dos metadados do yaml) se pelo menos 2 técnicas foram listadas."""
        techniques = prompt_data.get("techniques_applied", [])
        assert isinstance(techniques, list), "'techniques_applied' deve ser uma lista"
        assert len(techniques) >= 2, (
            f"Mínimo de 2 técnicas requeridas, encontradas: {len(techniques)}"
        )

        techniques_lower = " ".join(techniques).lower()
        assert "few-shot" in techniques_lower, "Few-shot Learning é obrigatório"

    def test_prompt_structure_is_valid(self, prompt_data):
        """Valida a estrutura usando o helper do projeto (bônus)."""
        is_valid, errors = validate_prompt_structure(prompt_data)
        assert is_valid, f"Estrutura inválida: {errors}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
