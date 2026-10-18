# Pull, Otimização e Avaliação de Prompts com LangChain e LangSmith

Desafio de Prompt Engineering: fazer **pull** de um prompt de baixa qualidade do
LangSmith Prompt Hub, **refatorá-lo** com técnicas avançadas, fazer **push** de
volta e **avaliar** a qualidade com 5 métricas customizadas, atingindo **≥ 0.8 em
todas elas**.

| Métrica | Resultado | Status |
| --- | --- | --- |
| Helpfulness (derivada) | **0.90** | ✅ |
| Correctness (derivada) | **0.90** | ✅ |
| F1-Score (base) | **0.91** | ✅ |
| Clarity (base) | **0.90** | ✅ |
| Precision (base) | **0.89** | ✅ |
| **Média geral** | **0.8996** | ✅ |

> Helpfulness = média(Clarity, Precision) · Correctness = média(F1-Score, Precision)

---

## Tecnologias

- **Python 3.10+** (executado com 3.12)
- **LangChain** (`langchain-core`, `langchain-openai`, `langchain-google-genai`)
- **LangSmith** — Prompt Hub, tracing, datasets e experimentos
- **Modelos (Gemini):** geração com `gemini-3.5-flash`, avaliação com `gemini-3.5-flash-lite`

---

## Como Executar

### Pré-requisitos

- Python 3.10+
- Conta no [LangSmith](https://smith.langchain.com) com um **handle público** criado
  (`USERNAME_LANGSMITH_HUB`)
- Uma API key de LLM: [Google Gemini](https://aistudio.google.com/app/apikey)
  (recomendado, tem tier grátis) **ou** [OpenAI](https://platform.openai.com/api-keys)

### 1. Ambiente virtual

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Variáveis de ambiente

Copie o template e preencha:

```bash
cp .env.example .env
```

```dotenv
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=lsv2_pt_...
LANGSMITH_PROJECT=mba-ia-pull-evaluation-prompt
USERNAME_LANGSMITH_HUB=seu_handle

LLM_PROVIDER=google
GOOGLE_API_KEY=...
LLM_MODEL=gemini-3.5-flash
EVAL_MODEL=gemini-3.5-flash-lite
```

> Para criar o **handle público**: LangSmith → **Prompts** → abra um prompt →
> **⋯** (ao lado de *Playground*) → **Make Public** → defina o handle. Ele é
> definitivo.

### 3. Fluxo completo

```bash
# 1) Pull do prompt semente (baixa qualidade)
python src/pull_prompts.py        # salva prompts/bug_to_user_story_v1.yml

# 2) (já incluído) refatoração em prompts/bug_to_user_story_v2.yml

# 3) Push do prompt otimizado (público)
python src/push_prompts.py        # publica {handle}/bug_to_user_story_v2

# 4) Avaliação + experimento no LangSmith
python src/evaluate.py

# 5) Testes de validação do prompt
pytest tests/test_prompts.py
```

### Ordem dos artefatos gerados

Cada `python src/evaluate.py` cria um **experimento** no LangSmith, no dataset
`{LANGSMITH_PROJECT}-eval`, gravando as 5 notas como *feedback* por exemplo.

---

## Técnicas Aplicadas (Fase 2)

O prompt `v2` parte de um `v1` propositalmente ruim (instruções vagas, sem persona,
sem formato, sem exemplos, `{bug_report}` duplicado). Apliquei as técnicas abaixo.

### 1. Role Prompting (obrigatória de contexto)

**O que é:** definir uma persona e um contexto detalhado para orientar tom, nível de
detalhe e vocabulário.

**Por que:** bugs chegam com jargão técnico e contexto de negócio misturados. Uma
persona sênior de Produto dá ao modelo um "padrão de qualidade" claro para decidir o
que vai para a User Story e o que vai para o contexto técnico.

**Como apliquei** (`system_prompt`):

```text
Você é Ana, uma Product Manager sênior com 10 anos de experiência em
metodologias ágeis (Scrum e Kanban) e em times de engenharia de software.
Sua especialidade é transformar relatos de bugs ... em User Stories claras,
acionáveis e prontas para entrar no backlog de desenvolvimento.
```

### 2. Few-shot Learning (obrigatória)

**O que é:** fornecer exemplos de entrada/saída que ancoram formato e nível de
detalhe esperados.

**Por que:** o maior ganho de qualidade. Os exemplos fixam o esqueleto da resposta e
ensinam, por demonstração, o comportamento esperado em casos simples, técnicos e de
"sugestão que não é bug".

**Como apliquei:** 3 exemplos completos no `system_prompt` (entrada + saída):
`Exemplo 1` (bug simples de UI), `Exemplo 2` (bug com métricas/contexto técnico) e
`Exemplo 3` (relato que não é bug — edge case).

### 3. Chain of Thought (CoT)

**O que é:** instruir o modelo a raciocinar passo a passo antes de responder.

**Por que:** análise de bug exige decompor o relato em persona, ação, valor,
detalhes técnicos, severidade e nº de problemas. O CoT melhora
`Correctness`/`F1-Score` ao evitar que detalhes importantes sejam omitidos.

**Como apliquei:** uma seção **"COMO PENSAR (raciocínio passo a passo)"** com 6 itens,
seguida de "Não mostre esse raciocínio na resposta final" (mantém a saída limpa).

### 4. Skeleton of Thought

**O que é:** fixar um esqueleto/estrutura para a resposta, preenchido pelo modelo.

**Por que:** garante saída consistente e testável, e faz a resposta aderir ao
formato de referência do dataset (que é exatamente
`Como um ... eu quero ... para que ...` + `Critérios de Aceitação`).

**Como apliquei:** seção **"FORMATO OBRIGATÓRIO DA SAÍDA (Markdown)"** com o template
e a regra de só incluir `Contexto Técnico:` quando o relato trouxer detalhes técnicos.

### 5. Edge Case Handling

**O que é:** regras explícitas para entradas fora do caso "ideal".

**Por que:** o dataset tem bugs simples, médios e complexos, com múltiplos problemas,
logs e métricas. Sem tratamento, o modelo omite impacto ou inventa dados.

**Como apliquei:**
- relato vazio/genérico → devolve o esqueleto padrão sem inventar fatos;
- pedido de melhoria (não é defeito) → tratado como funcionalidade;
- múltiplos bugs → critérios agrupados por tema (`A. Segurança`, `B. Integração`, …)
  + seção `Contexto do Bug`.

### Regras explícitas de comportamento

Além das técnicas, o `v2` define 10 regras numeradas (idioma/tom, persona específica,
3–7 critérios, critérios testáveis, preservar dados técnicos, **não inventar**,
quando incluir `Contexto Técnico`, agrupamento em bugs complexos, não adicionar
`Tasks`/estimativas, responder só com o artefato).

---

## Comparação: prompt v1 → prompt v2

| Aspecto | `v1` (baixa qualidade) | `v2` (otimizado) |
| --- | --- | --- |
| Persona | ausente | Product Manager sênior (Role Prompting) |
| Formato | livre ("crie uma user story") | template obrigatório Markdown |
| Exemplos | nenhum | 3 exemplos Few-shot |
| Raciocínio | nenhum | CoT em 6 passos |
| Critérios de aceitação | não citados | `Dado/Quando/Então/E`, 3–7 itens |
| Contexto técnico | ignorado | preservado quando aplicável |
| Edge cases | não tratados | 3 regras explícitas |
| Duplicação | `{bug_report}` no system **e** no user | `{bug_report}` só no user |
| Alinhamento ao gabarito | baixo | alto (formato idêntico ao dataset) |

O `v1` duplicava a variável `{bug_report}` (system + user) e não dava nenhuma
instrução de formato — daí F1/Clarity/Precision baixos. O `v2` alinha a saída ao
gabarito do dataset e adiciona as técnicas acima.

---

## Resultados Finais

### Métricas (experimento final)

```text
==================================================
Prompt: dimmy/bug_to_user_story_v2
==================================================

Métricas Derivadas:
  - Helpfulness: 0.90 ✓
  - Correctness: 0.90 ✓

Métricas Base:
  - F1-Score: 0.91 ✓
  - Clarity:  0.90 ✓
  - Precision:0.89 ✓

📊 MÉDIA GERAL: 0.8996
✅ STATUS: APROVADO - Todas as métricas >= 0.8
```

### Links no LangSmith

- **Dataset público (qualquer pessoa abre)** — 15 exemplos + experimentos:
  https://smith.langchain.com/public/a1af3f16-711d-4ea7-8caf-52a3a822206d/d/598163c5-b39e-4c30-bdd9-23e1661752af
- **Prompt público (Hub):** https://smith.langchain.com/hub/dimmy/bug_to_user_story_v2
- **Experimento (workspace):**
  https://smith.langchain.com/o/9e07403e-a984-4c17-8b28-db9f0fe733d5/datasets/598163c5-b39e-4c30-bdd9-23e1661752af/compare?selectedSessions=2e94dfb0-5f28-4343-b49e-1ad22217268b

### Screenshots

> Adicione aqui as capturas de tela (pasta `screenshots/`):
> 1. Dataset de avaliação com **15 exemplos**;
> 2. Experimento do `v2` com as **5 notas ≥ 0.8**;
> 3. **Tracing detalhado** de pelo menos 3 exemplos (aba *Runs/Traces* do experimento).

---

## Evidências no LangSmith

- **Dataset:** `mba-ia-pull-evaluation-prompt-eval` — 15 exemplos
  (5 simples, 7 médios, 3 complexos).
- **Experimento:** `dimmy-bug_to_user_story_v2-*` — 15 execuções com feedback das 5
  métricas por exemplo.
- **Tracing:** cada chamada do experimento gera um *run* com input `bug_report`,
  output da User Story e as notas; abra o experimento e clique em qualquer linha
  para ver o trace completo.

Para regerar um link público (o link muda ao compartilhar de novo):

```python
from langsmith import Client
print(Client().share_dataset(dataset_name="mba-ia-pull-evaluation-prompt-eval")["url"])
```

---

## Testes de Validação

```bash
pytest tests/test_prompts.py
```

Cobertura (`tests/test_prompts.py`):

| Teste | Verifica |
| --- | --- |
| `test_prompt_has_system_prompt` | campo `system_prompt` existe e não está vazio |
| `test_prompt_has_role_definition` | persona definida (ex.: "Você é … Product Manager") |
| `test_prompt_mentions_format` | exige formato User Story / Markdown |
| `test_prompt_has_few_shot_examples` | ≥ 2 exemplos de entrada/saída (Few-shot) |
| `test_prompt_no_todos` | nenhum marcador `[TODO]` |
| `test_minimum_techniques` | ≥ 2 técnicas em `techniques_applied` |
| `test_prompt_structure_is_valid` | estrutura validada por `utils.validate_prompt_structure` |

Resultado: **7 passed**.

---

## Estrutura do projeto

```
mba-ia-pull-evaluation-prompt/
├── .env.example
├── requirements.txt
├── README.md
├── prompts/
│   ├── bug_to_user_story_v1.yml   # pull do Hub (baixa qualidade)
│   └── bug_to_user_story_v2.yml   # prompt otimizado
├── datasets/
│   └── bug_to_user_story.jsonl    # 15 exemplos
├── src/
│   ├── pull_prompts.py            # implementado
│   ├── push_prompts.py            # implementado
│   ├── evaluate.py                # pronto
│   ├── metrics.py                 # pronto (5 métricas)
│   └── utils.py                   # pronto
├── tests/
│   └── test_prompts.py            # 7 testes implementados
└── tools/
    └── gemini_proxy.py            # OPCIONAL (ver nota abaixo)
```

---

## Nota sobre cota do Gemini (contorno opcional)

O tier **gratuito** do Gemini limita ~**5 req/min** e ~**20 req/dia** por
projeto/modelo. Uma avaliação completa consome ~**60 chamadas** (15 gerações +
45 juízes), o que estoura esse limite.

Com uma key/plano com cota adequada, basta usar `LLM_PROVIDER=google` (caminho
normal). No ambiente em que este projeto foi executado, a cota gratuita não era
suficiente, então usei `tools/gemini_proxy.py`: um proxy local compatível com a API
`/v1/chat/completions` da OpenAI que encaminha para o Gemini e **rotaciona várias
keys** (de projetos distintos), respeitando o limite por minuto.

```bash
export GEMINI_PROXY_KEYS="AIza...,AIza...,AIza..."   # ou tools/keys.json
python tools/gemini_proxy.py                          # 127.0.0.1:8765
# .env: LLM_PROVIDER=openai, OPENAI_API_KEY=dummy,
#       OPENAI_API_BASE=http://127.0.0.1:8765/v1
python src/evaluate.py
```

Esse proxy é apenas um utilitário de conveniência e **não** altera
`src/evaluate.py`, `src/metrics.py` nem `src/utils.py`.
