# Detecção de Spam em SMS com Aprendizado de Máquina

**Projeto AV2 — Disciplina de Machine Learning**

## Problema e Objetivo

Classificar automaticamente mensagens SMS como **spam** ou **ham** (legítima) usando aprendizado supervisionado. A variável-alvo é binária: `spam=1`, `ham=0`.

**Métrica primária:** F1-score da classe spam — adequada ao desbalanceamento (13,4% spam).

## Dataset

**SMS Spam Collection** (UCI Machine Learning Repository)  
- 5.572 mensagens (5.169 após remoção de duplicatas)
- Distribuição: 86,6% ham / 13,4% spam
- Idioma: inglês

**Download manual** (não versionado no repositório):
```bash
# O notebook baixa o dataset automaticamente na primeira execução.
# Para baixar manualmente:
wget "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
unzip "sms+spam+collection.zip" -d data/raw/
```

## Resultados

| Modelo | Acurácia | F1 (spam) | AUC-ROC | CV F1 médio |
|--------|----------|-----------|---------|-------------|
| Naive Bayes | 0,9662 | 0,8472 | 0,9809 | 0,8554 ± 0,0299 |
| Logistic Regression | 0,9613 | 0,8276 | 0,9909 | 0,7275 ± 0,0479 |
| **Linear SVM** | **0,9778** | **0,9084** | **0,9932** | **0,9065 ± 0,0250** |

Teste McNemar (LR vs SVM): **p=0,0014** — SVM Linear é estatisticamente superior.

## Instalação e Execução

### Opção 1 — Docker (recomendado — ambiente reproduzível)

```bash
# Construir e iniciar o container
docker compose up --build

# Acessar o Jupyter no navegador:
# http://localhost:8888
```

Abra `notebooks/machine_learning.ipynb` e execute todas as células em ordem (`Kernel → Restart & Run All`).  
O dataset já está incluído no container. Não é necessário instalar nada além do Docker.

### Opção 2 — Google Colab

1. Abra o notebook `notebooks/machine_learning.ipynb` no Google Colab.
2. Execute todas as células em ordem (Ctrl+F9 ou *Runtime → Run all*).

### Opção 3 — Ambiente local (Python 3.10+)

```bash
pip install -r requirements.txt
jupyter notebook --notebook-dir=. notebooks/machine_learning.ipynb
```

### Reprodutibilidade

Todos os componentes estocásticos usam `random_state=42`. Versão de Python: **3.10**.  
Resultados esperados estão registrados em `experiments/experiments.csv`.

## Estrutura do Repositório

```
projeto-ml-av2/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/          # Dataset bruto (não versionado — ver download acima)
│   └── processed/    # Dados transformados gerados pelo pipeline
├── notebooks/
│   └── machine_learning.ipynb   # Notebook principal (executar no Colab)
├── src/
│   ├── data/          # Carregamento e download do dataset
│   ├── features/      # Pré-processamento e vetorização
│   ├── models/        # Treinamento dos modelos
│   ├── evaluation/    # Métricas e testes estatísticos
│   └── visualization/ # Funções de plotagem
├── experiments/
│   └── experiments.csv   # Rastreio de experimentos com parâmetros e métricas
├── article/
│   ├── artigo.md         # Artigo técnico-científico
│   ├── referencias.bib   # Referências BibTeX
│   ├── figures/          # Figuras geradas pelo pipeline
│   └── tables/           # Tabelas de resultados
└── docs/
    ├── decisoes-tecnicas.md    # Justificativas metodológicas
    └── dicionario-de-dados.md  # Descrição das features e dataset
```

## Limitações

- Dataset em inglês (2012) — não generaliza para português ou spam moderno sem retreinamento.
- Hiperparâmetros padrão sem busca sistemática (grid search).
- Conjunto de teste pequeno para a classe spam (131 amostras).

Consulte `docs/decisoes-tecnicas.md` para justificativas completas e `article/artigo.md` para a análise acadêmica detalhada.
