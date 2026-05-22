# Dicionário de Dados — SMS Spam Collection

**Fonte:** UCI Machine Learning Repository  
**URL:** https://archive.ics.uci.edu/dataset/228/sms+spam+collection  
**Publicação:** Almeida et al., 2011

## Dataset bruto (`data/raw/SMSSpamCollection`)

| Coluna | Tipo | Valores | Descrição |
|--------|------|---------|-----------|
| `label` | string | `ham`, `spam` | Classe da mensagem |
| `message` | string | texto livre | Conteúdo da mensagem SMS |

- **Total de amostras:** 5.572
- **Após remoção de duplicatas:** 5.169
- **Distribuição:** ham 86,59% / spam 13,41% — dataset desbalanceado
- **Idioma:** inglês (britânico)
- **Período:** mensagens coletadas até ~2012

## Features derivadas (EDA)

| Feature | Tipo | Descrição |
|---------|------|-----------|
| `msg_len` | int | Número de caracteres da mensagem original |
| `has_number` | int (0/1) | Presença de sequência numérica |
| `has_link` | int (0/1) | Presença de URL ou domínio |
| `has_currency` | int (0/1) | Presença de símbolo monetário (£, $, €) |
| `clean` | string | Texto após pré-processamento NLP |
| `label_enc` | int (0/1) | Label encodado: spam=1, ham=0 |

## Representação vetorial (`data/processed/`)

| Artefato | Descrição |
|----------|-----------|
| TF-IDF matrix | 4.961 features (unigramas + bigramas, min_df=2, max=5.000) |
| Divisão | 80% treino (4.135) / 20% teste (1.034), estratificada, seed=42 |

## Limitações conhecidas

- Dataset exclusivamente em inglês; modelos não generalizam para português sem retreinamento.
- Mensagens coletadas até ~2012 (padrões de spam atuais podem diferir).
- Desbalanceamento (13,4% spam) exige atenção à métrica de avaliação.
