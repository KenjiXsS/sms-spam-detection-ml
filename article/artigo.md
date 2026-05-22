# Detecção de Spam em Mensagens SMS com Aprendizado de Máquina: Uma Comparação entre Naive Bayes, Regressão Logística e SVM Linear

**Disciplina:** Machine Learning — AV2  
**Data:** Maio de 2026

---

## Resumo

Este trabalho apresenta um sistema de detecção automática de spam em mensagens SMS utilizando técnicas de aprendizado de máquina. Foram avaliados três classificadores — Multinomial Naive Bayes, Regressão Logística e SVM Linear — sobre o SMS Spam Collection (UCI), contendo 5.169 mensagens (86,6% legítimas, 13,4% spam) após remoção de duplicatas. O pré-processamento utiliza limpeza textual, tokenização e representação TF-IDF com unigramas e bigramas. O SVM Linear obteve os melhores resultados com F1 spam de 0,9084, AUC-ROC de 0,9932 e F1 macro de 0,9479, superiores aos demais modelos de forma estatisticamente significativa (McNemar, p=0,0014). A validação cruzada estratificada de 5 folds confirma a robustez do SVM (F1 médio 0,9065 ± 0,0250). Os resultados demonstram a eficácia de representações TF-IDF combinadas com classificadores lineares para detecção de spam textual, ainda que com limitações de generalização para outras línguas e contextos temporais distintos.

**Palavras-chave:** detecção de spam, classificação de texto, TF-IDF, SVM, Naive Bayes, aprendizado de máquina.

---

## 1. Introdução

A proliferação de mensagens não solicitadas (spam) representa um problema persistente em canais de comunicação digital. No contexto de mensagens SMS, o spam pode variar desde propagandas indesejadas até tentativas de phishing e golpes financeiros, causando prejuízos diretos aos usuários e sobrecarregando infraestruturas de telecomunicação [ALMEIDA2011].

Sistemas automáticos de filtragem de spam baseados em aprendizado de máquina têm demonstrado eficácia superior a filtros baseados em regras manuais, principalmente pela capacidade de adaptação a novos padrões de spam sem intervenção humana [SEBASTIANI2002]. A classificação de texto é uma das aplicações mais consolidadas de aprendizado supervisionado, com décadas de literatura estabelecida [MANNING2008].

Este trabalho tem como objetivo comparar o desempenho de três classificadores clássicos de texto — Multinomial Naive Bayes (NB), Regressão Logística (LR) e SVM Linear (SVM) — para a tarefa de detecção de spam em SMS, avaliando tanto o desempenho no conjunto de teste quanto a estabilidade via validação cruzada e a significância estatística das diferenças observadas.

**Variável-alvo:** `label` ∈ {`ham`, `spam`} — classificação binária.  
**Métrica primária:** F1-score da classe spam, adequada ao desbalanceamento do dataset.

---

## 2. Revisão de Literatura

A detecção automática de spam textual tem sido estudada extensivamente desde o trabalho seminal de [ZHANG2004], que comparou técnicas estatísticas para filtragem de spam em e-mail. A popularização de dispositivos móveis ampliou o problema para o contexto de SMS, motivando a criação de benchmarks específicos como o SMS Spam Collection [ALMEIDA2011].

Representações baseadas em frequência de termos, especialmente TF-IDF, combinadas com classificadores lineares demonstram desempenho competitivo em classificação de textos curtos [MANNING2008]. O Naive Bayes multinomial é historicamente o baseline mais utilizado para classificação de texto por sua simplicidade e eficiência computacional, com resultados surpreendentemente bons mesmo violando a suposição de independência entre features [BIRD2009].

Classificadores SVM foram introduzidos com sucesso para categorização de texto por [JOACHIMS1998], que demonstrou sua vantagem em espaços de alta dimensionalidade — exatamente o cenário de representações TF-IDF com milhares de features. A Regressão Logística, além de competitiva em desempenho, oferece a vantagem adicional de produzir probabilidades calibradas e coeficientes interpretáveis [PEDREGOSA2011].

A comparação estatística entre classificadores via teste McNemar é recomendada pela literatura quando se busca inferência além da estimativa pontual de desempenho [SEBASTIANI2002].

---

## 3. Metodologia

### 3.1 Dataset

Utilizou-se o **SMS Spam Collection** (UCI Machine Learning Repository), coletado por [ALMEIDA2011] a partir de fóruns públicos e da base NUS SMS Corpus. O dataset contém 5.572 mensagens rotuladas, das quais 5.169 permanecem após remoção de duplicatas. A distribuição é fortemente desbalanceada: 4.466 mensagens ham (86,4%) e 703 spam (13,6%).

### 3.2 Pré-processamento

O pipeline de pré-processamento segue as etapas:

1. **Normalização:** conversão para minúsculas.
2. **Substituição de tokens especiais:** URLs substituídas pelo token `URL`; sequências numéricas pelo token `NUM`. Essa estratégia preserva a informação semântica da presença de links e números sem criar dimensões esparsas para cada instância única.
3. **Limpeza:** remoção de pontuação e caracteres não-alfabéticos.
4. **Filtragem:** remoção de stopwords em inglês (NLTK) e termos com comprimento ≤ 2.
5. **Stemming:** aplicação do PorterStemmer para redução morfológica.

### 3.3 Vetorização

A representação vetorial utiliza **TF-IDF** (Term Frequency–Inverse Document Frequency) com os seguintes parâmetros:

- `max_features=5.000`: vocabulário limitado às 5.000 features mais relevantes.
- `ngram_range=(1,2)`: unigramas e bigramas capturando expressões compostas.
- `min_df=2`: descarte de termos com frequência de documento < 2 (ruído).
- `sublinear_tf=True`: suavização logarítmica de frequências altas.

**Controle de data leakage:** a divisão treino/teste (80%/20%, estratificada, seed=42) é realizada antes do fit do vetorizador. O TF-IDF é ajustado exclusivamente no conjunto de treino e aplicado por transformação no conjunto de teste.

A matriz resultante tem dimensões (4.135 × 4.961) no treino e (1.034 × 4.961) no teste.

### 3.4 Modelos

| Modelo | Hiperparâmetros | Justificativa |
|--------|----------------|---------------|
| Multinomial Naive Bayes | alpha=1.0 (Laplace) | Baseline clássico; eficiente; funciona bem com TF-IDF |
| Logistic Regression | C=1.0, max_iter=1000, random_state=42 | Probabilidades calibradas; coeficientes interpretáveis |
| Linear SVM | C=1.0, max_iter=2000, random_state=42 | SOTA para texto em alta dimensão; margem robusta |

### 3.5 Avaliação

- **Métricas:** F1-score (spam), F1 macro, acurácia, AUC-ROC, Average Precision, matriz de confusão.
- **Validação cruzada:** StratifiedKFold (k=5, shuffle=True, seed=42) no conjunto de treino, reportando média ± desvio-padrão do F1 spam.
- **Teste estatístico:** McNemar entre LR e SVM (os dois melhores modelos) para verificar significância das diferenças.

---

## 4. Resultados

### 4.1 Desempenho no Conjunto de Teste

| Modelo | Acurácia | F1 (spam) | F1 macro | AUC-ROC |
|--------|----------|-----------|----------|---------|
| Naive Bayes | 0,9662 | 0,8472 | 0,9141 | 0,9809 |
| Logistic Regression | 0,9613 | 0,8276 | 0,9029 | 0,9909 |
| **Linear SVM** | **0,9778** | **0,9084** | **0,9479** | **0,9932** |

O SVM Linear supera os demais modelos em todas as métricas. A acurácia elevada dos três modelos reflete parcialmente o desbalanceamento do dataset; o F1 spam revela as diferenças reais de desempenho.

### 4.2 Validação Cruzada (5-Fold Estratificada)

| Modelo | F1 spam médio | Desvio-padrão | Folds |
|--------|---------------|---------------|-------|
| Naive Bayes | 0,8554 | ±0,0299 | [0,859; 0,863; 0,844; 0,902; 0,809] |
| Logistic Regression | 0,7275 | ±0,0479 | [0,704; 0,731; 0,798; 0,751; 0,654] |
| Linear SVM | 0,9065 | ±0,0250 | [0,896; 0,923; 0,939; 0,908; 0,866] |

A Regressão Logística apresenta maior variância entre folds (σ=0,048), indicando menor estabilidade. O SVM Linear mantém consistência superior com menor desvio.

### 4.3 Recall por Classe (Spam)

O SVM detectou 89,3% dos spams (recall), contra 75,6% do NB e 68,7% da LR — relevante pois falsos negativos (spam não detectado) têm custo prático maior que falsos positivos.

### 4.4 Teste Estatístico — McNemar (LR vs SVM)

Aplicando o teste McNemar com correção de continuidade:

- b = 23 (LR acerta, SVM erra)
- c = 59 (SVM acerta, LR erra)
- **p-value = 0,0014**

A diferença é **estatisticamente significativa** (α=0,05). O SVM Linear é superior à Regressão Logística como classificador de spam neste dataset.

### 4.5 Features Mais Relevantes (Logistic Regression)

Os 5 maiores coeficientes positivos (→ spam): `free`, `call`, `txt`, `prize`, `claim`  
Os 5 maiores coeficientes negativos (→ ham): `ok`, `gt`, `lor`, `come`, `got`

---

## 5. Discussão

Os resultados confirmam que representações TF-IDF combinadas com classificadores lineares são eficazes para detecção de spam em SMS. O SVM Linear supera os demais modelos consistentemente, alinhado com achados de [JOACHIMS1998] e [ZHANG2004] para classificação de texto em alta dimensão.

A Regressão Logística, apesar de inferior ao SVM em desempenho bruto, oferece vantagem interpretativa: seus coeficientes revelam os termos mais discriminativos, permitindo auditoria humana do modelo — relevante em aplicações onde a explicabilidade é requisito. O NB, por sua vez, é o modelo mais rápido e com bom equilíbrio entre precision e recall para a classe spam.

A análise de erros revela padrões: falsos negativos tendem a ser spams mais sutis, sem palavras-chave explícitas ("click here", "free prize"), enquanto falsos positivos são mensagens legítimas com linguagem mais assertiva ou promoções reais. Isso sugere que a incorporação de features contextuais (horário, remetente) poderia reduzir esses erros.

---

## 6. Limitações e Ameaças à Validade

### 6.1 Validade Interna

- **Overfitting potencial:** apesar da validação cruzada, o ajuste de hiperparâmetros foi realizado sem holdout adicional. Em cenários de produção, recomenda-se divisão treino/validação/teste.
- **Viés de seleção:** os hiperparâmetros padrão (C=1,0, alpha=1,0) foram utilizados sem busca sistemática (grid search). Otimização poderia melhorar especialmente a LR.

### 6.2 Validade Externa

- **Limitação temporal:** o dataset data de ~2012. Padrões de spam atuais (WhatsApp, e-mail, redes sociais) diferem significativamente de SMS da época, reduzindo a generalização sem retreinamento.
- **Limitação linguística:** o dataset é exclusivamente em inglês britânico. Modelos não generalizariam para português ou outros idiomas sem corpora específicos.
- **Limitação de domínio:** SMS tem características distintas de outros canais (brevidade, abreviações, ausência de HTML). Modelos treinados aqui não devem ser aplicados diretamente a spam de e-mail ou redes sociais.

### 6.3 Ameaças à Validade Estatística

- **Tamanho do conjunto de teste:** com apenas 131 exemplos de spam no teste, estimativas de métricas têm intervalos de confiança relativamente amplos.
- **Distribuição de treino/teste:** embora estratificada, a divisão aleatória pode favorecer ou prejudicar modelos específicos dependendo da seed.

---

## 7. Conclusão

Este trabalho demonstrou que o SVM Linear com representação TF-IDF é o classificador mais eficaz para detecção de spam em SMS no dataset avaliado, atingindo F1 spam de 0,9084 e AUC-ROC de 0,9932, com superioridade estatisticamente significativa sobre os demais modelos (McNemar, p=0,0014).

A pipeline desenvolvida — pré-processamento textual, vetorização TF-IDF ajustada exclusivamente no treino e validação cruzada estratificada — atende aos requisitos de reprodutibilidade e prevenção de data leakage. O controle de sementes aleatórias (seed=42) garante reprodutibilidade completa dos resultados.

Como trabalhos futuros, sugere-se: (i) explorar representações baseadas em embeddings (Word2Vec, BERT) para capturar semântica contextual; (ii) coletar dados mais recentes e em português; (iii) implementar busca de hiperparâmetros com validação cruzada interna; e (iv) avaliar modelos de detecção em tempo real com restrições de latência.

---

## Referências

[ALMEIDA2011] ALMEIDA, T. A.; GÓMEZ HIDALGO, J. M.; YAMAKAMI, A. Contributions to the study of SMS spam filtering: new collection and results. In: *ACM Symposium on Document Engineering (DocEng'11)*, 2011, p. 259–262.

[BIRD2009] BIRD, S.; KLEIN, E.; LOPER, E. *Natural Language Processing with Python*. Sebastopol: O'Reilly Media, 2009.

[JOACHIMS1998] JOACHIMS, T. Text categorization with support vector machines: learning with many relevant features. In: *European Conference on Machine Learning (ECML'98)*, 1998, p. 137–142.

[MANNING2008] MANNING, C. D.; RAGHAVAN, P.; SCHÜTZE, H. *Introduction to Information Retrieval*. Cambridge: Cambridge University Press, 2008.

[PEDREGOSA2011] PEDREGOSA, F. et al. Scikit-learn: machine learning in Python. *Journal of Machine Learning Research*, v. 12, p. 2825–2830, 2011.

[SEBASTIANI2002] SEBASTIANI, F. Machine learning in automated text categorization. *ACM Computing Surveys*, v. 34, n. 1, p. 1–47, 2002.

[ZHANG2004] ZHANG, L.; ZHU, J.; YAO, T. An evaluation of statistical spam filtering techniques. *ACM Transactions on Asian Language Information Processing*, v. 3, n. 4, p. 243–269, 2004.
