# Decisões Técnicas e Justificativas Metodológicas

## 1. Escolha do Dataset

**Decisão:** SMS Spam Collection (UCI, Almeida et al., 2011)  
**Justificativa:** Dataset público, amplamente utilizado como benchmark em detecção de spam textual, com rótulos confiáveis. Permite comparação direta com literatura existente.

## 2. Pré-processamento de Texto

**Decisão:** Pipeline: lowercase → substituição de URLs por token `URL` → substituição de números por `NUM` → remoção de pontuação → remoção de stopwords → stemming (PorterStemmer)  
**Justificativa:** Reduz dimensionalidade e agrupa variantes morfológicas. Substituição de URLs e números por tokens especiais preserva informação semântica (presença de link é forte indicador de spam) sem criar dimensões esparsas para cada URL/número único.

**Trade-off:** Stemming pode prejudicar interpretabilidade dos termos (ex.: "free" → "free", "winner" → "winner"), mas melhora generalização ao reduzir vocabulário.

## 3. Vetorização: TF-IDF

**Decisão:** TF-IDF com unigramas e bigramas (ngram_range=(1,2)), max_features=5.000, min_df=2, sublinear_tf=True  
**Justificativa:** TF-IDF penaliza termos muito frequentes e destaca termos discriminativos. Bigramas capturam expressões compostas como "click here" ou "free prize". `min_df=2` remove termos que aparecem em apenas uma mensagem (ruído). `sublinear_tf` aplica log(tf) para suavizar frequências altas.

**Trade-off:** Bag-of-words não preserva ordem completa das palavras, mas é suficiente para classificação de spam onde a presença de termos-chave é mais relevante que a sintaxe.

## 4. Divisão Treino/Teste

**Decisão:** 80/20, estratificada por classe, random_state=42  
**Justificativa:** Divisão estratificada mantém a proporção ham/spam (86,6%/13,4%) em ambos os conjuntos, evitando distribuição acidental diferente no teste. Split realizado ANTES de qualquer fit do vetorizador para evitar data leakage.

## 5. Escolha dos Modelos

| Modelo | Justificativa |
|--------|---------------|
| Multinomial Naive Bayes | Baseline clássico para classificação de texto; assume independência entre features (aceitável para TF-IDF); eficiente e interpretável |
| Logistic Regression | Forte para classificação binária linear; produz probabilidades calibradas; coeficientes permitem análise de importância de features |
| Linear SVM | Estado da arte para texto em alta dimensão; margem de separação robusta; sem `predict_proba` nativo, mas `decision_function` serve para ranking |

**Trade-off interpretabilidade vs desempenho:** Logistic Regression é mais interpretável; Linear SVM é superior em F1 e AUC mas não produz probabilidades diretamente.

## 6. Métricas de Avaliação

**Métrica primária:** F1-score da classe spam  
**Justificativa:** Dataset desbalanceado (13,4% spam). Acurácia isolada seria enganosa — um modelo que classifica tudo como "ham" teria 86,6% de acurácia com F1 spam = 0. F1 equilibra Precision e Recall. Falsos negativos (spam não detectado) têm custo maior que falsos positivos (ham bloqueado), justificando monitorar também o Recall.

**Métricas secundárias:** AUC-ROC, Average Precision, matriz de confusão, validação cruzada.

## 7. Validação Cruzada

**Decisão:** StratifiedKFold com 5 splits, shuffle=True, random_state=42  
**Justificativa:** Validação cruzada estratificada garante representatividade de classes em cada fold. 5 folds oferece bom equilíbrio entre variância da estimativa e custo computacional. Realizada sobre o conjunto de treino — conjunto de teste reservado para avaliação final.

## 8. Teste Estatístico

**Decisão:** Teste McNemar entre os dois melhores modelos (LR vs SVM)  
**Justificativa:** McNemar é o teste adequado para comparar dois classificadores no mesmo conjunto de teste — compara discordâncias em vez de médias, sendo mais sensível para amostras pareadas. Resultado: p=0,0014, diferença estatisticamente significativa (α=0,05), Linear SVM superior.
