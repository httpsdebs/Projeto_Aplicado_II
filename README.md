# Projeto Aplicado I - Detecção de Fraudes em Transações de Cartão de Crédito  

## 1. Introdução e Justificativa  

**Contexto:**  
O crescimento das transações financeiras digitais trouxe também um aumento nos casos de fraude. Detectar esses eventos é um desafio devido ao alto volume de dados e à complexidade dos padrões de fraude.  

**Relevância:**  
A detecção de fraudes é fundamental para reduzir perdas financeiras, proteger consumidores e manter a confiança no sistema financeiro. Métodos manuais não são escaláveis, reforçando a necessidade de soluções baseadas em ciência de dados.  

**Objetivo do Projeto:**  
Desenvolver um modelo de análise capaz de identificar transações fraudulentas com alta precisão, reduzindo tanto falsos positivos quanto falsos negativos.  

**Contribuição da Ciência de Dados:**  
A aplicação de técnicas de ciência de dados permite criar modelos preditivos escaláveis e mais eficientes para a detecção de fraudes em tempo real.  

---

## 2. Definição do Problema  

- **Problema Central:**  
  Como identificar padrões em grandes volumes de dados de transações financeiras que diferenciem operações legítimas de fraudulentas.  

- **Desafios Esperados:**  
  - Desbalanceamento de classes (fraudes são raras).  
  - Anonimização dos dados (variáveis PCA V1–V28).  
  - Natureza dinâmica da fraude (padrões em constante mudança).  
  - Volume elevado de dados.  

---

## 3. Descrição do Dataset  

- **Nome:** Credit Card Fraud Detection (Kaggle)  
- **Origem:** Université Libre de Bruxelles (ULB), Europa.  
- **Período:** Setembro de 2013.  
- **Tamanho:** ~285.000 transações, 31 variáveis.  

**Variáveis Principais:**  
- `Time`: Segundos desde a primeira transação.  
- `V1` a `V28`: Variáveis numéricas (componentes PCA).  
- `Amount`: Valor da transação.  
- `Class`: Variável alvo (0 = legítima, 1 = fraude).  

**Características Relevantes:**  
- Dados anonimizados.  
- Forte desbalanceamento (poucas fraudes em relação ao total).  

---

## 4. Metodologia Proposta  

### Fase 1 – Coleta e Entendimento dos Dados  
- Download do dataset e inspeção inicial.  

### Fase 2 – Análise Exploratória de Dados (AED)  
- Verificação de valores ausentes.  
- Distribuições das variáveis (histogramas, boxplots).  
- Análise do desbalanceamento da variável `Class`.  

### Fase 3 – Pré-processamento  
- Tratamento de dados ausentes (se houver).  
- Normalização (Amount e Time).  
- Técnicas para lidar com desbalanceamento (SMOTE, undersampling, oversampling).  

### Fase 4 – Modelagem  
- Algoritmos: Regressão Logística, Árvores de Decisão, Random Forest, Gradient Boosting, SVM, Redes Neurais.  
- Divisão em treino e teste.  
- Validação dos modelos.  

### Fase 5 – Avaliação e Otimização  
- Métricas: Precisão, Recall, F1-Score, AUC-ROC.  
- Ajuste de hiperparâmetros.  

### Fase 6 – Conclusões  
- Discussão dos resultados.  
- Limitações do estudo.  
- Sugestões para trabalhos futuros.  

---

## 5. Cronograma  

- **Semana 1-2:** Definição do escopo e planejamento.  
- **Semana 3-4:** Coleta e AED.  
- **Semana 5-6:** Pré-processamento e engenharia de features.  
- **Semana 7-8:** Modelagem e treinamento.  
- **Semana 9-10:** Avaliação e refinamento.  
- **Semana 11-12:** Documentação e apresentação final.  

---

## Integrantes do Grupo  

- Déborah Silvério Alves Morales - RA: 10728563  
- Diógenes  
- Lucas Iglezias dos Anjos - RA: 10433522  
- Luiz Benlardi Neto - RA: 10724617  
