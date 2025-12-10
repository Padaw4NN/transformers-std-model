# Transformer Model - Classificação de Texto

Este projeto implementa um modelo Transformer do zero usando PyTorch para classificação de texto no dataset AG News.

## O que é o Modelo Transformer?

O Transformer é uma arquitetura de rede neural revolucionária introduzida em 2017 no paper "Attention is All You Need". Diferente de RNNs e LSTMs, ele processa sequências inteiras em paralelo usando mecanismos de atenção.

### Principais Componentes

#### 1. **Multi-Head Attention**
- Permite que o modelo preste atenção a diferentes partes da sequência simultaneamente
- Divide a representação em múltiplas "cabeças" (heads) que capturam diferentes aspectos do contexto
- Calcula scores de atenção usando Query, Key e Value
- Fórmula: `Attention(Q, K, V) = softmax(QK^T / √d_k)V`

#### 2. **Positional Encoding**
- Adiciona informação sobre a posição dos tokens na sequência
- Usa funções seno e cosseno para gerar encodings únicos para cada posição
- Essencial porque o Transformer não tem noção inerente de ordem sequencial

#### 3. **Position-wise Feed-Forward Network**
- Duas camadas lineares com ativação GELU entre elas
- Aplicada independentemente a cada posição da sequência
- Aumenta a capacidade de expressão do modelo

#### 4. **Encoder Layer**
- Combina Multi-Head Attention + Feed-Forward Network
- Usa conexões residuais (skip connections) e Layer Normalization
- Estrutura: `X = LayerNorm(X + Attention(X))` → `X = LayerNorm(X + FFN(X))`

#### 5. **Transformer Classifier**
- Stack de múltiplas camadas de Encoder
- Embedding de tokens e Positional Encoding na entrada
- Global Average Pooling para agregar a sequência
- Cabeça de classificação no topo

### Arquitetura do Projeto

```
Input Text
    ↓
Token Embedding (vocab_size → d_model)
    ↓
Positional Encoding
    ↓
Encoder Layer 1 (Multi-Head Attention + FFN)
    ↓
Encoder Layer 2
    ↓
    ...
    ↓
Encoder Layer N
    ↓
Global Average Pooling
    ↓
Classification Head (Linear → GELU → Linear)
    ↓
Output (num_classes)
```

## Configuração do Ambiente Python

### Pré-requisitos
- Python 3.8 ou superior
- pip ou uv (gerenciador de pacotes)

### Criando o Ambiente Virtual

```bash
# Criar ambiente virtual
python -m venv venv

# Ativar o ambiente virtual
# macOS/Linux:
source venv/bin/activate

# Windows:
venv\Scripts\activate
```

### Instalando Dependências

```bash
# Opção 1: Usando pip
pip install -r requirements.txt

# Opção 2: Usando uv (mais rápido)
uv pip install -r requirements.txt
```

### Principais Dependências

- **torch**: Framework de deep learning
- **datasets**: Biblioteca da Hugging Face para carregar datasets
- **numpy**: Computação numérica
- **matplotlib** & **seaborn**: Visualização de dados
- **scikit-learn**: Métricas de avaliação

## 🚀 Executando o Projeto

1. Ative o ambiente virtual:
```bash
source venv/bin/activate  # macOS/Linux
```

2. Abra o notebook:
```bash
jupyter notebook transformer_model.ipynb
```

3. Execute as células sequencialmente para:
   - Carregar o dataset AG News (4 categorias: World, Sports, Business, Sci/Tech)
   - Construir vocabulário e tokenizar textos
   - Treinar o modelo Transformer
   - Avaliar performance e visualizar resultados

## Dataset

O projeto usa o **AG News**, um dataset de classificação de notícias com:
- 120.000 amostras de treino
- 7.600 amostras de teste
- 4 categorias de notícias

## Hiperparâmetros do Modelo

- `d_model`: 512 (dimensão do modelo)
- `num_heads`: 8 (número de cabeças de atenção)
- `num_layers`: 6 (número de camadas de encoder)
- `d_ff`: 2048 (dimensão da feed-forward network)
- `max_len`: 512 (comprimento máximo da sequência)
- `dropout`: 0.1

## Estrutura do Código

- `MultiHeadAttention`: Implementa o mecanismo de atenção multi-cabeça
- `PositionalEncoding`: Adiciona informação posicional aos embeddings
- `PositionwiseFeedForward`: Rede feed-forward position-wise
- `EncoderLayer`: Camada completa do encoder
- `TransformerClassifier`: Modelo completo para classificação

## Resultados Esperados

O modelo deve atingir boa acurácia na classificação de notícias após o treinamento, demonstrando a eficácia da arquitetura Transformer para tarefas de NLP.

---

**Nota**: Este projeto é educacional e implementa o Transformer do zero para fins de aprendizado. Para aplicações em produção, considere usar bibliotecas otimizadas como Hugging Face Transformers.
