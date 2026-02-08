# ANP Production Analysis

Ferramenta de análise de produção de petróleo e gás natural com dados públicos da ANP (Agência Nacional do Petróleo, Gás Natural e Biocombustíveis).

Baixa automaticamente os dados oficiais de produção por poço, aplica modelos de decline curve (Arps), detecta anomalias operacionais e compara ramp-up de produção entre campos.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red)
![License](https://img.shields.io/badge/License-MIT-green)

## Funcionalidades

**Visão geral** — Produção agregada por campo, ranking dos top produtores, mapa de calor mensal.

**Decline Curve Analysis** — Ajuste de modelos exponencial e hiperbólico de Arps (scipy curve_fit), seleção automática do melhor modelo por R², projeção de produção futura.

**Detecção de anomalias** — Identifica quedas significativas (>20%), paradas totais (produção = 0) e picos atípicos via IQR em rolling window.

**Comparação de campos** — Normaliza curvas de ramp-up pelo mês de primeiro óleo, calcula métricas (tempo até plateau, taxa de ramp-up, produção máxima).

## Dados

Fonte: [ANP - Produção por Poço](https://www.gov.br/anp/pt-br/centrais-de-conteudo/dados-abertos/producao-de-petroleo-e-gas-natural-por-poco)

- Dados mensais, nível de poço, desde 2005
- Cobertura: offshore, pré-sal e terra
- Publicados com lag de ~2 meses
- Download automático com cache local

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

Na primeira execução, a aplicação baixa os dados da ANP (~50-200 MB dependendo dos anos selecionados). Os arquivos ficam em cache em `data/`.

## Stack

- **pandas** — manipulação de dados
- **scipy** — curve fitting (Arps models)
- **plotly** — visualizações interativas
- **streamlit** — dashboard web
- **requests** — download dos dados

## Estrutura

```
├── app.py                    # Streamlit dashboard
├── analysis/
│   ├── loader.py             # Download + parse dos CSVs da ANP
│   ├── decline.py            # Arps decline curve fitting
│   ├── anomaly.py            # Detecção de anomalias
│   └── compare.py            # Comparação de ramp-up
├── data/                     # Cache de dados (gitignored)
└── .streamlit/config.toml    # Tema visual
```

## Licença

MIT
