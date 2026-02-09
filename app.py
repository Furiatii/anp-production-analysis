"""ANP Production Analysis - Streamlit Dashboard."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

from analysis.loader import load_data, get_available_fields
from analysis.decline import fit_decline_curve, get_decline_dates
from analysis.anomaly import detect_anomalies, anomaly_summary
from analysis.compare import compute_rampup, top_fields_by_production, field_monthly_production

st.set_page_config(
    page_title="ANP Production Analysis",
    page_icon="🛢️",
    layout="wide",
)

# ── CSS ──────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .stMetric { border: 1px solid #E2E8F0; border-radius: 8px; padding: 12px; }
    div[data-testid="stMetricValue"] { font-size: 1.8rem; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────────

st.sidebar.title("ANP Production Analysis")
st.sidebar.caption("Dados públicos de produção de petróleo e gás da ANP")

available_years = list(range(2015, 2026))
selected_years = st.sidebar.multiselect(
    "Anos",
    options=available_years,
    default=[2020, 2021, 2022, 2023, 2024, 2025],
)

ambiente_options = ["Mar", "Pré-Sal"]
selected_ambientes = st.sidebar.multiselect(
    "Ambiente",
    options=["Mar", "Pré-Sal", "Terra"],
    default=ambiente_options,
)

# ── Data loading ─────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False, ttl=3600)
def cached_load(years: tuple, ambientes: tuple):
    return load_data(list(years), list(ambientes))


if not selected_years:
    st.warning("Selecione pelo menos um ano na barra lateral.")
    st.stop()

with st.spinner("Baixando e processando dados da ANP..."):
    progress = st.progress(0, text="Baixando arquivos...")

    def update_progress(current, total):
        if total > 0:
            progress.progress(current / total, text=f"Arquivo {current}/{total}")

    # Use tuple for caching
    df = cached_load(tuple(sorted(selected_years)), tuple(sorted(selected_ambientes)))
    progress.empty()

if df.empty:
    st.error("Nenhum dado encontrado. Verifique a conexão ou os filtros selecionados.")
    st.stop()

# Field selector
all_fields = get_available_fields(df)
top_fields = top_fields_by_production(df, n=20)

selected_fields = st.sidebar.multiselect(
    "Campos (top 20 por produção)",
    options=top_fields,
    default=top_fields[:5] if top_fields else [],
)

# ── Tabs ─────────────────────────────────────────────────────────────────────

tab0, tab1, tab2, tab3, tab4 = st.tabs([
    "Sobre a Análise", "Visão Geral", "Decline Curve", "Anomalias", "Comparação de Campos",
])

# ══════════════════════════════════════════════════════════════════════════════
# Tab 0: About the Analysis
# ══════════════════════════════════════════════════════════════════════════════

with tab0:
    st.header("Sobre a Análise")

    st.markdown("""
### De onde vêm esses dados?

A ANP (Agência Nacional do Petróleo) publica mensalmente quanto cada poço de petróleo
produziu no Brasil. Os dados incluem produção de óleo (em barris por dia), gás natural,
água, e informações sobre o campo, bacia, operador e instalação (FPSO, plataforma fixa, etc.).

Essa ferramenta baixa automaticamente esses dados do portal público da ANP e aplica
três tipos de análise.

---

### 1. Visão Geral

Mostra o panorama da produção offshore brasileira:

- **Produção por campo ao longo do tempo**: cada linha é um campo (Búzios, Mero, Tupi...).
  Quando a linha sobe, o campo está produzindo mais. Quando desce, está em declínio ou parada.
- **Top produtores**: ranking dos campos com maior produção média no período selecionado.
- **Mapa de calor**: cada célula mostra a intensidade de produção de um campo em um mês.
  Cores mais quentes = mais produção. Útil para identificar visualmente padrões sazonais
  ou interrupções.

---

### 2. Decline Curve Analysis (Curva de Declínio)

Todo campo de petróleo segue um ciclo: a produção sobe (ramp-up), atinge um pico,
e depois cai naturalmente ao longo dos anos conforme o reservatório perde pressão.

O engenheiro J.J. Arps descreveu essa queda com dois modelos matemáticos:

- **Exponencial**: a produção cai a uma taxa fixa por mês. Modelo mais simples e conservador.
- **Hiperbólico**: a taxa de queda vai diminuindo com o tempo. Mais realista para a maioria dos campos.

**O que os parâmetros significam:**

| Parâmetro | O que é | Exemplo |
|-----------|---------|---------|
| **qi** | Produção no pico (bbl/dia) | qi = 50.000 significa que o campo produzia 50 mil barris/dia no auge |
| **Di** | Taxa de declínio mensal | Di = 0.02 significa queda de ~2% ao mês |
| **b** | Fator de Arps | b = 0 é exponencial, b > 0 o declínio desacelera (mais otimista) |
| **R²** | Qualidade do ajuste | Quanto mais perto de 1.0, melhor o modelo explica os dados |

**Como ler o gráfico:**
- Pontos escuros = produção real registrada pela ANP
- Linha azul tracejada = modelo ajustado (exponencial ou hiperbólico)
- Linha vermelha pontilhada = projeção futura baseada no modelo

Se o R² for baixo (< 0.5), significa que o campo não segue um padrão claro de declínio,
talvez porque ainda esteja em ramp-up ou tenha muitas interrupções.

**Para que serve:** estimar quando um campo vai parar de ser rentável e planejar investimentos.

---

### 3. Detecção de Anomalias

Identifica meses onde a produção de um campo fez algo fora do padrão:

| Tipo | O que detecta | Possíveis causas |
|------|---------------|------------------|
| **Queda** | Produção caiu mais de 20% de um mês pro outro | Manutenção programada, falha de equipamento, problema no poço |
| **Parada** | Produção foi a zero | Shutdown completo, manutenção maior, acidente operacional |
| **Pico** | Produção saltou muito acima do normal (outlier estatístico) | Novo poço conectado, retorno de manutenção, teste de produção |

**Como ler o gráfico:**
- Linha = produção mensal do campo
- Marcadores vermelhos (X) = quedas significativas
- Marcadores roxos = paradas totais
- Marcadores verdes = picos atípicos

O limiar de queda é ajustável (padrão: 20%). Os picos são detectados pelo método IQR
(interquartil range) em janela móvel de 6 meses.

---

### 4. Comparação de Campos (Ramp-up)

Quando um campo novo começa a produzir, a produção sobe gradualmente ao longo de
meses até atingir o plateau (capacidade máxima).

Essa aba normaliza todas as curvas pelo "mês zero" (primeiro óleo) para permitir
comparação direta, mesmo que os campos tenham começado a produzir em anos diferentes.

**Métricas da tabela:**

| Métrica | O que significa |
|---------|-----------------|
| **Primeiro óleo** | Quando o campo começou a produzir |
| **Produção máxima** | Maior produção mensal registrada (bbl/dia) |
| **Meses até pico** | Quanto tempo levou do primeiro óleo ao pico |
| **Taxa de ramp-up** | Velocidade média de subida (bbl/dia por mês) |
| **Produção atual** | Último valor registrado |

**Para que serve:** avaliar eficiência operacional. Um campo que atinge o plateau mais
rápido geralmente indica melhor planejamento de desenvolvimento e conexão de poços.
""")

    st.divider()
    st.caption("Fonte dos dados: [ANP - Produção por Poço]"
               "(https://www.gov.br/anp/pt-br/centrais-de-conteudo/"
               "dados-abertos/producao-de-petroleo-e-gas-natural-por-poco)")

# ══════════════════════════════════════════════════════════════════════════════
# Tab 1: Overview
# ══════════════════════════════════════════════════════════════════════════════

with tab1:
    st.header("Visão Geral da Produção")

    with st.expander("O que estou vendo?"):
        st.markdown(
            "Panorama da produção offshore brasileira. Cada linha no gráfico é um campo "
            "(Búzios, Mero, Tupi...). O ranking mostra os maiores produtores por média "
            "no período. O mapa de calor ajuda a identificar padrões sazonais e interrupções."
        )

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Campos", df["campo"].nunique())
    with col2:
        st.metric("Poços", df["poco_anp"].nunique())
    with col3:
        latest = df[df["data"] == df["data"].max()]
        total_oil = latest["petroleo_bbl_dia"].sum()
        st.metric("Prod. último mês (bbl/dia)", f"{total_oil:,.0f}")
    with col4:
        date_range = f"{df['data'].min():%b/%Y} - {df['data'].max():%b/%Y}"
        st.metric("Período", date_range)

    st.subheader("Produção total por campo ao longo do tempo")

    if selected_fields:
        plot_df = field_monthly_production(df, selected_fields)
    else:
        plot_df = field_monthly_production(df, top_fields[:10])

    if not plot_df.empty:
        fig = px.line(
            plot_df,
            x="data", y="production", color="campo",
            labels={"data": "Data", "production": "Petróleo (bbl/dia)", "campo": "Campo"},
        )
        fig.update_layout(height=500, hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

    # Top producers bar chart
    st.subheader("Top produtores (média no período)")

    avg_prod = (
        df.groupby("campo")["petroleo_bbl_dia"]
        .mean()
        .sort_values(ascending=True)
        .tail(15)
        .reset_index()
    )
    fig_bar = px.bar(
        avg_prod,
        x="petroleo_bbl_dia", y="campo",
        orientation="h",
        labels={"petroleo_bbl_dia": "Média de produção (bbl/dia)", "campo": "Campo"},
        color_discrete_sequence=["#0369A1"],
    )
    fig_bar.update_layout(height=450, yaxis=dict(categoryorder="total ascending"))
    st.plotly_chart(fig_bar, use_container_width=True)

    # Heatmap: monthly production by field
    st.subheader("Mapa de calor mensal")
    if selected_fields:
        heat_data = field_monthly_production(df, selected_fields)
    else:
        heat_data = field_monthly_production(df, top_fields[:10])

    if not heat_data.empty:
        heat_pivot = heat_data.pivot_table(
            index="campo", columns="data", values="production", aggfunc="sum",
        )
        fig_heat = px.imshow(
            heat_pivot.values,
            labels=dict(x="Data", y="Campo", color="bbl/dia"),
            x=[d.strftime("%Y-%m") for d in heat_pivot.columns],
            y=heat_pivot.index.tolist(),
            aspect="auto",
            color_continuous_scale="YlOrRd",
        )
        fig_heat.update_layout(height=400)
        st.plotly_chart(fig_heat, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# Tab 2: Decline Curve
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.header("Decline Curve Analysis")
    st.caption("Ajuste de curvas de declínio de Arps (exponencial e hiperbólico)")

    with st.expander("O que estou vendo?"):
        st.markdown(
            "A produção de um campo cai naturalmente ao longo do tempo. "
            "Os modelos de Arps ajustam uma curva matemática a essa queda e projetam "
            "a produção futura. **qi** = produção no pico, **Di** = taxa de declínio, "
            "**R²** = qualidade do ajuste (perto de 1.0 = bom). "
            "Se o R² for baixo, o campo pode ainda estar em ramp-up."
        )

    decline_field = st.selectbox(
        "Campo para análise de declínio",
        options=selected_fields if selected_fields else top_fields[:10],
        key="decline_field",
    )
    forecast_months = st.slider("Meses de projeção", 6, 36, 24)

    if decline_field:
        result = fit_decline_curve(df, decline_field, forecast_months=forecast_months)
        dates = get_decline_dates(df, decline_field)

        if result and dates is not None:
            # Metrics
            mc1, mc2, mc3, mc4 = st.columns(4)
            with mc1:
                st.metric("Modelo", result.model.capitalize())
            with mc2:
                st.metric("R²", f"{result.r_squared:.4f}")
            with mc3:
                st.metric("qi (bbl/dia)", f"{result.qi:,.0f}")
            with mc4:
                st.metric("Di (taxa de declínio)", f"{result.di:.4f}")

            if result.b > 0:
                st.metric("b (fator de Arps)", f"{result.b:.3f}")

            # Plot
            fig = go.Figure()

            # Actual production
            fig.add_trace(go.Scatter(
                x=dates,
                y=result.production_actual,
                mode="markers+lines",
                name="Produção real",
                marker=dict(color="#0F172A", size=4),
                line=dict(color="#0F172A", width=1),
            ))

            # Fitted curve
            fig.add_trace(go.Scatter(
                x=dates,
                y=result.fitted,
                mode="lines",
                name=f"Ajuste ({result.model})",
                line=dict(color="#0369A1", width=2, dash="dash"),
            ))

            # Forecast
            forecast_dates = pd.date_range(
                dates[-1], periods=forecast_months + 1, freq="MS",
            )[1:]
            fig.add_trace(go.Scatter(
                x=forecast_dates,
                y=result.forecast,
                mode="lines",
                name="Projeção",
                line=dict(color="#DC2626", width=2, dash="dot"),
            ))

            fig.update_layout(
                height=500,
                xaxis_title="Data",
                yaxis_title="Petróleo (bbl/dia)",
                hovermode="x unified",
            )
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.info(f"Dados insuficientes para ajuste de declínio em **{decline_field}**.")


# ══════════════════════════════════════════════════════════════════════════════
# Tab 3: Anomalies
# ══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.header("Detecção de Anomalias")

    with st.expander("O que estou vendo?"):
        st.markdown(
            "Meses onde a produção fez algo fora do padrão. "
            "**Queda** = caiu mais de 20% (manutenção, falha). "
            "**Parada** = foi a zero (shutdown). "
            "**Pico** = saltou muito acima do normal (novo poço, retorno de manutenção). "
            "Os marcadores no gráfico mostram onde cada anomalia ocorreu."
        )

    anomaly_field = st.selectbox(
        "Campo",
        options=["Todos"] + (selected_fields if selected_fields else top_fields[:10]),
        key="anomaly_field",
    )
    threshold = st.slider("Limiar de queda (%)", 10, 50, 20) / 100
    anomaly_types = st.multiselect(
        "Tipos de anomalia",
        options=["queda", "parada", "pico"],
        default=["queda", "parada", "pico"],
    )

    campo_arg = None if anomaly_field == "Todos" else anomaly_field
    anomalies = detect_anomalies(df, campo=campo_arg, drop_threshold=threshold)

    if not anomalies.empty and anomaly_types:
        anomalies = anomalies[anomalies["tipo"].isin(anomaly_types)]

    summary = anomaly_summary(anomalies)

    ac1, ac2, ac3, ac4 = st.columns(4)
    with ac1:
        st.metric("Total", summary["total"])
    with ac2:
        st.metric("Quedas (>{}%)".format(int(threshold * 100)), summary["queda"])
    with ac3:
        st.metric("Paradas", summary["parada"])
    with ac4:
        st.metric("Picos", summary["pico"])

    if not anomalies.empty:
        # Timeline with anomalies
        campo_for_plot = anomaly_field if anomaly_field != "Todos" else (selected_fields[0] if selected_fields else top_fields[0])
        prod_data = field_monthly_production(df, [campo_for_plot])

        if not prod_data.empty:
            fig = go.Figure()
            campo_prod = prod_data[prod_data["campo"] == campo_for_plot]
            fig.add_trace(go.Scatter(
                x=campo_prod["data"],
                y=campo_prod["production"],
                mode="lines",
                name="Produção",
                line=dict(color="#0F172A"),
            ))

            # Overlay anomalies
            campo_anomalies = anomalies[anomalies["campo"] == campo_for_plot]
            colors = {"queda": "#DC2626", "parada": "#7C3AED", "pico": "#059669"}
            for tipo in anomaly_types:
                tipo_df = campo_anomalies[campo_anomalies["tipo"] == tipo]
                if not tipo_df.empty:
                    fig.add_trace(go.Scatter(
                        x=tipo_df["data"],
                        y=tipo_df["producao"],
                        mode="markers",
                        name=tipo.capitalize(),
                        marker=dict(color=colors.get(tipo, "#666"), size=10, symbol="x"),
                    ))

            fig.update_layout(
                height=450,
                xaxis_title="Data",
                yaxis_title="Petróleo (bbl/dia)",
                hovermode="x unified",
            )
            st.plotly_chart(fig, use_container_width=True)

        # Table
        st.subheader("Detalhamento")
        display_df = anomalies.copy()
        if "data" in display_df.columns:
            display_df["data"] = display_df["data"].dt.strftime("%Y-%m")
        if "variacao_pct" in display_df.columns:
            display_df["variacao_pct"] = display_df["variacao_pct"].round(1)
        st.dataframe(
            display_df.rename(columns={
                "campo": "Campo", "data": "Data", "producao": "Produção (bbl/dia)",
                "producao_anterior": "Produção anterior", "variacao_pct": "Variação (%)",
                "tipo": "Tipo",
            }),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("Nenhuma anomalia detectada com os filtros selecionados.")


# ══════════════════════════════════════════════════════════════════════════════
# Tab 4: Field Comparison
# ══════════════════════════════════════════════════════════════════════════════

with tab4:
    st.header("Comparação de Campos")
    st.caption("Curvas de ramp-up normalizadas pelo mês de primeiro óleo")

    with st.expander("O que estou vendo?"):
        st.markdown(
            "Compara como diferentes campos subiram de produção. "
            "Todas as curvas começam no 'mês zero' (primeiro óleo), "
            "permitindo comparar mesmo que os campos tenham começado em anos diferentes. "
            "Um ramp-up mais rápido geralmente indica melhor planejamento operacional."
        )

    compare_fields = st.multiselect(
        "Campos para comparar",
        options=selected_fields if selected_fields else top_fields[:15],
        default=(selected_fields[:3] if selected_fields else top_fields[:3]),
        key="compare_fields",
    )

    if compare_fields:
        curves_df, metrics = compute_rampup(df, compare_fields)

        if not curves_df.empty:
            fig = px.line(
                curves_df,
                x="months_since_first_oil",
                y="production",
                color="campo",
                labels={
                    "months_since_first_oil": "Meses desde primeiro óleo",
                    "production": "Petróleo (bbl/dia)",
                    "campo": "Campo",
                },
            )
            fig.update_layout(height=500, hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

            # Metrics table
            st.subheader("Métricas de Ramp-up")
            metrics_data = []
            for m in metrics:
                metrics_data.append({
                    "Campo": m.campo,
                    "Primeiro óleo": m.first_oil_date.strftime("%b/%Y"),
                    "Produção máxima (bbl/dia)": f"{m.peak_production:,.0f}",
                    "Meses até pico": m.months_to_peak,
                    "Taxa de ramp-up (bbl/dia/mês)": f"{m.ramp_up_rate:,.0f}",
                    "Produção atual (bbl/dia)": f"{m.current_production:,.0f}",
                    "Meses de produção": m.total_months,
                })
            st.dataframe(
                pd.DataFrame(metrics_data),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("Dados insuficientes para os campos selecionados.")
    else:
        st.info("Selecione campos para comparar.")


# ── Footer ───────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    "Dados: [ANP - Produção por Poço](https://www.gov.br/anp/pt-br/centrais-de-conteudo/"
    "dados-abertos/producao-de-petroleo-e-gas-natural-por-poco) | "
    "Feito por [Gabriel Furiati](https://github.com/Furiatii)"
)
