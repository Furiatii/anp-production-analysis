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
from analysis.watercut import compute_watercut, compute_gor, watercut_summary
from analysis.fpso import fpso_monthly_production, top_installations, fpso_summary

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

tab0, tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Sobre a Análise", "Visão Geral", "Decline Curve", "Anomalias",
    "Comparação de Campos", "Water Cut & GOR", "Visão por FPSO",
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

---

### 5. Water Cut (BSW) e GOR

**Water Cut (BSW)** mede a porcentagem de água no líquido total produzido:

    BSW = água / (água + óleo) × 100%

Todo poço de petróleo produz água junto com o óleo. No início da vida do poço, a proporção
de água é baixa. Com o tempo, a água do reservatório começa a chegar ao poço (water breakthrough)
e o BSW sobe. Quando o BSW fica muito alto (acima de 90-95%), o poço pode deixar de ser
economicamente viável.

**GOR (Gas-Oil Ratio)** mede quanto gás é produzido para cada unidade de óleo:

    GOR = gás (m³/dia) / óleo (m³/dia)

Mudanças no GOR indicam comportamento do reservatório. Um GOR subindo pode significar que
o gás do reservatório está se expandindo (gas cap expansion) ou que o poço está produzindo
de uma zona com mais gás.

| Indicador | Sinal | Possível causa |
|-----------|-------|----------------|
| BSW subindo | Water breakthrough | Reservatório maduro, injeção de água chegando |
| BSW estável | Produção saudável | Boa gestão do reservatório |
| GOR subindo | Mais gás por barril | Expansão do cap de gás, depleção |
| GOR estável | Reservatório estável | Mecanismo de drive consistente |

---

### 6. Visão por FPSO

Agrega a produção de todos os poços conectados a cada instalação (FPSO, plataforma fixa, etc.).

Um FPSO (Floating Production, Storage and Offloading) é a unidade que recebe o petróleo
dos poços submarinos, processa e armazena até ser transferido para navios.

**Métricas por FPSO:**

| Métrica | O que significa |
|---------|-----------------|
| **Produção total** | Soma da produção de todos os poços conectados |
| **Poços ativos** | Quantos poços estão produzindo naquele mês |
| **Produção por poço** | Média de produção por poço conectado |
| **Eficiência operacional** | % do tempo que a plataforma ficou produzindo no mês |
| **BSW** | Water cut da plataforma (água no total líquido) |

**Para que serve:** comparar desempenho entre plataformas e identificar quais operam
com mais eficiência. Por exemplo, o FPSO P-78 (Búzios) pode ser comparado ao P-70
do mesmo campo para avaliar ramp-up.
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


# ══════════════════════════════════════════════════════════════════════════════
# Tab 5: Water Cut & GOR
# ══════════════════════════════════════════════════════════════════════════════

with tab5:
    st.header("Water Cut (BSW) e Gas-Oil Ratio (GOR)")

    with st.expander("O que estou vendo?"):
        st.markdown(
            "**Water Cut (BSW)** = % de água no líquido total. Sobe ao longo da vida do campo. "
            "**GOR** = razão gás/óleo (m³/m³). Mudanças indicam comportamento do reservatório. "
            "BSW subindo = water breakthrough (reservatório maduro). GOR subindo = expansão do cap de gás."
        )

    wc_fields = st.multiselect(
        "Campos",
        options=selected_fields if selected_fields else top_fields[:15],
        default=(selected_fields[:5] if selected_fields else top_fields[:5]),
        key="wc_fields",
    )

    if wc_fields:
        wc_tab, gor_tab = st.tabs(["Water Cut (BSW)", "GOR"])

        with wc_tab:
            wc_df = compute_watercut(df, wc_fields)
            if not wc_df.empty:
                fig_wc = px.line(
                    wc_df,
                    x="data", y="watercut", color="campo",
                    labels={"data": "Data", "watercut": "Water Cut (%)", "campo": "Campo"},
                )
                fig_wc.update_layout(height=500, hovermode="x unified")
                fig_wc.add_hline(y=50, line_dash="dot", line_color="red",
                                 annotation_text="50% BSW", annotation_position="top right")
                st.plotly_chart(fig_wc, use_container_width=True)

                # Summary table
                st.subheader("Resumo por Campo")
                wc_summary = watercut_summary(wc_df)
                if not wc_summary.empty:
                    st.dataframe(
                        wc_summary.rename(columns={
                            "campo": "Campo",
                            "bsw_atual": "BSW atual (%)",
                            "bsw_medio": "BSW médio (%)",
                            "tendencia_pp": "Tendência (p.p.)",
                            "status": "Status",
                        }),
                        use_container_width=True,
                        hide_index=True,
                    )
            else:
                st.info("Sem dados de água para os campos selecionados.")

        with gor_tab:
            gor_df = compute_gor(df, wc_fields)
            if not gor_df.empty:
                fig_gor = px.line(
                    gor_df,
                    x="data", y="gor", color="campo",
                    labels={"data": "Data", "gor": "GOR (m³/m³)", "campo": "Campo"},
                )
                fig_gor.update_layout(height=500, hovermode="x unified")
                st.plotly_chart(fig_gor, use_container_width=True)

                # GOR summary: current + average per field
                gor_summary = []
                for campo, grp in gor_df.groupby("campo"):
                    grp = grp.sort_values("data")
                    gor_summary.append({
                        "Campo": campo,
                        "GOR atual (m³/m³)": f"{grp['gor'].iloc[-1]:,.0f}",
                        "GOR médio (m³/m³)": f"{grp['gor'].mean():,.0f}",
                        "Produção atual (bbl/dia)": f"{grp['oleo_bbl_dia'].iloc[-1]:,.0f}",
                    })
                st.dataframe(pd.DataFrame(gor_summary), use_container_width=True, hide_index=True)
            else:
                st.info("Sem dados de gás para os campos selecionados.")
    else:
        st.info("Selecione campos para analisar.")


# ══════════════════════════════════════════════════════════════════════════════
# Tab 6: FPSO View
# ══════════════════════════════════════════════════════════════════════════════

with tab6:
    st.header("Visão por Instalação (FPSO)")
    st.caption("Produção agregada por plataforma, eficiência operacional e poços ativos")

    with st.expander("O que estou vendo?"):
        st.markdown(
            "Produção total de cada FPSO/plataforma, somando todos os poços conectados. "
            "**Eficiência** = % do tempo que produziu no mês. "
            "**Produção por poço** = média por poço conectado. "
            "Permite comparar plataformas do mesmo campo (ex: P-67 vs P-70 vs P-78 em Búzios)."
        )

    top_inst = top_installations(df, n=30)

    selected_inst = st.multiselect(
        "Instalações",
        options=top_inst,
        default=top_inst[:5] if top_inst else [],
        key="fpso_select",
    )

    if selected_inst:
        fpso_df = fpso_monthly_production(df, selected_inst)

        if not fpso_df.empty:
            # Production over time
            st.subheader("Produção total por instalação")
            fig_fpso = px.line(
                fpso_df,
                x="data", y="oleo_bbl_dia", color="instalacao",
                labels={"data": "Data", "oleo_bbl_dia": "Petróleo (bbl/dia)", "instalacao": "Instalação"},
            )
            fig_fpso.update_layout(height=500, hovermode="x unified")
            st.plotly_chart(fig_fpso, use_container_width=True)

            # Wells and production per well
            col_a, col_b = st.columns(2)

            with col_a:
                st.subheader("Poços ativos")
                fig_wells = px.line(
                    fpso_df,
                    x="data", y="n_pocos", color="instalacao",
                    labels={"data": "Data", "n_pocos": "Poços ativos", "instalacao": "Instalação"},
                )
                fig_wells.update_layout(height=400, hovermode="x unified")
                st.plotly_chart(fig_wells, use_container_width=True)

            with col_b:
                st.subheader("Produção por poço")
                fig_ppw = px.line(
                    fpso_df,
                    x="data", y="oleo_por_poco", color="instalacao",
                    labels={"data": "Data", "oleo_por_poco": "bbl/dia por poço", "instalacao": "Instalação"},
                )
                fig_ppw.update_layout(height=400, hovermode="x unified")
                st.plotly_chart(fig_ppw, use_container_width=True)

            # Efficiency
            st.subheader("Eficiência operacional")
            fig_eff = px.line(
                fpso_df,
                x="data", y="eficiencia", color="instalacao",
                labels={"data": "Data", "eficiencia": "Eficiência (%)", "instalacao": "Instalação"},
            )
            fig_eff.update_layout(height=400, hovermode="x unified")
            fig_eff.add_hline(y=95, line_dash="dot", line_color="green",
                             annotation_text="95% target", annotation_position="top right")
            st.plotly_chart(fig_eff, use_container_width=True)

            # Summary table
            st.subheader("Resumo das Instalações")
            summary = fpso_summary(df, selected_inst)
            if not summary.empty:
                display = summary.copy()
                display["primeiro_registro"] = display["primeiro_registro"].dt.strftime("%b/%Y")
                st.dataframe(
                    display.rename(columns={
                        "instalacao": "Instalação",
                        "prod_atual_bbl_dia": "Produção atual (bbl/dia)",
                        "n_pocos": "Poços ativos",
                        "bsw_pct": "BSW (%)",
                        "eficiencia_pct": "Eficiência (%)",
                        "primeiro_registro": "Primeiro registro",
                    }),
                    use_container_width=True,
                    hide_index=True,
                )
        else:
            st.info("Sem dados para as instalações selecionadas.")
    else:
        st.info("Selecione instalações para analisar.")


# ── Footer ───────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    "Dados: [ANP - Produção por Poço](https://www.gov.br/anp/pt-br/centrais-de-conteudo/"
    "dados-abertos/producao-de-petroleo-e-gas-natural-por-poco) | "
    "Feito por [Gabriel Furiati](https://github.com/Furiatii)"
)
