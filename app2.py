import streamlit as st
import os
import json
import sqlite3
from memory_processor import MemoriaChatProcessor

# Inicialização das estruturas
os.makedirs("db", exist_ok=True)
os.makedirs("memorias", exist_ok=True)
os.makedirs("templates", exist_ok=True)

DB_PATH = "db/jesus_chat_memorias.sqlite"

st.set_page_config(page_title="Jesus Chat Memórias", layout="wide")

# Hero-style topo visual
st.markdown("""
    <div style='background-color: #00008B; padding: 2rem; border-radius: 10px; text-align: center;'>
        <h1 style='color: white; font-size: 3rem;'>✝️ JESUS CHAT MEMÓRIAS</h1>
        <p style='color: #87CEEB; font-size: 1.2rem;'>Transformando conversas em sabedoria espiritual</p>
    </div>
    <br>
""", unsafe_allow_html=True)

# Navegação principal
aba = st.sidebar.radio("📂 Menu Principal", ["Início", "Processar JSON", "Memórias", "Clusters", "Entidades", "Grafo Semântico", "Métricas"])

if aba == "Início":
    st.header("📜 Introdução")
    st.markdown("""
    Esta plataforma permite:
    - Enviar conversas em JSON
    - Gerar resumos e métricas
    - Visualizar memórias já salvas
    - Navegar por entidades e clusters
    - Ver a estrutura semântica da conversa
    """)

elif aba == "Processar JSON":
    st.header("📤 Enviar JSON da Conversa")
    st.markdown("Faça o upload de um arquivo `.json` gerado por uma conversa com IA.")
    file = st.file_uploader("Clique abaixo para enviar", type=["json"], label_visibility="collapsed")
    if file:
        try:
            data = json.load(file)
            processor = MemoriaChatProcessor()
            result = processor.process_template(data)
            st.success("Memória processada!")

            col1, col2 = st.columns(2)
            with col1:
                st.metric("🎯 Tópicos", len(result['summary'].get('topics', [])))
                st.metric("🧠 Entidades", len(result['summary'].get('entities', [])))
            with col2:
                st.metric("🔢 Tokens", result['metrics']['total_tokens'])
                st.metric("📅 Data", result['metadata']['timestamp'])

            st.subheader("🧠 Resumo")
            st.markdown(result['summary']['brief'])

            st.download_button("⬇️ Baixar HTML", data=result["html_view"], file_name="memoria.html")

        except Exception as e:
            st.error(f"Erro: {e}")

elif aba == "Memórias":
    st.header("📁 Memórias Geradas")
    htmls = [f for f in os.listdir("memorias") if f.endswith(".html")]
    if not htmls:
        st.warning("Nenhuma memória encontrada.")
    for file in htmls:
        with st.expander(f"📄 {file}"):
            st.markdown(f"- [Abrir no navegador](memorias/{file})", unsafe_allow_html=True)
            st.code(open(f"memorias/{file}", encoding="utf-8").read()[:500] + "...", language='html')

elif aba == "Clusters":
    st.header("🔷 Clusters de Tópicos")
    if os.path.exists(DB_PATH):
        conn = sqlite3.connect(DB_PATH)
        clusters = conn.execute("SELECT id, name, keywords, importance FROM topic_clusters").fetchall()
        for c in clusters:
            st.markdown(f"**{c[1]}** — *Importância:* {c[3]:.2f}<br>Palavras-chave: `{c[2]}`", unsafe_allow_html=True)
        conn.close()
    else:
        st.info("Banco de dados não encontrado.")

elif aba == "Entidades":
    st.header("🧬 Entidades Detectadas")
    if os.path.exists(DB_PATH):
        conn = sqlite3.connect(DB_PATH)
        entities = conn.execute("SELECT name, type, mentions FROM entities ORDER BY mentions DESC").fetchall()
        for e in entities:
            st.markdown(f"- **{e[0]}** (*{e[1]}*) — {e[2]} ocorrência(s)")
        conn.close()
    else:
        st.info("Banco não disponível.")

elif aba == "Grafo Semântico":
    st.header("🕸️ Estrutura de Conceitos (Nós e Ligações)")
    if os.path.exists(DB_PATH):
        conn = sqlite3.connect(DB_PATH)
        nodes = conn.execute("SELECT id, label, type FROM nodes").fetchall()
        edges = conn.execute("SELECT source, target, relationship FROM edges").fetchall()
        st.markdown("**Nós**")
        for n in nodes:
            st.markdown(f"- `{n[0]}`: {n[1]} ({n[2]})")
        st.markdown("**Ligações**")
        for e in edges:
            st.markdown(f"- {e[0]} → {e[1]} : *{e[2]}*")
        conn.close()
    else:
        st.warning("Banco de dados ausente.")

elif aba == "Métricas":
    st.header("📊 Métricas Consolidadas")
    if os.path.exists(DB_PATH):
        conn = sqlite3.connect(DB_PATH)
        met = conn.execute("SELECT * FROM metrics").fetchall()
        for row in met:
            st.markdown(f"- Chat `{row[0]}`: Resolução={row[1]:.2f}, Precisão={row[2]:.2f}, Completude={row[3]:.2f}, Eficiência={row[4]:.2f}")
        conn.close()
    else:
        st.warning("Banco de dados não carregado.")
