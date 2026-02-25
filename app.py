# app.py - Blood Report Analyzer (MODIFIED - compact abnormal format)

import streamlit as st
import pandas as pd
from io import StringIO
from datetime import datetime
import mysql.connector
from pathlib import Path

# LangChain imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain_groq import ChatGroq

st.set_page_config(page_title="🩸 Blood Report Analyzer", layout="wide")

# ── 1. SECURE SECRETS CHECK
required_secrets = ["DB_HOST", "DB_PORT", "DB_USER", "DB_PASSWORD", "DB_NAME", "GROQ_API_KEY"]
missing = [s for s in required_secrets if s not in st.secrets]
if missing:
    st.error(f"🚨 Missing {len(missing)} required secrets")
    st.stop()

# ── 2. SSL CERTIFICATE SETUP
def setup_ssl_cert():
    cert_path = "isrgrootx1.pem"
    if not Path(cert_path).exists():
        cert_content = st.secrets.get("TIDB_SSL_CA", "")
        if cert_content:
            Path(cert_path).write_text(cert_content)
    return cert_path

# ── 3. DATABASE CONNECTION
@st.cache_resource
def get_db_connection():
    ssl_ca_path = setup_ssl_cert()
    conn = mysql.connector.connect(
        host=st.secrets["DB_HOST"],
        port=int(st.secrets["DB_PORT"]),
        user=st.secrets["DB_USER"],
        password=st.secrets["DB_PASSWORD"],
        database=st.secrets["DB_NAME"],
        ssl_ca=ssl_ca_path,
        ssl_verify_cert=True,
        ssl_verify_identity=True,
        connect_timeout=30
    )
    return conn

# ── 4. EMBEDDINGS
@st.cache_resource(show_spinner=False)
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

embeddings = load_embeddings()

# ── 5. SESSION STATE
if "rag_chain" not in st.session_state: st.session_state.rag_chain = None
if "messages" not in st.session_state: st.session_state.messages = []
if "df" not in st.session_state: st.session_state.df = None

# ── 6. UI
st.title("🩸 Blood Report Analyzer – Groq + TiDB")
st.caption("Paste report → Edit → Process → Ask questions")

with st.sidebar:
    st.success("All systems ready")
    st.info("Paste report in CSV-like format")

tab1, tab2 = st.tabs(["📊 Analyze Report", "ℹ️ Instructions"])

with tab1:
    default_text = """Test,Result,Unit,Reference Range,Flag
Hemoglobin,12.4,g/dL,13.0-17.0,L
WBC,8.2,10^3/µL,4.0-11.0,
Glucose Fasting,102,mg/dL,70-99,H
Creatinine,1.1,mg/dL,0.6-1.2,
ALT,45,U/L,7-56,
Total Cholesterol,210,mg/dL,<200,H"""

    raw_text = st.text_area("1. Paste your blood report here", height=220, value=default_text)

    if st.button("🔍 2. Parse Table", type="primary", use_container_width=True):
        if raw_text.strip():
            try:
                df = pd.read_csv(StringIO(raw_text), sep=None, engine="python")
                df = df.dropna(how="all").fillna("")
                st.session_state.df = df
                st.success(f"✅ Parsed {len(df)} tests")
            except Exception as e:
                st.error(f"Parse error: {e}")

    if st.session_state.df is not None:
        st.subheader("3. Edit values if needed")

        edited_df = st.data_editor(
            st.session_state.df,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "Test": st.column_config.TextColumn("Test", required=True),
                "Result": st.column_config.NumberColumn("Result", step=0.01, format="%.1f"),
                "Unit": st.column_config.TextColumn("Unit"),
                "Reference Range": st.column_config.TextColumn("Reference range"),
                "Flag": st.column_config.SelectboxColumn("Flag", options=["", "H", "L", "H*", "L*", "Abnormal"]),
            }
        )

        if st.button("🚀 4. Process & Save", type="primary", use_container_width=True):
            with st.spinner("Processing report & building AI assistant..."):

                # ── Build compact abnormal lines (your requested format) ──
                abnormal_lines = []

                for _, row in edited_df.iterrows():
                    test = row["Test"].strip()
                    result = row["Result"]
                    unit = row["Unit"].strip()
                    ref = str(row["Reference Range"]).strip()
                    flag = str(row["Flag"]).strip().upper()

                    if flag in ["H", "L", "H*", "L*", "ABNORMAL"]:
                        result_str = f"{result:.1f} {unit}" if unit else f"{result:.1f}"
                        
                        # Clean reference range display
                        ref_clean = ref.replace(" - ", "-").replace(" -", "-").replace("- ", "-")
                        if ref_clean.startswith(("<", ">")):
                            range_part = ref_clean
                        else:
                            range_part = ref_clean.replace("-", " - ")

                        line = f"{test}: {result_str} ({range_part}, {flag})"
                        abnormal_lines.append(f"• {line}")

                abnormal_summary = "\n".join(abnormal_lines)
                if not abnormal_summary:
                    abnormal_summary = "• No abnormal values found."

                # ── Build full report text for vector store ──
                report_lines = ["Test | Result | Unit | Reference Range | Flag"]
                for _, row in edited_df.iterrows():
                    vals = [str(row[c]) for c in edited_df.columns if pd.notna(row[c]) and str(row[c]).strip()]
                    report_lines.append(" | ".join(vals))

                full_report_text = "\n".join(report_lines)

                # ── RAG setup ──
                splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
                chunks = splitter.split_text(full_report_text)
                docs = [Document(page_content=ch) for ch in chunks]

                vectorstore = FAISS.from_documents(docs, embeddings)
                retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

                # Improved prompt with compact abnormal summary at the top
                prompt = ChatPromptTemplate.from_template(
                    """You are a careful lab report assistant.
Answer using ONLY the provided blood report data.
Never diagnose diseases. Never give medical advice.
Only report values, units, ranges and flags.

Abnormal results (most important):
{abnormal_summary}

Full report data:
{context}

Question: {input}

Answer concisely and clearly (include units and flags when relevant):"""
                )

                llm = ChatGroq(
                    model="llama-3.3-70b-versatile",
                    temperature=0.1,
                    api_key=st.secrets["GROQ_API_KEY"]
                )

                question_answer_chain = create_stuff_documents_chain(llm, prompt)
                rag_chain = create_retrieval_chain(retriever, question_answer_chain)

                st.session_state.rag_chain = rag_chain
                st.session_state.abnormal_summary = abnormal_summary   # optional - for later use

                # ── Save to TiDB ──
                try:
                    conn = get_db_connection()
                    cursor = conn.cursor()
                    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    inserted = 0
                    for _, row in edited_df.iterrows():
                        cursor.execute("""
                            INSERT INTO blood_reports
                            (timestamp, test_name, result, unit, ref_range, flag)
                            VALUES (%s, %s, %s, %s, %s, %s)
                        """, (
                            ts,
                            row["Test"],
                            float(row["Result"]) if pd.notna(row["Result"]) else None,
                            row["Unit"],
                            row["Reference Range"],
                            row["Flag"]
                        ))
                        inserted += 1

                    conn.commit()
                    conn.close()
                    st.success(f"✅ AI ready — {inserted} tests saved")
                except Exception as e:
                    st.error(f"Database error: {e}")

        # ── Chat interface ──
        if st.session_state.get("rag_chain"):
            st.markdown("---")
            st.subheader("5. Ask questions about this report")

            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            if query := st.chat_input("Example: Why is hemoglobin low?  or  Is cholesterol high?"):
                st.session_state.messages.append({"role": "user", "content": query})
                with st.chat_message("user"):
                    st.markdown(query)

                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        try:
                            response = st.session_state.rag_chain.invoke({
                                "input": query,
                                "abnormal_summary": st.session_state.get("abnormal_summary", "No abnormal summary available.")
                            })
                            answer = response["answer"].strip()
                            st.markdown(answer)
                            st.session_state.messages.append({"role": "assistant", "content": answer})
                        except Exception as e:
                            st.error(f"AI error: {e}")

with tab2:
    st.markdown("""
    ### Quick guide
    1. Paste blood report (copy from PDF/Excel/WhatsApp as CSV-like text)
    2. Click Parse Table
    3. Edit values if needed
    4. Click Process & Save
    5. Ask any question about your results

    Abnormal values are shown in compact format like:  
    **Hemoglobin: 12.4 g/dL (13.0 - 17.0, L)**

    The AI only uses your report — no diagnosis, no advice.
    """)
