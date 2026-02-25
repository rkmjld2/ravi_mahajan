# app.py - Blood Report Analyzer (FINAL VERSION - GitHub SAFE)
import streamlit as st
import pandas as pd
from io import StringIO
import time
from datetime import datetime
import mysql.connector
from pathlib import Path

# LangChain imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
#from langchain.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
#from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain_groq import ChatGroq

st.set_page_config(page_title="🩸 Blood Report Analyzer", layout="wide")

# ── 1. SECURE SECRETS CHECK (GitHub SAFE - NO credentials shown)
required_secrets = ["DB_HOST", "DB_PORT", "DB_USER", "DB_PASSWORD", "DB_NAME", "GROQ_API_KEY"]
missing = [s for s in required_secrets if s not in st.secrets]
if missing:
    st.error(f"🚨 Missing {len(missing)} required secrets")
    st.info("""
    **FIX: Streamlit Cloud → Settings → Secrets → Paste from README.md**
    1. Click "Settings" tab
    2. Scroll to "Secrets" section  
    3. Copy-paste 6 lines from README.md
    4. Click "Save" → Refresh app
    """)
    st.stop()

st.session_state.groq_api_key = st.secrets["GROQ_API_KEY"]

# ── 2. SSL CERTIFICATE SETUP
def setup_ssl_cert():
    cert_path = "isrgrootx1.pem"
    if not Path(cert_path).exists():
        cert_content = st.secrets.get("TIDB_SSL_CA", "")
        if cert_content:
            Path(cert_path).write_text(cert_content)
            st.success("✅ SSL cert created from secrets")
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
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

embeddings = load_embeddings()

# ── 5. SESSION STATE
if "rag_chain" not in st.session_state: st.session_state.rag_chain = None
if "messages" not in st.session_state: st.session_state.messages = []
if "df" not in st.session_state: st.session_state.df = None

# ── 6. UI
st.title("🩸 Blood Report Analyzer – Groq + TiDB Cloud")
st.caption("✅ Production-ready | 🔒 Secure secrets | 💾 Saves to your TiDB database")

with st.sidebar:
    st.markdown("### ✅ Status")
    st.success("All systems ready")
    st.info("Paste report → Edit → Process → Ask AI")

tab1, tab2 = st.tabs(["📊 Upload & Analyze", "ℹ️ Instructions"])

with tab1:
    raw_text = st.text_area(
        "1. Paste blood report (CSV format)",
        height=250,
        value="""Test,Result,Unit,Reference Range,Flag
Hemoglobin,12.4,g/dL,13.0-17.0,L
WBC,8.2,10^3/µL,4.0-11.0,
Glucose Fasting,102,mg/dL,70-99,H
Creatinine,1.1,mg/dL,0.6-1.2,
ALT,45,U/L,7-56,
Total Cholesterol,210,mg/dL,<200,H""",
        help="Copy table from PDF/Excel/WhatsApp"
    )

    if st.button("🔍 2. Parse Table", type="primary", use_container_width=True):
        if raw_text.strip():
            try:
                df = pd.read_csv(StringIO(raw_text), sep=None, engine="python")
                df = df.dropna(how="all")
                st.session_state.df = df
                st.success(f"✅ Parsed {len(df)} tests")
            except Exception as e:
                st.error(f"❌ Parse error: {str(e)}")

    if st.session_state.df is not None:
        st.subheader("3. ✏️ Edit Results")
        edited_df = st.data_editor(
            st.session_state.df,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "Test": st.column_config.TextColumn("Test name", required=True),
                "Result": st.column_config.NumberColumn("Result", step=0.01),
                "Unit": st.column_config.TextColumn("Unit"),
                "Reference Range": st.column_config.TextColumn("Reference range"),
                "Flag": st.column_config.SelectboxColumn(
                    "Flag", options=["", "H", "L", "H*", "L*", "Abnormal"]
                ),
            }
        )

        if st.button("🚀 4. Process & Save to TiDB", type="primary", use_container_width=True):
            with st.spinner("Building AI + Saving to database..."):
                # AI RAG Chain (your original logic)
                lines = ["Test | Result | Unit | Reference Range | Flag"]
                for _, row in edited_df.iterrows():
                    row_str = " | ".join(str(val) for val in row if pd.notna(val))
                    lines.append(row_str)
                
                full_text = "\n".join(lines)
                splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
                chunks = splitter.split_text(full_text)
                docs = [Document(page_content=ch) for ch in chunks]
                
                vectorstore = FAISS.from_documents(docs, embeddings)
                retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

                prompt = ChatPromptTemplate.from_template("""
You are a lab assistant. Answer using ONLY the report data below.
Never diagnose diseases. Report values, flags, ranges only.
##added here
 # New prompt for recommendations

                    rec_prompt_template = """You are a general health information assistant.
Based ONLY on the abnormal lab values below:

For each abnormal value:
- Suggest common lifestyle and diet changes
- Mention the most common medicine class doctors sometimes consider
- If the condition is very well-known, you may give 1–2 extremely common generic medicine examples (only ferrous sulfate for iron, metformin for glucose, atorvastatin/rosuvastatin for cholesterol — nothing else)
- ALWAYS start medicine mention with: "Doctors sometimes consider medicines from the class of..."
- NEVER use words like "take", "prescribe", "you should", "recommended dose"
- NEVER give dosage, duration, brand names, or any instruction to use medicine

MANDATORY ENDING (must appear exactly):
"This is NOT medical advice. NEVER take any medicine based on this information. Only a qualified doctor can diagnose you, decide if any treatment is needed, and prescribe the correct medicine if appropriate."

Abnormal values from report:
{abnormal_context}

Answer in bullet points, be concise and cautious."""


# close here 
Report: {context}
Question: {input}
Answer (include units/flags):""")

                llm = ChatGroq(
                    model="llama-3.3-70b-versatile",
                    temperature=0.1,
                    api_key=st.session_state.groq_api_key
                )
                qa_chain = create_stuff_documents_chain(llm, prompt)
                st.session_state.rag_chain = create_retrieval_chain(retriever, qa_chain)

            # Save to YOUR TiDB database (matches medical1_app.sql)
            try:
                conn = get_db_connection()
                cursor = conn.cursor()
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                inserted_count = 0
                for _, row in edited_df.iterrows():
                    cursor.execute("""
                        INSERT INTO blood_reports 
                        (timestamp, test_name, result, unit, ref_range, flag) 
                        VALUES (%s, %s, %s, %s, %s, %s)
                    """, (
                        timestamp, 
                        row.get("Test", ""), 
                        float(row.get("Result", 0)),
                        row.get("Unit", ""), 
                        row.get("Reference Range", ""),
                        row.get("Flag", "")
                    ))
                    inserted_count += 1
                conn.commit()
                conn.close()
                st.success(f"✅ AI ready! 💾 Saved {inserted_count} tests to TiDB!")
            except Exception as e:
                st.error(f"❌ Database error: {str(e)}")

    # Chat interface
    if st.session_state.rag_chain:
        st.markdown("---")
        st.subheader("5. 💬 Ask about your report")
        
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if query := st.chat_input("What do you want to know? (e.g. 'Is cholesterol high?')"):
            st.session_state.messages.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(query)

            with st.chat_message("assistant"):
                with st.spinner("AI analyzing..."):
                    response = st.session_state.rag_chain.invoke({"input": query})
                    answer = response["answer"].strip()
                    st.markdown(answer)
            
            st.session_state.messages.append({"role": "assistant", "content": answer})

with tab2:
    st.markdown("""
    ### How to use:
    1. **Paste** blood test report (PDF/Excel/WhatsApp)
    2. **Parse** → Edit values in table  
    3. **Process** → AI analyzes + saves to TiDB
    4. **Ask** questions about your results
    5. **Download** Q&A session
    
    ### Your TiDB database receives:
    ```sql
    INSERT INTO blood_reports (timestamp, test_name, result, unit, ref_range, flag)
    ```
    """)

