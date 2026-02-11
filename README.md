# 🩸 Blood Report Analyzer
## Quick Deploy

1. Fork this repo
2. Deploy on Streamlit Cloud
3. Add secrets (Settings → Secrets):

```toml
[connections.databases.default]
host     = "gateway01.ap-southeast-1.prod.aws.tidbcloud.com"
port     = 4000
username = "ax6KHc1BNkyuaor.root"
password = "your-database-password"
database = "medical1_app"

[secrets]
GROQ_API_KEY = "your-groq-api-key"
TIDB_SSL_CA = '''
-----BEGIN CERTIFICATE-----
...your certificate block...
-----END CERTIFICATE-----
'''

PINECONE_API_KEY = "your-pinecone-api-key"
PINECONE_ENV = "us-east1-gcp"
LANGCHAIN_API_KEY = "your-langchain-api-key"
LANGCHAIN_PROJECT = "rag-app"
LANGCHAIN_TRACING = "true"


