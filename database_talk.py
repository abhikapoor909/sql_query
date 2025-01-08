import streamlit as st
from langchain.agents import create_sql_agent
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_types import AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="LangChain: Chat with SQL DB", page_icon="🦜")
st.title("🦜 LangChain: Chat with SQL DB")

# MySQL connection inputs
mysql_host = st.sidebar.text_input("Provide MySQL Host", value="localhost")
mysql_user = st.sidebar.text_input("MySQL User", value="root")
mysql_password = st.sidebar.text_input("MySQL Password (leave blank if none)", type="password")
mysql_db = st.sidebar.text_input("MySQL Database")

if not (mysql_host and mysql_user and mysql_db):
    st.info("Please enter all required MySQL database details.")
    st.stop()

# Configure the LLM
from langchain_groq import ChatGroq
llm = ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key)

# Function to configure the MySQL database
@st.cache_resource(ttl="2h")
def configure_mysql_db(host, user, password, db):
    try:
        if password:
            connection_string = f"mysql+pymysql://{user}:{password}@{host}/{db}"
        else:
            connection_string = f"mysql+pymysql://{user}@{host}/{db}"

        return SQLDatabase(create_engine(connection_string))
    except Exception as e:
        st.error(f"Failed to connect to the database: {e}")
        st.stop()

# Configure the database
db = configure_mysql_db(mysql_host, mysql_user, mysql_password, mysql_db)

# Set up the toolkit and agent
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
)

# Chat interface
if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# User input
user_query = st.chat_input(placeholder="Ask anything from the database")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        streamlit_callback = StreamlitCallbackHandler(st.container())
        try:
            response = agent.run(user_query, callbacks=[streamlit_callback])
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.write(response)
        except Exception as e:
            st.error(f"An error occurred: {e}")