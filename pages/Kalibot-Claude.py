import openai
import streamlit as st
import pinecone
import tiktoken
import json
from datetime import datetime
from datetime import datetime
import pytz
import os
from github import Github
import time
import google.generativeai as genai
import anthropic

st.set_page_config(page_title="Kalibot-Claude", page_icon="🤖")

st.title("Kalibot-Claude")

##### APIS #####
git_tok = Github(os.environ["github_key"])
openai.api_key = os.environ["openai_secret_key"]
genai.configure(api_key=os.environ["gemini_key"])

anthropicclient = anthropic.Anthropic(
    # defaults to os.environ.get("ANTHROPIC_API_KEY")
    api_key=os.environ["claude_key"],
)

pinecone.init(
    api_key= os.environ["pinecone_secret_key"],  # app.pinecone.io (console)
    environment="gcp-starter"  # next to API key in console
)
index_name = 'kalicube-test'
index = pinecone.Index(index_name)
################

primer = os.environ["primer_content"]

def update_json_file(git_tok, tail_id, user_prompt, bot_answer, token_usage):
    
    france_tz = pytz.timezone("Europe/Paris")
    time_france = datetime.now(france_tz)
    dt_string = time_france.strftime("%d/%m/%Y %H:%M:%S")
    
    json_main_structure =   {
        "promptId": tail_id+1,
        "promptDateTime": dt_string,
        "prompt": 'KALICUBEPRO:CLAUDE-'+user_prompt,
        "promptAnswer": bot_answer,
        "totalTokenUsage": token_usage
    }

    repo = git_tok.get_repo("damreykalicube/KalibotTesting")
    file_content = repo.get_contents('promptdata.json')
    data = file_content.decoded_content.decode()
    data = json.loads(data)
    data.append(json_main_structure)
    
    final_json_data = json.dumps(data, indent=4, separators=(',',': '))
    #repo.delete_file(file_content.path, "removed need update", file_content.sha, branch="main")
    #repo.create_file("promptdata.json", "commit", final_json_data)
    repo.update_file("promptdata.json", "commit", final_json_data, file_content.sha)
    
def init_json_file(git_tok): # getting the promptId
    repo = git_tok.get_repo("damreykalicube/KalibotTesting")
    file_content = repo.get_contents('promptdata.json')
    data = file_content.decoded_content.decode()
    data = json.loads(data)
    
    #print(data[-1])
    return data[-1]["promptId"]

def num_tokens_from_string(sentences):
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(sentences))
    return num_tokens

def process_vector_query(prompt, index):
    query = prompt
    embed_model = "text-embedding-ada-002"
    res = openai.Embedding.create(
        input=[query],
        engine=embed_model
    )
    xq = res['data'][0]['embedding']
    res = index.query(xq, top_k=20, include_metadata=True)
    contexts = []
    
    #contexts = [item['metadata']['text'] for item in res['matches']]
    for item in res['matches']:
      if (item['metadata'].get('completion')):
        contexts.append("".join(item['metadata']['completion']))
      else:
        contexts.append(item['metadata']['text'])

    contexts[:] = (validContext for validContext in contexts if len(validContext.split('.')) > 1)

    augmented_query = "\n\n---\n\n".join(contexts[:5])+"\n\n-----\n\nUsing the context above please do or answer the following "+query
    
    return augmented_query

if "openai_model" not in st.session_state:
    #st.session_state["openai_model"] = "gpt-4-1106-preview"
    st.session_state["openai_model"] = "gpt-4o-2024-05-13"
    st.session_state["gemini_model"] = "gemini-1.5-flash"
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    current_id_tail = init_json_file(git_tok)
    totalToken = 0
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)
        augmented_query = process_vector_query(prompt, index)
        totalToken += num_tokens_from_string(augmented_query) + num_tokens_from_string(primer)
        
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        collected_text = ""

        with anthropicclient.messages.stream(
            messages=[
                {
                    "role": "user",
                    "content": primer+"\n"+augmented_query,
                },
            ],
            model="claude-3-5-sonnet-20240620",
            max_tokens=8192,
            temperature=.2,
        ) as response:
          for chunk in response.text_stream:
                #print(f'{chunk} - end')
                #print("what")
                collected_text += chunk
                message_placeholder.markdown(collected_text + "▌")
        totalToken += num_tokens_from_string(collected_text)
        message_placeholder.markdown(collected_text)
        update_json_file(git_tok, current_id_tail, prompt, collected_text, totalToken)
    st.session_state.messages.append({"role": "assistant", "content": collected_text})
