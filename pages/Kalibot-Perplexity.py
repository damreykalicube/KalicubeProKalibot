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
import requests
import time

st.set_page_config(page_title="Kalibot Version 2", page_icon="🤖")

st.title("Kalibot Version 2")
git_tok = Github(os.environ["github_key"])

pinecone.init(
    api_key= os.environ["pinecone_secret_key"],
    environment="gcp-starter"
)
openai.api_key = os.environ["openai_secret_key"]
index_name = 'kalicube-test'
index = pinecone.Index(index_name)

primer = os.environ["primer_content"]

def perplexity_call(messages, output_placeholder):
    
    temptext = ""
    perplexity_api_key=os.environ["perplexity_key"]
    url = "https://api.perplexity.ai/chat/completions"
    headers = {
        "Authorization": f"Bearer {perplexity_api_key}",
        "Content-Type": "application/json"
    }
    # Request payload
    payload = {
        "model": "llama-3.1-sonar-small-128k-online",
        "messages": messages,
        "temperature": 0.2,
        "stream": True  # Enable streaming
    }
    #message_placeholder = st.empty()
    response = requests.post(url, headers=headers, json=payload, stream=True)
    #print(response.text)
    if response.status_code == 200:
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8').strip()
                if line.startswith("data: "):
                    line = line[6:]
                if line == "[DONE]":
                    break
                try:
                    result = json.loads(line)
                    if 'choices' in result and len(result['choices']) > 0:
                        delta = result['choices'][0].get('delta', "")
                        if 'content' in delta:
                            temptext += delta['content']
                            output_placeholder.markdown(temptext + " ")
                except json.JSONDecodeError:
                    print(f"Error decoding JSON: {line}")
        return temptext
    else:
        print(f"Error: {response.status_code}")
        #print(response.text)


def update_json_file(git_tok, tail_id, user_prompt, bot_answer, token_usage):
    
    france_tz = pytz.timezone("Europe/Paris")
    time_france = datetime.now(france_tz)
    dt_string = time_france.strftime("%d/%m/%Y %H:%M:%S")
    
    json_main_structure =   {
        "promptId": tail_id+1,
        "promptDateTime": dt_string,
        "prompt": "PERPLEXITY-"+user_prompt,
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

    augmented_query = "\n\n---\n\n".join(contexts[:10])+"\n\n-----\n\n"+query
    
    return augmented_query

if "openai_model" not in st.session_state:
    #st.session_state["openai_model"] = "gpt-4-1106-preview"
    st.session_state["openai_model"] = "gpt-4o-2024-05-13"
    st.session_state["perplexity_model"] = "llama-3.1-sonar-small-128k-online"
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
        
        #print(f"{num_tokens_from_string(primer)}")
    with st.chat_message("assistant"):
        output_placeholder = st.empty()
        full_response = ""
        messages=[
                {"role": "system", "content": primer},
                {"role": "user", "content": augmented_query}
        ]
        full_response = perplexity_call(messages, output_placeholder)
        totalToken += num_tokens_from_string(full_response)
        #st.markdown(full_response)
        update_json_file(git_tok, current_id_tail, prompt, full_response, totalToken)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
