import os

import streamlit as st

from src.config import Configuration

from pgs.utils import get_configuration_from_env, get_embedder, get_knowledge_graph, get_responder

st.set_page_config(
    page_title="Chat",
    page_icon="🦜",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    ## 🦜 Chat With Knowledge Graph 🕸️

    After building a Knowledge Graph from your documents, you are now able to ask it questions.  
    The agent in charge to answer you is able to generate answers
    * grounded by similarity search on document chunks 
    * querying the Graph in its own native language, Cypher
    * querying for [communities](https://en.wikipedia.org/wiki/Louvain_method) inside the Graph 
    * looking for communities subgraphs
    * combining the previous approaches 
    """
)

CONF_PATH = f"{os.getcwd()}/configuration.json"

env = False
conf = None

st.session_state["answer_method"] = None
st.session_state["community_to_use"] = None
st.session_state["adjacent_chunks"] = False

answering_options = ["Similarity Search", "Cypher", "Communities", "Subgraph", "Combine"]
community_options = ["leiden", "louvain"]

try:
    conf = Configuration.from_file(CONF_PATH)
except Exception as e:
    conf = get_configuration_from_env()
    
if conf:
    embedder = get_embedder(conf.embedder_conf)

    knowledge_graph = get_knowledge_graph(conf, embedder)

    responder = get_responder(conf, knowledge_graph)

    if knowledge_graph._driver.verify_authentication():
        
        with st.expander(label="**Graph Metrics**", icon="📊", expanded=True):
            #TODO cache data here 
            a, b, c, d = st.columns(4, vertical_alignment="center")
            e, f, g, h = st.columns(4, vertical_alignment="center")

            a.metric(
                label="Docs in Graph",
                help="Number of documents ingested into the Knowledge Graph",
                value=knowledge_graph.number_of_docs,
            )
            b.metric(
                label="Labels in Graph",
                help="Number of entity Labels in the Knowledge Graph", 
                value=knowledge_graph.number_of_labels,
            )
            c.metric(
                label="Nodes",
                help="Number of Entities in the Graph", 
                value=knowledge_graph.number_of_nodes,
            )
            d.metric(
                label="Relationships",
                help="Number of Relationships between entities", 
                value=knowledge_graph.number_of_relationships,
            )
            e.metric(
                label="Leiden Communities",
                help="Number of Leiden Communities in the Graph",
                value=knowledge_graph.number_of_leiden_communities
            )
            f.metric(
                label="Leiden Modularity",
                help="The higher it is, the more connected communities in the Graph",
                value=round(knowledge_graph.leiden_modularity, 4)
            )
            g.metric(
                label="Louvain Communities",
                help="Number of Louvain Communities in the Graph",
                value=knowledge_graph.number_of_louvain_communities
            )
            h.metric(
                label="Louvain Modularity",
                help="The higher it is, the more connected communities in the Graph",
                value=round(knowledge_graph.louvain_modularity, 4)
            )


with st.sidebar:
    st.session_state["answer_method"] = st.radio(
        label="Select Answering Method",
        options=answering_options,
        help="Choose the method the Graph Agent will use to answer"
    )
    
    st.session_state["adjacent_chunks"] = st.checkbox(
        label="Use neighbouring Chunks",
        help="When performing similarity-based generation, the Agent will also look for Chunk nearing each other in the same Document."
    )

    st.session_state["community_to_use"] = st.selectbox(
        label="Select the reference Community",
        options=community_options, 
        help="Looking for Communities and Subgraphs requires to select one"
    )

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi, you can ask me questions about the Documents in the Knowledge Graph"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
    
# Accept user input
if prompt := st.chat_input("What are the available nodes in the Graph?"):
    
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
        
    chat_history = ""
    for m in st.session_state.messages:
        if m["role"] == "user":
            chat_history += f"User: {m['content']}\n"
        elif m["role"] == "assistant":
            chat_history += f"Assistant: {m['content']}\n"
        
    if st.session_state["answer_method"] == "Similarity Search":
        response = responder.answer_with_context(
            query=prompt, 
            use_adjacent_chunks=st.session_state["adjacent_chunks"],
            history=chat_history
        )
    elif st.session_state["answer_method"] == "Cypher":
        response = responder.answer_with_enhanced_cypher(
            query=prompt, 
            intermediate_steps=False,
            history=chat_history
        )
    elif st.session_state["answer_method"] == "Communities":
        response = responder.answer_with_community_reports(
            query=prompt, 
            use_adjacent_chunks=st.session_state["adjacent_chunks"],
            community_type=st.session_state["community_to_use"]
        )
    elif st.session_state["answer_method"] == "Subgraph":
        response = responder.answer_with_community_subgraph(
            query=prompt, 
            community_type=st.session_state["community_to_use"]
        )
    else:
        response = responder.answer(
            query=prompt, 
            use_adjacent_chunks=st.session_state["adjacent_chunks"]
        )
        
    with st.chat_message("assistant"):
        st.write(response)
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Add response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

           