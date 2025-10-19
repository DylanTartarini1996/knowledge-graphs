from langchain.prompts import PromptTemplate


def get_question_answering_prompt() -> PromptTemplate:
    
    prompt = """ 
        You are a helpful virtual assistant.  
        
        Your task is provide a relevant and precise answer to the user's question, given context information.  
        You might also be given the conversation record of your previous interactions with the user.
        
        Do not make things up or add any information on your own.  
        If the context is not relevant to the user's question, just say that you don't know. 
        Maintain the core information from the context.
        
        CHAT HISTORY: {history}  
        
        QUESTION: {question}
        
        CONTEXT: {context}
        
        HELPFUL ANSWER: 
    """
    
    template = PromptTemplate.from_template(prompt)
    
    template.input_variables = ["history", "question", "context"]
    
    return template
    

def get_rephrase_prompt() -> PromptTemplate:

    prompt = """
        Your task is to rephrase a user's question based on the schema of a Graph Database that will be given to you. 
        You might also be given the conversation record of previous interaction between the user and a virtual assistant, to provide you additional context. 
        
        The schema is made of node labels and relationships available in the Graph. 

        Remember that in a Knowledge Graph there are Documents and Chunks. 
        * a node with label `Document` always has a property `filename` (every Document has a name);
        * a node with label `Chunk` is connected via a `PART_OF` relationship to a node with the `Document` label (Chunks are pieces of text coming from a Document);
        * a node with label `Chunk` always has a `text` property; 
        * a node with label `Chunk` is usually connected to other nodes with label `Chunk` by a `NEXT` relationship (Chunks are ordered in a sequential order);
        * a node with label `Chunk` might be connected to other nodes in the Graph by a `MENTIONS` relationship (text in Chunks might mention some relevant entities). 

        Do not mention anything else, just rephrase the question from the user to be as coherent as possible with the schema of the graph.
        Do not make things up or add any information on your own. 

        CHAT HISTORY: {history}
        AVAILABLE NODE LABELS: {graph_labels}
        AVAILABLE RELATIONSHIPS: {graph_relationships}
        QUESTION: {question}

        REPHRASED_QUESTION: 
    """

    template = PromptTemplate.from_template(prompt)

    template.input_variables = ['history', 'graph_labels', 'graph_relationships', 'question']

    return template


def get_qa_prompt_with_subgraph() -> PromptTemplate:
    
    prompt = """ 
        You are a helpful virtual assistant.  
        Your task is provide a relevant and precise answer to the user's question, given context information from a Knowledge Graph.  
        You might also be given the conversation record of your previous interactions with the user.
        
        In the context you will find:  
        * one or more SUMMARY OF COMMUNITY CHUNKS;
        * the COMMUNITY GRAPH represented as a list of dictionaries;
        * the CHUNKS in that community;
        * the MENTIONED ENTITIES in each chunk.
        
        Do not make things up or add any information on your own.  
        If the context is not relevant to the user's question, just say that you don't know. 
        Maintain the core information from the context.  
        
        CHAT HISTORY: {history} 
        
        QUESTION: {question}
        
        CONTEXT: {context}
        
        HELPFUL ANSWER: 
    """
    
    template = PromptTemplate.from_template(prompt)
    
    template.input_variables = ["history", "context", "question"]
    
    return template


def get_summarization_prompt() -> PromptTemplate:

    prompt = """
        Your task is to synthetize a clear and helpful answer to a question.

        The sources of information to use for your task come from a Vector Database and from a Graph Database.
        You might also be given the conversation record of your previous interactions with the user.
        
        In your task, you MUST use the context obtained from a vector search on the Vector Database 
        and the query results given running a Cypher Query on the Graph Database.  
        If one of the sources is empty, just answer the question using the other source. 

        Do not mention anything else, just summarize a precise, clear and helpful answer. 
        Do not make things up or add any information on your own. 

        CHAT HISTORY: {history} 
        
        QUESTION: {question}

        RETRIEVED CONTEXT: {retrieved_context}

        QUERY RESULT ON GRAPH: {query_result}

        ANSWER: 
    """

    template = PromptTemplate.from_template(prompt)

    template.input_variables = ['history', 'question', 'retrieved_context', 'query_result']

    return template