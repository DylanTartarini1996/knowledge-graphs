from langchain.prompts import PromptTemplate

def get_summarize_community_prompt() -> PromptTemplate:
    
    prompt = """
        Your task is to synthetize a summary from the given context.

        Do not mention anything else, just summarize a precise, clear and helpful summary. 
        Do not make things up or add any information on your own. 

        CONTEXT: {context}

        SUMMARY: 
    """

    template = PromptTemplate.from_template(prompt)

    template.input_variables = ['context']

    return template