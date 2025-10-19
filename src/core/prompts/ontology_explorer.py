from langchain.prompts import PromptTemplate

def get_ontology_creation_prompt()-> PromptTemplate:
    """ 
    Looks to a subset of chunks from one or more `ProcessedDocument` 
    and use them as context to suggest an `Ontology`.
    """
    prompt= """
        You are a top-tier algorithm designed for extracting information in structured formats to build a Knowledge Graph.

        You will be given a description of the domain and a list of texts from the same domain: 
        your task is to use them as context to create an `Ontology`, which is a representation 
        of knowledge in the form of allowed labels (and their descriptions) as well as allowed relationships.

        Remember that an Ontology is defined as:  

        ```
        class Ontology: 
            '''
            Used to describe arbitrary, project-specific allowed labels and relationships.
            '''
            allowed_labels: List[str]
            labels_descriptions: Dict[str, str]
            allowed_relations: List[str]
        ```

        You MUST return ONLY the Ontology inferred from the context given to you. Do not add anything else.
        If the description of the domain is empty, just use the context texts as a source to infer the Ontology. 

        ### BEGIN! 
        DOMAIN DESCRIPTION: {domain_description}
        CONTEXT TEXTS: {texts}
    """

    template = PromptTemplate.from_template(prompt)

    template.input_variables = ['domain_description','texts']

    return template