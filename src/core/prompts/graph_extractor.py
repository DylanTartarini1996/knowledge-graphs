from langchain.prompts import PromptTemplate

def get_graph_extractor_prompt()-> PromptTemplate:
    """ 
    Parses the instructions to give as input to the LLM in charge of 
    Relations Extractions. 
    """
    prompt= """
        You are a top-tier algorithm designed for extracting information in structured formats to build a Knowledge Graph.

        Your task is to extract informations in the form of Nodes and Relationships from an INPUT TEXT.

        - NODES represent entities and concepts.
        - RELATIONSHIPS represents the connections between nodes.
        - PROPERTIES characterize nodes or relationships.

        ------
        RULES:
        ------

        1. FORMAT
        - You MUST return ONLY the Graph extracted from the INPUT TEXT. Do not add anything else. 
        - Remember that a Graph is defined as 

        ````
        class Graph(Serializable):
            '''
            Represents a graph consisting of nodes and relationships.

            Attributes:
                nodes (List[Node]): A list of nodes in the graph.
                relationships (List[Relationship]): A list of relationships in the graph.
            '''
            nodes: List[Node]
            relationships: List[Relationship]
        ````

        where Nodes and Relationships are defined as

        ````
        class Node(Serializable):
            '''
            Represents a node in a graph with associated properties.

            Attributes:
                id (str): A unique identifier for the node.
                type (str): The type or label of the node.
                properties (Optional[Dict[str, str]]): Additional properties associated with the node.
            '''
            id: str
            type: str
            properties: Optional[Dict[str, str]]
        ````

        and 

        ````
        class Relationship(Serializable):
            '''
            Represents a directed relationship between two nodes in a Graph.

            Attributes:
                source (str): The source node of the relationship.
                target (str): The target node of the relationship.
                type (str): The type of the relationship.
                properties (Optional[Dict[str, str]]): Additional properties associated with the relationship.
            '''
            source: str
            target: str
            type: str
            properties: Optional[Dict[str, str]]
        ````

        2. ALLOWED LABELS AND RELATIONSHIPS
        - If provided with allowed labels and relationship types then you MUST use only those as possible outcomes. 
        - If labels and relationships are not provided, you are free to use any label and relationship you see fit. 

        ------------
        
        ALLOWED NODE LABELS: {allowed_labels}
        LABELS DESCRIPTIONS: {labels_descriptions}
        ALLOWED RELATIONSHIPS TYPES: {allowed_relationships}

        ## Begin Extraction!
        INPUT TEXT: {input_text}
    """

    template = PromptTemplate.from_template(prompt)

    template.input_variables = ['input_text', 'allowed_labels', 'labels_descriptions', 'allowed_relationships']

    return template

# 2. NUMERICAL DATA AND DATES
#         - Numerical data, like age or other related information, should be incorporated as attributes or properties of the respective nodes.
#         -  Do not create separate nodes for dates or numerical values. Always attach them as attributes or properties of nodes.
#         - Properties must be in a key-value format.
#         - Never use escaped single or double quotes within property values.
#         - Use camelCase for property keys, e.g., `birthDate`.


#        3. ENTITY CONSISTENCY
#         When extracting entities, it's vital to ensure consistency.
#         If an entity, such as "John Doe", is mentioned multiple times in the text but is referred to by different names or pronouns (e.g., "Joe", "he"),
#         always use the most complete identifier for that entity throughout the knowledge graph. In this example, use "John Doe" as the entity ID.
#         Remember, the knowledge graph should be coherent and easily understandable, so maintaining consistency in entity references is crucial.