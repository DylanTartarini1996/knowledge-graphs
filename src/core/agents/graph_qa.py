from typing import List, Optional, Any, Dict, Tuple, Union

from langchain_core.messages import BaseMessage
from langchain_neo4j.chains.graph_qa.cypher import GraphCypherQAChain

from src.config import LLMConf
from src.core.graph.graph_queries import get_adjacent_chunks, get_mentioned_entities, filter_graph_by_communities
from src.core.graph.knowledge_graph import KnowledgeGraph
from src.core.prompts.graph_qa import get_qa_prompt_with_subgraph, get_question_answering_prompt, get_rephrase_prompt, get_summarization_prompt
from src.factory.llm import fetch_llm
from src.schema import ChatMode, Chunk, Message
from src.utils.logger import get_logger


logger = get_logger(__name__)


class GraphAgentResponder:
    """
    Agent powered by up to three LLMs, is able to answer a user's question
    navigating the `KnowledgeGraph` via Cypher Queries as well as via Vector Search.
    """

    # TODO update this agent to be aware of history, be able to rephrase according to it and answer based on a modality

    def __init__(
        self, 
        qa_llm_conf: LLMConf,
        cypher_llm_conf: LLMConf, 
        graph: KnowledgeGraph,
        rephrase_llm_conf: Optional[LLMConf]=None
    ):
        self.graph = graph
        self.qa_llm = fetch_llm(qa_llm_conf)
        self.cypher_llm = fetch_llm(cypher_llm_conf)
        self.qa_prompt = get_question_answering_prompt()
        self.qa_prompt_with_subgraph = get_qa_prompt_with_subgraph()

        self.summarize_prompt = get_summarization_prompt()

        self.graph_qa_chain = GraphCypherQAChain.from_llm(
            qa_llm=self.qa_llm, 
            cypher_llm=self.cypher_llm,
            graph=self.graph, 
            verbose=True,
            allow_dangerous_requests=True,
            validate_cypher=True, 
            return_intermediate_steps=True
        )
        self.rephrase_llm = None
        if rephrase_llm_conf:
            self.rephrase_llm = fetch_llm(rephrase_llm_conf)
            self.rephrase_prompt = get_rephrase_prompt()
            self.rephrase_prompt.partial_variables = {
                "graph_labels": self.graph.labels,
                "graph_relationships": self.graph.relationships
            }

    @staticmethod
    def parse_context(chunks: List[Chunk]) -> str:
        # TODO 
        context = ""
        # parse context in a meaningful way
        return context 
    
    

    def rephrase_question(self, history: List[Message], chat_mode: ChatMode) -> str:
        """ 
        Rephrase the user's query giving the history of the conversation 
        and the chat mode to be employed when aswering the query 
        """
        rephrased_question = ""
        # TODO 
        return rephrased_question
            
        
    def answer_with_cypher(
        self, 
        query: str, 
        intermediate_steps: bool=False, 
        ) -> str | Tuple[str, Union[list, None]]:
        """ 
        Uses only the Cypher chain to answer the user's question.

        -------
        params:
        -------
        - `query`: `str`
            The query to be answered
        - `intermediate_steps`: `bool`
            Whether to return the intermediate steps from the thought process of the LLM
        """
        
        if self.rephrase_llm:
            try: 
                rephrased_question = self.rephrase_llm.invoke(input=self.rephrase_prompt.format(question=query, history=history)).content
                logger.info(f"Rephrased Question: {rephrased_question}")
            except Exception as e:
                logger.warning(f"Failed to rephrase user question with exception: {e}")
                rephrased_question = None
        else:
            rephrased_question = None
            
        try:
            graph_qa_output = self.graph_qa_chain._call(
                inputs={"query": rephrased_question} if rephrased_question is not None else {"query": query}
            )
            if intermediate_steps:
                return graph_qa_output["result"], graph_qa_output["intermediate_steps"]
            else: 
                return graph_qa_output["result"]
        except Exception as e:
            logger.warning(f"Problem Answering with CYPHER chain: {e}")
            
            
    def answer_with_context(self, query: str, use_adjacent_chunks: bool=False) -> Tuple[str, List[Chunk]]:
        """ 
        Uses only vanilla RAG to answer the user's question.  
        If `use_adjacent_chunks=True` will query the graph for additional context  
        compared to the Chunks retrieved by the similarity search.  

        Latency will be higher due to expanded context. 

        --------
        returns:
        --------
        - `Tuple[str, List[Chunk] | None]`
            Answer and list of text chunks used by the LLM to provide the answer
        """
        context = ""
        chunks = []
        # TODO use history here
        try:
            context_docs = self.graph.vector_store.similarity_search(query=query)
        except Exception as e:
            logger.warning(f"Failed to retrieve context with exception: {e}")
            context_docs = []
        
        if use_adjacent_chunks:
            for doc in context_docs:
    
                current_chunk = Chunk(
                    chunk_id=doc.metadata["chunk_id"],
                    text=doc.page_content,
                    filename=doc.metadata["filename"]
                )
                # search adjacent chunks 
                with self.graph._driver.session() as session:
                    prev_chunk, current_chunk, next_chunk = get_adjacent_chunks(session, current_chunk)
                    session.close()

                chunks.extend([prev_chunk, current_chunk, next_chunk])
                
            # TODO 
            context = self.parse_context(chunks)

        else: 
            for doc in context_docs:

                context += f"\n {doc.page_content}"
            
        answer: BaseMessage = self.qa_llm.invoke(
            input=self.qa_prompt.format(
                question=query,  # TODO 
                context=context
            )
        )

        return answer.content, chunks
    
    
    def answer_with_community_reports(
        self, 
        query: str, 
        use_adjacent_chunks: bool=False, 
        community_type: str="leiden"
        ) -> Tuple[str, List[Chunk]]: 
        """ 
        Queries two vector indexes to get the user's answer out of an ensemble of contexts:
            1. one made of a list of `CommunityReport`
            2. one made of a list of `Chunk` from the same communities of the reports. 
            
        If `use_adjacent_chunks=True` will query the graph for additional context 
        compared to the Chunks retrieved by the similarity search. Latency will be higher 
        due to expanded context. 
        
        --------
        params:
        --------
        - `query`: `str`
            The query to be answered
        - `use_adjacent_chunks`: `bool`
            Whether to use chunks adjacent to the ones retrieved to answer the query
        - `community_type`: `str`
            The community type to look for in the graph; available types are `louvain` and `leiden`.
        
        --------
        returns:
        --------
        - `Tuple[str, List[Chunk] | None]`
            Answer and list of text chunks used by the LLM to provide the answer
        """
        # TODO specialize parsing context depending on the chat modality 
        try:
            reports_and_scores = self.graph.cr_store.similarity_search_with_relevance_scores(
                query=query, 
                k=3, 
                filter={"community_type": community_type},
                score_threshold=0.8
            )
            
            logger.info(f"Retrieved {len(reports_and_scores)} Community Reports")
            
        except Exception as e:
            logger.warning(f"Failed to retrieve Community Reports with exception: {e}")
            
        for report, score in reports_and_scores:
            
            context += f"SUMMARY OF CHUNKS: \n {report.page_content} \n"
            
            try: 
                # fetch only similar chunks in the community 
                community_chunks = self.graph.vector_store.similarity_search(
                    query=query,
                    filter={f"community_{community_type}": report.metadata['community_id']}
                )
                logger.info(f"Retrieved {len(community_chunks)} Chunks for community: {report.metadata['community_id']}")
                
            
            except Exception as e:
                logger.warning(f"Failed to enrich context with chunks from community: {report.metadata['community_id']}")
                
            context += f"CHUNKS: \n"
            
            if not use_adjacent_chunks:
                
                for chunk in community_chunks:
                    context += f"{chunk.page_content} \n"
                    
            else: 
                for chunk in community_chunks:

                    current_chunk = Chunk(
                        chunk_id=chunk.metadata["chunk_id"],
                        text=chunk.page_content,
                        filename=chunk.metadata["filename"]
                    )
                    # search adjacent chunks 
                    with self.graph._driver.session() as session:
                        prev_chunk, current_chunk, next_chunk = get_adjacent_chunks(session, current_chunk)
                        session.close()
                    
                    context += f"{prev_chunk.text} \n" if prev_chunk is not None else ""
                    context += f"{current_chunk.text} \n"
                    context += f"{next_chunk.text} \n" if next_chunk is not None else ""
                
        answer: BaseMessage = self.qa_llm.invoke(
            input=self.qa_prompt.format(
                question=query, 
                context=context
            )
        )

        # TODO extend list of chunks
        chunks = []
        return answer.content, chunks
            
        
    def answer_with_community_subgraph(
        self, 
        query: str, 
        community_type: str = "leiden",
        ) -> Tuple[str, List[Chunk]]: 
        """ 
        Answers after querying for communities:  
        
        * read the most relevant community reports 
        * fetch chunks belonging to the most relevant community (the one from the community report)
        * follow the MENTIONS relationship of each Chunk and obtain a dictionary 
        * fetch the community subgraph under the form of another dictionary 
        * passes the dictionaries + the report to a reconciler agent to decide how to answer 

        -------
        params:
        -------
        - `query`: `str`
            The query to be answered
        - `community_type`: `str`
            The community type to look for in the graph; available types are `louvain` and `leiden`.
        
        --------
        returns:
        --------
        - `Tuple[str, List[Chunk] | None]`
            Answer and list of text chunks used by the LLM to provide the answer
        """
        context = ""

        # TODO parse content specifically to chat mode

        try:
            reports = self.graph.cr_store.similarity_search(
                query=query, 
                k=1, 
                filter={"community_type": community_type},
            )
            for report in reports:
                logger.info(f"Retrieved Community Reports of type {community_type} with community id: {report.metadata['community_id']}")
            
        except Exception as e:
            logger.warning(f"Failed to retrieve Community Reports with exception: {e}")
            
            
        for report in reports:  
            
            context += f"SUMMARY OF COMMUNITY CHUNKS: \n {report.page_content} \n"
            
            with self.graph._driver.session() as session:
                
                community_subgraph = filter_graph_by_communities(
                    session, 
                    community_ids=[report.metadata['community_id']], 
                    community_type=community_type
                )
                
                session.close()
                
            context += f"COMMUNITY GRAPH: {community_subgraph} \n --------------------------------------- \n "
            
            context += f"COMMUNITY CHUNKS: "
             
            try: 
                # fetch only similar chunks in the community 
                community_chunks = self.graph.vector_store.similarity_search(
                    query=query,
                    filter={f"community_{community_type}": report.metadata['community_id']}
                )
                logger.info(f"Retrieved {len(community_chunks)} Chunks for community: {report.metadata['community_id']}")
                
                for chunk in community_chunks:
                    
                    context += f" \n --------------------------------------- \n CHUNK CONTENT: \n {chunk.page_content} \n "
                    context += f"MENTIONED ENTITIES: \n"
                    
                    current_chunk = Chunk(
                        chunk_id=chunk.metadata["chunk_id"],
                        text=chunk.page_content,
                        filename=chunk.metadata["filename"]
                    )
                    
                    with self.graph._driver.session() as session:
                        mentioned_entities = get_mentioned_entities(session, current_chunk, use_elementId=False)
                        session.close()
                        
                        for ent_dict in mentioned_entities:
                            context += f"{ent_dict['name']} \n"
                        
            except Exception as e:
                logger.warning(f"Failed to enrich context with chunks from community: {report.metadata['community_id']}")

                
        answer: BaseMessage = self.qa_llm.invoke(
            input=self.qa_prompt_with_subgraph.format(
                question=query, 
                context=context, 
            )
        )
        
        chunks = [] # TODO extend chunks 

        return answer.content, chunks


    def combined_answer(
        self, 
        query: str, 
        use_adjacent_chunks: bool=False, 
        filter: Optional[Dict[str, Any]]=None,
        ) -> Tuple[str, List[Chunk]]:
        """ 
        Answers the user query performing text generation after having retrieved
        context both via Vector Search and Cypher Queries. 
        Results from both this methods are synthetized in a comprehensive answer.

        -------
        params:
        -------
        - `query`: `str`
            The query to be answered
        - `use_adjacent_chunks`: `bool`
            Whether to use chunks adjacent to the ones retrieved to answer the query
        - `filter`: ` Optional[Dict[str, Any]]`
            Optional filtering map
        --------
        returns:
        --------
        - `Tuple[str, List[Chunk] | None]`
            Answer and list of text chunks used by the LLM to provide the answer
        """
        context = ""
        # TODO parse context
        try:
            context_docs = self.graph.vector_store.similarity_search(query=query, filter=filter)
        except Exception as e:
            logger.warning(f"Failed to retrieve context with exception: {e}")
            context_docs = []
        
        if use_adjacent_chunks:
            for doc in context_docs:
    
                current_chunk = Chunk(
                    chunk_id=doc.metadata["chunk_id"],
                    text=doc.page_content,
                    filename=doc.metadata["filename"]
                )
                # search adjacent chunks 
                with self.graph._driver.session() as session:
                    prev_chunk, current_chunk, next_chunk = get_adjacent_chunks(session, current_chunk)
                    session.close()
                
                context += f"\n {prev_chunk.text}" if prev_chunk is not None else ""
                context += f"\n {current_chunk.text}"
                context += f"\n {next_chunk.text}" if next_chunk is not None else ""
        else: 
            for doc in context_docs:
                context += f"\n {doc.page_content}"
        
        try: 
            cypher_chain_answer, cypher_steps = self.answer_with_cypher(query=query, intermediate_steps=True)
        except TypeError:
            cypher_steps = None
            logger.warning("Unable to run Cypher chain for this question")
        
        final_answer: BaseMessage = self.qa_llm.invoke(
            input=self.summarize_prompt.format(
                question=query, 
                retrieved_context=context, 
                query_result=cypher_steps
            )
        )

        chunks = [] # TODO extend list of chunks

        return final_answer.content, chunks
    


    def get_answer(
        self, 
        query: str, 
        chat_mode: ChatMode, 
        chat_history: Optional[List[Message]] = []
        ) -> Tuple[str, Union[List[Chunk] | None]]:
        """ 
        Gets an answer to the user's query with a given Chat Mode. 

        --------
        params:
        --------
        - `query`: `str`. 
            The input from the user
        - `chat_mode`: `ChatMode`  
            The chat mode to employ when answering the user's query
        - `history`: `Optional[List[Message]]`. 
            The history of the conversation, when available. Used to rephrase the query depending on chat mode        
        --------
        returns:
        --------
        - `Tuple[str, List[Chunk] | None]`
            Answer and list of text chunks used by the LLM to provide the answer
        """

        if len(chat_history) > 0: 
            rephrased_question = self.rephrase_question(chat_history, chat_mode)

        chunks = None

        if chat_mode == ChatMode.SIMILARITY:
            answer, chunks = self.answer_with_context(
                query=rephrased_question if rephrased_question else query, 
                use_adjacent_chunks=True
            )

        elif chat_mode == ChatMode.CYPHER:
            answer = self.answer_with_cypher(
                query=rephrased_question if rephrased_question else query,
                intermediate_steps=False
            )

        elif chat_mode == ChatMode.COMMUNITIES:
            answer, chunks = self.answer_with_community_reports(
                query=rephrased_question if rephrased_question else query,
                use_adjacent_chunks=True,
                community_type="leiden"
            )

        elif chat_mode == ChatMode.SUBGRAPH:
            answer, chunks = self.answer_with_community_subgraph(
                query=rephrased_question if rephrased_question else query,
                community_type="leiden"
            )
        elif chat_mode == ChatMode.COMBINE:
            answer, chunks = self.combined_answer(
                query=rephrased_question if rephrased_question else query,
                use_adjacent_chunks=True
            )

        return answer, chunks
        








        