from typing import List
from src.core.graph.knowledge_graph import KnowledgeGraph
from src.factory.embeddings import get_embeddings
from src.factory.llm import fetch_llm
from src.config import LLMConf, EmbedderConf
from src.core.graph.graph_model import Community, CommunityReport
from src.core.prompts.communities import get_summarize_community_prompt
from src.utils.logger import get_logger


logger = get_logger(__name__)


class CommunitiesSummarizer:
    """ 
    Agent in charge of producing summaries of Community Reports. 
    """
    
    def __init__(
        self, 
        llm_conf: LLMConf, 
        embeddings_conf: EmbedderConf
        ):
        self.llm = fetch_llm(llm_conf)
        self.embeddings = get_embeddings(embeddings_conf)
        self.summarize_community_prompt = get_summarize_community_prompt()
        
        
    def get_reports(self, communities: List[Community]) -> List[CommunityReport]:
        """ 
        Generate Community Reports for available communities in the Graph. 
        """
        reports = []
        
        for comm in communities:
            
            report = self.get_community_report(comm)
            
            reports.append(report)
            
        return reports
            
    
    def get_community_report(self, community: Community) -> CommunityReport | None:
        """ 
        Generates a CommunityReport for a given community, out of chunks available in said community. 
        It will also embed the summary to make it retrievable
        """
        if not community.chunks: 
            logger.warning(f"There are no Chunks to summarize for community {community.community_type}: {community.community_id}")
            return None
        
        chunks_content = ""
        for chunk in community.chunks:
            chunks_content+=chunk.text.replace("\n\n","\n")+("\n\n")
        
        try:    
            summary = self.llm.invoke(
                input=self.summarize_community_prompt.format(
                    context=chunks_content
                )
            ).content
        except Exception as e:
            logger.warning(f"Issue summarizing Chunks for community {community.community_type}: {community.community_id}: {e}")
            return None
        
        try:
            summary_embeddings = self.embeddings.embed_documents([summary])[0]
        except Exception as e:
            logger.warning(f"Issue embedding Summary for community {community.community_type}: {community.community_id}: {e}")
        
        report = CommunityReport(
            communtiy_type=community.community_type,
            community_id=community.community_id,
            summary=summary,
            community_size=community.community_size,
            summary_embeddings=summary_embeddings
        )
        
        return report
        
    
        
        
        
        
        