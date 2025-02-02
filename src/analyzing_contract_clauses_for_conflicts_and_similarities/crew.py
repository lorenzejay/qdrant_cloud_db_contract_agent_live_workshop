from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

# from crewai_tools import QdrantVectorSearchTool
import os
from dotenv import load_dotenv
from analyzing_contract_clauses_for_conflicts_and_similarities.tools.qdrant_vector_search_tool import (
    QdrantVectorSearchTool,
)

load_dotenv()


@CrewBase
class AnalyzingContractClausesForConflictsAndSimilaritiesCrew:
    """AnalyzingContractClausesForConflictsAndSimilarities crew"""

    vector_search_tool = QdrantVectorSearchTool(
        collection_name="contracts_business_2",
        qdrant_url=os.getenv("QDRANT_URL"),
        qdrant_api_key=os.getenv("QDRANT_API_KEY"),
    )

    @agent
    def data_retrieval_analysis_specialist(self) -> Agent:
        return Agent(
            config=self.agents_config["data_retrieval_analysis_specialist"],
            tools=[self.vector_search_tool],
        )

    @agent
    def source_citer_specialist(self) -> Agent:
        return Agent(
            config=self.agents_config["source_citer_specialist"],
            tools=[self.vector_search_tool],
        )

    @agent
    def report_generation_specialist(self) -> Agent:
        return Agent(
            config=self.agents_config["report_generation_specialist"],
            tools=[self.vector_search_tool],
        )

    @task
    def retrieve_contracts_task(self) -> Task:
        return Task(
            config=self.tasks_config["retrieve_contracts_task"],
        )

    @task
    def source_citer_task(self) -> Task:
        return Task(
            config=self.tasks_config["source_citer_task"],
        )

    @task
    def generate_report_task(self) -> Task:
        return Task(
            config=self.tasks_config["generate_report_task"],
        )

    @crew
    def crew(self) -> Crew:
        """Creates the AnalyzingContractClausesForConflictsAndSimilarities crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
