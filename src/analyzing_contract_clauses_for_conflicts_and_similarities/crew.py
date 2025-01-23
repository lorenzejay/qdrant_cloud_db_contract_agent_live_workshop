from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

from analyzing_contract_clauses_for_conflicts_and_similarities.tools.qdrant_vector_search_tool import (
    QdrantVectorSearchTool,
)

import os
from dotenv import load_dotenv

load_dotenv()


@CrewBase
class AnalyzingContractClausesForConflictsAndSimilaritiesCrew:
    """AnalyzingContractClausesForConflictsAndSimilarities crew"""

    vector_search_tool = QdrantVectorSearchTool(
        qdrant_url=os.getenv("QDRANT_URL"),
        qdrant_api_key=os.getenv("QDRANT_API_KEY"),
        collection_name="contracts24",
    )

    @agent
    def data_retrieval_analysis_specialist(self) -> Agent:
        return Agent(
            config=self.agents_config["data_retrieval_analysis_specialist"],
            tools=[self.vector_search_tool],
        )

    @agent
    def report_generation_specialist(self) -> Agent:
        return Agent(
            config=self.agents_config["report_generation_specialist"],
        )

    @task
    def retrieve_contracts_task(self) -> Task:
        return Task(
            config=self.tasks_config["retrieve_contracts_task"],
            tools=[self.vector_search_tool],
        )

    @task
    def analyze_clauses_task(self) -> Task:
        return Task(
            config=self.tasks_config["analyze_clauses_task"],
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
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
        )
