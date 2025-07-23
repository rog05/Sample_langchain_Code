# !pip install crewai==0.28.8 crewai_tools==0.1.6 langchain_community==0.0.29 transformers huggingface_hub -q

from crewai import Agent, Task, Crew, Process
from crewai_tools import ScrapeWebsiteTool, SerperDevTool
from langchain_community.chat_models import ChatHuggingFace

# ------------------ Set Environment Keys ------------------

# ------------------ Tools Setup ------------------
search_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()

# ------------------ Agent Definitions ------------------
data_analyst_agent = Agent(
    role="Data Analyst",
    goal="Monitor and analyze market data in real-time to identify trends and predict market movements.",
    backstory="Specializing in financial markets, this agent employs statistical modeling and machine learning techniques to provide critical insights. Renowned for its proficiency in data analysis, the Data Analyst Agent serves as a pivotal resource for informing trading decisions.",
    verbose=True,
    allow_delegation=True,
    tools=[scrape_tool, search_tool]
)

trading_strategy_agent = Agent(
    role="Trading Strategy Developer",
    goal="Develop and test various trading strategies leveraging insights from the Data Analyst Agent.",
    backstory="Possessing a deep understanding of financial markets and quantitative analysis, this agent formulates and optimizes trading strategies. It assesses the performance of diverse approaches to identify the most profitable and risk-averse options.",
    verbose=True,
    allow_delegation=True,
    tools=[scrape_tool, search_tool]
)

execution_agent = Agent(
    role="Trade Advisor",
    goal="Recommend optimal trade execution strategies based on approved trading plans.",
    backstory="Specializing in the analysis of timing, price, and logistical details of potential trades, this agent evaluates these factors to provide well-founded recommendations. Its expertise ensures that trades are executed efficiently and in alignment with the overall strategy.",
    verbose=True,
    allow_delegation=True,
    tools=[scrape_tool, search_tool]
)

risk_management_agent = Agent(
    role="Risk Advisor",
    goal="Evaluate and provide insights on the risks associated with potential trading activities.",
    backstory="With extensive expertise in risk assessment models and market dynamics, this agent thoroughly examines the potential risks of proposed trades. It delivers comprehensive analyses of risk exposure and recommends safeguards to ensure that trading activities align with the firm's risk tolerance.",
    verbose=True,
    allow_delegation=True,
    tools=[scrape_tool, search_tool]
)

# ------------------ Task Definitions ------------------
data_analysis_task = Task(
    description=(
        "Continuously monitor and analyze market data for the selected stock ({stock_selection}). "
        "Employ statistical modeling and machine learning techniques to identify trends and predict market movements."
    ),
    expected_output=(
        "Generate insights and alerts regarding significant market opportunities or threats for {stock_selection}."
    ),
    agent=data_analyst_agent,
)

strategy_development_task = Task(
    description=(
        "Develop and refine trading strategies based on insights from the Data Analyst and user-defined risk tolerance ({risk_tolerance}). "
        "Incorporate trading preferences ({trading_strategy_preference}) in the strategy development process."
    ),
    expected_output=(
        "A set of potential trading strategies for {stock_selection} that align with the user's risk tolerance."
    ),
    agent=trading_strategy_agent,
)

execution_planning_task = Task(
    description=(
        "Analyze approved trading strategies to determine the optimal execution methods for {stock_selection}, "
        "considering current market conditions and pricing strategies."
    ),
    expected_output=(
        "Comprehensive execution plans detailing how and when to execute trades for {stock_selection}."
    ),
    agent=execution_agent,
)

risk_assessment_task = Task(
    description=(
        "Evaluate the risks associated with the proposed trading strategies and execution plans for {stock_selection}. "
        "Provide a detailed analysis of potential risks and recommend mitigation strategies."
    ),
    expected_output=(
        "A comprehensive risk analysis report detailing potential risks and mitigation recommendations for {stock_selection}."
    ),
    agent=risk_management_agent,
)

# ------------------ Hugging Face Model for Crew ------------------
manager_llm = ChatHuggingFace(
    repo_id="mistralai/Mistral-7B-Instruct-v0.1",  # You can change this to another HF model
    model_kwargs={"temperature": 0.7, "max_new_tokens": 512}
)

# ------------------ Crew Setup ------------------
financial_trading_crew = Crew(
    agents=[
        data_analyst_agent,
        trading_strategy_agent,
        execution_agent,
        risk_management_agent
    ],
    tasks=[
        data_analysis_task,
        strategy_development_task,
        execution_planning_task,
        risk_assessment_task
    ],
    manager_llm=manager_llm,
    process=Process.hierarchical,
    verbose=True
)

# ------------------ Input for Simulation ------------------
financial_trading_inputs = {
    'stock_selection': 'TCS',
    'initial_capital': '100000',
    'risk_tolerance': 'Medium',
    'trading_strategy_preference': 'Day Trading',
    'news_impact_consideration': True
}

# ------------------ Kickoff ------------------
result = financial_trading_crew.kickoff(inputs=financial_trading_inputs)

# ------------------ Display ------------------
Markdown(result)
