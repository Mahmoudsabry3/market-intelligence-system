# Multi-Agent Market Intelligence System

## 1. System Overview

The Multi-Agent Market Intelligence System is an advanced, modular framework designed to automatically generate comprehensive market intelligence reports based on user queries. The system employs a chain of specialized agents, each responsible for a distinct phase in the report generation process, from query understanding to final report formatting.

The system aims to provide actionable market insights by collecting, analyzing, and synthesizing data from multiple sources including financial APIs, news articles, social media, and industry databases. It features robust fallback mechanisms to ensure operation even when external APIs are unavailable.

## 2. System Architecture

The system follows a sequential multi-agent architecture with four primary components:

```
User Query → Query Understanding Agent → Data Collection Agent → Data Analysis Agent → Report Generation Agent → Final Report
```

Each agent performs a specialized function in the pipeline:

1. **Query Understanding Agent**: Interprets natural language queries and extracts structured parameters.
2. **Data Collection Agent**: Gathers relevant market data from multiple sources.
3. **Data Analysis Agent**: Processes and analyzes the collected data to generate insights.
4. **Report Generation Agent**: Creates a formatted report based on the analyzed data.

## 3. Detailed Component Documentation

### 3.1 Query Understanding Agent

#### Purpose
Processes natural language queries and extracts structured information about markets and focus areas.

#### Input/Output
- **Input**: Natural language query string
- **Output**: Dictionary with structured query information

#### Implementation
```python
class QueryUnderstandingAgent:
    """
    First agent in the workflow that processes the user's high-level query
    and extracts structured information about the market and focus areas.
    
    Enhanced with simulated BERT-like NLP capabilities to better understand query intent,
    extract detailed parameters, and handle complex comparative queries.
    """
```

#### Key Methods

1. **`_extract_with_llm(query)`**: Uses OpenAI to extract entities and intent from queries.
   - Attempts to use a language model for sophisticated query understanding
   - Falls back to rule-based extraction if API is unavailable

2. **`_fallback_extraction(query)`**: Simulates BERT-like entity extraction as a fallback.
   - Identifies market, geography, timeframe, and focus areas
   - Uses pattern matching and predefined entity types

3. **`_simulate_intent_classification(query)`**: Determines the primary intent of the query.
   - Classifies into categories: market_analysis, competitive_analysis, trend_forecast, comparative_study
   - Uses keyword matching to determine the most likely intent

4. **`process(query)`**: Main entry point that orchestrates the query understanding process.
   - Applies advanced NLP techniques when available
   - Falls back to simulated capabilities when needed
   - Returns structured query with confidence score

#### Example Output
```python
{
    "intent": "trend_forecast",
    "parameters": {
        "market": "fintech",
        "focus": "trends",
        "geography": ["global"],
        "timeframe": "current"
    },
    "confidence": 0.89
}
```

### 3.2 Data Collection Agent

#### Purpose
Collects relevant market data from multiple sources based on the structured query parameters.

#### Input/Output
- **Input**: Dictionary with structured query information
- **Output**: Dictionary with collected data from multiple sources

#### Implementation
```python
class DataCollectionAgent:
    """
    Second agent in the workflow that collects relevant data based on
    the structured query from the Query Understanding Agent.
    
    Enhanced with capabilities for:
    - Real-time data collection from multiple sources (web, APIs, databases)
    - AI-powered filtering to remove outdated or irrelevant information
    - Smart aggregation of information across sources
    """
```

#### Key Methods

1. **`_retrieve_news_articles(market, focus, max_results=10)`**: Collects relevant news articles.
   - Uses News API when available
   - Filters articles based on relevance to the market and focus
   - Falls back to simulated web scraping when API is unavailable

2. **`_simulate_web_scraping(market, focus)`**: Generates simulated article data when real APIs are unavailable.
   - Creates realistic article structures with titles, sources, dates
   - Applies relevance filtering to simulate AI-based filtering

3. **`_summarize_articles_with_llm(articles)`**: Generates AI summaries of collected articles.
   - Uses OpenAI API to summarize article content
   - Handles errors gracefully and includes fallback mechanisms

4. **`_get_financial_data(market, focus)`**: Retrieves financial data for relevant companies.
   - Maps markets to ticker symbols
   - Uses Yahoo Finance API to get real financial metrics
   - Calculates market trends, growth rates, and competitive landscape

5. **`_simulate_api_data(market, focus)`**: Provides simulated market data when real APIs fail.
   - Creates realistic market metrics based on the market type
   - Ensures consistent data formats with real API responses

6. **`_get_social_sentiment(market)`**: Analyzes social media sentiment.
   - Uses Twitter and Reddit APIs when available
   - Collects and analyzes posts, hashtags, and trending topics
   - Falls back to simulated sentiment data when needed

7. **`_simulate_social_listening(market)`**: Generates realistic social sentiment data.
   - Creates sentiment scores, trending topics, and sample sizes
   - Ensures format consistency with real API responses

8. **`process(structured_query)`**: Orchestrates the entire data collection process.
   - Extracts parameters from the structured query
   - Collects data from multiple sources with fallbacks
   - Compiles all data into a comprehensive structure

#### Example Output
```python
{
    "market_name": "fintech",
    "focus": "trends",
    "query_geography": ["global"],
    "data_collection_timestamp": "2023-12-15 10:30:45",
    "recent_articles": [...],
    "market_growth": "11.9% annually",
    "market_size": "$128.88 billion",
    "competitors": [...],
    "social_sentiment": {...},
    "trends": [...],
    "challenges": [...],
    "data_sources": ["web_search", "financial_apis", "social_media", ...]
}
```

### 3.3 Data Analysis Agent

#### Purpose
Analyzes the collected data to generate insights and recommendations for the market.

#### Input/Output
- **Input**: Dictionary with collected data
- **Output**: Dictionary with analyzed data and insights

#### Implementation
```python
class DataAnalysisAgent:
    """
    Third agent in the workflow that analyzes the collected data and
    generates insights and recommendations based on the market analysis.
    
    Enhanced with capabilities for:
    - Advanced data analysis and pattern recognition
    - AI-powered recommendation generation
    - Smart aggregation of insights across different data sources
    """
```

#### Key Methods

1. **`process(collected_data)`**: Analyzes the collected data.
   - Extracts key metrics from the collected data
   - Generates insights based on market growth, size, and trends
   - Currently a simplified implementation that could be enhanced in future versions

#### Example Output
```python
{
    # All collected data plus:
    "insights": "The fintech market is experiencing 11.9% annually growth and has a market size of $128.88 billion."
}
```

### 3.4 Report Generation Agent

#### Purpose
Formats the analyzed data into a comprehensive, well-structured market intelligence report.

#### Input/Output
- **Input**: Dictionary with analyzed data
- **Output**: Formatted report as a string (Markdown format)

#### Implementation
```python
class ReportGenerationAgent:
    """
    Fourth agent in the workflow that generates a formatted report based on
    the analyzed data from the Data Analysis Agent.
    
    Enhanced with capabilities for:
    - Formatting the report in various styles and formats
    - Integrating external data sources for additional context
    - Smart aggregation of report components
    """
```

#### Key Methods

1. **`process(analyzed_data)`**: Generates a structured report.
   - Creates a markdown-formatted report with multiple sections
   - Extracts and formats information from the analyzed data
   - Includes executive summary, market overview, competitor analysis, and recommendations

#### Report Sections
1. **Report Header**: Title and generation timestamp
2. **Executive Summary**: Key market metrics and high-level overview
3. **Market Overview**: Data quality assessment and recent news
4. **Market Trends**: Analysis of current and emerging trends
5. **Market Challenges**: Assessment of obstacles and potential solutions
6. **Competitive Landscape**: Table of key players and detailed competitor profiles
7. **Social Media Sentiment Analysis**: Overview of market sentiment from social platforms
8. **Recommendations and Opportunities**: Strategic recommendations based on the analysis
9. **Data Sources**: List of data sources used in the report
10. **Disclaimer**: Legal disclaimer regarding the AI-generated content

## 4. Supporting Functions

### 4.1 PDF Generation

#### Purpose
Converts the generated report into PDF format for easy sharing and presentation.

#### Implementation
```python
def save_report_as_pdf(report_text, filename):
    """
    Save the generated report as a PDF file.
    """
```

#### Key Features
- Converts Markdown to HTML with formatting
- Attempts to save as PDF using pdfkit
- Falls back to HTML if PDF generation fails
- Provides a final fallback to plain text if needed

### 4.2 Process Query Function

#### Purpose
Orchestrates the entire workflow from user query to final report.

#### Implementation
```python
def process_query(query, save_pdf=True):
    """
    Main function that processes a user query through the entire agent workflow.
    """
```

#### Key Steps
1. Initializes all agent components
2. Processes the query through each agent sequentially
3. Saves the final report in PDF format if requested
4. Returns the formatted report string

## 5. Data Flow and Integration

The system maintains a consistent data structure throughout the pipeline, with each agent adding or transforming specific aspects:

1. **User Query (string)** → QueryUnderstandingAgent
2. **Structured Query (dict)** → DataCollectionAgent
3. **Collected Data (dict)** → DataAnalysisAgent
4. **Analyzed Data (dict)** → ReportGenerationAgent
5. **Final Report (string)** → PDF/HTML/Text Output

Integration with external services is managed through API clients:
- News API for article retrieval
- Yahoo Finance API for financial data
- Twitter API for social media sentiment
- Reddit API for forum discussions
- OpenAI API for summarization and analysis

Each external dependency includes robust fallback mechanisms to ensure system reliability even when APIs are unavailable.

## 6. Error Handling and Resilience

The system incorporates comprehensive error handling at multiple levels:

1. **API-Level Resilience**: All external API calls include try-except blocks with appropriate fallbacks.
2. **Service-Level Fallbacks**: When an entire service is unavailable, simulated data generation provides realistic alternatives.
3. **Exponential Backoff**: API retries use exponential backoff to handle rate limiting (e.g., in Yahoo Finance calls).
4. **Format Validation**: Data structure validation ensures consistent formats throughout the pipeline.
5. **Report Generation Fallbacks**: Multi-tier fallbacks for report saving (PDF → HTML → Text).

## 7. Usage Instructions

### Basic Usage

```python
# Import the system
from multi_agent_report_system import process_query

# Generate a report with a natural language query
query = "Generate a strategic intelligence report for the fintech sector in Southeast Asia, focusing on digital payments"
report = process_query(query)

# Print or display the report
print(report)
```

### Command-Line Usage

The system can be run directly from the command line:

```bash
python multi_agent_report_system.py
```

When run this way, it will prompt for a query and then generate the report.

### Configuration

API keys are configured via environment variables:
- `OPENAI_API_KEY`: For LLM-based summarization and analysis
- `NEWS_API_KEY`: For news article retrieval
- `TWITTER_API_KEY`, `TWITTER_API_SECRET`, etc.: For Twitter sentiment analysis
- `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`: For Reddit data collection

## 8. Future Enhancements

The system architecture allows for several potential enhancements:

1. **Advanced NLP Integration**: Deeper integration with more sophisticated NLP models
2. **Real-time Data Updates**: Streaming data capabilities for up-to-the-minute insights
3. **Interactive Reports**: Dynamic HTML reports with interactive visualizations
4. **Multi-market Comparison**: Enhanced capabilities for comparing different markets
5. **Customizable Templates**: User-configurable report templates and formats
6. **Expanded Data Sources**: Integration with additional industry-specific data providers

## 9. Dependencies

The system relies on the following key libraries:
- `requests`: For API communication
- `openai`: For AI-assisted summarization and analysis
- `yfinance`: For financial data retrieval
- `tweepy`: For Twitter API integration
- `praw`: For Reddit API integration
- `markdown2`: For Markdown to HTML conversion
- `pdfkit`: For HTML to PDF conversion
- `jinja2`: For template rendering

Dependencies should be installed before system use. 