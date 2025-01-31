import gradio as gr
import re
from litellm import completion
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from io import BytesIO
import tempfile
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
from markdown2 import markdown
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from io import BytesIO
import tempfile
from datetime import datetime
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import json
import os
import logging
import sys
from typing import Dict, List
import requests
from bs4 import BeautifulSoup
from collections import defaultdict
from dotenv import load_dotenv
from llama_index.core import (
    VectorStoreIndex,
    Document,
    StorageContext,
    load_index_from_storage,
    Settings
)
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
import logging 
import os
from dotenv import load_dotenv
import sys
import openai
import yfinance as yf
import json
from openai.types.beta import Assistant, Thread
from openai.types.beta.threads import Run
import json
import logging
import requests

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
log_handler = logging.StreamHandler(sys.stdout)
log = logging.getLogger(__name__)
log.addHandler(log_handler)
log.setLevel(logging.INFO)
client = openai.OpenAI(api_key=OPENAI_API_KEY)

def yahoo_finance(ticker="", period=""):
    msft = yf.Ticker(ticker)
    result = msft.history(period=period).head()
    result = result.to_json()
    print("yahoo finance")
    print(result)
    return json.dumps(result)

def alpha_vantage_stock_data(symbol="", interval=""):
    print("alpha vantage")
    base_url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_INTRADAY",
        "symbol": symbol,
        "interval": interval,
        "apikey": ALPHA_VANTAGE_API_KEY
    }
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        data = response.json()
        return json.dumps(data, indent=4)
    else:
        log.error(f"Failed to fetch data from Alpha Vantage. Status code: {response.status_code}")
        return json.dumps({"error": "Unable to fetch data"})
    
def alpha_vantage_forex_data(from_currency="", to_currency=""):
    print("alpha vantage forex - free daily")
    base_url = "https://www.alphavantage.co/query"
    params = {
        "function": "CURRENCY_EXCHANGE_RATE",
        "from_currency": from_currency,
        "to_currency": to_currency,
        "apikey": ALPHA_VANTAGE_API_KEY
    }
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        data = response.json()
        return json.dumps(data, indent=4)
    else:
        log.error(f"Failed to fetch forex data from Alpha Vantage. Status code: {response.status_code}")
        return json.dumps({"error": "Unable to fetch forex data"})

def world_bank_data(indicator="", country=""):
    print("world bank data")
    url = f"http://api.worldbank.org/v2/country/{country}/indicator/{indicator}?format=json"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if len(data) > 1:
            return json.dumps(data[1], indent=4)
        else:
            return json.dumps({"error": "No data available for the given indicator and country"})
    else:
        return json.dumps({"error": f"API request failed with status code {response.status_code}"})


def news_api(country="", category=""):
    NEWS_API_KEY = os.getenv("NEWS_API_KEY")
    base_url = "https://newsapi.org/v2/top-headlines"
    
    params = {
        "country": country,
        "category": category,
        "apiKey": os.getenv("NEWS_API_KEY")
    }
    
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        data = response.json()
        return json.dumps(data, indent=4)
    else:
        logging.error(f"Failed to fetch news data. Status code: {response.status_code}")
        return json.dumps({"error": "Unable to fetch news data"})
    
def get_insider_roster(symbol):
    print("insider func")
    url = f"https://financial-modeling-prep.p.rapidapi.com/v4/insider-roaster/{symbol}"
    headers = {
        "x-rapidapi-key": os.getenv("INSIDER_API_KEY"),
        "x-rapidapi-host": "financial-modeling-prep.p.rapidapi.com"
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  
        print(response.json())
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None


tool_callables = {
    "yahoo_finance": yahoo_finance,
    "alpha_vantage_stock_data": alpha_vantage_stock_data,
    "alpha_vantage_forex_data": alpha_vantage_forex_data,
    "world_bank_data": world_bank_data,
    "news_api": news_api,
    "get_insider_roster": get_insider_roster,
}

tools = [
    {
        "type": "function",
        "function": {
            "name": "yahoo_finance",
            "description": "Get the related finance data from yahoo finance",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Ticker format of a stock",
                    },
                    "period": {
                        "type": "string",
                        "description": "Time period",
                    },
                },
                "required": ["ticker", "period"],
              
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "alpha_vantage_stock_data",
            "description": "Fetch stock data from Alpha Vantage API",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Stock symbol (e.g., MSFT for Microsoft)"
                    },
                    "interval": {
                        "type": "string",
                        "description": "Time interval for stock data (e.g., 1min, 5min, daily)"
                    },
                },
                "required": ["symbol", "interval"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "alpha_vantage_forex_data",
            "description": "Fetch forex data (currency exchange rate) from Alpha Vantage API",
            "parameters": {
                "type": "object",
                "properties": {
                    "from_currency": {
                        "type": "string",
                        "description": "Base currency (e.g., USD)"
                    },
                    "to_currency": {
                        "type": "string",
                        "description": "Quote currency (e.g., EUR)"
                    },
                },
                "required": ["from_currency", "to_currency"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "world_bank_data",
            "description": "Fetch macroeconomic and social data from the World Bank API",
            "parameters": {
                "type": "object",
                "properties": {
                    "indicator": {
                        "type": "string",
                        "description": "The World Bank indicator code (e.g., GDP, population)"
                    },
                    "country": {
                        "type": "string",
                        "description": "The country code (e.g., USA, TUR)"
                    },
                },
                "required": ["indicator", "country"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "news_api",
            "description": "Fetch top headlines from News API",
            "parameters": {
                "type": "object",
                "properties": {
                    "country": {
                        "type": "string",
                        "description": "Two-letter country code"
                    },
                    "category": {
                        "type": "string",
                        "description": "News category (e.g., business, technology)"
                    }
                },
                "required": ["country"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_insider_roster",
            "description": "Get financial data",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Symbol of the related financial object"
                    },
                },
                "required": ["symbol"]
            }
        }
    },
]

openai_assistant: Assistant = client.beta.assistants.create(
    model="gpt-4-turbo",
    instructions="you are a financial agent helper",
    name="finance",
    tools=tools
)


def run(query: str, max_turns: int = 3) -> str:
    openai_assistant = client.beta.assistants.create(
        model="gpt-4-turbo",
        instructions=f"you are a finance assistant, who provides information about the stock market. Today's date is {datetime.now().strftime('%Y-%m-%d')}",
        name="tutor",
        tools=tools
    )
    thread: Thread = client.beta.threads.create()

    client.beta.threads.messages.create(
        thread_id=thread.id, role="user", content=query
    )

    run: Run = client.beta.threads.runs.create_and_poll(
        thread_id=thread.id,
        assistant_id=openai_assistant.id,
    )

    for turn in range(max_turns):

        messages = client.beta.threads.messages.list(
            thread_id=thread.id,
            run_id=run.id,
            order="desc",
            limit=1,
        )

        if run.status == "completed":
            assistant_res: str = next(
                (
                    content.text.value
                    for content in messages.data[0].content
                    if content.type == "text"
                ),
                None,
            )

            return assistant_res

        if run.status == "requires_action":
            func_tool_outputs = []

            for tool in run.required_action.submit_tool_outputs.tool_calls:
                args = (
                    json.loads(tool.function.arguments)
                    if tool.function.arguments
                    else {}
                )
                func_output = tool_callables[tool.function.name](**args)

                func_tool_outputs.append(
                    {"tool_call_id": tool.id, "output": str(func_output)}
                )

            run = client.beta.threads.runs.submit_tool_outputs_and_poll(
                thread_id=thread.id, run_id=run.id, tool_outputs=func_tool_outputs
            )

            continue

        else:
            if run.status == "failed":
                log.error(
                    f"OpenAIFunctionAgent turn-{turn+1} | Run failure reason: {run.last_error}"
                )

            raise Exception(
                f"Failed to generate text due to: {run.last_error}"
            )

    raise MaxTurnsReachedException()


class MaxTurnsReachedException(Exception):
    def __init__(self):
        super().__init__("Reached maximum number of turns")

logger = logging.getLogger(__name__)
responses_and_graphs = []  

FINANCIAL_URLS = [
        "https://www.bloomberg.com", "https://www.cnbc.com", "https://finance.yahoo.com", 
        "https://www.tradingview.com", "https://data.worldbank.org", "https://www.imf.org/en/Data",
        "https://www.statista.com/topics/1723/finance/", "https://coinmarketcap.com", "https://www.coindesk.com", 
        "https://www.cryptocompare.com", "https://www.nerdwallet.com", 
        "https://www.borsaistanbul.com", "https://www.tcmb.gov.tr", "https://ec.europa.eu/eurostat", 
        "https://www.khanacademy.org/economics-finance-domain", "https://hbr.org", "https://stats.oecd.org",
        "https://www.worldbank.org", "https://www.imf.org", "https://fred.stlouisfed.org", 
        "https://www.nber.org", "https://www.ft.com/global-economy", "https://www.bbc.com/news/business/economy"
    ]
Settings.llm = OpenAI(model="gpt-4o", temperature=0.2)
Settings.embed_model = OpenAIEmbedding()

class FinancialRAGSystem:
    def __init__(self, persist_dir: str = "./financial_data_store"):
        self.persist_dir = persist_dir
        self.index = None
        self.node_parser = SimpleNodeParser.from_defaults(
            chunk_size=1000,
            chunk_overlap=200
        )

    def scrape_financial_data(self, urls: List[str]) -> List[Document]:
        documents = []
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        for url in urls:
            try:
                logger.info(f"Scraping {url}")
                response = requests.get(url, headers=headers, timeout=10)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.text, 'html.parser')
                for script in soup(["script", "style"]):
                    script.decompose()
                
                text = soup.get_text(separator='\n')
                clean_text = "\n".join(
                    line.strip() for line in text.splitlines() 
                    if line.strip() and len(line.strip()) > 50
                )
                
                doc = Document(
                    text=clean_text,
                    metadata={"source": url}
                )
                documents.append(doc)
                
            except Exception as e:
                logger.error(f"Error scraping {url}: {e}")
        
        return documents

    def build_or_load_index(self, urls: List[str] = None):
        if os.path.exists(self.persist_dir):
            logger.info("Loading existing index...")
            storage_context = StorageContext.from_defaults(persist_dir=self.persist_dir)
            self.index = load_index_from_storage(storage_context)
        else:
            if not urls:
                raise ValueError("URLs required to build new index")
            
            logger.info("Building new index...")
            documents = self.scrape_financial_data(urls)
        
            self.index = VectorStoreIndex.from_documents(
                documents,
                show_progress=True
            )
            self.index.storage_context.persist(persist_dir=self.persist_dir)

    def query_system(self, query: str, response_mode: str = "compact") -> str:
        if not self.index:
            raise ValueError("Index not initialized. Call build_or_load_index first.")
        
        try:
            query_engine = self.index.as_query_engine(
                response_mode=response_mode,
                similarity_top_k=5,
                node_postprocessors=[],
                streaming=True
            )
            
            response = query_engine.query(
                f"""You are an expert financial analyst and market specialist with deep knowledge of global markets, currencies, commodities, and economic indicators. Your task is to provide a comprehensive analysis based on the following request:

                    {query}

                    Please structure your response in the following detailed format:

                    1. MARKET OVERVIEW
                    - Current Market Sentiment: [Bullish/Bearish/Neutral]
                    - Key Market Indicators:
                        * VIX (Volatility Index)
                        * Major Index Performances
                        * Global Market Trends

                    2. CURRENCY ANALYSIS
                    - Base Currency: [Specify]
                    - Key Exchange Rates:
                        * USD/TRY
                        * EUR/TRY
                        * GBP/TRY
                    - Currency Trend Analysis
                    - Forex Market Impact

                    3. COMMODITY INSIGHTS
                    - Gold Prices and Trends:
                        * XAU/USD
                        * Gold Gram/TRY
                    - Oil Markets:
                        * Brent Oil
                        * WTI Crude
                    - Other Relevant Commodities:
                        * Silver
                        * Natural Gas
                        * Industrial Metals

                    4. ECONOMIC INDICATORS
                    - Inflation Rates
                    - Interest Rates
                    - Employment Data
                    - GDP Growth
                    - Industrial Production

                    5. TECHNICAL ANALYSIS
                    - Support/Resistance Levels
                    - Moving Averages
                    - RSI Indicators
                    - Trading Volume Analysis

                    6. RISK ASSESSMENT
                    - Market Risks
                    - Geopolitical Factors
                    - Economic Policy Impacts
                    - Volatility Measures

                    7. FUTURE OUTLOOK
                    - Short-term Projections (1-3 months)
                    - Medium-term Outlook (3-6 months)
                    - Long-term Forecast (6-12 months)

                    8. ACTIONABLE INSIGHTS
                    - Key Trading/Investment Opportunities
                    - Risk Management Strategies
                    - Portfolio Diversification Recommendations

                    Please provide your analysis based on the most recent data from trusted sources including:
                    - Bloomberg
                    - Reuters
                    - Central Bank Data (TCMB, FED, ECB)
                    - Major Financial Institutions
                    - Market Analysis Platforms

                    Additional Guidelines:
                    - Cite specific sources for key data points
                    - Include relevant time stamps for market data
                    - Highlight any unusual market movements or anomalies
                    - Provide context for significant market events
                    - Consider both technical and fundamental factors
                    - Analyze cross-market correlations
                    - Include relevant regulatory/policy impacts

                    Your analysis should be:
                    1. Data-driven and objective
                    2. Contextually relevant
                    3. Forward-looking
                    4. Risk-aware
                    5. Actionable

                    """
            )
            
            return str(response)
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return f"Error processing your query: {str(e)}"
        
        
class FinancialDataAgent:
    def __init__(self):
        self.model = None

    def set_model(self, model_name):
        self.model = model_name
        if model_name.startswith("claude-"):
            self.provider = "anthropic" 
        elif model_name.startswith("gpt-"):
            self.provider = "openai" 
        else:
            self.provider = "unknown" 

    def get_answer(self, question):
        messages = [
            {
                "role": "system",
                "content": (
                    f"""
                    You are a highly knowledgeable financial assistant with expertise in corporate finance, investment analysis, personal financial planning, and global economic trends. Your goal is to provide precise, actionable, and well-reasoned financial advice in a clear and professional tone. You can analyze complex datasets, interpret financial statements, and offer insights on topics like budgeting, savings, debt management, risk assessment, and market trends.

                    When responding:

                    Ensure clarity and simplicity, regardless of the complexity of the topic.
                    Use numerical examples, charts, or calculations where applicable to illustrate key points.
                    Highlight potential risks and opportunities in financial decisions.
                    Reference credible financial frameworks, tools, or practices to support your answers.
                    Adapt your tone and style to fit the context—formal for corporate clients, approachable for personal finance inquiries.
                    Scenario Examples:

                    A client asks, "What investment options should I consider for high returns within a 5-year timeframe?"
                    A company needs advice on managing liquidity during a market downturn.
                    An individual wants guidance on creating a retirement savings plan.
                    An entrepreneur seeks insights on funding options for a new business.
                    Provide comprehensive, insightful responses to queries like these, ensuring value in every interaction.
                    
                    Today is {datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}.
                    
                    IF YOU WANTED TO GIVE A PYTHON CODE, DO NOT ADD ANY EXPLANATIONS, JUST GIVE THE CODE!!!!
                """
                ),
            },
            {
                "role": "user",
                "content": 
                    question
            },
        ]
        response = completion(
            model=self.model,
            messages=messages,
            max_tokens=4096
        )
        return response["choices"][0]["message"]["content"]

    def generate_code_for_graph(self, question, graph_type):
        graph_code_prompt = (
            f"{question}\n"
            f"Please generate Python code to create a {graph_type.lower()}. Just give the code, do not make any explanations before or after the code. Ensure the code is complete and syntactically correct."
        )
        return self.get_answer(graph_code_prompt)

agent = FinancialDataAgent()

def generate_response_and_code(selected_model, question, graph_type):
    agent.set_model(selected_model)
    llm_response = agent.get_answer(question)
    print("LLM Response:\n", llm_response)
    graph_image_path = None 

    if graph_type and graph_type != "No Graphic Requested":
        if graph_type == "Prediction w/ Linear Regression":
            try:
                data_request = (
                    "Provide a small example dataset (in JSON format) for a simple linear regression task. "
                    "The dataset should have two numerical variables, 'X' and 'y', with 20 data points. "
                    "DO NOT ADD ANY EXPLANATIONS BEFORE OR AFTER, JUST GIVE THE DATASET!!!!!!!"
                )
                data_response = agent.get_answer(data_request)
                data_response = data_response.replace("```json", "").replace("```", "").strip()
                print("Data Response:\n", data_response)

                try:
                    dataset = json.loads(data_response) 

                    if isinstance(dataset, list):
                        points = dataset  
                    else:
                        raise ValueError("Unexpected JSON format: Expected a list of dictionaries.")

                    X = np.array([point["X"] for point in points]).reshape(-1, 1)
                    y = np.array([point["y"] for point in points])
                except json.JSONDecodeError as json_error:
                    raise ValueError(f"Error parsing JSON data: {str(json_error)}")
                except KeyError as key_error:
                    raise ValueError(f"Missing expected keys in data: {str(key_error)}")

                X_train, X_test = X[:-5], X[-5:]
                y_train, y_test = y[:-5], y[-5:]

                regr = linear_model.LinearRegression()
                regr.fit(X_train, y_train)
                y_pred = regr.predict(X_test)

                print("Coefficients:", regr.coef_)
                print("Intercept:", regr.intercept_)
                print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
                print("R2 Score:", r2_score(y_test, y_pred))

                plt.figure()
                plt.scatter(X_test, y_test, color="black", label="Actual Data")
                plt.plot(X_test, y_pred, color="blue", linewidth=3, label="Prediction")
                plt.legend()
                plt.title("Linear Regression Prediction")
                plt.xlabel("Feature (X)")
                plt.ylabel("Target (y)")

                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
                    graph_image_path = temp_file.name
                    plt.savefig(graph_image_path)
                    plt.close()

                llm_response += f"\nCoefficients: {regr.coef_}\n"
                llm_response += f"Intercept: {regr.intercept_}\n"
                llm_response += f"Mean squared error: {mean_squared_error(y_test, y_pred):.2f}\n"
                llm_response += f"Coefficient of determination (R2): {r2_score(y_test, y_pred):.2f}\n"
            except Exception as e:
                llm_response += f"\nError during linear regression execution: {str(e)}"


        else:
            graph_code = agent.generate_code_for_graph(question, graph_type)
            graph_code = graph_code.replace("```python", "")
            graph_code = graph_code.replace("```", "")
            graph_code = graph_code.replace("``}", "")
            print("Generated Graph Code:\n", graph_code)
            
            try:
                exec_globals = {}
                exec_locals = {"plt": plt}
                exec(graph_code, exec_globals, exec_locals)
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
                    graph_image_path = temp_file.name
                    plt.savefig(graph_image_path)
                    plt.close()
            except Exception as e:
                llm_response += f"\nError in executing graph code: {str(e)}"
                graph_image_path = None

    responses_and_graphs.append({"response": llm_response, "graph": graph_image_path})
    return llm_response, graph_image_path


def save_all_responses_and_graphs_to_pdf():
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        pdf_path = temp_pdf.name
        pdf = SimpleDocTemplate(pdf_path, pagesize=letter)
        elements = []
        styles = getSampleStyleSheet()
        
        elements.append(Paragraph("Financial Report", styles['Heading1']))
        elements.append(Spacer(1, 24))

        for item in responses_and_graphs:
            llm_response = item["response"]
            graph_image_path = item.get("graph")

            html_response = markdown(llm_response)
            elements.append(Paragraph(html_response, styles['BodyText']))
            elements.append(Spacer(1, 12))

            if graph_image_path:
                img = Image(graph_image_path, width=400, height=300)
                elements.append(img)
                elements.append(Spacer(1, 24))

        pdf.build(elements)
    return pdf_path

def download_pdf():
    return save_all_responses_and_graphs_to_pdf()

def decide_action(model_dropdown, question_input, graph_type_radio, response_type):
    ro, go = generate_response_and_code(model_dropdown, question_input, graph_type_radio)
    
    if (response_type == "Get Information from Finance Agent"):
        response_output, graph_code_output = generate_response_and_code(model_dropdown, question_input, graph_type_radio)
        return response_output, graph_code_output 
    elif (response_type == "Get Info w/ RAG"):
        rag_system = FinancialRAGSystem()
        rag_system.build_or_load_index(urls=FINANCIAL_URLS)
        response = rag_system.query_system(question_input)
        code_graph = generate_code_for_graph()
        return response, go
    elif (response_type == "Get Info w/ API Calls"):
        answer = run(question_input)
        return answer, go
        
        


with gr.Blocks() as demo:
    with gr.Row():
        model_dropdown = gr.Dropdown(
            choices=["perplexity/llama-3.1-sonar-large-128k-online", "claude-3-opus-20240229", "claude-3-sonnet-20240229", 
                     "claude-3-haiku-20240307", "gpt-4o", "gpt-3.5-turbo", "gpt-4o-mini", "gpt-4-turbo", "gpt-4"],
            value="perplexity/llama-3.1-sonar-large-128k-online",
            label="Model"
        )
        question_input = gr.Textbox(lines=2, placeholder="Select...", label="Question")
        
        response_type = gr.Radio(
            ["Get Information from Finance Agent", "Get Info w/ RAG", "Get Info w/ API Calls"], 
            label="Select Search Type"
        )
        graph_type_radio = gr.Radio(
            ["Line Graph", "Bar Graph", "Pie Chart", "Area Graph", "Spread Chart", 
             "Histogram", "Box Plot", "Heatmap", "Prediction w/ Linear Regression", 
             "No Graphic Requested"], 
            label="Select Graph Type"
        )
    
    with gr.Row():
        response_output = gr.Textbox(label="Response")
        graph_code_output = gr.Image(label="Generated Graph")
    
    generate_btn = gr.Button("Generate Response and Code")
    
 
    generate_btn.click(
        decide_action, 
        inputs=[model_dropdown, question_input, graph_type_radio, response_type], 
        outputs=[response_output, graph_code_output]
    )
    
    with gr.Row():
        download_btn = gr.Button("Download All Responses and Graphs as PDF")
        pdf_file_output = gr.File(label="Download PDF")
        download_btn.click(
            download_pdf, 
            inputs=[], 
            outputs=pdf_file_output
        )

demo.launch()
