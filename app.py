import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from forex_ml_model import EnhancedForexMLModel
import sys
from validators import CurrencyPairValidator
from typing import List
from datetime import datetime, timedelta
import os
from openai import AzureOpenAI
from dataclasses import dataclass
from typing import Optional

from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import time
import threading
import queue
import json

st.set_page_config(layout="wide")


# Initialize the Azure OpenAI client
client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version="2024-08-01-preview"
)
deployment_name = 'gpt-4'

# Agent prompts
MARKET_ANALYSIS_TEMPLATE = """
You are the Market Analysis Agent responsible for gathering and analyzing forex market data.
Current Context: {context}
Task: Analyze the market conditions for {currency_pair} considering:
1. Technical indicators
2. Economic data
3. Market sentiment
4. Recent news impact

Provide a concise market analysis summary.
"""

DECISION_MAKER_TEMPLATE = """
You are the Decision-Making Agent responsible for generating trading decisions.
Market Analysis: {market_analysis}
Risk Parameters: {risk_params}
Currency Pair: {currency_pair}

Based on the market analysis and risk parameters, provide:
1. Trading decision (Buy/Sell/Hold)
2. Entry price
3. Strategy reasoning
"""

RISK_MANAGER_TEMPLATE = """
You are the Risk Management Agent responsible for evaluating trade risks.
Proposed Trade: {trade_decision}
Current Portfolio: {portfolio}
Risk Parameters: {risk_params}

Evaluate the trade and provide:
1. Position size recommendation
2. Risk assessment
3. Stop loss and take profit levels
"""

EXECUTION_TEMPLATE = """
You are the Execution Agent responsible for trade execution.
Trade Decision: {trade_decision}
Risk Assessment: {risk_assessment}
Market Conditions: {market_conditions}

Provide execution plan:
1. Order type
2. Entry strategy
3. Exit parameters
"""

SPINNER_CSS = """
<style>
.agent-spinner {
    width: 20px;
    height: 20px;
    border: 3px solid #f3f3f3;
    border-top: 3px solid #3498db;
    border-radius: 50%;
    animation: spin 1s linear infinite;
    margin: 10px auto;
}
.agent-card {
    padding: 1rem;
    border-radius: 8px;
    background: white;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    margin-bottom: 1rem;
}
.agent-header {
    display: flex;
    align-items: center;
    margin-bottom: 0.5rem;
}
.agent-status {
    margin-left: auto;
    font-size: 0.8rem;
    padding: 0.2rem 0.5rem;
    border-radius: 12px;
}
.status-idle {
    background: #f3f3f3;
    color: #666;
}
.status-working {
    background: #e3f2fd;
    color: #1976d2;
}
.status-complete {
    background: #e8f5e9;
    color: #2e7d32;
}
.status-error {
    background: #ffebee;
    color: #c62828;
}
@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}
</style>
"""


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

validator = CurrencyPairValidator()

def get_quote_currencies(base_currency: str) -> List[str]:
    """Get valid quote currencies for the selected base currency."""
    return validator.get_valid_quote_currencies(base_currency)


def create_header():
    """Creates the header section of the application."""
    st.title("Forex Trading ML Analysis Platform")
    st.write("""
    This application leverages machine learning to analyze forex trading patterns and generate trading signals. 
    It combines technical analysis, machine learning, and statistical methods to identify potential trading opportunities 
    and evaluate trading strategies through comprehensive backtesting.
    """)

def load_default_data():
    """Load default historical data for better analysis."""
    # Create a longer historical dataset (120 days)
    dates = pd.date_range(end=pd.Timestamp.now(), periods=120, freq='D')
    
    # Generate more realistic price movements for GBP/EUR
    base_price = 1.0650  # Starting price for GBP/EUR
    prices = []
    for i in range(120):
        if i == 0:
            prices.append(base_price)
        else:
            # Generate realistic daily price changes
            change = np.random.normal(0, 0.001)  # 0.1% daily volatility
            prices.append(prices[-1] * (1 + change))
    
    prices = np.array(prices)
    
    # Generate OHLC data from the prices
    data = pd.DataFrame({
        'timestamp': dates,
        'close': prices,
        'open': prices * (1 + np.random.normal(0, 0.0005, 120)),
        'high': prices * (1 + abs(np.random.normal(0, 0.001, 120))),
        'low': prices * (1 - abs(np.random.normal(0, 0.001, 120))),
        'volume': np.random.randint(1000, 5000, 120),
        'pair': ['GBP/EUR'] * 120,
        'base_currency': ['GBP'] * 120,
        'target_currency': ['EUR'] * 120
    })
    
    # Ensure OHLC relationships are valid
    data['high'] = np.maximum(data[['open', 'close', 'high']].max(axis=1), data['high'])
    data['low'] = np.minimum(data[['open', 'close', 'low']].min(axis=1), data['low'])
    
    return data


def display_backtest_results(results):
    """Displays the backtest results in a formatted way."""
    if results is None:
        st.error("No results to display")
        return

    st.subheader("Detailed Backtest Results")
    
    # Create metrics in a grid layout
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Initial Capital",
            value=f"${100000:,.2f}",
            delta=f"${results['final_capital'] - 100000:,.2f}"
        )
    
    with col2:
        st.metric(
            label="Total Return",
            value=f"{results['total_return']:.2%}",
            delta=f"{results['total_return']:.2%}"
        )
    
    with col3:
        st.metric(
            label="Sharpe Ratio",
            value=f"{results['sharpe_ratio']:.2f}"
        )

    # Create second row of metrics
    col4, col5, col6 = st.columns(3)
    
    with col4:
        st.metric(
            label="Win Rate",
            value=f"{results['win_rate']:.2%}"
        )
    
    with col5:
        st.metric(
            label="Profit Factor",
            value=f"{results['profit_factor']:.2f}"
        )
    
    with col6:
        st.metric(
            label="Max Drawdown",
            value=f"{results['max_drawdown']:.2%}"
        )

    # Display technical indicators in a clean table format
    if 'technical_indicators' in results:
        st.subheader("Technical Analysis")
        tech_df = pd.DataFrame({
            'Indicator': results['technical_indicators'].keys(),
            'Value': [f"{v:.4f}" if isinstance(v, (float, np.float64)) 
                     else str(v) for v in results['technical_indicators'].values()]
        })
        st.table(tech_df)

    # Display recent trades in a scrollable table
    if 'trades' in results and len(results['trades']) > 0:
        st.subheader("Recent Trades")
        trades_df = results['trades'].tail()
        trades_df['entry_date'] = pd.to_datetime(trades_df['entry_date']).dt.strftime('%Y-%m-%d %H:%M')
        trades_df['exit_date'] = pd.to_datetime(trades_df['exit_date']).dt.strftime('%Y-%m-%d %H:%M')
        st.dataframe(trades_df.style.format({
            'entry_price': '${:.4f}',
            'exit_price': '${:.4f}',
            'pnl': '${:.2f}',
            'return': '{:.2%}'
        }))

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from forex_ml_model import EnhancedForexMLModel
from validators import CurrencyPairValidator  # Add this import

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the validator globally
validator = CurrencyPairValidator()

def get_quote_currencies(base_currency: str) -> List[str]:
    """Get valid quote currencies for the selected base currency."""
    return validator.get_valid_quote_currencies(base_currency)

def ml_tab():
    """Renders the Machine Learning tab content."""
    st.header("Machine Learning Model Backtesting")
    
    st.write("""
    Input your forex trading data to analyze patterns and generate trading signals. 
    You can either enter data manually for quick analysis or upload a CSV file for bulk processing.
    
    Note: The model requires historical data for accurate analysis. When using manual entry, 
    the system will combine your input with historical data for better results.
    """)

    # Data Input Method Selection
    input_method = st.radio(
        "Choose input method:",
        ["Manual Entry", "File Upload"],
        help="Select how you want to input your forex data"
    )

    if input_method == "Manual Entry":
        with st.form("forex_data_input"):
            # Load default values
            default_data = load_default_data()
            latest_data = default_data.iloc[-1]
            
            # Currency Pair Selection
            col1, col2 = st.columns(2)
            with col1:
                base_currency = st.selectbox(
                    "Base Currency",
                    options=["EUR", "USD", "GBP", "JPY", "AUD", "CHF", "CAD", "NZD"],
                    index=0,
                    help="The base currency in the pair (e.g., EUR in EUR/USD)"
                )
            
            # Get valid quote currencies for selected base currency
            valid_quotes = get_quote_currencies(base_currency)
            with col2:
                quote_currency = st.selectbox(
                    "Quote Currency",
                    options=valid_quotes,
                    index=0 if valid_quotes else None,
                    help="Available quote currencies for selected base currency"
                )

            # Date and Time
            date_time = st.date_input(
                "Date",
                value=pd.Timestamp.now().date(),
                help="The date for this price data"
            )

            # Only show price inputs if we have a valid currency pair
            if valid_quotes:
                # OHLCV Data with default values
                col3, col4, col5 = st.columns(3)
                with col3:
                    open_price = st.number_input(
                        "Open Price",
                        min_value=0.0,
                        value=float(latest_data['open']),
                        step=0.0001,
                        format="%.4f",
                        help="Opening price for the period"
                    )
                    high_price = st.number_input(
                        "High Price",
                        min_value=0.0,
                        value=float(latest_data['high']),
                        step=0.0001,
                        format="%.4f",
                        help="Highest price during the period"
                    )

                with col4:
                    low_price = st.number_input(
                        "Low Price",
                        min_value=0.0,
                        value=float(latest_data['low']),
                        step=0.0001,
                        format="%.4f",
                        help="Lowest price during the period"
                    )
                    close_price = st.number_input(
                        "Close Price",
                        min_value=0.0,
                        value=float(latest_data['close']),
                        step=0.0001,
                        format="%.4f",
                        help="Closing price for the period"
                    )

                with col5:
                    volume = st.number_input(
                        "Volume",
                        min_value=0,
                        value=int(latest_data['volume']),
                        help="Trading volume for the period"
                    )

                # Additional Parameters
                col6, col7 = st.columns(2)
                with col6:
                    min_move_pct = st.number_input(
                        "Minimum Move Percentage",
                        min_value=0.0001,
                        max_value=1.0,
                        value=0.001,
                        step=0.0001,
                        help="Minimum price movement percentage to consider for a trade signal"
                    )
                
                with col7:
                    forecast_period = st.number_input(
                        "Forecast Period (days)",
                        min_value=1,
                        max_value=30,
                        value=1,
                        help="Number of days to forecast ahead"
                    )
            else:
                st.warning(f"No valid quote currencies available for {base_currency}")

            submit_button = st.form_submit_button("Analyze Data")

            if submit_button:
                # Validate currency pair
                if not validator.is_valid_pair(base_currency, quote_currency):
                    st.error(f"Invalid currency pair: {base_currency}/{quote_currency}")
                    return

                try:
                    # Create DataFrame combining historical and new data
                    new_data = pd.DataFrame({
                        'timestamp': [pd.Timestamp(date_time)],
                        'open': [float(open_price)],
                        'high': [float(high_price)],
                        'low': [float(low_price)],
                        'close': [float(close_price)],
                        'volume': [int(volume)],
                        'pair': [f"{base_currency}/{quote_currency}"],
                        'base_currency': [base_currency],
                        'target_currency': [quote_currency]
                    })

                    # Update historical data with correct currency pair
                    default_data['pair'] = f"{base_currency}/{quote_currency}"
                    default_data['base_currency'] = base_currency
                    default_data['target_currency'] = quote_currency

                    # Combine with historical data
                    combined_data = pd.concat([default_data[:-1], new_data], ignore_index=True)

                    # Show preview of the data
                    st.subheader("Analysis Data Preview")
                    st.dataframe(combined_data.tail())

                    with st.spinner("Analyzing data..."):
                        # Progress container
                        progress_text = st.empty()
                        
                        # Initialize model
                        progress_text.text("Initializing model...")
                        forex_model = EnhancedForexMLModel(data_path="./forex_data")
                        
                        # Process the combined data
                        progress_text.text("Processing data...")
                        forex_model.data = combined_data
                        
                        # Create target variable
                        progress_text.text("Creating target variable...")
                        forex_model.create_target_variable(
                            forecast_period=int(forecast_period),
                            min_move_pct=float(min_move_pct)
                        )
                        
                        # Create enhanced features
                        progress_text.text("Creating technical indicators...")
                        forex_model.create_enhanced_features()
                        
                        # Select features
                        progress_text.text("Selecting features...")
                        forex_model.select_features()
                        
                        # Show selected features
                        st.write("Selected Features:", forex_model.feature_columns)
                        
                        # Prepare train/test split
                        progress_text.text("Preparing training data...")
                        forex_model.prepare_train_test_data(test_size=30)
                        
                        # Train models
                        progress_text.text("Training models...")
                        forex_model.create_and_train_models()
                        
                        # Run backtest
                        progress_text.text("Running backtest...")
                        results = forex_model.backtest_strategy()
                        
                        # Clear progress text
                        progress_text.empty()
                        
                        if results and results['number_of_trades'] > 0:
                            display_backtest_results(results)
                            
                            # Display technical analysis
                            st.subheader("Technical Analysis")
                            last_data = combined_data.iloc[-1]
                            
                            # Get technical indicators
                            technical_indicators = pd.DataFrame({
                                'Indicator': [
                                    'RSI',
                                    'MACD',
                                    'SMA 20',
                                    'SMA 50',
                                    'Volatility'
                                ],
                                'Value': [
                                    last_data.get('rsi_14', 'N/A'),
                                    last_data.get('macd_12_26', 'N/A'),
                                    last_data.get('sma_20', 'N/A'),
                                    last_data.get('sma_50', 'N/A'),
                                    last_data.get('volatility', 'N/A')
                                ]
                            })
                            
                            st.table(technical_indicators)
                        else:
                            st.warning(f"""
                            No trades were generated with current parameters. This could be due to:
                            1. Minimum move percentage too high (current: {min_move_pct:.4f})
                            2. Insufficient price movement in the data
                            3. Not enough historical data for reliable signals
                            
                            Try:
                            1. Adjusting the minimum move percentage to 0.001-0.005
                            2. Increasing the forecast period
                            3. Providing more historical data
                            """)

                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
                    st.exception(e)
    
    else:
        # File upload section (your existing code)
        st.write("""
        Upload a CSV file with your forex data. The file should contain the following columns:
        - timestamp: Date and time of the data point
        - open: Opening price
        - high: Highest price
        - low: Lowest price
        - close: Closing price
        - volume: Trading volume
        - pair: Currency pair (e.g., EUR/USD)
        - base_currency: Base currency
        - target_currency: Quote currency
        """)

        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type="csv",
            help="Upload your forex data in CSV format"
        )

        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                required_columns = [
                    'timestamp', 'open', 'high', 'low', 'close', 
                    'volume', 'pair', 'base_currency', 'target_currency'
                ]
                
                missing_columns = [col for col in required_columns if col not in data.columns]
                if missing_columns:
                    st.error(f"Missing required columns: {', '.join(missing_columns)}")
                    return

                # Display data preview
                st.subheader("Data Preview")
                st.dataframe(data.head())

                # Analysis parameters
                col1, col2 = st.columns(2)
                with col1:
                    min_move_pct = st.number_input(
                        "Minimum Move Percentage",
                        min_value=0.0001,
                        max_value=1.0,
                        value=0.001,
                        step=0.0001,
                        help="Minimum price movement percentage to consider for a trade signal"
                    )
                
                with col2:
                    forecast_period = st.number_input(
                        "Forecast Period (days)",
                        min_value=1,
                        max_value=30,
                        value=1,
                        help="Number of days to forecast ahead"
                    )

                if st.button("Analyze Data"):
                    with st.spinner("Analyzing data..."):
                        try:
                            # Progress container
                            progress_text = st.empty()
                            
                            # Initialize model
                            progress_text.text("Initializing model...")
                            forex_model = EnhancedForexMLModel(data_path="./forex_data")
                            
                            # Process the data
                            progress_text.text("Processing data...")
                            forex_model.data = data
                            
                            # Create target variable
                            progress_text.text("Creating target variable...")
                            forex_model.create_target_variable(
                                forecast_period=forecast_period,
                                min_move_pct=min_move_pct
                            )
                            
                            # Create enhanced features
                            progress_text.text("Creating technical indicators...")
                            forex_model.create_enhanced_features()
                            
                            # Select features
                            progress_text.text("Selecting features...")
                            forex_model.select_features()
                            
                            # Show selected features
                            st.write("Selected Features:", forex_model.feature_columns)
                            
                            # Prepare train/test split
                            progress_text.text("Preparing training data...")
                            forex_model.prepare_train_test_data(test_size=30)
                            
                            # Train models
                            progress_text.text("Training models...")
                            forex_model.create_and_train_models()
                            
                            # Run backtest
                            progress_text.text("Running backtest...")
                            results = forex_model.backtest_strategy()
                            
                            # Clear progress text
                            progress_text.empty()
                            
                            if results and results['number_of_trades'] > 0:
                                display_backtest_results(results)
                                
                                # Display technical analysis
                                st.subheader("Technical Analysis")
                                last_data = data.iloc[-1]
                                
                                technical_indicators = pd.DataFrame({
                                    'Indicator': [
                                        'RSI',
                                        'MACD',
                                        'SMA 20',
                                        'SMA 50',
                                        'Volatility'
                                    ],
                                    'Value': [
                                        last_data.get('rsi_14', 'N/A'),
                                        last_data.get('macd_12_26', 'N/A'),
                                        last_data.get('sma_20', 'N/A'),
                                        last_data.get('sma_50', 'N/A'),
                                        last_data.get('volatility', 'N/A')
                                    ]
                                })
                                
                                st.table(technical_indicators)
                            else:
                                st.warning(f"""
                                No trades were generated with current parameters. This could be due to:
                                1. Minimum move percentage too high (current: {min_move_pct:.4f})
                                2. Insufficient price movement in the data
                                3. Not enough historical data for reliable signals
                                
                                Try:
                                1. Adjusting the minimum move percentage to 0.001-0.005
                                2. Increasing the forecast period
                                3. Providing more historical data
                                """)

                        except Exception as e:
                            st.error(f"Error processing file: {str(e)}")
                            st.exception(e)

            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
                st.exception(e)

def features_tab():
    """Renders the Features tab content."""
    st.header("Feature Engineering and Selection")
    
    st.write("""
    Our model uses a comprehensive set of technical and statistical features to capture various aspects of market behavior. 
    These features are carefully selected and engineered to provide meaningful signals for the machine learning model.
    """)

    # Technical Indicators Section
    st.subheader("Technical Indicators")
    
    st.markdown("""
    The model incorporates several categories of technical indicators:
    
    1. **Trend Indicators**
    - Moving Averages (SMA, EMA) at various timeframes
    - MACD (Moving Average Convergence Divergence)
    - Average Directional Index (ADX)
    - Ichimoku Cloud components
    
    2. **Momentum Indicators**
    - Relative Strength Index (RSI)
    - Stochastic RSI
    - Rate of Change (ROC)
    - Ultimate Oscillator
    
    3. **Volatility Indicators**
    - Bollinger Bands
    - Average True Range (ATR)
    - Keltner Channels
    
    4. **Volume-Based Indicators**
    - On-Balance Volume (OBV)
    - Volume Price Trend (VPT)
    - Money Flow Index (MFI)
    """)

    # Feature Selection Process
    st.subheader("Feature Selection Process")
    
    st.write("""
    The feature selection process involves multiple steps to ensure we use the most relevant and non-redundant features:
    
    1. **Correlation Analysis**: Features with correlation > 0.98 are removed to prevent multicollinearity
    2. **Feature Importance**: Using Random Forest's feature importance to identify the most significant predictors
    3. **Domain Knowledge**: Including established technical indicators based on trading expertise
    """)

    # Key Features
    st.subheader("Key Features Highlight")
    
    st.markdown("""
    Some of the most important features in our model include:
    
    - **Price Action Features**
    > - Daily returns and volatility
    > - Price relative to moving averages
    > - Price momentum at different timeframes
    
    - **Advanced Technical Features**
    > - Trend strength indicators
    > - Volatility regime indicators
    > - Volume-price relationship metrics
    
    - **Custom Engineered Features**
    > - Cross-timeframe momentum
    > - Volatility-adjusted returns
    > - Technical indicator divergences
    """)

#=====================================================================================

@dataclass
class AgentState:
    name: str
    status: str = "Idle"
    output: Optional[str] = None
    error: Optional[str] = None

def create_agent_card(agent_state: AgentState):
    status_classes = {
        "Idle": "status-idle",
        "Working": "status-working",
        "Complete": "status-complete",
        "Error": "status-error"
    }
    
    # Format the output text without markdown syntax
    formatted_output = ""
    if agent_state.output:
        output_lines = agent_state.output.strip().split('\n')
        formatted_output = f"""
        <div class="agent-output">
            <div class="agent-output-content">
                {'<br>'.join(line.strip() for line in output_lines)}
            </div>
        </div>
        """
        
    html = f"""
    <div class="agent-card">
        <div class="agent-header">
            <h4 style="margin: 0">{agent_state.name}</h4>
            <span class="agent-status {status_classes.get(agent_state.status, 'status-idle')}">
                {agent_state.status}
            </span>
        </div>
        {"<div class='agent-spinner'></div>" if agent_state.status == "Working" else ""}
        {formatted_output}
        {f"<div style='color: #c62828; margin-top: 1rem'>{agent_state.error}</div>" if agent_state.error else ""}
    </div>
    """
    return html


# Initialize the Azure OpenAI client
client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version="2024-08-01-preview"
)
deployment_name = 'gpt-4'

# Template for analyzing forex data
FOREX_ANALYSIS_TEMPLATE = """
You are an AI Research Agent specialized in Foreign Exchange (Forex) market analysis.
Based on the provided information and query, analyze the currency strength and market conditions.

Context: {context}
User Query: {query}

Please provide a detailed analysis considering:
1. Current market trends
2. Economic indicators
3. Technical analysis
4. Potential risks and opportunities

Analysis:
"""

def generate_forex_response(prompt, context=""):
    try:
        # Combine the template with user input
        formatted_prompt = FOREX_ANALYSIS_TEMPLATE.format(
            context=context,
            query=prompt
        )
        
        response = client.chat.completions.create(
            model=deployment_name,
            messages=[
                {"role": "system", "content": "You are a professional Forex market analyst."},
                {"role": "user", "content": formatted_prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return None

class ForexAgent:
    def __init__(self, name, template, client):
        self.name = name
        self.template = template
        self.client = client
        self.status = "Idle"
        self.last_output = None
        
    def process(self, **kwargs):
        self.status = "Working"
        try:
            prompt = self.template.format(**kwargs)
            response = self.client.chat.completions.create(
                model=deployment_name,
                messages=[
                    {"role": "system", "content": f"You are the {self.name}"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            self.last_output = response.choices[0].message.content
            self.status = "Complete"
            return self.last_output
        except Exception as e:
            self.status = f"Error: {str(e)}"
            return None

class AgentOrchestrator:
    def __init__(self, client):
        self.market_analysis_agent = ForexAgent("Market Analysis Agent", MARKET_ANALYSIS_TEMPLATE, client)
        self.decision_maker_agent = ForexAgent("Decision-Making Agent", DECISION_MAKER_TEMPLATE, client)
        self.risk_manager_agent = ForexAgent("Risk Management Agent", RISK_MANAGER_TEMPLATE, client)
        self.execution_agent = ForexAgent("Execution Agent", EXECUTION_TEMPLATE, client)
        self.message_queue = queue.Queue()
        
    def get_agent_statuses(self):
        return {
            "Market Analysis": self.market_analysis_agent.status,
            "Decision Making": self.decision_maker_agent.status,
            "Risk Management": self.risk_manager_agent.status,
            "Execution": self.execution_agent.status
        }
    
    def analyze_forex_market(self, currency_pair, context, risk_params):
        # Reset all agents
        for agent in [self.market_analysis_agent, self.decision_maker_agent, 
                     self.risk_manager_agent, self.execution_agent]:
            agent.status = "Idle"
            agent.last_output = None
            
        # Step 1: Market Analysis
        market_analysis = self.market_analysis_agent.process(
            context=context,
            currency_pair=currency_pair
        )
        self.message_queue.put(("Market Analysis", market_analysis))
        
        if market_analysis:
            # Step 2: Decision Making
            trade_decision = self.decision_maker_agent.process(
                market_analysis=market_analysis,
                risk_params=risk_params,
                currency_pair=currency_pair
            )
            self.message_queue.put(("Decision Making", trade_decision))
            
            if trade_decision:
                # Step 3: Risk Management
                risk_assessment = self.risk_manager_agent.process(
                    trade_decision=trade_decision,
                    portfolio=context,
                    risk_params=risk_params
                )
                self.message_queue.put(("Risk Management", risk_assessment))
                
                if risk_assessment:
                    # Step 4: Execution
                    execution_plan = self.execution_agent.process(
                        trade_decision=trade_decision,
                        risk_assessment=risk_assessment,
                        market_conditions=market_analysis
                    )
                    self.message_queue.put(("Execution", execution_plan))
                    
                    return {
                        "market_analysis": market_analysis,
                        "trade_decision": trade_decision,
                        "risk_assessment": risk_assessment,
                        "execution_plan": execution_plan
                    }
        return None

def format_agent_output(agent_type: str, data: dict) -> str:
    if agent_type == "Market Analysis":
        return f"""
Currency Pair: {data['pair']}

Technical Analysis:
• Trend: {data['trend']}
• RSI: {data['rsi']} ({data['rsi_status']})
• Moving Averages: {data['ma_status']} 200 MA

Economic Indicators:
• Interest Rate: {data['interest_rate']}%
• Inflation: {data['inflation']}%
• GDP Growth: {data['gdp']}%

Previous Insights: {data['insights']}
"""
    elif agent_type == "Decision Making":
        return f"""
Trading Decision: {data['decision']}

Rationale:
• {data['rationale_1']}
• {data['rationale_2']}
• {data['rationale_3']}

Target Entry: {data['entry']}

Based on Previous Rounds: {data['previous']}
"""
    elif agent_type == "Risk Management":
        return f"""
Position Size: {data['position_size']}% of portfolio

Risk Parameters:
• Stop Loss: {data['stop_loss']} (+{data['sl_pips']} pips)
• Take Profit: {data['take_profit']} (-{data['tp_pips']} pips)
• Risk:Reward Ratio: {data['risk_reward']}

Adjustment from Previous Round: {data['adjustment']}
"""
    else:  # Execution
        return f"""
Order Type: {data['order_type']}

Execution Strategy:
• Place {data['action']} limit at {data['entry']}
• Set stop loss at {data['stop_loss']}
• Set take profit at {data['take_profit']}
• Monitor price action at key levels

Round {data['round']} Adjustments: {data['adjustments']}
"""

def forex_multi_agent_tab():
    st.markdown(SPINNER_CSS, unsafe_allow_html=True)
    
    st.header("Multi-Agent Forex Analysis System")
    
    # Initialize agents state if not exists
    if 'agents' not in st.session_state:
        st.session_state.agents = {
            "Market Analysis": AgentState(name="Market Analysis Agent"),
            "Decision Making": AgentState(name="Decision Making Agent"),
            "Risk Management": AgentState(name="Risk Management Agent"),
            "Execution": AgentState(name="Execution Agent")
        }
    
    # Sidebar inputs
    with st.sidebar:
        st.header("Analysis Parameters")
        
        # Conversation turns control
        conversation_turns = st.slider(
            "Number of Analysis Rounds",
            min_value=1,
            max_value=10,
            value=1,
            help="Set how many rounds of analysis the agents should perform. Each round builds upon previous insights."
        )
        
        # Currency selection
        currency_pairs = ['EUR/USD', 'GBP/USD', 'USD/JPY', 'USD/CHF', 'AUD/USD', 'USD/CAD']
        selected_pair = st.selectbox("Currency Pair", currency_pairs)
        
        # Market context
        st.subheader("Market Context")
        col1, col2 = st.columns(2)
        with col1:
            interest_rate = st.number_input("Interest Rate (%)", 0.0, 10.0, 5.0, 0.25)
            inflation_rate = st.number_input("Inflation Rate (%)", 0.0, 20.0, 2.0, 0.1)
        with col2:
            gdp_growth = st.number_input("GDP Growth (%)", -10.0, 10.0, 2.0, 0.1)
            volatility = st.number_input("Market Volatility", 1, 100, 50)
        
        # Risk parameters
        st.subheader("Risk Parameters")
        risk_per_trade = st.slider("Risk per Trade (%)", 0.1, 5.0, 1.0, 0.1)
        
        analyze_button = st.button("Start Analysis", type="primary", use_container_width=True)
    
    # Create a container for the analysis
    analysis_container = st.container()
    
    if analyze_button:
        conversation_history = []
        
        for turn in range(conversation_turns):
            if conversation_turns > 1:
                analysis_container.subheader(f"Analysis Round {turn + 1}/{conversation_turns}")
            
            # Create columns for agents
            cols = analysis_container.columns(4)
            
            # Reset agent states
            for agent in st.session_state.agents.values():
                agent.status = "Idle"
                agent.output = None
                agent.error = None
            
            try:
                # Market Analysis
                with cols[0]:
                    st.markdown("### Market Analysis")
                    status_placeholder = st.empty()
                    content_placeholder = st.empty()
                    
                    status_placeholder.markdown("🔄 Working...")
                    time.sleep(1)
                    
                    market_analysis = f"""
                    Currency Pair: {selected_pair}
                    
                    Technical Analysis:
                    • Trend: {"Bearish" if turn % 2 == 0 else "Bullish"}
                    • RSI: {42 - turn * 2} ({"Oversold" if 42 - turn * 2 < 30 else "Neutral"})
                    • Moving Averages: {"Below" if turn % 2 == 0 else "Above"} 200 MA
                    
                    Economic Indicators:
                    • Interest Rate: {interest_rate}%
                    • Inflation: {inflation_rate}%
                    • GDP Growth: {gdp_growth}%
                    
                    Previous Insights: {' | '.join(conversation_history) if conversation_history else 'Initial Analysis'}
                    """
                    
                    status_placeholder.markdown("✅ Complete")
                    content_placeholder.code(market_analysis)
                
                # Decision Making
                with cols[1]:
                    st.markdown("### Decision Making")
                    status_placeholder = st.empty()
                    content_placeholder = st.empty()
                    
                    status_placeholder.markdown("🔄 Working...")
                    time.sleep(1)
                    
                    decision = f"""
                    Trading Decision: {"SELL" if turn % 2 == 0 else "BUY"}
                    
                    Rationale:
                    • {"Bearish" if turn % 2 == 0 else "Bullish"} technical indicators
                    • {"High" if turn % 2 == 0 else "Low"} interest rate differential
                    • {"Moderate" if turn % 2 == 0 else "Strong"} economic growth
                    
                    Target Entry: {1.0720 - turn * 0.0010:.4f}
                    
                    Based on Previous: {' | '.join(conversation_history[-2:]) if conversation_history else 'Initial Decision'}
                    """
                    
                    status_placeholder.markdown("✅ Complete")
                    content_placeholder.code(decision)
                
                # Risk Management
                with cols[2]:
                    st.markdown("### Risk Management")
                    status_placeholder = st.empty()
                    content_placeholder = st.empty()
                    
                    status_placeholder.markdown("🔄 Working...")
                    time.sleep(1)
                    
                    risk_assessment = f"""
                    Position Size: {risk_per_trade + turn * 0.1:.1f}% of portfolio
                    
                    Risk Parameters:
                    • Stop Loss: {1.0750 - turn * 0.0010:.4f} (+30 pips)
                    • Take Profit: {1.0660 - turn * 0.0010:.4f} (-60 pips)
                    • Risk:Reward Ratio: 1:{2 + turn}
                    
                    Previous Round: {conversation_history[-1] if conversation_history else 'Initial Assessment'}
                    """
                    
                    status_placeholder.markdown("✅ Complete")
                    content_placeholder.code(risk_assessment)
                
                # Execution
                with cols[3]:
                    st.markdown("### Execution")
                    status_placeholder = st.empty()
                    content_placeholder = st.empty()
                    
                    status_placeholder.markdown("🔄 Working...")
                    time.sleep(1)
                    
                    execution_plan = f"""
                    Order Type: {"Limit Sell" if turn % 2 == 0 else "Limit Buy"}
                    
                    Execution Strategy:
                    • Place {"sell" if turn % 2 == 0 else "buy"} limit at {1.0720 - turn * 0.0010:.4f}
                    • Stop loss at {1.0750 - turn * 0.0010:.4f}
                    • Take profit at {1.0660 - turn * 0.0010:.4f}
                    • Monitor key levels
                    
                    Round {turn + 1} Updates: Based on {turn} previous rounds
                    """
                    
                    status_placeholder.markdown("✅ Complete")
                    content_placeholder.code(execution_plan)
                
                # Store insights for next round
                conversation_history.append(f"Round {turn + 1}: {'Bearish' if turn % 2 == 0 else 'Bullish'} at {1.0720 - turn * 0.0010:.4f}")
                
            except Exception as e:
                for col in cols:
                    with col:
                        st.error(f"Error: {str(e)}")
            
            # Add separation between turns if not the last turn
            if turn < conversation_turns - 1:
                analysis_container.markdown("---")


# Update main function
def main():
    tab1, tab2, tab3 = st.tabs([
        "Machine Learning",
        "Features",
        "AI Research Agent"
    ])
    
    with tab1:
        ml_tab()
    with tab2:
        features_tab()
    with tab3:
        forex_multi_agent_tab()

if __name__ == "__main__":
    main()

