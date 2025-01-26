import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score, 
                           precision_score, recall_score, f1_score, roc_auc_score)
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class EnhancedForexMLModel:
    def __init__(self, data_path: str = "./forex_data", random_state: int = 42):
        """Initialize the Enhanced Forex ML Model."""
        self.data_path = Path(data_path)
        self.random_state = random_state
        
        # Create directories
        self.model_dir = self.data_path / 'models'
        self.model_dir.mkdir(exist_ok=True)
        (self.model_dir / 'plots').mkdir(exist_ok=True)
        
        # Initialize containers
        self.models = {}
        self.cv_results = {}
        self.best_model = None
        self.feature_importance = None
        self.scaler = None
        self.selected_features = None
        self.feature_importance_history = []
        self.performance_metrics = []
        
        # Setup logging
        self._setup_logging()

    def _setup_logging(self):
        """Setup logging configuration."""
        log_file = self.data_path / 'forex_ml.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)


    def load_and_prepare_data(self):
        """Load and prepare the forex data."""
        try:
            # Load data
            data_file = self.data_path / 'processed_data' / 'forex_data_with_features.csv'
            self.data = pd.read_csv(data_file)
            self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
            
            # Sort by timestamp
            self.data = self.data.sort_values(['pair', 'timestamp'])
            self.logger.info(f"Loaded data with shape: {self.data.shape}")
            
            return True
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            return False


    def create_target_variable(self, forecast_period: int = 1, min_move_pct: float = 0.001):
        """Create target variable with improved labeling."""
        # Calculate future returns for each pair separately
        self.data['future_returns'] = self.data.groupby('pair')['close'].shift(-forecast_period) / self.data['close'] - 1
        
        # Add volatility-adjusted threshold
        lookback = 20
        rolling_std = self.data.groupby('pair')['close'].transform(
            lambda x: x.pct_change().rolling(lookback).std()
        )
        
        # Dynamic threshold based on volatility
        dynamic_threshold = np.maximum(min_move_pct, rolling_std)
        
        # Create target variable with dynamic threshold
        self.data['target'] = 0
        self.data.loc[self.data['future_returns'] > dynamic_threshold, 'target'] = 1
        self.data.loc[self.data['future_returns'] < -dynamic_threshold, 'target'] = 0
        
        # Remove rows where we can't calculate target
        self.data = self.data.dropna(subset=['target'])
        
        class_dist = self.data['target'].value_counts(normalize=True)
        self.logger.info(f"Created target variable with {len(self.data)} samples")
        self.logger.info(f"Class distribution:\n{class_dist}")

    def _calculate_atr(self, data, period=14):
        """Calculate Average True Range."""
        high = data['high']
        low = data['low']
        close = data['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr

    def create_enhanced_features(self):
        """Create enhanced features with more sophisticated technical indicators."""
        try:
            self.logger.info("Starting feature creation...")
            self.logger.info(f"Initial data shape: {self.data.shape}")
            self.logger.info(f"Initial columns: {list(self.data.columns)}")
            
            # Create a copy of the data
            enhanced_data = self.data.copy()
            
            # Process each currency pair separately
            for pair in enhanced_data['pair'].unique():
                self.logger.info(f"Processing pair: {pair}")
                mask = enhanced_data['pair'] == pair
                pair_data = enhanced_data[mask].copy()
                
                # Basic price features
                pair_data['returns'] = pair_data['close'].pct_change()
                pair_data['log_returns'] = np.log(pair_data['close']).diff()
                
                # Moving averages
                for window in [5, 10, 20, 50]:
                    pair_data[f'sma_{window}'] = pair_data['close'].rolling(window=window).mean()
                    pair_data[f'ema_{window}'] = pair_data['close'].ewm(span=window, adjust=False).mean()
                
                # Volatility
                pair_data['daily_range'] = (pair_data['high'] - pair_data['low']) / pair_data['close']
                pair_data['volatility'] = pair_data['returns'].rolling(window=20).std()
                
                # RSI
                delta = pair_data['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                pair_data['rsi'] = 100 - (100 / (1 + rs))
                
                # MACD
                exp1 = pair_data['close'].ewm(span=12, adjust=False).mean()
                exp2 = pair_data['close'].ewm(span=26, adjust=False).mean()
                pair_data['macd'] = exp1 - exp2
                pair_data['macd_signal'] = pair_data['macd'].ewm(span=9, adjust=False).mean()
                pair_data['macd_hist'] = pair_data['macd'] - pair_data['macd_signal']
                
                # Additional features
                pair_data['bb_middle'] = pair_data['close'].rolling(window=20).mean()
                bb_std = pair_data['close'].rolling(window=20).std()
                pair_data['bb_upper'] = pair_data['bb_middle'] + (bb_std * 2)
                pair_data['bb_lower'] = pair_data['bb_middle'] - (bb_std * 2)
                
                # Log new columns
                new_columns = [col for col in pair_data.columns if col not in enhanced_data.columns]
                self.logger.info(f"Created features for {pair}:")
                self.logger.info(f"New columns added: {new_columns}")
                
                # Important: Update the enhanced_data with new columns
                for col in new_columns:
                    enhanced_data.loc[mask, col] = pair_data[col]
            
            # Log NaN values before filling
            nan_counts = enhanced_data.isna().sum()
            self.logger.info("NaN counts before filling:")
            self.logger.info(nan_counts[nan_counts > 0].to_string())
            
            # Forward fill NaN values
            enhanced_data = enhanced_data.fillna(method='ffill')
            # Backward fill remaining NaN values
            enhanced_data = enhanced_data.fillna(method='bfill')
            
            # Verify data quality
            if enhanced_data.isna().any().any():
                remaining_nans = enhanced_data.isna().sum()
                self.logger.error("Remaining NaN values found:")
                self.logger.error(remaining_nans[remaining_nans > 0].to_string())
                raise ValueError("NaN values remain after feature creation")
            
            # Update the main dataframe with enhanced data
            self.data = enhanced_data
            
            # Log final state
            self.logger.info(f"Final data shape: {self.data.shape}")
            self.logger.info(f"Final columns: {list(self.data.columns)}")
            self.logger.info("Enhanced features created successfully")
            
        except Exception as e:
            self.logger.error(f"Error in create_enhanced_features: {str(e)}")
            self.logger.exception("Full traceback:")
            raise e



    def select_features(self):
        """Select relevant features and remove redundant ones."""
        try:
            self.logger.info("Starting feature selection...")
            
            # Features to exclude
            exclude_columns = {
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'pair', 'base_currency', 'target_currency', 'target',
                'future_returns'
            }
            
            # Get all numeric columns
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            self.logger.info(f"All numeric columns: {list(numeric_cols)}")
            
            # Get feature columns
            self.feature_columns = [col for col in numeric_cols if col not in exclude_columns]
            self.logger.info(f"Initial feature columns: {self.feature_columns}")
            
            if len(self.feature_columns) == 0:
                # If no features, try to recreate them
                self.logger.warning("No features found, attempting to recreate features...")
                self.create_enhanced_features()
                
                # Try feature selection again
                numeric_cols = self.data.select_dtypes(include=[np.number]).columns
                self.feature_columns = [col for col in numeric_cols if col not in exclude_columns]
                
                if len(self.feature_columns) == 0:
                    raise ValueError("No features available after recreation attempt")
            
            # Remove highly correlated features
            if len(self.feature_columns) > 1:
                correlation_matrix = self.data[self.feature_columns].corr().abs()
                upper = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
                high_corr_features = [column for column in upper.columns if any(upper[column] > 0.98)]
                
                self.feature_columns = [col for col in self.feature_columns 
                                    if col not in high_corr_features]
                
                self.logger.info(f"Features after correlation removal: {self.feature_columns}")
            
            # Save selected features
            with open(self.model_dir / 'selected_features.txt', 'w') as f:
                f.write('\n'.join(self.feature_columns))
                
            self.logger.info(f"Final feature count: {len(self.feature_columns)}")
            self.logger.info(f"Final features: {self.feature_columns}")
            
            return True
                
        except Exception as e:
            self.logger.error(f"Error in select_features: {str(e)}")
            raise e


    def prepare_train_test_data(self, test_size: int = 90):
        """Prepare training and testing datasets."""
        try:
            # Ensure we have features selected
            if not hasattr(self, 'feature_columns') or not self.feature_columns:
                self.create_enhanced_features()
                self.select_features()
            
            # Split date
            split_date = self.data['timestamp'].max() - timedelta(days=test_size)
            
            # Create train/test split
            train_data = self.data[self.data['timestamp'] <= split_date]
            test_data = self.data[self.data['timestamp'] > split_date]
            
            # Verify we have data in both sets
            if len(train_data) == 0 or len(test_data) == 0:
                raise ValueError("Insufficient data for train/test split")
            
            # Prepare features and target
            X_train = train_data[self.feature_columns]
            y_train = train_data['target']
            X_test = test_data[self.feature_columns]
            y_test = test_data['target']
            
            # Verify we have features
            if X_train.empty or X_test.empty:
                raise ValueError("No features available for training")
            
            # Scale features
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Convert back to DataFrame
            self.X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
            self.X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
            self.y_train = y_train
            self.y_test = y_test
            
            self.logger.info(f"Training set size: {len(self.X_train)}, Test set size: {len(self.X_test)}")
            self.logger.info(f"Training class distribution:\n{y_train.value_counts(normalize=True)}")
            self.logger.info(f"Test class distribution:\n{y_test.value_counts(normalize=True)}")
            
        except Exception as e:
            self.logger.error(f"Error in prepare_train_test_data: {str(e)}")
            raise e


    def prepare_train_test_data(self, test_size: int = 90):
        """Prepare training and testing datasets."""
        # Split date
        split_date = self.data['timestamp'].max() - timedelta(days=test_size)
        
        # Create train/test split
        train_data = self.data[self.data['timestamp'] <= split_date]
        test_data = self.data[self.data['timestamp'] > split_date]
        
        # Prepare features and target
        X_train = train_data[self.feature_columns]
        y_train = train_data['target']
        X_test = test_data[self.feature_columns]
        y_test = test_data['target']
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert back to DataFrame
        self.X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        self.X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
        self.y_train = y_train
        self.y_test = y_test
        
        self.logger.info(f"Training set size: {len(self.X_train)}, Test set size: {len(self.X_test)}")
        self.logger.info(f"Training class distribution:\n{y_train.value_counts(normalize=True)}")
        self.logger.info(f"Test class distribution:\n{y_test.value_counts(normalize=True)}")

    def create_and_train_models(self):
        """Create and train multiple models with improved parameters for imbalanced data."""
        base_models = {
            'random_forest': RandomForestClassifier(
                n_estimators=200,  # Reduced from 1000 for faster training
                max_depth=6,
                min_samples_split=10,
                min_samples_leaf=5,
                class_weight='balanced',  # Changed from balanced_subsample
                random_state=self.random_state,
                n_jobs=-1
            ),
            'xgboost': xgb.XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=1,
                gamma=0.1,
                reg_alpha=0.1,
                reg_lambda=1,
                scale_pos_weight=5,  # Added to handle class imbalance
                random_state=self.random_state
            ),
            'lightgbm': LGBMClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                num_leaves=16,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_samples=10,
                reg_alpha=0.1,
                reg_lambda=1,
                class_weight='balanced',
                random_state=self.random_state,
                n_jobs=-1,
                verbose=-1  # Reduce verbosity
            )
        }
        
        # Train each model
        best_f1 = 0
        for name, model in base_models.items():
            self.logger.info(f"Training {name}...")
            model.fit(self.X_train, self.y_train)
            self.models[name] = model
            
            # Evaluate on validation set
            y_pred = model.predict(self.X_test)
            f1 = f1_score(self.y_test, y_pred)
            
            # Track best model
            if f1 > best_f1:
                best_f1 = f1
                self.best_model = (name, model)
                self.logger.info(f"New best model: {name} (F1: {f1:.3f})")


    def evaluate_models(self):
        """Evaluate all trained models with improved metrics."""
        results = {}
        best_f1 = 0
        
        for name, model in self.models.items():
            self.logger.info(f"Evaluating {name}...")
            
            # Make predictions
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            roc_auc = roc_auc_score(self.y_test, y_pred_proba)
            
            results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'confusion_matrix': confusion_matrix(self.y_test, y_pred),
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            self.logger.info(f"{name} Metrics:")
            self.logger.info(f"Accuracy: {accuracy:.3f}")
            self.logger.info(f"Precision: {precision:.3f}")
            self.logger.info(f"Recall: {recall:.3f}")
            self.logger.info(f"F1 Score: {f1:.3f}")
            self.logger.info(f"ROC AUC: {roc_auc:.3f}")
            
            # Update best model if needed
            if f1 > best_f1:
                best_f1 = f1
                self.best_model = (name, model)
                self.logger.info(f"New best model: {name}")
            
            # Plot confusion matrix
            plt.figure(figsize=(8, 6))
            sns.heatmap(confusion_matrix(self.y_test, y_pred), 
                    annot=True, fmt='d', cmap='Blues')
            plt.title(f'{name} Confusion Matrix')
            plt.savefig(self.model_dir / f'plots/{name}_confusion_matrix.png')
            plt.close()
        
        return results

    def calculate_technical_indicators(self, data):
        """Calculate key technical indicators for analysis."""
        df = data.copy()
        
        # Calculate RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Calculate MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        
        # Calculate SMAs
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        
        # Calculate Volatility
        df['volatility'] = df['close'].pct_change().rolling(window=20).std() * np.sqrt(252)
        
        return df

    def backtest_strategy(self):
        """Enhanced backtesting strategy with realistic trade sizing."""
        if self.best_model is None:
            self.logger.warning("No best model selected")
            return None
        
        try:
            name, model = self.best_model
            test_data = self.data.iloc[-len(self.X_test):].copy()
            
            # Calculate technical indicators for analysis
            test_data = self.calculate_technical_indicators(test_data)
            
            predictions = model.predict(self.X_test)
            probabilities = model.predict_proba(self.X_test)[:, 1]
            
            # Rest of your existing backtest_strategy code...
            
            # When returning results, include the technical indicators
            results['technical_indicators'] = {
                'RSI': test_data['rsi'].iloc[-1],
                'MACD': test_data['macd'].iloc[-1],
                'MACD Signal': test_data['macd_signal'].iloc[-1],
                'SMA 20': test_data['sma_20'].iloc[-1],
                'SMA 50': test_data['sma_50'].iloc[-1],
                'Volatility': test_data['volatility'].iloc[-1]
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in backtest strategy: {str(e)}")
            self.logger.exception(e)
            return None


    def backtest_strategy(self):
        """Enhanced backtesting strategy with realistic trade sizing."""
        if self.best_model is None:
            self.logger.warning("No best model selected")
            return None
        
        try:
            name, model = self.best_model
            predictions = model.predict(self.X_test)
            probabilities = model.predict_proba(self.X_test)[:, 1]
            
            # Initialize results
            trades = []
            capital = 100000.0  # Starting capital
            current_capital = capital
            max_capital = capital
            min_capital = capital
            
            # Risk management parameters
            risk_per_trade = 0.02      # Risk 2% per trade
            stop_loss_pct = 0.005      # 0.5% stop loss
            take_profit_pct = 0.01     # 1% take profit
            min_probability = 0.60      # Minimum probability threshold
            
            test_data = self.data.iloc[-len(self.X_test):].reset_index(drop=True)
            
            # Calculate technical indicators
            test_data = self.calculate_technical_indicators(test_data)
            daily_returns = []
            
            for i in range(len(predictions)-1):
                try:
                    if predictions[i] == 1 and probabilities[i] > min_probability:
                        # Calculate position size based on risk
                        entry_price = test_data.loc[i, 'close']
                        stop_loss = entry_price * (1 - stop_loss_pct)
                        risk_amount = current_capital * risk_per_trade
                        position_size = risk_amount / (entry_price - stop_loss)
                        
                        # Set take profit
                        take_profit = entry_price * (1 + take_profit_pct)
                        
                        # Next day's prices
                        next_high = test_data.loc[i+1, 'high']
                        next_low = test_data.loc[i+1, 'low']
                        next_close = test_data.loc[i+1, 'close']
                        
                        # Determine exit price and update position
                        if next_low <= stop_loss:
                            exit_price = stop_loss
                            pnl = -risk_amount  # Maximum loss
                        elif next_high >= take_profit:
                            exit_price = take_profit
                            pnl = position_size * (take_profit - entry_price)
                        else:
                            exit_price = next_close
                            pnl = position_size * (exit_price - entry_price)
                        
                        current_capital += pnl
                        max_capital = max(max_capital, current_capital)
                        min_capital = min(min_capital, current_capital)
                        
                        trades.append({
                            'entry_date': test_data.loc[i, 'timestamp'],
                            'exit_date': test_data.loc[i+1, 'timestamp'],
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'position_size': position_size,
                            'pnl': pnl,
                            'return': pnl / position_size,
                            'capital': current_capital
                        })
                    
                    # Calculate daily return
                    daily_returns.append((current_capital - capital) / capital)
                    
                except Exception as e:
                    self.logger.error(f"Error processing trade at index {i}: {str(e)}")
                    continue
            
            if len(trades) > 0:
                trades_df = pd.DataFrame(trades)
                winning_trades = trades_df[trades_df['pnl'] > 0]
                losing_trades = trades_df[trades_df['pnl'] < 0]
                
                # Calculate metrics
                total_return = (current_capital - capital) / capital
                max_drawdown = (max_capital - min_capital) / max_capital
                sharpe_ratio = np.sqrt(252) * (np.mean(daily_returns) / np.std(daily_returns)) if np.std(daily_returns) != 0 else 0
                
                last_indicators = test_data.iloc[-1]
                
                return {
                    'total_return': total_return,
                    'sharpe_ratio': sharpe_ratio,
                    'max_drawdown': max_drawdown,
                    'win_rate': len(winning_trades) / len(trades),
                    'profit_factor': abs(winning_trades['pnl'].sum() / losing_trades['pnl'].sum()) if len(losing_trades) > 0 and losing_trades['pnl'].sum() != 0 else np.inf,
                    'number_of_trades': len(trades),
                    'avg_winner': winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0,
                    'avg_loser': losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0,
                    'final_capital': current_capital,
                    'trades': trades_df,
                    'technical_indicators': {
                        'RSI': last_indicators['rsi'],
                        'MACD': last_indicators['macd'],
                        'SMA_20': last_indicators['sma_20'],
                        'SMA_50': last_indicators['sma_50'],
                        'Volatility': last_indicators['volatility']
                    }
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error in backtest strategy: {str(e)}")
            self.logger.exception(e)
            return None


def main():
    # Initialize model
    ml_model = EnhancedForexMLModel()
    
    # Load and prepare data
    if not ml_model.load_and_prepare_data():
        return
    
    # Create target variable
    ml_model.create_target_variable(min_move_pct=0.001)
    
    # Create enhanced features
    ml_model.create_enhanced_features()
    
    # Select features
    ml_model.select_features()
    
    # Prepare train/test data
    ml_model.prepare_train_test_data(test_size=90)
    
    # Train models
    print("\nTraining models...")
    ml_model.create_and_train_models()
    
    # Evaluate models
    print("\nEvaluating models...")
    results = ml_model.evaluate_models()
    
    # Analyze feature importance
    if hasattr(ml_model.best_model[1], 'feature_importances_'):
        importances = pd.DataFrame({
            'feature': ml_model.feature_columns,
            'importance': ml_model.best_model[1].feature_importances_
        })
        importances = importances.sort_values('importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(importances.head(10))
        
        # Plot feature importance
        plt.figure(figsize=(12, 6))
        sns.barplot(data=importances.head(20), x='importance', y='feature')
        plt.title('Top 20 Most Important Features')
        plt.tight_layout()
        plt.savefig(ml_model.model_dir / 'plots/feature_importance.png')
        plt.close()
    
    # Run backtest
    print("\nRunning backtest...")
    backtest_results = ml_model.backtest_strategy()
    
    # Print detailed results
    print("\nDetailed Backtest Results:")
    print("-" * 50)
    print(f"Initial Capital: ${100000:,.2f}")
    print(f"Final Capital: ${backtest_results['final_capital']:,.2f}")
    print(f"Total Return: {backtest_results['total_return']:.2%}")
    print(f"Sharpe Ratio: {backtest_results['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {backtest_results['max_drawdown']:.2%}")
    print("-" * 50)
    print("Trade Statistics:")
    print(f"Number of Trades: {backtest_results['number_of_trades']}")
    print(f"Win Rate: {backtest_results['win_rate']:.2%}")
    print(f"Profit Factor: {backtest_results['profit_factor']:.2f}")
    print(f"Average Winner: ${backtest_results['avg_winner']:,.2f}")
    print(f"Average Loser: ${backtest_results['avg_loser']:,.2f}")
    
    # Save models and results
    print("\nSaving models and results...")
    models_save_path = ml_model.model_dir / 'final_models'
    models_save_path.mkdir(exist_ok=True)
    
    # Save best model
    joblib.dump(ml_model.best_model[1], 
                models_save_path / f'{ml_model.best_model[0]}_best_model.joblib')
    
    # Save scaler
    joblib.dump(ml_model.scaler, models_save_path / 'scaler.joblib')
    
    # Save feature list
    with open(models_save_path / 'feature_list.txt', 'w') as f:
        f.write('\n'.join(ml_model.feature_columns))
    
    # Save backtest results
    results_df = pd.DataFrame([backtest_results])
    results_df.to_csv(models_save_path / 'backtest_results.csv', index=False)
    
    print("\nTraining and evaluation completed successfully!")

def predict_new_data(model_path: str, data: pd.DataFrame) -> dict:
    """
    Make predictions on new data using saved model.
    
    Parameters:
    -----------
    model_path : str
        Path to the saved model directory
    data : pd.DataFrame
        New data to make predictions on
        
    Returns:
    --------
    dict
        Dictionary containing predictions and probabilities
    """
    model_path = Path(model_path)
    
    # Load model and scaler
    model = joblib.load(model_path / 'final_models/random_forest_best_model.joblib')
    scaler = joblib.load(model_path / 'final_models/scaler.joblib')
    
    # Load feature list
    with open(model_path / 'final_models/feature_list.txt', 'r') as f:
        features = f.read().splitlines()
    
    # Prepare features
    X = data[features]
    X_scaled = scaler.transform(X)
    
    # Make predictions
    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)[:, 1]
    
    return {
        'predictions': predictions,
        'probabilities': probabilities
    }

def calculate_statistics(predictions: dict, actual_prices: pd.Series) -> dict:
    """
    Calculate trading statistics based on predictions.
    
    Parameters:
    -----------
    predictions : dict
        Dictionary containing predictions and probabilities
    actual_prices : pd.Series
        Series of actual prices
        
    Returns:
    --------
    dict
        Dictionary containing trading statistics
    """
    signals = predictions['predictions']
    probs = predictions['probabilities']
    
    trades = []
    for i in range(len(signals)-1):
        if signals[i] == 1:
            entry_price = actual_prices.iloc[i]
            exit_price = actual_prices.iloc[i+1]
            
            profit = (exit_price - entry_price) / entry_price
            trades.append({
                'entry_price': entry_price,
                'exit_price': exit_price,
                'profit': profit,
                'probability': probs[i]
            })
    
    if trades:
        trades_df = pd.DataFrame(trades)
        return {
            'total_trades': len(trades),
            'winning_trades': len(trades_df[trades_df['profit'] > 0]),
            'average_profit': trades_df['profit'].mean(),
            'profit_std': trades_df['profit'].std(),
            'sharpe': trades_df['profit'].mean() / trades_df['profit'].std() if trades_df['profit'].std() != 0 else 0
        }
    else:
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'average_profit': 0,
            'profit_std': 0,
            'sharpe': 0
        }

if __name__ == "__main__":
    main()
