import logging
from typing import List, Tuple, Set

class CurrencyPairValidator:
    """Validates and manages forex currency pairs"""
    
    def __init__(self):
        # Define valid currency pairs based on common forex trading pairs
        self.valid_pairs: Set[str] = {
            'EUR/USD', 'GBP/USD', 'USD/JPY', 'USD/CHF',
            'GBP/EUR', 'EUR/GBP', 'EUR/JPY', 'GBP/JPY',
            'AUD/USD', 'USD/CAD', 'NZD/USD'
        }
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
    def is_valid_pair(self, base: str, quote: str) -> bool:
        """
        Check if a currency pair is valid.
        
        Args:
            base: Base currency code
            quote: Quote currency code
            
        Returns:
            bool: True if the pair is valid, False otherwise
        """
        pair = f"{base}/{quote}"
        reverse_pair = f"{quote}/{base}"
        
        is_valid = pair in self.valid_pairs or reverse_pair in self.valid_pairs
        if not is_valid:
            self.logger.warning(f"Invalid currency pair attempted: {pair}")
        
        return is_valid
        
    def get_valid_quote_currencies(self, base: str) -> List[str]:
        """
        Get list of valid quote currencies for a given base currency.
        
        Args:
            base: Base currency code
            
        Returns:
            List[str]: List of valid quote currencies
        """
        valid_quotes = []
        for pair in self.valid_pairs:
            currencies = pair.split('/')
            if currencies[0] == base:
                valid_quotes.append(currencies[1])
            elif currencies[1] == base:
                valid_quotes.append(currencies[0])
        return valid_quotes
