import os
import pandas as pd
import numpy as np
from fredapi import Fred
from datetime import datetime, timedelta
from typing import List

from neurons.llm.information_retrieval import (
    get_newscatcher_articles,
    clean_search_queries
)

class FredAnalyzer:
    def __init__(self):
        self.fred = Fred(api_key=os.getenv('FRED_API_KEY'))
        self.cache = {}
        self.cache_duration = timedelta(hours=1)
        self.last_cache_update = {}

    async def get_historical_data(self, series_id: str, start_date: str, end_date: str = None):
        """Fetch historical data for a given FRED series"""
        cache_key = f"{series_id}_{start_date}_{end_date}"
        
        if (cache_key in self.cache and 
            cache_key in self.last_cache_update and 
            datetime.now() - self.last_cache_update[cache_key] < self.cache_duration):
            return self.cache[cache_key]

        data = self.fred.get_series(
            series_id,
            start_date=start_date,
            end_date=end_date
        )
        
        self.cache[cache_key] = data
        self.last_cache_update[cache_key] = datetime.now()
        
        return data

    def analyze_trends(self, data: pd.Series) -> dict:
        """Analyze trends in the time series data"""
        if len(data) < 2:
            return {
                'trend': 0,
                'momentum': 0,
                'volatility': 0
            }

        returns = data.pct_change().dropna()
        trend = 1 if data.iloc[-1] > data.iloc[-2] else -1        
        momentum = returns.mean()
        volatility = returns.std()

        return {
            'trend': trend,
            'momentum': momentum,
            'volatility': volatility
        }

    async def get_market_sentiment(self, series_id: str) -> float:
        """
        Analyze market sentiment using Newscatcher API
        Returns a sentiment score between -1 and 1
        """
        # Generate search queries based on the series
        search_queries = self._generate_search_queries(series_id)
        clean_search_queries(search_queries)

        # Set date range for recent articles
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        retrieval_dates = [start_date, end_date]

        try:
            # Get articles using Newscatcher
            articles = get_newscatcher_articles(
                search_queries,
                retrieval_dates,
                num_articles=5,
                max_results=10
            )

            if not articles:
                return 0

            sentiments = []
            for article in articles:
                text = f"{article.title} {article.excerpt if article.excerpt else ''}"
                
                sentiment = self._analyze_text_sentiment(text)
                sentiments.append(sentiment)

            return np.mean(sentiments) if sentiments else 0

        except Exception as e:
            print(f"Error in sentiment analysis: {e}")
            return 0

    def _generate_search_queries(self, series_id: str) -> List[str]:
        """Generate relevant search queries based on the FRED series ID"""
        series_terms = {
            'DTB4WK': ['4-week treasury bill', 'one month t-bill', 'short term treasury rates'],
            'DGS3MO': ['3-month treasury yield', 'three month treasury rate'],
            'DGS6MO': ['6-month treasury yield', 'treasury bill rates'],
            'DGS1': ['1-year treasury yield', 'treasury market outlook'],
            'DGS2': ['2-year treasury yield', 'treasury note rates'],
            'DGS5': ['5-year treasury yield', 'medium term treasury outlook'],
            'DGS10': ['10-year treasury yield', 'benchmark treasury rate'],
            'DGS30': ['30-year treasury yield', 'long term treasury bonds'],
            'DFII5': ['5-year TIPS yield', 'inflation protected securities'],
            'DFII10': ['10-year TIPS yield', 'inflation linked bonds'],
        }

        default_terms = [
            'US Treasury yields',
            'Federal Reserve monetary policy',
            'bond market analysis'
        ]

        search_terms = series_terms.get(series_id, []) + default_terms
        return search_terms[:3]

    def _analyze_text_sentiment(self, text: str) -> float:
        """
        Analyze sentiment of text using keyword matching
        Returns a score between -1 and 1
        """
        text = text.lower()
        
        positive_words = {
            'increase': 0.5, 'rise': 0.5, 'gain': 0.5, 'higher': 0.5, 'surge': 0.7,
            'rally': 0.6, 'boost': 0.5, 'improve': 0.4, 'positive': 0.5, 'bullish': 0.7,
            'upward': 0.5, 'strong': 0.5, 'strength': 0.5, 'optimistic': 0.6
        }
        
        negative_words = {
            'decrease': -0.5, 'fall': -0.5, 'drop': -0.5, 'lower': -0.5, 'decline': -0.5,
            'weak': -0.5, 'bearish': -0.7, 'downward': -0.5, 'negative': -0.5,
            'worsen': -0.6, 'concern': -0.4, 'risk': -0.4, 'pessimistic': -0.6
        }

        sentiment_score = 0
        word_count = 0

        for word, score in positive_words.items():
            if word in text:
                sentiment_score += score
                word_count += 1

        for word, score in negative_words.items():
            if word in text:
                sentiment_score += score
                word_count += 1

        return sentiment_score / max(1, word_count)

    def get_correlations(self, series_id: str, target_value: float) -> dict:
        """Analyze correlations with other relevant economic indicators"""
        related_series = {
            'GDP': 'GDP',
            'Inflation': 'CPIAUCSL',
            'Unemployment': 'UNRATE'
        }
        
        correlations = {}
        main_series = self.fred.get_series(series_id, start_date='2020-01-01')
        
        for name, series_id in related_series.items():
            try:
                related_data = self.fred.get_series(series_id, start_date='2020-01-01')
                df = pd.DataFrame({
                    'main': main_series,
                    'related': related_data
                }).dropna()
                
                if not df.empty:
                    correlations[name] = df['main'].corr(df['related'])
            except Exception:
                correlations[name] = 0
                
        return correlations

    def extract_series_id(self, description: str) -> str:
        """Extract FRED series ID from event description"""
        start = description.find('series_id=') + 10
        end = description.find(')', start)
        return description[start:end] if start > 10 and end > start else None

    def calculate_probability(self, 
                            series_id: str,
                            target_value: float,
                            trend_data: dict,
                            sentiment: float,
                            correlations: dict) -> float:
        """
        Calculate final probability using all available signals
        Returns probability between 0 and 1
        """
        # Weights for different factors
        weights = {
            'trend': 0.3,
            'momentum': 0.2,
            'sentiment': 0.2,
            'correlations': 0.2,
            'volatility': 0.1
        }

        trend_prob = (trend_data['trend'] + 1) / 2

        momentum_prob = 1 / (1 + np.exp(-10 * trend_data['momentum']))

        sentiment_prob = (sentiment + 1) / 2

        correlation_prob = (sum(correlations.values()) + 1) / 2

        volatility_factor = 1 - min(1, trend_data['volatility'] * 10)
        base_prob = (
            weights['trend'] * trend_prob +
            weights['momentum'] * momentum_prob +
            weights['sentiment'] * sentiment_prob +
            weights['correlations'] * correlation_prob
        )
        
        final_prob = 0.5 + (base_prob - 0.5) * volatility_factor

        return min(1, max(0, final_prob))

    async def predict(self, event_description: str, target_value: float) -> float:
        """Main prediction function"""
        series_id = self.extract_series_id(event_description)
        if not series_id:
            return 0.5

        try:
            historical_data = await self.get_historical_data(
                series_id,
                start_date=(datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            )

            trend_data = self.analyze_trends(historical_data)

            sentiment = await self.get_market_sentiment(series_id)

            correlations = self.get_correlations(series_id, target_value)

            probability = self.calculate_probability(
                series_id,
                target_value,
                trend_data,
                sentiment,
                correlations
            )

            return probability

        except Exception as e:
            print(f"Error in FRED prediction: {e}")
            return 0.5