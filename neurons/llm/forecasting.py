import asyncio
from datetime import datetime
import re

from . import ranking, summarize, ensemble
from .config.keys import OPENAI_KEY
from .prompts.prompts import PROMPT_DICT


class Forecaster:
    RETRIEVAL_CONFIG = {
        "NUM_SEARCH_QUERY_KEYWORDS": 3,
        "MAX_WORDS_NEWSCATCHER": 5,
        "MAX_WORDS_GNEWS": 8,
        "SEARCH_QUERY_MODEL_NAME": "gpt-4-1106-preview",
        "SEARCH_QUERY_TEMPERATURE": 0.0,
        "SEARCH_QUERY_PROMPT_TEMPLATES": [
            PROMPT_DICT["search_query"]["0"],
            PROMPT_DICT["search_query"]["1"],
        ],
        "NUM_ARTICLES_PER_QUERY": 5,
        "SUMMARIZATION_MODEL_NAME": "gpt-3.5-turbo-1106",
        "SUMMARIZATION_TEMPERATURE": 0.2,
        "SUMMARIZATION_PROMPT_TEMPLATE": PROMPT_DICT["summarization"]["9"],
        "NUM_SUMMARIES_THRESHOLD": 10,
        "PRE_FILTER_WITH_EMBEDDING": True,
        "PRE_FILTER_WITH_EMBEDDING_THRESHOLD": 0.32,
        "RANKING_MODEL_NAME": "gpt-3.5-turbo-1106",
        "RANKING_TEMPERATURE": 0.0,
        "RANKING_PROMPT_TEMPLATE": PROMPT_DICT["ranking"]["0"],
        "RANKING_RELEVANCE_THRESHOLD": 4,
        "RANKING_COSINE_SIMILARITY_THRESHOLD": 0.5,
        "SORT_BY": "date",
        "RANKING_METHOD": "llm-rating",
        "RANKING_METHOD_LLM": "title_250_tokens",
        "NUM_SUMMARIES_THRESHOLD": 20,
        "EXTRACT_BACKGROUND_URLS": True,
    }

    REASONING_CONFIG = {
        "BASE_REASONING_MODEL_NAMES": ["gpt-4-1106-preview", "gpt-4-1106-preview"],
        "BASE_REASONING_TEMPERATURE": 1.0,
        "BASE_REASONING_PROMPT_TEMPLATES": [
            [
                PROMPT_DICT["binary"]["scratch_pad"]["1"],
                PROMPT_DICT["binary"]["scratch_pad"]["2"],
            ],
            [
                PROMPT_DICT["binary"]["scratch_pad"]["new_3"],
                PROMPT_DICT["binary"]["scratch_pad"]["new_6"],
            ],
        ],
        "ALIGNMENT_MODEL_NAME": "gpt-3.5-turbo-1106",
        "ALIGNMENT_TEMPERATURE": 0,
        "ALIGNMENT_PROMPT": PROMPT_DICT["alignment"]["0"],
        "AGGREGATION_METHOD": "meta",
        "AGGREGATION_PROMPT_TEMPLATE": PROMPT_DICT["meta_reasoning"]["0"],
        "AGGREGATION_TEMPERATURE": 0.2,
        "AGGREGATION_MODEL_NAME": "gpt-4",
        "AGGREGATION_WEIGTHTS": None,
    }

    async def get_prediction(self, market):
        if OPENAI_KEY is None:
            return None

        question = market.event.description
        background_info = ''
        resolution_criteria = ''

        start_time = int(datetime.utcnow().timestamp())
        start_date = datetime.fromtimestamp(start_time).strftime('%Y-%m-%d')
        end_time = (datetime.fromtimestamp(market.event.starts or market.event.resolve_date))
        end_date = end_time.strftime('%Y-%m-%d')
        retrieval_dates = [start_date, end_date]

        urls_in_background = []

        (
            ranked_articles,
            all_articles,
            search_queries_list_gnews,
            search_queries_list_nc,
        ) = await ranking.retrieve_summarize_and_rank_articles(
            question,
            background_info,
            resolution_criteria,
            retrieval_dates,
            urls=urls_in_background,
            config=self.RETRIEVAL_CONFIG,
            return_intermediates=True,
        )

        all_summaries = summarize.concat_summaries(
            ranked_articles[: self.RETRIEVAL_CONFIG["NUM_SUMMARIES_THRESHOLD"]]
        )

        today_to_close_date = [
            datetime.utcnow().strftime('%Y-%m-%d'),
            retrieval_dates[1]
        ]
        ensemble_dict = await ensemble.meta_reason(
            question=question,
            background_info=background_info,
            resolution_criteria=resolution_criteria,
            today_to_close_date_range=today_to_close_date,
            retrieved_info=all_summaries,
            reasoning_prompt_templates=self.REASONING_CONFIG["BASE_REASONING_PROMPT_TEMPLATES"],
            base_model_names=self.REASONING_CONFIG["BASE_REASONING_MODEL_NAMES"],
            base_temperature=self.REASONING_CONFIG["BASE_REASONING_TEMPERATURE"],
            aggregation_method=self.REASONING_CONFIG["AGGREGATION_METHOD"],
            answer_type="probability",
            weights=self.REASONING_CONFIG["AGGREGATION_WEIGTHTS"],
            meta_model_name=self.REASONING_CONFIG["AGGREGATION_MODEL_NAME"],
            meta_prompt_template=self.REASONING_CONFIG["AGGREGATION_PROMPT_TEMPLATE"],
            meta_temperature=self.REASONING_CONFIG["AGGREGATION_TEMPERATURE"],
        )

        return float(ensemble_dict["meta_prediction"])