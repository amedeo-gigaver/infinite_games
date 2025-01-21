from datetime import datetime

from . import ensemble, ranking, summarize
from .config.keys import GOOGLE_AI_KEY, OPENAI_KEY
from .prompts.prompts import PROMPT_DICT


def _get_retrieval_config(model_setup: dict):
    return {
        "NUM_SEARCH_QUERY_KEYWORDS": 3,
        "MAX_WORDS_NEWSCATCHER": 5,
        "MAX_WORDS_GNEWS": 8,
        "SEARCH_QUERY_MODEL_NAME": model_setup["SEARCH_QUERY_MODEL_NAME"],
        "SEARCH_QUERY_TEMPERATURE": 0.0,
        "SEARCH_QUERY_PROMPT_TEMPLATES": [
            PROMPT_DICT["search_query"]["0"],
            PROMPT_DICT["search_query"]["1"],
        ],
        "NUM_ARTICLES_PER_QUERY": 5,
        "SUMMARIZATION_MODEL_NAME": model_setup["SUMMARIZATION_MODEL_NAME"],
        "SUMMARIZATION_TEMPERATURE": 0.2,
        "SUMMARIZATION_PROMPT_TEMPLATE": PROMPT_DICT["summarization"]["9"],
        "NUM_SUMMARIES_THRESHOLD": 10,
        "PRE_FILTER_WITH_EMBEDDING": True,
        "PRE_FILTER_WITH_EMBEDDING_THRESHOLD": 0.32,
        "RANKING_MODEL_NAME": model_setup["RANKING_MODEL_NAME"],
        "RANKING_TEMPERATURE": 0.0,
        "RANKING_PROMPT_TEMPLATE": PROMPT_DICT["ranking"]["0"],
        "RANKING_RELEVANCE_THRESHOLD": 3,
        "RANKING_COSINE_SIMILARITY_THRESHOLD": 0.5,
        "SORT_BY": "date",
        "RANKING_METHOD": "llm-rating",
        "RANKING_METHOD_LLM": "title_250_tokens",
        "NUM_SUMMARIES_THRESHOLD": 20,
        "EXTRACT_BACKGROUND_URLS": True,
    }


def _get_reasoning_config(model_setup: dict):
    return {
        "BASE_REASONING_MODEL_NAMES": model_setup["BASE_REASONING_MODEL_NAMES"],
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
        "ALIGNMENT_MODEL_NAME": model_setup["ALIGNMENT_MODEL_NAME"],
        "ALIGNMENT_TEMPERATURE": 0,
        "ALIGNMENT_PROMPT": PROMPT_DICT["alignment"]["0"],
        "AGGREGATION_METHOD": "meta",
        "AGGREGATION_PROMPT_TEMPLATE": PROMPT_DICT["meta_reasoning"]["0"],
        "AGGREGATION_TEMPERATURE": 0.2,
        "AGGREGATION_MODEL_NAME": model_setup["AGGREGATION_MODEL_NAME"],
        "AGGREGATION_WEIGTHTS": None,
    }


class Forecaster:
    model_setups = {
        # Budget version of OPENAI models
        0: {
            "SEARCH_QUERY_MODEL_NAME": "gpt-4o-mini-2024-07-18",
            "SUMMARIZATION_MODEL_NAME": "gpt-4o-mini-2024-07-18",
            "RANKING_MODEL_NAME": "gpt-3.5-turbo-1106",
            "BASE_REASONING_MODEL_NAMES": ["gpt-3.5-turbo-1106", "gpt-3.5-turbo-1106"],
            "ALIGNMENT_MODEL_NAME": "gpt-4o-mini-2024-07-18",
            "AGGREGATION_MODEL_NAME": "gpt-3.5-turbo-1106",
        },
        # A gpt-4 more expensive version of OPENAI models
        1: {
            "SEARCH_QUERY_MODEL_NAME": "gpt-4-1106-preview",
            "SUMMARIZATION_MODEL_NAME": "gpt-3.5-turbo-1106",
            "RANKING_MODEL_NAME": "gpt-3.5-turbo-1106",
            "BASE_REASONING_MODEL_NAMES": ["gpt-4-1106-preview", "gpt-4-1106-preview"],
            "ALIGNMENT_MODEL_NAME": "gpt-3.5-turbo-1106",
            "AGGREGATION_MODEL_NAME": "gpt-4",
        },
        2: {
            "SEARCH_QUERY_MODEL_NAME": "gemini-pro",
            "SUMMARIZATION_MODEL_NAME": "gemini-pro",
            "RANKING_MODEL_NAME": "gemini-pro",
            "BASE_REASONING_MODEL_NAMES": ["gemini-pro", "gemini-pro"],
            "ALIGNMENT_MODEL_NAME": "gemini-pro",
            "AGGREGATION_MODEL_NAME": "gemini-pro",
        },
        3: {
            "SEARCH_QUERY_MODEL_NAME": "gemini-pro",
            "SUMMARIZATION_MODEL_NAME": "gemini-pro",
            "RANKING_MODEL_NAME": "gemini-pro",
            "BASE_REASONING_MODEL_NAMES": ["gpt-3.5-turbo-1106", "gpt-3.5-turbo-1106"],
            "ALIGNMENT_MODEL_NAME": "gemini-pro",
            "AGGREGATION_MODEL_NAME": "gemini-pro",
        },
        # You can add more setups here.
    }

    async def get_prediction(self, market, models_setup_option: int = 0):
        if (OPENAI_KEY is None and GOOGLE_AI_KEY is None) or models_setup_option not in [
            0,
            1,
            2,
            3,
        ]:
            return None

        retrieval_config = _get_retrieval_config(self.model_setups[models_setup_option])
        reasoning_config = _get_reasoning_config(self.model_setups[models_setup_option])

        question = market.event.description
        background_info = ""
        resolution_criteria = ""

        start_time = int(datetime.now().timestamp())
        start_date = datetime.fromtimestamp(start_time - 48 * 60 * 60).strftime("%Y-%m-%d")
        end_time = datetime.fromtimestamp(market.event.cutoff)
        end_date = end_time.strftime("%Y-%m-%d")
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
            config=retrieval_config,
            return_intermediates=True,
        )

        all_summaries = summarize.concat_summaries(
            ranked_articles[: retrieval_config["NUM_SUMMARIES_THRESHOLD"]]
        )

        today_to_close_date = [datetime.utcnow().strftime("%Y-%m-%d"), retrieval_dates[1]]
        ensemble_dict = await ensemble.meta_reason(
            question=question,
            background_info=background_info,
            resolution_criteria=resolution_criteria,
            today_to_close_date_range=today_to_close_date,
            retrieved_info=all_summaries,
            reasoning_prompt_templates=reasoning_config["BASE_REASONING_PROMPT_TEMPLATES"],
            base_model_names=reasoning_config["BASE_REASONING_MODEL_NAMES"],
            base_temperature=reasoning_config["BASE_REASONING_TEMPERATURE"],
            aggregation_method=reasoning_config["AGGREGATION_METHOD"],
            answer_type="probability",
            weights=reasoning_config["AGGREGATION_WEIGTHTS"],
            meta_model_name=reasoning_config["AGGREGATION_MODEL_NAME"],
            meta_prompt_template=reasoning_config["AGGREGATION_PROMPT_TEMPLATE"],
            meta_temperature=reasoning_config["AGGREGATION_TEMPERATURE"],
        )
        print(ensemble_dict["meta_prediction"])
        return float(ensemble_dict["meta_prediction"])
