'''
Configurations of the Qbot
'''
from types import SimpleNamespace

TEAM = None
PROJECT = "llmchat"
JOB_TYPE = "production"

default_config = SimpleNamespace(
    project=PROJECT,
    entity=TEAM,
    job_type=JOB_TYPE,
    vector_store_artifact="boyueshen/llmchat/vector_store:latest",
    chat_prompt_artifact="boyueshen/llmchat/chat_prompt:latest",
    chat_temperature=0.3,
    max_fallback_retries=1,
    model_name="gpt-3.5-turbo",
)