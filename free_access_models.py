import os
import warnings
from typing import Any, Optional
import numpy as np

try:
    from google.colab import userdata
    os.environ['HUGGINGFACEHUB_API_TOKEN'] = userdata.get('HUGGINGFACEHUB_API_TOKEN')
    os.environ['HF_TOKEN'] = os.environ['HUGGINGFACEHUB_API_TOKEN']
except:
    ValueError('It looks like you are not on Google Colab. If you are, make sure to set the environment variables "HUGGINGFACEHUB_API_TOKEN" and "HF_TOKEN" to your Huggingface tokens. Both will be needed. Then, you can run this again.')


from langchain_community.llms import HuggingFaceHub
from langchain_community.chat_models.huggingface import ChatHuggingFace
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings, HuggingFaceEmbeddings
from langchain.schema import HumanMessage, AIMessage, SystemMessage

from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel,Extra

from transformers import pipeline

# Set model ID
model_id = "mistralai/Mistral-7B-Instruct-v0.2"
#model_id = 'gpt2'

# Hub versions
def LLMMistral():
    llm = HuggingFaceHub(
        repo_id=model_id,
        task="text-generation",
        model_kwargs={
            "max_new_tokens": 512,
            "top_k": 30,
            "temperature": 0.1,
            "repetition_penalty": 1.03,
            "return_full_text": False
        },
    )
    return llm

def ChatMistral(llm=None):

    if llm is None:
        llm = LLMMistral()
    chat_model = ChatHuggingFace(llm=llm, system_message=SystemMessage("You are a helpful assistant and are extremely concise in your responses."))

    return chat_model

def MistralEmbeddings():
  embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=os.environ['HF_TOKEN'], model_name=model_id
)
  return embeddings


################## Local Versions #################

#
def LLMMistralLocal():
  llm = HuggingFacePipeline.from_model_id(
      model_id=model_id,
      task="text-generation",
      device=0,
      pipeline_kwargs={"max_new_tokens": 512,
            "top_k": 30,
            "do_sample": True,
            "temperature": 0.1,
            "repetition_penalty": 1.03,
            "return_full_text": False,'pad_token_id':2},
  )
  return llm

class ChatHuggingFaceLocal(ChatHuggingFace):

  llm: HuggingFacePipeline

  def __init__(self, **kwargs):
        super(type(self).__bases__[0], self).__init__(**kwargs) #we want the grandparent init

        from transformers import AutoTokenizer

        self.tokenizer = (
            AutoTokenizer.from_pretrained(self.model_id)
            if self.tokenizer is None
            else self.tokenizer
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token_id = self.llm.pipeline.model.config.eos_token_id
         

def ChatMistralLocal(llm=None):
  if llm is None:
    llm = LLMMistralLocal()
  chat_model = ChatHuggingFaceLocal(llm=llm, model_id=model_id, system_message=SystemMessage("You are a helpful assistant"))
  return chat_model

class HuggingFaceEmbeddingsLocal(BaseModel, Embeddings):

  fx: Optional[Any] = None

  def __init__(self, model, tokenizer, **kwargs):
      super().__init__(**kwargs)
      self.fx = pipeline('feature-extraction', model=model, tokenizer=tokenizer, device=0)
  
  class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
  
  def embed_documents(self, texts):
    embeddings = self.fx(texts)

    nembeddings = [np.array(embedding).mean(1).squeeze().tolist() for embedding in embeddings]
    return nembeddings
  
  def embed_query(self, text: str):
    
    embed = self.embed_documents([text])[0]
    nembed = np.expand_dims(embed, axis=0)
    return embed
  

## Local embeddings
def MistralEmbeddingsLocal(chat_model):
  
  model = chat_model.llm.pipeline.model
  tokenizer = chat_model.llm.pipeline.tokenizer
  embeddings = HuggingFaceEmbeddingsLocal(model, tokenizer)
  return embeddings
