# Databricks notebook source
import torch

# COMMAND ----------

if 'config' not in locals():
  config = {}

# COMMAND ----------

# DBTITLE 1,Use Case
config['use-case']="qabot_chatbot_summit"

# COMMAND ----------

# Define the model we would like to use
config['model_id'] = 'meta-llama/Llama-2-13b-chat-hf'
username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')
config['use_azure_formrecognizer'] = False

# COMMAND ----------

# DBTITLE 1,Create database
config['database_name'] = 'qabot_chatbot_summit'

# create database if not exists
_ = spark.sql(f"create database if not exists {config['database_name']}")

# set current datebase context
_ = spark.catalog.setCurrentDatabase(config['database_name'])

# COMMAND ----------

# DBTITLE 1,Set Environmental Variables for tokens
import os

if "Llama-2" in config['model_id']:
  config['HUGGING_FACE_HUB_TOKEN'] = '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'

# COMMAND ----------

# DBTITLE 1,Set document path
config['loc'] = f"/dbfs/FileStore/insurance_policy_doc/"
config['vector_store_path'] = f"/dbfs/Users/{username}/qabot_chatbot_summit/vector_store/{config['model_id']}/{config['use-case']}" # /dbfs/... is a local file system representation

# COMMAND ----------

if config['use_azure_formrecognizer'] == True:
  config['formendpoint'] = 'XXXXXXXXXXX'
  config['formkey'] = 'XXXXXXXXX'

# COMMAND ----------

# DBTITLE 1,mlflow settings
import mlflow
config['registered_model_name'] = f"{config['use-case']}"
config['model_uri'] = f"models:/{config['registered_model_name']}/production"
_ = mlflow.set_experiment('/Users/{}/{}'.format(username, config['registered_model_name']))

# COMMAND ----------

# DBTITLE 1,Set model configs
if config['model_id'] == 'meta-llama/Llama-2-13b-chat-hf' :
  # Setup prompt template ####
  config['embedding_model'] = 'BAAI/bge-large-en'
  config['model_kwargs'] = {}
  
  # Model parameters
  config['pipeline_kwargs']={"temperature":  0.10,
                            "max_new_tokens": 256}
  
  config['template'] = """<s><<SYS>>
  You are a assistant built to answer policy related questions based on the context provided, the context is a document and use no other information.If the context does not provide enough relevant information to determine the answer, just say I don't know. If the context is irrelevant to the question, just say I don't know. If the query doesn't form a complete question, just say I don't know.Only answer the question asked and do not repeat the question
  <</SYS>>[INST] Given the context: {context}. Answer the question {question} ?\n
  [/INST]""".strip()

