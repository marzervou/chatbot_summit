# Chatbot Summit 2023 Q/A Bot

This a repo to create a QA Retrieval Bot using both Open LLM .\
To get started please change the configuration notebook in **util/notebook-config.py**. 

Please look at the documentation Below on configuration for specific LLM's

Please change the **util/notebook-config.py** to point to the location where your files are stored.
```
config['loc']  = <"Location where the files are stored">
```

## LLM's Supported
Currently the code supports the following version 
- [Llama-2-70-b HF-chat version with 8-bit Quantization](#runnig-the-code-using-llama-2-models)

## Runtime Tested
The following code is tested on ML DBR GPU 13.2 Runtime

## Cluster Configurations
The Code to run open LLMS has been tested on the below single node cluster configurations:
- AWS : g5-12xlarge [4 A10's]
- Azure : NC24Ads_A100_v4 [1 A100]

The TGI pipeline has support to run on older Generation GPU's like the V100's but has not been tested extensively

## Coverting PDF to txt
There are two ways to convert the PDF to TXT

- **Using Azure Form Recognizer**
To use form recognizer you need to add 
```
config['use_azure_formrecognizer'] = True
and add the URL and KEY from the Azure portal 
config['formendpoint'] 
config['formkey']
```
- Using Langchain PDF converter
set
```
config['use_azure_formrecognizer'] = False
```

## Runnig the code using LLAMA-2 Models:
To use LLAMA-2 models, you need to agree to the terms and condition of HuggingFace and provide an API key to download the models
Refer to these steps to download the key : https://huggingface.co/docs/api-inference/quicktour#get-your-api-token and set the below parameters
```
config['model_id'] = 'meta-llama/Llama-2-XXb-chat-hf'
config['HUGGING_FACE_HUB_TOKEN'] = '<your HF AI API Key>'
```
Note : to need to keep 02_load_model Notebook running to have the API running


## Embedding Model
The open LLM embedding can be changed by over-riding the Dictionary in utils/notebook-config.py
