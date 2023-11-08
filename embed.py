import os
from dotenv import load_dotenv

from genai.extensions.langchain import LangChainInterface
from genai.schemas import GenerateParams
from genai.credentials import Credentials
from langchain.embeddings import OpenAIEmbeddings
from chromadb.utils import embedding_functions


def get_ibm_granite_context():
    system_prompt = """
    - You are a helpful AI assistant and provide the answer for the question based on the given context.
    - You answer the question as truthfully as possible using the provided text, and if the answer is not contained within the text below, you say "I don't know".
    """ 

    ## This will wrap the default prompts that are internal to llama-index
    #query_wrapper_prompt = SimpleInputPrompt(">>QUESTION<<{query_str}\n>>ANSWER<<")
    #query_wrapper_prompt = Prompt("[INST] {query_str} [/INST]")

    #print("Changing default model")
    # Change default model
    #embed_model = LangchainEmbedding(HuggingFaceEmbeddings())
    #embed_model='meta-llama/llama-2-13b-chat'
    #embed_model='google/flan-t5-xxl'
    embed_model='ibm/granite-13b-chat-v1'

    #print(f"Getting server environment variables")

    load_dotenv()
    api_key = os.getenv("GENAI_KEY", None) 
    api_url = os.getenv("GENAI_API", None)
    creds = Credentials(api_key, api_endpoint=api_url)

    print(f"Initializing embedder with api_key: {api_key}, api_url: {api_url}")

    # server_url = os.getenv('TGIS_SERVER_URL', 'http://localhost') # Get server url from env else default
    # server_port = os.getenv('TGIS_SERVER_PORT', '8049') # Get server port from env else default
    # print(f"Initializing TGIS predictor with server_url: {server_url}, server_port: {server_port}")
    # inference_server_url=f"{server_url}:{server_port}/"
    # print(f"Inference Service URL: {inference_server_url}")


    # llm = LangChainLLM(
    #     llm=HuggingFaceTextGenInference(
    #         inference_server_url=api_url,
    #         max_new_tokens=256,
    #         temperature=temperature,
    #         repetition_penalty=repetition_penalty,
    #         server_kwargs={},
    #     ),
    # )


    params = GenerateParams(decoding_method="greedy",temperature=0.5,max_new_tokens=1024,min_new_tokens=256,repetition_penalty=1.9)

    llm = LangChainInterface(model=embed_model, params=params, credentials=creds)

    print("Creating service_context")
    # service_context = ServiceContext.from_defaults(chunk_size=1024, llm=llm, 
    #                                                query_wrapper_prompt=query_wrapper_prompt,
    #                                                system_prompt=system_prompt,
    #                                                embed_model=embed_model)
    ans=llm("what is the best kind of car?")
    print(ans)


    embedder = embedding_functions.OpenAIEmbeddingFunction(
        model_name="text-embedding-ada-002"
    )

huggingface_ef = embedding_functions.HuggingFaceEmbeddingFunction(
    api_key="YOUR_API_KEY",
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

    # return service_context

get_ibm_granite_context()
