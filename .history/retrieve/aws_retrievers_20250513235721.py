"""
retriever/aws_retrievers.py
This file contains functions to interact with AWS services like OpenSearch and Pinecone.
"""

from functools import cache
from opensearchpy import OpenSearch, AWSV4SignerAuth, RequestsHttpConnection
import boto3

AWS_PROFILE_NAME = "sigir-participant"
AWS_REGION_NAME = "us-east-1"

def get_ssm_value(key: str, profile: str = AWS_PROFILE_NAME, region: str = AWS_REGION_NAME) -> str:
    """Get a cleartext value from AWS SSM."""
    session = boto3.Session(profile_name=profile, region_name=region)
    ssm = session.client("ssm")
    return ssm.get_parameter(Name=key)["Parameter"]["Value"]

def get_ssm_secret(key: str, profile: str = AWS_PROFILE_NAME, region: str = AWS_REGION_NAME):
    session = boto3.Session(profile_name=profile, region_name=region)
    ssm = session.client("ssm")
    return ssm.get_parameter(Name=key, WithDecryption=True)["Parameter"]["Value"]


from typing import List, Literal, Tuple
from multiprocessing.pool import ThreadPool
import boto3
from pinecone import Pinecone
import torch
from functools import cache
from transformers import AutoModel, AutoTokenizer

PINECONE_INDEX_NAME = "fineweb10bt-512-0w-e5-base-v2"
PINECONE_NAMESPACE="default"

@cache
def has_mps():
    return torch.backends.mps.is_available()

@cache
def has_cuda():
    return torch.cuda.is_available()

@cache
def get_tokenizer(model_name: str = "intfloat/e5-base-v2"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer

@cache
def get_model(model_name: str = "intfloat/e5-base-v2"):
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    if has_mps():
        model = model.to("mps")
    elif has_cuda():
        model = model.to("cuda")
    else:
        model = model.to("cpu")
    return model

def average_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def embed_query(query: str,
                query_prefix: str = "query: ",
                model_name: str = "intfloat/e5-base-v2",
                pooling: Literal["cls", "avg"] = "avg",
                normalize: bool =True) -> list[float]:
    return batch_embed_queries([query], query_prefix, model_name, pooling, normalize)[0]

def batch_embed_queries(queries: List[str], query_prefix: str = "query: ", model_name: str = "intfloat/e5-base-v2", pooling: Literal["cls", "avg"] = "avg", normalize: bool =True) -> List[List[float]]:
    with_prefixes = [" ".join([query_prefix, query]) for query in queries]
    tokenizer = get_tokenizer(model_name)
    model = get_model(model_name)
    with torch.no_grad():
        encoded = tokenizer(with_prefixes, padding=True, return_tensors="pt", truncation="longest_first")
        encoded = encoded.to(model.device)
        model_out = model(**encoded)
        if pooling == "cls":
            embeddings = model_out.last_hidden_state[:, 0]
        elif pooling == "avg":
            embeddings = average_pool(model_out.last_hidden_state, encoded["attention_mask"])
        else:
            raise ValueError(f"Unsupported pooling method: {pooling}")
        if normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return embeddings.tolist()

@cache
def get_pinecone_index(index_name: str = PINECONE_INDEX_NAME):
    pc = Pinecone(api_key=get_ssm_secret("/pinecone/ro_token"))
    index = pc.Index(name=index_name)
    return index

def query_pinecone(query: str, top_k: int = 10, namespace: str = PINECONE_NAMESPACE) -> dict:
    index = get_pinecone_index()
    results = index.query(
        vector=embed_query(query),
        top_k=top_k,
        include_values=False,
        namespace=namespace,
        include_metadata=True
    )

    return results

def batch_query_pinecone(queries: list[str], top_k: int = 10, namespace: str = PINECONE_NAMESPACE, n_parallel: int = 10) -> list[dict]:
    """Batch query a Pinecone index and return the results.

    Internally uses a ThreadPool to parallelize the queries.
    """
    index = get_pinecone_index()
    embeds = batch_embed_queries(queries)
    pool = ThreadPool(n_parallel)
    results = pool.map(lambda x: index.query(vector=x, top_k=top_k, include_values=False, namespace=namespace, include_metadata=True), embeds)
    return results

def show_pinecone_results(results):
    for match in results["matches"]:
        print("chunk:", match["id"], "score:", match["score"])
        print(match["metadata"]["text"])
        print()


from functools import cache
from opensearchpy import OpenSearch, AWSV4SignerAuth, RequestsHttpConnection

OPENSEARCH_INDEX_NAME = "fineweb10bt-512-0w-e5-base-v2"

@cache
def get_client(profile: str = AWS_PROFILE_NAME, region: str = AWS_REGION_NAME):
    credentials = boto3.Session(profile_name=profile).get_credentials()
    auth = AWSV4SignerAuth(credentials, region=region)
    host_name = get_ssm_value("/opensearch/endpoint", profile=profile, region=region)
    aos_client = OpenSearch(
        hosts=[{"host": host_name, "port": 443}],
        http_auth=auth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
    )
    return aos_client

def query_opensearch(query: str, top_k: int = 10) -> dict:
    """Query an OpenSearch index and return the results."""
    client = get_client()
    results = client.search(index=OPENSEARCH_INDEX_NAME, body={"query": {"match": {"text": query}}, "size": top_k})
    return results

def batch_query_opensearch(queries: list[str], top_k: int = 10, n_parallel: int = 10) -> list[dict]:
    """Sends a list of queries to OpenSearch and returns the results. Configuration of Connection Timeout might be needed for serving large batches of queries"""
    client = get_client()
    request = []
    for query in queries:
        req_head = {"index": OPENSEARCH_INDEX_NAME}
        req_body = {
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["text"],
                }
            },
            "size": top_k,
        }
        request.extend([req_head, req_body])

    return client.msearch(body=request)

def show_opensearch_results(results: dict):
    for match in results["hits"]["hits"]:
        print("chunk:", match["_id"], "score:", match["_score"])
        print(match["_source"]["text"])
        print()

if __name__ == '__main__':
    # OpenSearch retriever
    #results = query_opensearch("What is a second brain?")
    #show_opensearch_results(results)
    #batch_results = batch_query_opensearch(["What is a second brain?", "how does a brain work?", "Where is Paris?"], top_k=1)
    #for results in batch_results['responses']:
    #    show_opensearch_results(results)

    # PineCone Retriever
    #results = query_pinecone("What is a second brain?")
    #show_pinecone_results(results)
    #batch_results = batch_query_pinecone(["What is a second brain?", "how does a brain work?", "Where is Paris?"], top_k=2)
    #for results in batch_results:
    #   show_pinecone_results(results)
