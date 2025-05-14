"""
Main execution file for LiveRAG challenge with multiple Falcon model options
"""
import os
import sys
from dotenv import load_dotenv
import argparse
import json
import time
import socket
from typing import Dict, List, Any, Optional
import tqdm
import pandas as pd
import requests
import jsonschema
from jsonschema import validate

from config import (
    LLM_MODEL_ID,
    RESULTS_DIR,
    DEFAULT_TOP_K_RETRIEVAL,
    DEFAULT_TOP_K_FINAL,
    BM25_INDEX_PATH,
    MAX_NEW_TOKENS,
    TEMPERATURE,
    TOP_P,
    DO_SAMPLE
)
from utils import setup_logger, log_execution_time, save_results
from auth import setup_huggingface_auth
from retriever_utils import cleanup
import pyterrier as pt

# Add the current directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Import AWS retriever functions directly
# Copy the functions from aws_retrievers.py that you need
from functools import cache
import boto3


"""
    #############
    How to run: remote AWS vs. local PyTerrier.
    #############
Test jsonl file (5 QAs): "5QAs_DMinput.jsonl"

1. To run with local PyTerrier index:
python main.py --input 5QAs_DMinput.jsonl --output results.json --index-path /path/to/index

2. To run with AWS resources:
python main.py --input 5QAs_DMinput.jsonl --output results.json --use-aws --retriever-type hybrid



    #############
    Different ways to run RAG:
    #############
>> Hybrid RAG (OpenSearch + Pinecone), Sparse BM25 Only RAG (OpenSearch), or Dense Only RAG (Pinecone).
--retriever-type hybrid
--retriever-type sparse
--retriever-type dense

>> DIFFERENT HYBRID EXAMPLES:
Using ec2: 
python main.py --input 5QAs_DMinput.jsonl --output results_ec2.json --use-aws --retriever-type hybrid --top-k-dense 30 --top-k-sparse 30 --top-k-final 5 --falcon-mode ec2 --ec2-endpoint http://YOUR-EC2-IP:8000/generate

Using AI71 falcon:
python main.py --input 5QAs_DMinput.jsonl --output results_ai71.json --use-aws --retriever-type hybrid --top-k-dense 30 --top-k-sparse 30 --top-k-final 5 --falcon-mode ai71 --ai71-api-key YOUR_AI71_API_KEY

Using local GPU for falcon (if available):
python main.py --input 5QAs_DMinput.jsonl --output results_local.json --use-aws --retriever-type hybrid --top-k-dense 30 --top-k-sparse 30 --top-k-final 5 --falcon-mode local



"""




# Setup logger
logger = setup_logger("main")

# Initialize PyTerrier if needed
if not pt.java.started():
    pt.java.init()

# Define constants for AWS resources
AWS_PROFILE_NAME = "sigir-participant"
AWS_REGION_NAME = "us-east-1"
OPENSEARCH_INDEX_NAME = "fineweb10bt-512-0w-e5-base-v2"
PINECONE_INDEX_NAME = "fineweb10bt-512-0w-e5-base-v2"
PINECONE_NAMESPACE = "default"

# Add the parent directory to Python's module search path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# Import AWS retriever functions if available
try:
    from retrieve.aws_retrievers import (
        query_opensearch, query_pinecone, 
        get_ssm_value, get_ssm_secret,
        batch_query_opensearch, batch_query_pinecone,
        PINECONE_NAMESPACE
    )
    AWS_RETRIEVERS_AVAILABLE = True
    logger.info("AWS retrievers successfully imported")
except ImportError as e:
    logger.warning(f"AWS retriever functions not available: {e}")
    AWS_RETRIEVERS_AVAILABLE = False


# Add this just after the AWS retrievers import section:
print("AWS Retrievers import successful, testing connectivity...")
try:
    # Test connection with OpenSearch
    test_result = query_opensearch("test", top_k=1)
    print(f"OpenSearch test successful, got {len(test_result.get('hits', {}).get('hits', []))} hits")
    
    # Test connection with Pinecone
    test_result = query_pinecone("test", top_k=1)
    print(f"Pinecone test successful, got {len(test_result.get('matches', []))} matches")
except Exception as e:
    print(f"AWS connectivity test failed: {e}")
    import traceback
    traceback.print_exc()

# Wrapper for OpenSearch retriever
class OpenSearchRetriever(pt.Transformer):
    def __init__(self, top_k: int = 10):
        """
        Parameters:
          top_k: Number of top documents to retrieve per query.
        """
        self.top_k = top_k
    
    def transform(self, queries_df: pd.DataFrame) -> pd.DataFrame:
        """
        For each query in the input DataFrame, calls query_opensearch
        and converts the results to a DataFrame with required columns.
        """
        results_rows = []
        for _, row in queries_df.iterrows():
            query = row["query"] if "query" in row.index else row.get("text", "")
            qid = row["qid"] if "qid" in row.index else None
            # Call the provided OpenSearch querying function
            os_results = query_opensearch(query, top_k=self.top_k)
            hits = os_results.get("hits", {}).get("hits", [])
            for rank, hit in enumerate(hits):
                results_rows.append({
                    "qid": qid,
                    "query": query,
                    "docno": hit["_id"],
                    "score": hit["_score"],
                    "rank": rank,
                    "text": hit["_source"].get("text", "")
                })
        return pd.DataFrame(results_rows)

# Wrapper for Pinecone retriever
class PineConeRetriever(pt.Transformer):
    def __init__(self, top_k: int = 10, namespace: str = PINECONE_NAMESPACE):
        """
        Parameters:
          top_k: Number of top documents to retrieve per query.
          namespace: Pinecone namespace to use.
        """
        self.top_k = top_k
        self.namespace = namespace
    
    def transform(self, queries_df: pd.DataFrame) -> pd.DataFrame:
        """
        For each query in the input DataFrame, calls query_pinecone
        and converts the results into a DataFrame with required columns.
        """
        results_rows = []
        for _, row in queries_df.iterrows():
            query = row["query"] if "query" in row.index else row.get("text", "")
            qid = row["qid"] if "qid" in row.index else None
            # Call the provided Pinecone querying function
            pc_results = query_pinecone(query, top_k=self.top_k, namespace=self.namespace)
            matches = pc_results.get("matches", [])
            for rank, match in enumerate(matches):
                results_rows.append({
                    "qid": qid,
                    "query": query,
                    "docno": match["id"],
                    "score": match["score"],
                    "rank": rank,
                    "text": match["metadata"].get("text", "")
                })
        return pd.DataFrame(results_rows)

    
# Enhanced prompt generator with parallel processing and retries
class EnhancedPromptGenerator:
    """Enhanced prompt generator with multiple backend options, parallel processing, and retries"""
    
    def __init__(self, model_id: str = LLM_MODEL_ID, mode: str = "local", 
                 ec2_endpoint: str = None, ai71_api_key: str = None,
                 ai71_base_url: str = "https://api.ai71.ai/v1", n_parallel: int = 5, n_retries: int = 3):
        """
        Initialize the prompt generator with different backend options
        
        Parameters:
            model_id: Model identifier
            mode: One of "local", "ec2", or "ai71"
            ec2_endpoint: URL for EC2 API endpoint (required if mode="ec2")
            ai71_api_key: API key for AI71 (required if mode="ai71")
            ai71_base_url: Base URL for AI71 API (optional)
            n_parallel: Number of parallel requests (for ai71 mode)
            n_retries: Number of retries on failure (for ai71 mode)
        """
        self.model_id = model_id
        self.mode = mode
        self.ec2_endpoint = ec2_endpoint
        self.ai71_api_key = ai71_api_key
        self.ai71_base_url = ai71_base_url
        self.n_parallel = n_parallel
        self.n_retries = n_retries
        
        # Load model locally if needed
        if mode == "local":
            logger.info(f"Loading model locally: {model_id}")
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, 
                torch_dtype=torch.float16,  # Use float16 instead of bfloat16 for wider compatibility
                device_map="auto",
                low_cpu_mem_usage=True
            )
            
            # Set padding token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            logger.info(f"Using remote API for model: {model_id} in {mode} mode")
            
            # Initialize API client if using AI71
            if mode == "ai71" and ai71_api_key:
                try:
                    from ai71 import AI71
                    self.ai71_client = AI71(
                        api_key=ai71_api_key, 
                        base_url=self.ai71_base_url 
                    )
                    logger.info("AI71 client initialized successfully")
                except ImportError:
                    logger.warning("ai71 package not installed. Using requests library instead.")
                    self.ai71_client = None
    
    # Used for submission.
    def _build_prompt(self, query: str, context: List[Dict[str, Any]]) -> str:
            """Build prompt from query and context"""
            # Basic prompt template
            prompt = f"""You are an AI assistant tasked with answering questions based on the provided information.
            
    Information:
    """
            
            # Add context
            for i, passage in enumerate(context):
                prompt += f"""[{i+1}] {passage['text']}

    """
            
            # Add query
            prompt += f"""
    Question: {query}

    Answer the question based only on the provided information. Keep the answer concise, limited to 200 tokens. If the information doesn't contain the answer, say "I don't have enough information to answer this question."

    Answer:"""
            
            return prompt
    
    # Not used
    def _build_advanced_prompt(self, query: str, context: List[Dict[str, Any]]) -> str:
        """Build enhanced prompt from query and context"""
        prompt = f"""You are a highly knowledgeable AI assistant tasked with providing accurate, helpful answers based strictly on the provided information. Your goal is to be both relevant and faithful to the source material.

    Information:
    """
        # Add context with clear source markers
        for i, passage in enumerate(context):
            prompt += f"""[Source {i+1}]: {passage['text']}

    """
        
        prompt += f"""
    Question: {query}

    Instructions:
    1. Answer ONLY using the information provided in the sources above
    2. If you cannot find the complete answer in the sources, say: "I don't have enough information to fully answer this question." and explain what specific information is missing
    3. If the question is unclear or uses informal language, interpret it charitably and answer the most likely intended meaning
    4. Cite your sources using [Source X] notation when making specific claims
    5. Be direct and concise while ensuring all key information is included
    6. Match the technical level and tone of your response to the question style
    7. Stay strictly faithful to the source material - do not make assumptions or add external knowledge

    Answer: """

        return prompt
    

    # Not used.
    def _build_other_prompt(self, query: str, context: List[Dict[str, Any]]) -> str:
        """Build competition-optimized prompt for Falcon-3B-10B-Instruct"""
        
        # Start with task framing that works well with Falcon's training
        prompt = f"""Task: You are a precise and accurate AI assistant. Your job is to answer questions using ONLY the provided reference passages. Here are your reference passages:

    """
        # Add context more efficiently - Falcon tends to work better with numbered lists
        for i, passage in enumerate(context, 1):
            prompt += f"""[{i}] {passage['text'].strip()}

    """
        
        # Clean separation before question
        prompt += f"""
    Question: {query.strip()}

    Important Guidelines:
    - Answer ONLY based on the above passages
    - If information is missing, say "Based on the provided passages, I cannot fully answer this question."
    - Support ALL claims with passage references like [1], [2], etc.
    - Be precise and accurate

    Answer:"""

        return prompt
    
    def _complete_ai71(self, messages: list, max_tokens: int = MAX_NEW_TOKENS, temperature: float = TEMPERATURE, top_p: float = TOP_P):
        """Run single completion with retries for AI71"""
        retries = 0
        while True:
            try:
                return self.ai71_client.chat.completions.create(
                    model="tiiuae/falcon3-10b-instruct",
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p
                )
            except Exception as e:
                retries += 1
                if self.n_retries < retries:
                    raise e
                logger.warning(f"AI71 API call failed, retrying ({retries}/{self.n_retries}): {e}")
                time.sleep(retries)  # Exponential backoff


    def _batch_complete_ai71(self, list_of_messages: list, max_tokens: int = MAX_NEW_TOKENS, 
                            temperature: float = TEMPERATURE, top_p: float = TOP_P):
        """Run batch completions in parallel with AI71"""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        results = []
        
        with ThreadPoolExecutor(max_workers=self.n_parallel) as executor:
            # Submit requests
            futures = [
                executor.submit(
                    self._complete_ai71, 
                    messages, 
                    max_tokens, 
                    temperature, 
                    top_p
                )
                for messages in list_of_messages
            ]
            
            # Collect results as they complete
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"AI71 request failed: {e}")
                    results.append(None)
        
        return results
    

    def _complete_ec2(self, prompt: str, max_tokens: int = MAX_NEW_TOKENS, 
                      temperature: float = TEMPERATURE, top_p: float = TOP_P, 
                      do_sample: bool = DO_SAMPLE):
        """Run single completion with retries for EC2 endpoint"""
        retries = 0
        while True:
            try:
                response = requests.post(
                    self.ec2_endpoint,
                    json={
                        "prompt": prompt,
                        "max_new_tokens": max_tokens,
                        "temperature": temperature,
                        "top_p": top_p,
                        "do_sample": do_sample
                    },
                    timeout=60  # 60 second timeout
                )
                
                if response.status_code == 200:
                    return response.json()
                else:
                    raise Exception(f"EC2 API returned status {response.status_code}: {response.text}")
            except Exception as e:
                retries += 1
                if self.n_retries < retries:
                    raise e
                logger.warning(f"EC2 API call failed, retrying ({retries}/{self.n_retries}): {e}")
                time.sleep(retries)  # Exponential backoff

    def batch_generate(self, queries: List[str], contexts: List[List[Dict[str, Any]]], 
                       ids: List[Any], doc_ids: List[Any], question_types: List[Any], 
                       user_types: List[Any]) -> List[Dict[str, Any]]:
        """Generate answers for multiple queries in parallel using configured backend"""
        if len(queries) != len(contexts) or len(queries) != len(ids):
            raise ValueError("Length of queries, contexts, and ids must be the same")
        
        start_time = time.time()
        results = []
        
        # Prepare prompts and metadata
        prompts = [self._build_prompt(q, c) for q, c in zip(queries, contexts)]
        metadata = list(zip(ids, doc_ids, question_types, user_types, queries))
        
        try:
            generation_start = time.time()
            
            if self.mode == "local":
                # Process sequentially for local model (could be parallelized with multiple GPUs)
                for i, (prompt, meta) in enumerate(zip(prompts, metadata)):
                    qa_id, doc_id, question_type, user_type, query = meta
                    single_result = self.generate(query, contexts[i], qa_id, doc_id, question_type, user_type)
                    results.append(single_result)
            
            elif self.mode == "ec2":
                # Process in parallel with EC2 endpoint
                from concurrent.futures import ThreadPoolExecutor, as_completed
                
                with ThreadPoolExecutor(max_workers=self.n_parallel) as executor:
                    # Submit requests
                    futures = [
                        executor.submit(self._complete_ec2, prompt)
                        for prompt in prompts
                    ]
                    
                    # Collect results as they complete
                    for i, future in enumerate(as_completed(futures)):
                        try:
                            qa_id, doc_id, question_type, user_type, query = metadata[i]
                            
                            response = future.result()
                            answer = response.get("generated_text", "").strip()
                            
                            generation_time = time.time() - generation_start
                            results.append({
                                "answer": answer,
                                "qa_id": qa_id,
                                "doc_id": doc_id,
                                "question_type": question_type,
                                "user_type": user_type,
                                "input_prompt": prompts[i],
                                "query": query,
                                "generation_time": generation_time,
                                "total_time": time.time() - start_time,
                                "model_id": self.model_id,
                                "mode": self.mode
                            })
                        except Exception as e:
                            logger.error(f"Error processing EC2 result: {e}")
                            qa_id, doc_id, question_type, user_type, query = metadata[i]
                            results.append({
                                "answer": "I apologize, but I encountered an error while generating the answer.",
                                "qa_id": qa_id,
                                "doc_id": doc_id,
                                "question_type": question_type,
                                "user_type": user_type,
                                "input_prompt": prompts[i],
                                "query": query,
                                "error": str(e),
                                "generation_time": time.time() - generation_start,
                                "total_time": time.time() - start_time,
                                "model_id": self.model_id,
                                "mode": self.mode
                            })
            
            elif self.mode == "ai71":
                # Process in parallel with AI71
                if hasattr(self, 'ai71_client') and self.ai71_client:
                    # Convert prompts to message format
                    messages_list = [
                        [
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": prompt}
                        ]
                        for prompt in prompts
                    ]
                    
                    # Run batch completion
                    batch_results = self._batch_complete_ai71(messages_list)
                    
                    # Process results
                    for i, response in enumerate(batch_results):
                        qa_id, doc_id, question_type, user_type, query = metadata[i]
                        
                        if response is not None:
                            answer = response.choices[0].message.content.strip()
                        else:
                            answer = "I apologize, but I encountered an error while generating the answer."
                        
                        results.append({
                            "answer": answer,
                            "qa_id": qa_id,
                            "doc_id": doc_id,
                            "question_type": question_type,
                            "user_type": user_type,
                            "input_prompt": prompts[i],
                            "query": query,
                            "generation_time": time.time() - generation_start,
                            "total_time": time.time() - start_time,
                            "model_id": self.model_id,
                            "mode": self.mode
                        })
                else:
                    # Fallback to sequential processing with requests
                    for i, (prompt, meta) in enumerate(zip(prompts, metadata)):
                        qa_id, doc_id, question_type, user_type, query = meta
                        single_result = self.generate(query, contexts[i], qa_id, doc_id, question_type, user_type)
                        results.append(single_result)
            
            else:
                logger.error(f"Unknown generation mode: {self.mode}")
                # Return error results
                for i, meta in enumerate(metadata):
                    qa_id, doc_id, question_type, user_type, query = meta
                    results.append({
                        "answer": f"Error: Unknown generation mode {self.mode}",
                        "qa_id": qa_id,
                        "doc_id": doc_id,
                        "question_type": question_type,
                        "user_type": user_type,
                        "input_prompt": prompts[i],
                        "query": query,
                        "generation_time": 0,
                        "total_time": time.time() - start_time,
                        "model_id": self.model_id,
                        "mode": self.mode
                    })
        
        except Exception as e:
            logger.error(f"Error during batch generation: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            # Return error results for all queries
            for i, meta in enumerate(metadata):
                qa_id, doc_id, question_type, user_type, query = meta
                results.append({
                    "answer": "I apologize, but I encountered an error while generating the answer.",
                    "qa_id": qa_id,
                    "doc_id": doc_id,
                    "question_type": question_type,
                    "user_type": user_type,
                    "input_prompt": prompts[i] if i < len(prompts) else "",
                    "query": query,
                    "error": str(e),
                    "generation_time": time.time() - generation_start,
                    "total_time": time.time() - start_time,
                    "model_id": self.model_id,
                    "mode": self.mode
                })
        
        return results
    
    @log_execution_time
    def generate(self, query: str, context: List[Dict[str, Any]], qa_id, doc_id, question_type, user_type) -> Dict[str, Any]:
        """Generate answer for a single query using configured backend"""
        start_time = time.time()
        
        # Build prompt
        prompt = self._build_prompt(query, context)
        
        # Generate based on selected mode
        try:
            generation_start = time.time()
            
            if self.mode == "local":
                # Local generation using HuggingFace Transformers
                import torch
                
                # Tokenize input
                model_inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
                
                # Generate
                with torch.no_grad():
                    generation_output = self.model.generate(
                        **model_inputs,
                        max_new_tokens=MAX_NEW_TOKENS,
                        temperature=TEMPERATURE,
                        top_p=TOP_P,
                        do_sample=DO_SAMPLE,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                
                # Decode output
                answer = self.tokenizer.decode(
                    generation_output[0][model_inputs.input_ids.shape[1]:], 
                    skip_special_tokens=True
                ).strip()
                
            elif self.mode == "ec2":
                # EC2 API endpoint generation with retries
                if not self.ec2_endpoint:
                    logger.error("EC2 endpoint URL not provided")
                    answer = "Error: EC2 endpoint URL not configured"
                else:
                    response = self._complete_ec2(prompt)
                    answer = response.get("generated_text", "").strip()
            
            elif self.mode == "ai71":
                # AI71 API generation with retries
                if not self.ai71_api_key:
                    logger.error("AI71 API key not provided")
                    answer = "Error: AI71 API key not configured"
                elif hasattr(self, 'ai71_client') and self.ai71_client:
                    # Use AI71 Python SDK with retries
                    response = self._complete_ai71([
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ])
                    answer = response.choices[0].message.content.strip()
                else:
                    # Use requests as fallback with retries
                    retries = 0
                    while True:
                        try:
                            response = requests.post(
                                "https://api.ai71.ai/v1/chat/completions",
                                json={
                                    "model": "tiiuae/falcon3-10b-instruct",
                                    "messages": [
                                        {"role": "system", "content": "You are a helpful assistant."},
                                        {"role": "user", "content": prompt}
                                    ],
                                    "max_tokens": MAX_NEW_TOKENS,
                                    "temperature": TEMPERATURE,
                                    "top_p": TOP_P
                                },
                                headers={"Authorization": f"Bearer {self.ai71_api_key}"},
                                timeout=60
                            )
                            
                            if response.status_code == 200:
                                answer = response.json()["choices"][0]["message"]["content"].strip()
                                break
                            else:
                                raise Exception(f"AI71 API returned status {response.status_code}: {response.text}")
                        except Exception as e:
                            retries += 1
                            if self.n_retries < retries:
                                logger.error(f"AI71 API failed after {self.n_retries} retries: {e}")
                                answer = f"Error: AI71 API failed after {self.n_retries} retries"
                                break
                            logger.warning(f"AI71 API call failed, retrying ({retries}/{self.n_retries}): {e}")
                            time.sleep(retries)  # Exponential backoff
            
            else:
                logger.error(f"Unknown generation mode: {self.mode}")
                answer = f"Error: Unknown generation mode {self.mode}"
            
            generation_time = time.time() - generation_start
            
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            import traceback
            logger.error(traceback.format_exc())
            answer = "I apologize, but I encountered an error while generating the answer."
            generation_time = time.time() - generation_start
        
        # Prepare result
        result = {
            "answer": answer,
            "qa_id": qa_id, 
            "doc_id": doc_id,
            "question_type": question_type,
            "user_type": user_type,
            "input_prompt": prompt,
            "query": query,
            "generation_time": generation_time,
            "total_time": time.time() - start_time,
            "model_id": self.model_id,
            "mode": self.mode
        }
        
        return result

def parse_args(AWS_EC2_ENDPOINT:Optional[str] = None, AI_71_API_KEY: Optional[str] = None):
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="LiveRAG Challenge Runner")
    parser.add_argument("--input", type=str, default="input.jsonl",
                        help="Input file with queries (JSONL)")
    parser.add_argument("--output", type=str, default="output.json",
                        help="Output file for results (JSON)")
    parser.add_argument("--top-k-sparse", type=int, default=DEFAULT_TOP_K_RETRIEVAL,
                        help="Number of documents to retrieve from sparse index")
    parser.add_argument("--top-k-dense", type=int, default=DEFAULT_TOP_K_RETRIEVAL,
                        help="Number of documents to retrieve from dense index")
    parser.add_argument("--top-k-final", type=int, default=DEFAULT_TOP_K_FINAL,
                        help="Number of documents to use in final prompt")
    parser.add_argument("--hf-token", type=str, default=None,
                        help="Hugging Face API token for accessing Falcon model")
    parser.add_argument("--retriever-type", type=str, default="hybrid",
                        choices=["sparse", "dense", "hybrid", "local"],
                        help="Type of retriever to use (sparse=OpenSearch, dense=Pinecone, hybrid=both, local=PyTerrier)")
    parser.add_argument("--use-aws", action="store_true",
                        help="Use AWS resources (OpenSearch, Pinecone)")
    parser.add_argument("--index-path", type=str, default=BM25_INDEX_PATH,
                        help="Path to PyTerrier index (for local retriever)")
    
    # New arguments for multiple Falcon options
    parser.add_argument("--falcon-mode", type=str, default="local",
                        choices=["local", "ec2", "ai71"],
                        help="Mode for Falcon model: local, ec2, or ai71")
    parser.add_argument("--ec2-endpoint", type=str, default=AWS_EC2_ENDPOINT,
                        help="URL for EC2 API endpoint (e.g., http://your-ec2-ip:8000/generate)")
    parser.add_argument("--ai71-api-key", type=str, default=AI_71_API_KEY,
                        help="API key for AI71 platform")
    
    # Add competition submission flag
    parser.add_argument("--competition-format", action="store_true",
                        help="Format output for competition submission")
    parser.add_argument("--competition-output", type=str, default="competition_submission.json",
                        help="Output file for competition submission format")
    
    # Add these to parse_args() if they aren't already there
    parser.add_argument("--batch-size", type=int, default=5,
                        help="Batch size for parallel processing")
    parser.add_argument("--n-parallel", type=int, default=3,
                        help="Number of parallel requests")
    parser.add_argument("--n-retries", type=int, default=3,
                        help="Number of retries on failure")
    
    return parser.parse_args()


def load_competition_data(input_file: str) -> List[Dict[str, Any]]:
    """Load competition questions from the JSONL file"""

    try:
        data = []
        with open(input_file, 'r') as f:
            for line in f:
                item = json.loads(line.strip())
                # Make sure required fields exist
                if "id" not in item:
                    logger.warning(f"Item missing 'id' field: {item}")
                    continue
                if "question" not in item:
                    logger.warning(f"Item missing 'question' field: {item}")
                    continue
                data.append(item)
        logger.info(f"Loaded {len(data)} questions from {input_file}")
        return data
    except Exception as e:
        logger.error(f"Error loading input file: {e}")
        return []

def load_synthetic_data(input_file: str) -> List[Dict[str, str]]:
    """Load questions from the JSONL file"""
    try:
        data = []
        with open(input_file, 'r') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        logger.info(f"Loaded {len(data)} questions from {input_file}")
        return data
    except Exception as e:
        logger.error(f"Error loading input file: {e}")
        return []
    

def format_competition_submission(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Format results for competition submission"""
    formatted_results = []
    
    for result in results:
        # Extract document text and IDs
        passages = []
        if "context" in result:
            context = result["context"]
            for doc in context:
                passage = {
                    "passage": doc["text"],
                    "doc_IDs": []
                }
                
                # Try to extract doc_IDs from different possible formats
                doc_id = doc.get("id", "")
                if doc_id.startswith("doc-"):
                    # Extract UUID from format like: doc-<urn:uuid:...>::chunk-0
                    doc_id_parts = doc_id.split("::")
                    raw_id = doc_id_parts[0].replace("doc-", "")
                    passage["doc_IDs"].append(raw_id)
                else:
                    # Just use the raw ID
                    passage["doc_IDs"].append(doc_id)
                
                # Add metadata doc IDs if available
                if "metadata" in doc and "doc_ids" in doc["metadata"]:
                    for meta_id in doc["metadata"]["doc_ids"]:
                        if meta_id not in passage["doc_IDs"]:
                            passage["doc_IDs"].append(meta_id)
                
                passages.append(passage)
        
        # Format the competition submission item
        formatted_item = {
            "id": result.get("id", 0),
            "question": result.get("query", ""),
            "passages": passages,
            "final_prompt": result.get("input_prompt", ""),
            "answer": result.get("answer", "")
        }
        
        formatted_results.append(formatted_item)
    
    return formatted_results

def validate_competition_format(formatted_results: List[Dict[str, Any]]) -> bool:
    """Validate the competition submission format"""
    # Define JSON schema
    schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "Answer file schema",
        "type": "object",
        "properties": {
            "id": {
                "type": "integer",
                "description": "Question ID"
            },
            "question": {
                "type": "string",
                "description": "The question"
            },
            "passages": {
                "type": "array",
                "description": "Passages used and related FineWeb doc IDs, ordered by decreasing importance",
                "items": {
                    "type": "object",
                    "properties": {
                        "passage": {
                            "type": "string",
                            "description": "Passage text"
                        },
                        "doc_IDs": {
                            "type": "array",
                            "description": "Passage related FineWeb doc IDs, ordered by decreasing importance",
                            "items": {
                                "type": "string",
                                "description": "FineWeb doc ID, e.g., <urn:uuid:d69cbebc-133a-4ebe-9378-68235ec9f091>"
                            }
                        }
                    },
                    "required": ["passage", "doc_IDs"]
                }
            },
            "final_prompt": {
                "type": "string",
                "description": "Final prompt, as submitted to Falcon LLM"
            },
            "answer": {
                "type": "string",
                "description": "Your answer"
            }
        },
        "required": ["id", "question", "passages", "final_prompt", "answer"]
    }
    
    try:
        
        # Validate each item
        invalid_count = 0
        for item in formatted_results:
            try:
                validate(instance=item, schema=schema)
            except jsonschema.exceptions.ValidationError as e:
                logger.error(f"Validation error for item {item.get('id', 'unknown')}: {e}")
                invalid_count += 1
        
        if invalid_count > 0:
            logger.warning(f"{invalid_count} items failed validation")
            return False
        
        return True
    except ImportError:
        logger.warning("jsonschema package not available, skipping validation")
        return True
    

def save_competition_submission(formatted_results: List[Dict[str, Any]], output_file: str):
    """Save the competition submission to a file"""
    try:
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        
        # Save each item as JSONL
        with open(output_file, 'w') as f:
            for item in formatted_results:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        logger.info(f"Competition submission saved to {output_file}")
        return True
    except Exception as e:
        logger.error(f"Error saving competition submission: {e}")
        return False


def process_query_aws_for_competition(query: str, sparse_retriever, dense_retriever, generator, 
                  top_k_sparse: int, top_k_dense: int, top_k_final: int, 
                  retriever_type: str, query_id: int) -> Dict[str, Any]:
    """Process a single query through the RAG pipeline for competition submission"""
    start_time = time.time()
    
    # Retrieve documents
    try:
        # Clean up query
        cleaned_query = cleanup(query)
        context = []
        
        # Retrieve from sparse index (OpenSearch)
        if retriever_type in ["sparse", "hybrid"]:
            sparse_docs = sparse_retriever.search(cleaned_query)
            for _, row in sparse_docs.iterrows():
                if 'text' in row and row['text'] and len(row['text'].strip()) > 0:
                    context.append({
                        "id": row.get("docno", "unknown"),
                        "text": row.get("text", ""),
                        "score": float(row.get("score", 0.0)),
                        "metadata": {"source": "opensearch", "doc_ids": [row.get("docno", "unknown")]}
                    })
        
        # Retrieve from dense index (Pinecone)
        if retriever_type in ["dense", "hybrid"]:
            dense_docs = dense_retriever.search(cleaned_query)
            for _, row in dense_docs.iterrows():
                if 'text' in row and row['text'] and len(row['text'].strip()) > 0:
                    context.append({
                        "id": row.get("docno", "unknown"),
                        "text": row.get("text", ""),
                        "score": float(row.get("score", 0.0)),
                        "metadata": {"source": "pinecone", "doc_ids": [row.get("docno", "unknown")]}
                    })
        
        # Sort by score and take top_k_final
        context = sorted(context, key=lambda x: x["score"], reverse=True)[:top_k_final]
        
        # Generate answer
        result = generator.generate(query, context, f"q{query_id}", [], [], [])
        
        # Add additional fields needed for competition format
        result["id"] = query_id
        result["query"] = query
        result["context"] = context
        result["documents"] = [doc["id"] for doc in context]
        result["total_pipeline_time"] = time.time() - start_time
        
        return result
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return {
            "id": query_id,
            "query": query,
            "answer": "An error occurred while processing this query.",
            "error": str(e),
            "total_pipeline_time": time.time() - start_time
        }

def initialize_local_retriever(index_path: str):
    """Initialize BM25 retriever with existing index"""
    try:
        # First, check if properties file exists
        properties_path = os.path.join(index_path, "data.properties")
        if not os.path.exists(properties_path):
            # Try looking for relative properties path
            properties_path = index_path
        
        # Load the index
        index_ref = pt.IndexRef.of(properties_path)
        index = pt.IndexFactory.of(index_ref)
        logger.info(f"Loaded index with {index.getCollectionStatistics().getNumberOfDocuments()} documents")
        
        # Create the retriever pipeline
        retriever = pt.BatchRetrieve(index, wmodel="BM25") >> pt.text.get_text(index, "text")
        return retriever, index
    except Exception as e:
        logger.error(f"Error initializing retriever: {e}")
        return None, None

@log_execution_time
def process_query_local(query: str, retriever, generator, top_k: int, qa_id, doc_id, question_type, user_type) -> Dict[str, Any]:
    """Process a single query through the RAG pipeline using local PyTerrier index"""
    start_time = time.time()
    
    # Retrieve documents
    try:
        # Clean up query and retrieve documents
        cleaned_query = cleanup(query)
        documents = retriever.search(cleaned_query)
        
        # Format for generator
        context = []
        for _, row in documents.iterrows():
            if len(context) >= top_k:
                break
            
            if 'text' in row and row['text'] and len(row['text'].strip()) > 0:
                context.append({
                    "id": row.get("docno", "unknown"),
                    "text": row.get("text", ""),
                    "score": float(row.get("score", 0.0)),
                    "metadata": {"source": "bm25"}
                })
        
        # Generate answer
        result = generator.generate(query, context, qa_id, doc_id, question_type, user_type)
        
        # Add metadata
        result["documents"] = [doc["id"] for doc in context]
        result["total_pipeline_time"] = time.time() - start_time
        
        return result
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return {
            "answer": "An error occurred while processing this query.",
            "qa_id": qa_id,
            "doc_id": doc_id,
            "question_type": question_type,
            "user_type": user_type,
            "query": query,
            "error": str(e),
            "total_pipeline_time": time.time() - start_time
        }

@log_execution_time
def process_query_aws(query: str, sparse_retriever, dense_retriever, generator, 
                  top_k_sparse: int, top_k_dense: int, top_k_final: int, 
                  retriever_type: str, qa_id, doc_id, question_type, user_type) -> Dict[str, Any]:
    """Process a single query through the RAG pipeline using AWS resources"""
    start_time = time.time()
    
    # Retrieve documents
    try:
        # Clean up query
        cleaned_query = cleanup(query)
        context = []
        
        # Retrieve from sparse index (OpenSearch)
        if retriever_type in ["sparse", "hybrid"]:
            sparse_docs = sparse_retriever.search(cleaned_query)
            for _, row in sparse_docs.iterrows():
                if 'text' in row and row['text'] and len(row['text'].strip()) > 0:
                    context.append({
                        "id": row.get("docno", "unknown"),
                        "text": row.get("text", ""),
                        "score": float(row.get("score", 0.0)),
                        "metadata": {"source": "opensearch"}
                    })
        
        # Retrieve from dense index (Pinecone)
        if retriever_type in ["dense", "hybrid"]:
            dense_docs = dense_retriever.search(cleaned_query)
            for _, row in dense_docs.iterrows():
                if 'text' in row and row['text'] and len(row['text'].strip()) > 0:
                    context.append({
                        "id": row.get("docno", "unknown"),
                        "text": row.get("text", ""),
                        "score": float(row.get("score", 0.0)),
                        "metadata": {"source": "pinecone"}
                    })
        
        # Sort by score and take top_k_final
        context = sorted(context, key=lambda x: x["score"], reverse=True)[:top_k_final]
        
        # Generate answer
        result = generator.generate(query, context, qa_id, doc_id, question_type, user_type)
        
        # Add metadata
        result["documents"] = [doc["id"] for doc in context]
        result["total_pipeline_time"] = time.time() - start_time
        
        return result
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return {
            "answer": "An error occurred while processing this query.",
            "qa_id": qa_id,
            "doc_id": doc_id,
            "question_type": question_type,
            "user_type": user_type,
            "query": query,
            "error": str(e),
            "total_pipeline_time": time.time() - start_time
        }

def main():
    """Main execution function"""
    # Load environment variables
    load_dotenv()
    
    # Set keys from env
    AWS_EC2_ENDPOINT = os.getenv("AWS_EC2_ENDPOINT")
    AI_71_API_KEY = os.getenv("AI_71_API_KEY")
    
    args = parse_args(AWS_EC2_ENDPOINT=AWS_EC2_ENDPOINT, AI_71_API_KEY=AI_71_API_KEY)

    # Check falcon-mode and required parameters
    if args.falcon_mode == "ec2" and not args.ec2_endpoint:
        logger.error("EC2 endpoint URL must be provided when using --falcon-mode=ec2")
        return
    
    if args.falcon_mode == "ai71" and not args.ai71_api_key:
        logger.error("AI71 API key must be provided when using --falcon-mode=ai71")
        return
    
    # Critical check - fail fast if AWS was requested but not available
    if args.use_aws and not AWS_RETRIEVERS_AVAILABLE:
        logger.error("AWS retrievers were requested but could not be imported.")
        logger.error("Please ensure the retrieve/aws_retrievers.py file is compatible")
        return
    
    # Set up Hugging Face authentication if needed
    if args.hf_token:
        setup_huggingface_auth(args.hf_token)
    
    # Load data - determine if we're using competition format
    if args.competition_format:
        logger.info("==========Loading competition data==========")
        data = load_competition_data(args.input)
    else:
        data = load_synthetic_data(args.input)
        
    if not data:
        logger.error("No data loaded. Exiting.")
        return
    
    print(f"Loaded {len(data)} items from input file")
    if data:
        print(f"First item: {data[0]}")

    # Initialize appropriate retrievers
    use_aws = args.use_aws and AWS_RETRIEVERS_AVAILABLE
    
    if use_aws:
        logger.info("Using AWS retrievers (OpenSearch, Pinecone)")
        sparse_retriever = OpenSearchRetriever(top_k=args.top_k_sparse)
        dense_retriever = PineConeRetriever(top_k=args.top_k_dense)

    else:
        logger.info("Using local PyTerrier retriever")
        local_retriever, index = initialize_local_retriever(args.index_path)
        if not local_retriever:
            logger.error("Failed to initialize local retriever. Exiting.")
            return

    # After initializing retrievers:
    if use_aws:
        print("AWS retrievers initialized successfully")
        # Test retrieval
        test_query = "test"
        sparse_docs = sparse_retriever.search(test_query)
        print(f"Sparse retriever test: got {len(sparse_docs)} docs")
        dense_docs = dense_retriever.search(test_query)
        print(f"Dense retriever test: got {len(dense_docs)} docs")
    
    # Initialize generator with selected mode
    generator = EnhancedPromptGenerator(
        model_id=LLM_MODEL_ID,
        mode=args.falcon_mode,
        ec2_endpoint=args.ec2_endpoint,
        ai71_api_key=args.ai71_api_key,
        n_parallel=args.n_parallel,  # Default or use args.n_parallel if you added that argument
        n_retries=args.n_retries    # Default or use args.n_retries if you added that argument
    )
    

    results = []
    for i, item in enumerate(tqdm.tqdm(data, desc="Processing questions")):
        print("BEGINNING QUESTION PROCESSING")
        if args.competition_format:
            question = item.get("question", "")
            query_id = item.get("id", i)
        else:
            question = item.get("question", "")
            qa_id = item.get("qa_id", f"q{i}")
            doc_id = item.get("document_ids", "")
            question_type = item.get("question_categories", "")
            user_type = item.get("user_categories", "")
        
        # Process all questions
        print(f"About to process query: {question[:50]}...")
        if use_aws:
            print("Using process_query_aws...")
        else:
            print("Using process_query_local...")
            
        if not question:
            logger.warning(f"No question found in item {i}")
            continue
        
        logger.info(f"Processing question {i+1}/{len(data)}: {question[:50]}...")
        
        # Process query using appropriate function
        if use_aws:
            if args.competition_format:
                result = process_query_aws_for_competition(
                    query=question, 
                    sparse_retriever=sparse_retriever, 
                    dense_retriever=dense_retriever, 
                    generator=generator, 
                    top_k_sparse=args.top_k_sparse,
                    top_k_dense=args.top_k_dense,
                    top_k_final=args.top_k_final,
                    retriever_type=args.retriever_type,
                    query_id=query_id
                )
            else:
                result = process_query_aws(
                    query=question, 
                    sparse_retriever=sparse_retriever, 
                    dense_retriever=dense_retriever, 
                    generator=generator, 
                    top_k_sparse=args.top_k_sparse,
                    top_k_dense=args.top_k_dense,
                    top_k_final=args.top_k_final,
                    retriever_type=args.retriever_type,
                    qa_id=qa_id,
                    doc_id=doc_id,
                    question_type=question_type,
                    user_type=user_type
                )
        else:
            result = process_query_local(
                query=question,
                retriever=local_retriever,
                generator=generator,
                top_k=args.top_k_final,
                qa_id=qa_id if not args.competition_format else f"q{query_id}",
                doc_id=doc_id if not args.competition_format else [],
                question_type=question_type if not args.competition_format else [],
                user_type=user_type if not args.competition_format else []
            )
        
        # Add reference answer if available (for non-competition format)
        if "answer" in item and not args.competition_format:
            result["reference_answer"] = item["answer"]
        
        # Add item index
        result["item_index"] = i
        
        results.append(result)
        
        # Log progress every 10 items
        if (i + 1) % 10 == 0:
            logger.info(f"Completed {i+1}/{len(data)} questions")
    
    # Save results in standard format
    output_file = args.output or f"results_baseline_{int(time.time())}.json"
    save_results({"results": results}, output_file, RESULTS_DIR)
    
    # Handle competition format if requested
    if args.competition_format:
        logger.info("Formatting results for competition submission")
        formatted_results = format_competition_submission(results)
        
        # Validate formatted results
        is_valid = validate_competition_format(formatted_results)
        if not is_valid:
            logger.warning("Competition submission format validation failed")
        
        # Save competition submission
        competition_output = args.competition_output or f"competition_submission_{int(time.time())}.jsonl"
        save_competition_submission(formatted_results, competition_output)
    
    # Calculate statistics
    success_count = sum(1 for r in results if "error" not in r)
    avg_time = sum(r.get("total_pipeline_time", 0) for r in results if "total_pipeline_time" in r) / max(1, success_count)
    
    logger.info(f"Completed {success_count}/{len(results)} questions successfully")
    logger.info(f"Average processing time: {avg_time:.2f}s")
    logger.info(f"Results saved to {os.path.join(RESULTS_DIR, output_file)}")
    if args.competition_format:
        logger.info(f"Competition submission saved to {competition_output}")


if __name__ == "__main__":
    main()