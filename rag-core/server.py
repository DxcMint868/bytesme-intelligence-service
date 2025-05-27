from flask import Flask, Response, app, request
from httpx import HTTPError
from langchain.prompts import PromptTemplate
# from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document
# from langchain_openrouter.openrouter import OpenRouterLLM
from StreamableOpenRouterLLM import StreamableOpenRouterLLM
import time
import json
import psycopg2
from psycopg2.pool import SimpleConnectionPool
import sseclient
import requests
import os
import logging
from dotenv import load_dotenv
import random
import threading
import queue

load_dotenv(override=True)
app = Flask(__name__)

# Configs
VIETNAMESE_MODEL = "VoVanPhuc/sup-SimCSE-VietNamese-phobert-base"
EMBEDDING_DIMENSIONS = 768
GENERATIVE_MODEL_TEMPERATURE = 0.75
SCHEMA_NAME = "public"
COLLECTION_NAME = "product_pg_embeddings"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
_model = None


# Init db connection pool
db_pool = SimpleConnectionPool(
    minconn=1,
    maxconn=10,
    host=os.getenv("POSTGRES_HOST"),
    port=os.getenv("POSTGRES_PORT"),
    user=os.getenv("POSTGRES_USERNAME"),
    password=os.getenv("POSTGRES_PASSWORD"),
    dbname=os.getenv("POSTGRES_DATABASE"),
)


def get_db_connection(retries=2):
    last_exception = None
    for attempt in range(retries + 1):
        conn = None
        try:
            conn = db_pool.getconn()
            # Perform a quick check to see if the connection is alive
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
            # If the check passes, return the connection
            # Rollback any implicit transaction started by SELECT 1 if not in autocommit
            if not conn.autocommit:
                conn.rollback()
            logger.info(
                f"Successfully obtained and validated DB connection (attempt {attempt + 1}).")
            return conn
        except (psycopg2.OperationalError, psycopg2.InterfaceError) as e:
            last_exception = e
            logger.warning(
                f"Attempt {attempt + 1} to get/validate DB connection failed: {e}")
            if conn:
                # If we got a connection but it's bad, put it back to be closed by the pool
                db_pool.putconn(conn, close=True)
            if attempt < retries:
                logger.info("Retrying to get a new DB connection...")
                time.sleep(0.5)  # Small delay before retrying
            else:
                logger.error("Max retries reached for getting DB connection.")
                # Fall through to raise the last_exception outside the loop
        except Exception as e:  # Catch other potential errors from getconn() or the check
            last_exception = e
            logger.error(
                f"Attempt {attempt + 1}: Unexpected error during DB connection retrieval/validation: {e}", exc_info=True)
            if conn:
                # Ensure problematic connection is closed
                db_pool.putconn(conn, close=True)
            # Fall through to raise the last_exception if this is the last attempt

    # If all retries failed
    if last_exception:
        raise last_exception
    else:
        # This case should ideally not be reached if retries > 0 and an error always occurs
        raise psycopg2.OperationalError(
            "Failed to obtain a valid database connection after multiple attempts.")


def release_db_connection(conn):
    db_pool.putconn(conn)


def get_embedding_model():
    """
    Get or create a cached SentenceTransformer model optimized for Vietnamese.
    This prevents reloading the model multiple times during execution.
    """
    global _model
    if _model is None:
        try:
            print(
                f"Load pretrained SentenceTransformer: {VIETNAMESE_MODEL}")
            _model = SentenceTransformer(VIETNAMESE_MODEL, device="cpu")
            # fallback_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            # _model = SentenceTransformer(fallback_model, device="cpu")
            print(f"Successfully loaded model: {VIETNAMESE_MODEL}")
        except Exception as e:
            fallback_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            print(
                f"Failed to load {VIETNAMESE_MODEL}, falling back to {fallback_model}: {e}")
            _model = SentenceTransformer(fallback_model, device="cpu")
    return _model


# Define OpenRouter models
OPENROUTER_MODELS = [
    'deepseek/deepseek-r1:free',
    # 'qwen/qwq-32b:free'
    # 'qwen/qwen3-30b-a3b:free',
    'meta-llama/llama-4-maverick:free',
    'meta-llama/llama-4-scout:free',
    'cognitivecomputations/dolphin3.0-r1-mistral-24b:free'
]


# Return the generative model
def get_openrouter_model(selection_strategy="random", excepted_models=None):
    if selection_strategy == "random":
        available_models = [
            model for model in OPENROUTER_MODELS if not excepted_models or model not in excepted_models]
        if not available_models:
            raise ValueError(
                "No available OpenRouter models after filtering out excepted models")
        return random.choice(available_models)

    # elif selection_strategy == "round_robin":


# Initialize Langchain llm via OpenRouter provider
combined_api_key = os.getenv("OPENROUTER_API_KEY")
if not combined_api_key:
    raise ValueError("OPENROUTER_API_KEY environment variable is not set")

OPENROUTER_API_KEYS = combined_api_key.split(',')


def get_openrouter_api_key(excepted_keys=None):
    if excepted_keys is None:
        excepted_keys = []
    available_keys = [
        key for key in OPENROUTER_API_KEYS if key not in excepted_keys]
    if not available_keys:
        raise ValueError(
            "No available OpenRouter API keys after filtering out excepted keys")
    return random.choice(available_keys)


llm = StreamableOpenRouterLLM(api_key=get_openrouter_api_key(),
                              temperature=GENERATIVE_MODEL_TEMPERATURE,
                              model=get_openrouter_model(),
                              stream=True,
                              )
logger.info("Initialized OpenRouter LLM with model: "
            f"{llm.model} and temperature: {llm.temperature}")

# Define RAG prompt
PROMPT_TEMPLATE = """
You are an AI assistant for an F&B app. Use the following product information to answer the user's query. Provide concise, relevant responses.
Context: {context}
Query: {question}
Answer:
"""
prompt = PromptTemplate(template=PROMPT_TEMPLATE,
                        input_variables=["context", "question"])


def generate_with_llm(
        query,
        llm,
        conn,
        on_rate_limit=None,
        on_unauthorized=None,
        on_generic_error=None
):
    try:
        logger.info(f"Streaming query received: {query}")
        try:
            # Get documents and product details from vector search
            retrieved_docs_with_distance, product_details_list = retrieve_docs(
                conn, query, top_k=5)

            # Stream document to front-end to simulate real thinking
            content = ""
            for idx, (doc, distance) in enumerate(retrieved_docs_with_distance):
                content = doc.page_content[:200] + "..." if len(
                    doc.page_content) > 200 else doc.page_content
                content += "Đánh giá mức độ trùng khớp so với tìm kiếm người dùng: " + \
                    str(round(1 - distance, 4)) + "\n\n"
            yield f"data: {json.dumps({'type': 'thinking', 'chunk': content})}\n\n"

            # Prepare context for LLM
            if retrieved_docs_with_distance:
                context_parts = [doc.page_content for doc,
                                 dist in retrieved_docs_with_distance]
                context_str = "\n\n---\n\n".join(context_parts)
                logger.info(
                    f"Context for LLM (first 300 chars): {context_str[:300]}...")
            else:
                context_str = "No specific product information was found related to your query."
                logger.info(
                    "No documents found by vector search to form context.")

            # Format prompt
            full_prompt_text = prompt.format(
                context=context_str, question=query)
            logger.info(
                f"Full prompt for LLM (first 300 chars): {full_prompt_text[:300]}...")

            # Stream AI response
            llmOutput = ''
            for chunk in llm.stream(full_prompt_text):
                if isinstance(chunk, str):
                    text = chunk
                    llmOutput += text
                elif hasattr(chunk, "content"):
                    text = chunk.content
                    llmOutput += text
                else:
                    text = str(chunk)
                    llmOutput += text
                yield f"data: {json.dumps({'type': 'answer', 'chunk': text})}\n\n"

            # Stream product details
            for product in product_details_list:
                yield f"data: {json.dumps({'type': 'product', 'data': product})}\n\n"

            # Signal end of stream
            yield "data: [DONE]\n\n"
            logger.info("All streaming finished successfully")

            logger.info("LLM streaming finished successfully")
            logger.info(f"Full LLM output: {llmOutput}...")

        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code
            logger.error(
                f"HTTP error during streaming query: {e}, Status Code: {status_code}", exc_info=True)

            if status_code == 429:
                error_message = "Rate limit exceeded. Please try again later."
                if on_rate_limit:
                    on_rate_limit(e)
            elif status_code == 401:
                if on_unauthorized:
                    on_unauthorized(e)
            elif status_code == 500:
                if on_generic_error:
                    on_generic_error(e)

            yield f"data: {json.dumps({'type': 'answer', 'chunk': error_message, 'error': True})}\n\n"
            yield "data: [DONE]\n\n"

    except Exception as e:
        logger.error(
            f"Error in main thread during streaming: {e}", exc_info=True)
        yield f"data: {json.dumps({'type': 'error', 'source': 'main', 'message': str(e), 'critical': True})}\n\n"
        yield "data: [DONE]\n\n"
        if on_generic_error:
            on_generic_error(e)
    finally:
        if conn:
            release_db_connection(conn)


@app.route('/product/search/rag', methods=['GET'])
def stream_query_route():
    query = request.args.get('query')
    if not query:
        return Response(json.dumps({"message": "Query parameter is missing"}), status=400, mimetype='application/json')

    # Prepared injection dependencies
    conn = get_db_connection()

    def on_rate_limit(e):
        global llm

        logger.warning(f"Rate limit exceeded: {e}")
        fallback_llm = StreamableOpenRouterLLM(
            api_key=get_openrouter_api_key(excepted_keys=[llm.api_key]),
            temperature=GENERATIVE_MODEL_TEMPERATURE,
            model=llm.model,
            stream=True,
        )
        llm = fallback_llm
        logger.info(f"Switched to fallback model: {llm.model}")
        # Try again with fallback model
        generate_with_llm(query, llm, conn)

    def on_unauthorized(e):
        logger.error(f"Unauthorized access: {e}")
        return Response(json.dumps({"message": "Unauthorized access"}), status=401, mimetype='application/json')

    def on_generic_error(e):
        logger.error(f"Generic error occurred: {e}")
        return Response(json.dumps({"message": "An error occurred"}), status=500, mimetype='application/json')

    return Response(
        generate_with_llm(
            query,
            llm,
            conn,
            on_rate_limit=on_rate_limit,
            on_unauthorized=on_unauthorized,
            on_generic_error=on_generic_error
        ),
        mimetype='text/event-stream')


@app.route('/product/search/semantics', methods=['GET'])
def test_search_route():
    query = request.args.get('query', 'Caffeine-related products')
    logger.info(f'Query: {query}')
    top_k = int(request.args.get('top_k', 3))
    conn = get_db_connection()

    _, product_details_list = retrieve_docs(
        conn, query, top_k)

    # Return the product_details_list, which is a list of dictionaries and JSON serializable
    return Response(json.dumps(product_details_list), mimetype='application/json')


def embed_text(text):
    """Wrapper function to safely create embeddings"""
    try:
        model = get_embedding_model()
        embeddings = model.encode(text, normalize_embeddings=True)
        logger.info(
            f"Generated embeddings for text: {text[:50]}... with shape {embeddings.shape}")
        # Log first 5 values for debugging
        logger.info(f"Embedding: {embeddings[:5]}...")
        return embeddings.tolist()
    except Exception as e:
        print(f"Error embedding text: {e}")
        # Return zeros array as fallback (with proper dimension)
        return [0.0] * EMBEDDING_DIMENSIONS


def retrieve_docs(conn, query, top_k=3):
    """Retrieve documents and product details using vector distance."""
    try:
        query_embedding = embed_text(query)
        logger.info(
            f"Query: '{query}', Embedding (first 5 dims): {query_embedding[:5]}")

        if all(v == 0.0 for v in query_embedding):
            logger.warning(
                "Query embedding failed (all zeros), search results will be inaccurate")

        cursor = conn.cursor()
        cursor.execute(f"SET search_path TO {SCHEMA_NAME}, public;")

        sql_query = """
        SELECT
            e.document,
            e.cmetadata,
            e.embedding <-> %s::vector AS distance,
            p.product_id,
            p.product_total_ratings,
            p.product_overall_stars,
            p.product_total_orders,
			p.product_discount_percentage,
			p.product_unit_price,
            pi.product_image_url
        FROM
            langchain_pg_collection c
        JOIN
            langchain_pg_embedding e ON c.id = e.collection_id
        LEFT JOIN
            products p ON p.product_code = e.cmetadata->>'product_code'
        LEFT JOIN LATERAL
            (SELECT product_image_url
            FROM product_images
            WHERE product_images.product_id = p.product_id
            LIMIT 1) pi ON TRUE
        WHERE
			c.name = %s
            AND (p.product_stock_quantity > 0 OR p.product_stock_quantity IS NULL)
        ORDER BY
            distance ASC
        LIMIT %s
        """
        logger.info(
            f"Executing vector search with COLLECTION_NAME: {COLLECTION_NAME}, top_k: {top_k}")
        cursor.execute(sql_query, (query_embedding, COLLECTION_NAME, top_k))

        fetched_rows = cursor.fetchall()
        logger.info(
            f"Number of rows returned by vector search: {len(fetched_rows)}")

        documents_with_distance = []
        product_details_list = []

        for row_idx, row_data in enumerate(fetched_rows):
            document_text = row_data[0]
            metadata_json = row_data[1]
            distance = row_data[2]

            logger.info(
                f"Row {row_idx} - Document text (first 100 chars): {document_text[:100]}...")
            logger.info(f"Row {row_idx} - Raw Metadata JSON: {metadata_json}")
            logger.info(f"Row {row_idx} - Distance: {distance}")

            # Parse metadata
            if isinstance(metadata_json, dict):
                metadata = metadata_json
            elif metadata_json:
                try:
                    metadata = json.loads(metadata_json)
                except json.JSONDecodeError:
                    logger.error(
                        f"Row {row_idx} - Failed to parse metadata JSON: {metadata_json}")
                    metadata = {}
            else:
                metadata = {}

            logger.info(f"Row {row_idx} - Parsed Metadata: {metadata}")

            # Create document for LLM
            document = Document(page_content=document_text, metadata=metadata)
            documents_with_distance.append((document, float(distance)))

            # Extract product details for UI
            product_details = {
                "product_code": metadata.get("product_code", "Unknown"),
                "product_name": metadata.get("product_name", "Unknown"),
                "category_name": metadata.get("category_name", "Unknown"),
                "description": document_text,  # Use document text as description
                "product_id": row_data[3],
                "total_ratings": row_data[4],
                "overall_stars": row_data[5],
                "total_orders": row_data[6],
                "discount_percentage": row_data[7],
                # JSON string of sizes and prices, e.g. {"product_sizes": "S|M|L", "product_prices": "89000|94000|99000"}
                "sizes_prices": row_data[8],
                "image_url": row_data[9],
            }
            product_details_list.append(product_details)

        cursor.close()

        return documents_with_distance, product_details_list

    except Exception as e:
        logger.error(f"Error in retrieve_docs: {e}", exc_info=True)
        # if conn:
        # conn.rollback()
        return [], []


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
