from singletons.logger import get_logger
from singletons.embedding_model import embed_text
from singletons.db_conn import get_db_connection, release_db_connection
from singletons.session_manager import get_session_manager
from services.vectorstore import retrieve_docs_from_query
from langchain.prompts import PromptTemplate
import requests
import json

logger = get_logger()


# Define RAG prompt
CONVERSATIONAL_PROMPT_TEMPLATE = """
You are an AI assistant for Bytesme F&B app, which sells products such as cake, pastries, cookies, desserts, and drinks.
You are having a conversation with a customer. Use the conversation history and product information to provide helpful responses.

Conversation History:
{conversation_history}

User's questions context:
{context}

Current User question: {question}

Instructions:
- Start with a greeting, respond in a friendly, cute, teeny, playful tone
- Respond in the same language as the current user query
- Be conversational and refer to previous messages when relevant
- If the user is asking follow-up questions, build upon the previous context
- Provide concise, relevant responses about products
- If the user changes topic, acknowledge it and help with the new topic

Answer:
"""


# def generate_with_llm(
#         query,
#         llm,
#         on_rate_limit=None,
#         on_unauthorized=None,
#         on_generic_error=None
# ):
#     try:
#         conn = get_db_connection()
#         # Get documents and product details from vector search
#         retrieved_docs_with_distance, product_details_list = retrieve_docs_from_query(
#             conn, query, top_k=5)
#         # Stream document to front-end to simulate real thinking
#         content = ""
#         for idx, (doc, distance) in enumerate(retrieved_docs_with_distance):
#             content = doc.page_content[:200] + "..." if len(
#                 doc.page_content) > 200 else doc.page_content
#             content += "Đánh giá mức độ trùng khớp so với tìm kiếm người dùng: " + \
#                 str(round(1 - distance, 4)) + "\n\n"
#         yield f"data: {json.dumps({'type': 'thinking', 'chunk': content})}\n\n"
#         # Prepare context for LLM
#         if retrieved_docs_with_distance:
#             context_parts = [doc.page_content for doc,
#                              dist in retrieved_docs_with_distance]
#             context_str = "\n\n---\n\n".join(context_parts)
#             logger.info(
#                 f"Context for LLM (first 300 chars): {context_str[:300]}...")
#         else:
#             context_str = "No specific product information was found related to your query."
#             logger.info(
#                 "No documents found by vector search to form context.")

#         # Format prompt
#         prompt = PromptTemplate(template=CONVERSATIONAL_PROMPT_TEMPLATE,
#                                 input_variables=["context", "question"])
#         full_prompt_text = prompt.format(
#             context=context_str, question=query)
#         logger.info(
#             f"Full prompt for LLM (first 300 chars): {full_prompt_text[:300]}...")
#         # Stream AI response
#         llmOutput = ''
#         for chunk in llm.stream(full_prompt_text):
#             if isinstance(chunk, str):
#                 text = chunk
#                 llmOutput += text
#             elif hasattr(chunk, "content"):
#                 text = chunk.content
#                 llmOutput += text
#             else:
#                 text = str(chunk)
#                 llmOutput += text
#             yield f"data: {json.dumps({'type': 'answer', 'chunk': text})}\n\n"
#         # Stream product details
#         for product in product_details_list:
#             yield f"data: {json.dumps({'type': 'product', 'data': product})}\n\n"
#         # Signal end of stream
#         yield "data: [DONE]\n\n"
#         logger.info("All streaming finished successfully")
#         logger.info("LLM streaming finished successfully")
#         logger.info(f"Full LLM output: {llmOutput}...")
#     except requests.exceptions.HTTPError as e:
#         status_code = e.response.status_code
#         logger.error(
#             f"HTTP error during streaming query: {e}, Status Code: {status_code}", exc_info=True)
#         if status_code == 429:
#             if on_rate_limit:
#                 on_rate_limit(e)
#         elif status_code == 401:
#             if on_unauthorized:
#                 on_unauthorized(e)
#         elif status_code == 500:
#             if on_generic_error:
#                 on_generic_error(e)
#         # yield f"data: {json.dumps({'type': 'answer', 'chunk': error_message, 'error': True})}\n\n"
#         yield "data: [DONE]\n\n"

#     finally:
#         if conn:
#             release_db_connection(conn)


def generate_with_llm(
        query,
        llm,
        session_id: str,
        on_rate_limit=None,
        on_unauthorized=None,
        on_generic_error=None
):
    session_manager = get_session_manager()
    session = session_manager.get_or_create_session(session_id)

    try:
        conn = get_db_connection()

        # Add user message to conversation history
        session.add_message("user", query)

        # Get conversation history
        conversation_history = session.get_conversation_history()

        # Enhanced query for vector search - combine current query with recent context
        enhanced_query = query
        if len(session.messages) > 1:
            # If there's conversation history, enhance the search query
            recent_queries_context = " ".join(
                [msg.content for msg in session.messages[-1:] if msg.role == "user"])
            enhanced_query = f"{recent_queries_context} {query}"
            logger.info(
                f"Enhanced query for vector search: {enhanced_query}...")

        # Get documents and product details from vector search
        retrieved_docs_with_distance, product_details_list = retrieve_docs_from_query(
            conn, enhanced_query, top_k=4)

        # Update session context
        session.update_context_documents(retrieved_docs_with_distance)

        # Stream thinking content
        content = ""
        for idx, (doc, distance) in enumerate(retrieved_docs_with_distance):
            content = doc.page_content[:200] + "..." if len(
                doc.page_content) > 200 else doc.page_content
            content += f"Đánh giá mức độ trùng khớp: {round(1 - distance, 4)}\n\n"
        yield f"data: {json.dumps({'type': 'thinking', 'chunk': content})}\n\n"

        # Prepare context for LLM
        if retrieved_docs_with_distance:
            context_parts = [doc.page_content for doc,
                             dist in retrieved_docs_with_distance]
            context_str = "\n\n".join(context_parts)
        else:
            context_str = "No specific product information found."

        # Format prompt with conversation history
        prompt = PromptTemplate(
            template=CONVERSATIONAL_PROMPT_TEMPLATE,
            input_variables=["conversation_history", "context", "question"]
        )
        full_prompt_text = prompt.format(
            conversation_history=conversation_history,
            context=context_str,
            question=query,
        )
        print(
            f"Debug: session_id: {session_id}, convo history: {conversation_history}")

        logger.info(
            f"Conversational prompt (first 300 chars): {full_prompt_text[:300]}...")

        # Stream AI response
        llm_output = ''
        for chunk in llm.stream(full_prompt_text):
            if isinstance(chunk, str):
                text = chunk
            elif hasattr(chunk, "content"):
                text = chunk.content
            else:
                text = str(chunk)

            llm_output += text
            yield f"data: {json.dumps({'type': 'answer', 'chunk': text})}\n\n"

        # Add assistant response to conversation history
        session.add_message("assistant", llm_output)

        # Stream product details
        for product in product_details_list:
            yield f"data: {json.dumps({'type': 'product', 'data': product})}\n\n"

        # Send session ID before end of stream
        yield f"data: {json.dumps({'type': 'session_id', 'session_id': session_id})}\n\n"

        # Signal end of stream
        yield "data: [DONE]\n\n"
        logger.info(
            f"Conversational response completed for session: {session_id}")

    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code
        logger.error(
            f"HTTP error during streaming: {e}, Status: {status_code}")

        if status_code == 429 and on_rate_limit:
            on_rate_limit(e)
        elif status_code == 401 and on_unauthorized:
            on_unauthorized(e)
        elif on_generic_error:
            on_generic_error(e)

        yield "data: [DONE]\n\n"

    except Exception as e:
        logger.error(f"Error in conversational LLM: {e}", exc_info=True)
        yield f"data: {json.dumps({'type': 'error', 'message': 'An error occurred'})}\n\n"
        yield "data: [DONE]\n\n"

    finally:
        if conn:
            release_db_connection(conn)
