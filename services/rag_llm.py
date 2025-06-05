from singletons.logger import get_logger
from singletons.db_conn import get_db_connection, release_db_connection
from singletons.session_manager import get_session_manager
from services.vectorstore import retrieve_docs_from_query
from langchain.prompts import PromptTemplate
import requests
import json
from singletons.query_classifier import get_query_classifier

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


PRODUCT_FOCUSED_PROMPT = """
You are an AI assistant for Bytesme F&B app, which sells products such as cake, pastries, cookies, desserts, and drinks.
You are having a conversation with a customer about our products.

Conversation History:
{conversation_history}

User's questions context:
{context}

Current User Question: {question}

Instructions:
- Respond in the same language as the current user query,
- Respond in a fiendly, cute, teeny, heart-warming, playful tone
- Focus on helping the customer with product information
- Be friendly and helpful about our food and beverage offerings
- Provide specific details about products when available

Answer:
"""

GENERAL_CONVERSATION_PROMPT = """
You are an AI assistant for Bytesme F&B app. You are having a friendly conversation with a customer.

Conversation History:
{conversation_history}

Current User Question: {question}

Instructions:
- Respond in the same language as the current user query,
- Respond in a fiendly, cute, teeny, heart-warming, playful tone
- Be friendly, helpful, and conversational
- You can discuss general topics, store information, policies, etc.
- If the customer asks about products, you can mention that you'd be happy to help them find specific items

Answer:
"""


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
    query_classifier = get_query_classifier()

    try:
        conn = get_db_connection()

        # Add user message to conversation history
        session.add_message("user", query)
        conversation_history = session.get_conversation_history()

        # STEP 1: Classify the query
        classification = query_classifier.classify_query(query)

        # Send classification info to client
        yield f"data: {json.dumps({'type': 'thinking', 'chunk': classification})}\n\n"

        if classification['is_product_related']:
            # STEP 2A: Product-related query - perform RAG
            logger.info(
                f"Product-related query detected: {classification['reasoning']}")

            # Enhanced query for vector search
            enhanced_query = query
            if len(session.messages) > 1:
                recent_queries_context = " ".join(
                    [msg.content for msg in session.messages[:-1] if msg.role == "user"])
                enhanced_query = f"{recent_queries_context} {query}"
                logger.info(
                    f"Enhanced query for vector search: {enhanced_query}...")

            # Get documents and product details from vector search
            retrieved_docs_with_distance, product_details_list = retrieve_docs_from_query(
                conn, enhanced_query, top_k=4)

            # Update session context
            session.update_context_documents(retrieved_docs_with_distance)

            # Stream thinking content
            thinking_content = f"🔍 Đang tìm kiếm sản phẩm liên quan...\n"
            thinking_content += f"📊 Độ tin cậy phân loại: {classification['confidence']:.2f}\n\n"

            for idx, (doc, distance) in enumerate(retrieved_docs_with_distance):
                content = doc.page_content[:200] + "..." if len(
                    doc.page_content) > 200 else doc.page_content
                thinking_content += f"📄 Tài liệu {idx + 1}: {content}\n"
                thinking_content += f"🎯 Độ trùng khớp: {round(1 - distance, 4)}\n\n"

            yield f"data: {json.dumps({'type': 'thinking', 'chunk': thinking_content})}\n\n"

            # Prepare context for LLM
            if retrieved_docs_with_distance:
                context_parts = [doc.page_content for doc,
                                 dist in retrieved_docs_with_distance]
                context_str = "\n\n".join(context_parts)
            else:
                context_str = "Không tìm thấy thông tin sản phẩm cụ thể."

            # Use product-focused prompt
            prompt = PromptTemplate(
                template=PRODUCT_FOCUSED_PROMPT,
                input_variables=["conversation_history", "context", "question"]
            )
            full_prompt_text = prompt.format(
                conversation_history=conversation_history,
                context=context_str,
                question=query
            )

        else:
            # STEP 2B: General conversation - skip RAG
            logger.info(
                f"General conversation detected: {classification['reasoning']}")

            # Stream thinking content for general conversation
            thinking_content = f"💬 Câu hỏi chung - không cần tìm kiếm sản phẩm\n"
            thinking_content += f"📊 Độ tin cậy phân loại: {classification['confidence']:.2f}\n"
            thinking_content += f"🤖 Sử dụng AI thuần túy để trả lời\n\n"

            yield f"data: {json.dumps({'type': 'thinking', 'chunk': thinking_content})}\n\n"

            # Use general conversation prompt (no product context)
            prompt = PromptTemplate(
                template=GENERAL_CONVERSATION_PROMPT,
                input_variables=["conversation_history", "question"]
            )
            full_prompt_text = prompt.format(
                conversation_history=conversation_history,
                question=query
            )

            # No product details for general conversation
            product_details_list = []

        logger.info(
            f"Prompt type: {'Product-focused' if classification['is_product_related'] else 'General conversation'}")
        logger.info(
            f"Full prompt (first 300 chars): {full_prompt_text[:300]}...")

        # STEP 3: Stream LLM response
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

        # STEP 4: Stream product details (only for product-related queries)
        if classification['is_product_related'] and product_details_list:
            for product in product_details_list:
                yield f"data: {json.dumps({'type': 'product', 'data': product})}\n\n"

        # Send session ID and classification summary
        yield f"data: {json.dumps({'type': 'session_id', 'session_id': session_id, 'query_type': classification['category']})}\n\n"

        # Signal end of stream
        yield "data: [DONE]\n\n"
        logger.info(
            f"Response completed for session: {session_id} (type: {classification['category']})")

    except Exception as e:
        logger.error(f"Error in conversational LLM: {e}", exc_info=True)
        yield f"data: {json.dumps({'type': 'error', 'message': 'An error occurred'})}\n\n"
        yield "data: [DONE]\n\n"

    finally:
        if conn:
            release_db_connection(conn)
