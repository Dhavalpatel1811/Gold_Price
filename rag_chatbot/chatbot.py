# -*- coding: utf-8 -*-
"""
Investment advisor chatbot: routes each user message to one of four modes
(chat, investment advice, prediction lookup, book Q&A) and answers using
a local LLaMA3 model via Ollama, grounded with RAG context where relevant.

Requires Ollama running locally with the llama3 model pulled:
    ollama pull llama3
"""
from ollama import chat

from rag_engine import retrieve_context, setup_rag

print("Loading RAG indexes (book + prediction)...")
model_emb, book_index, book_chunks, pred_index, pred_chunks = setup_rag()
print("RAG system ready.\n")

SIMPLE_CHAT = [
    "hi", "hii", "hello", "hey", "hyy", "ok", "okay", "thanks", "thank you",
    "bye", "goodbye", "how are you", "what's up", "wassup",
]
INVESTMENT_ADVICE = [
    "should i invest", "what do you suggest", "recommendation",
    "what should i do", "invest or not", "buy or sell", "advice",
]
PREDICTION_QUERY = [
    "prediction", "forecast", "price tomorrow", "future price",
    "what will be", "trend", "next",
]


def detect_intent(user_input):
    """Classify the user's message into chat / investment / prediction / book_qa."""
    q = user_input.lower().strip()

    if any(word in q for word in SIMPLE_CHAT) and len(q.split()) <= 5:
        return "chat"
    if any(phrase in q for phrase in INVESTMENT_ADVICE):
        return "investment"
    if any(word in q for word in PREDICTION_QUERY):
        return "prediction"
    return "book_qa"


def _ask_llama(prompt):
    response = chat(model="llama3:latest", messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]


def get_response(user_input):
    """Route a user message to the right handler and return the chatbot's reply."""
    intent = detect_intent(user_input)

    if intent == "chat":
        return _ask_llama(f"Respond briefly and warmly to: {user_input}")

    if intent == "investment":
        context = retrieve_context(
            user_input, model_emb, book_index, book_chunks, pred_index, pred_chunks, k=2
        )
        prompt = f"""You're a financial advisor. Give a SHORT recommendation based on:

{context}

Question: {user_input}

Answer format:
Recommendation: [INVEST/AVOID/HOLD]
Reason: [2-3 sentences combining prediction trend + book principle]

Be concise and direct."""
        return _ask_llama(prompt)

    if intent == "prediction":
        context = retrieve_context(user_input, model_emb, pred_index=pred_index, pred_chunks=pred_chunks, k=1)
        prompt = f"Answer briefly using this prediction data:\n\n{context}\n\nQuestion: {user_input}"
        return _ask_llama(prompt)

    # book_qa
    context = retrieve_context(user_input, model_emb, book_index=book_index, book_chunks=book_chunks, k=2)
    prompt = f"Answer concisely using this book context:\n\n{context}\n\nQuestion: {user_input}"
    return _ask_llama(prompt)
