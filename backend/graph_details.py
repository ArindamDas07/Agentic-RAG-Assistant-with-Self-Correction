from typing import List, TypedDict, Literal
from pydantic import BaseModel, Field
from langchain_classic.retrievers import MultiQueryRetriever, ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor

# Document Processing
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Ollama Integrations
# from langchain_ollama import OllamaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq.chat_models import ChatGroq

# Core Components
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
import os
from dotenv import load_dotenv
from pathlib import Path
from backend.pdf_utils import embedding_model
load_dotenv()  # Load environment variables from .env

groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize Local LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
)

def get_retriever(user_id: str):
    """
    Load a per-user Chroma vector database and return a contextual compression retriever for RAG queries.

    This function performs the following steps:
    1. Checks if a Chroma vector database exists for the given `user_id`. Raises an error if not found.
    2. Loads the user's vector store from `chroma_db/{user_id}` using the predefined embedding model.
    3. Creates a Maximal Marginal Relevance (MMR) retriever to fetch diverse relevant documents.
    4. Wraps the MMR retriever in a `MultiQueryRetriever` to allow multiple LLM-driven query reformulations.
    5. Initializes an `LLMChainExtractor` as a document compressor.
    6. Returns a `ContextualCompressionRetriever` that combines multi-query retrieval with LLM-based compression.
    """
    
    vector_dir = Path("chroma_db") / user_id
    if not vector_dir.exists() or not any(vector_dir.iterdir()):
        raise ValueError(f"No vector database found for user {user_id}. Please process the PDFs first.")

    vector_store = Chroma(
        collection_name="chroma_collection",
        embedding_function=embedding_model,
        persist_directory=str(vector_dir)
    )

    mmr_retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "lambda_mult": 0.5}
    )

    multiquery_retriever = MultiQueryRetriever.from_llm(
        retriever=mmr_retriever,
        llm=llm
    )

    compressor = LLMChainExtractor.from_llm(llm)

    compression_retriever = ContextualCompressionRetriever(
        base_retriever=multiquery_retriever,
        base_compressor=compressor
    )

    return compression_retriever

class State(TypedDict):
    question: str
    docs: List[Document]
    user_id: str

    answer: str

    # 🔍 Hallucination check
    verdict: Literal["Supported", "Not Supported","Not_Sure"]
    reason: str

    # 🔁 Control
    regenerated: bool  # to avoid multiple loops

    # 🧾 Transparency (user-facing)
    final_status: Literal["clean", "corrected","failed","not_in_docs"]

def initialize_state(state):
    """
    Initialize a default state dictionary for managing RAG question-answering sessions.

    This function sets up the required keys to track the lifecycle of a question,
    the retrieved documents, and the model's response. It ensures consistent structure
    for downstream processing, including hallucination checking and regeneration logic.

    Args:
        state (dict): Placeholder for an initial state (not used in this implementation, included for API consistency).

    Returns:
        dict: A dictionary with default keys:
            - "regenerated" (bool): Indicates if the answer was regenerated; used for control flow.
            - "docs" (list): List of retrieved Document objects.
            - "answer" (str): Generated answer text, initially empty.
            - "verdict" (str): Hallucination check result; defaults to "Not_Sure".
            - "reason" (str): Explanation of hallucination verdict; initially empty.
            - "final_status" (str): Overall status of the response; defaults to "clean".
    """
    return {
        "regenerated": False,   # 🔁 critical for control flow
        "docs": [],             # avoid missing key
        "answer": "",           # safe default
        "verdict": "Not_Sure", # temporary placeholder
        "reason": "",           
        "final_status": "clean" # temporary (will be overwritten)
    }

def retrieve(state):
    """
    Retrieve relevant documents from the vector database for a user's query.

    This function extracts the question and user ID from the provided state,
    loads the user's Chroma vector store retriever, and returns the documents
    most relevant to the question using contextual compression.

    Args:
        state (dict): A dictionary containing the session state, expected to have:
            - "question" (str): The user's input question.
            - "user_id" (str): The unique identifier for the user's session/data.

    Returns:
        dict: A dictionary containing:
            - "docs" (list): List of retrieved and compressed Document objects 
              relevant to the user's question.
    """
    q = state['question']
    user_id = state['user_id']
    retriever = get_retriever(user_id)
    return{"docs":retriever.invoke(q)}

class GenerateAnswer(BaseModel):
    final_status: Literal["clean", "not_in_docs"]
    answer: str

generate_answer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strict RAG assistant.\n\n"

            "Your job:\n"
            "1. Answer ONLY using the provided context.\n"
            "2. If the answer is clearly present → final_status = clean\n"
            "3. If the answer is NOT present → final_status = not_in_docs AND answer = 'I don't know'\n\n"

            "STRICT RULES:\n"
            "- Use ONLY the context\n"
            "- Do NOT use prior knowledge\n"
            "- Do NOT guess or infer\n"
            "- Do NOT partially answer\n"
            "- If unsure → return 'I don't know'\n"
            "- Output MUST be valid JSON\n\n"

            "Output format:\n"
            "{{\n"
            '  "final_status": "clean" or "not_in_docs",\n'
            '  "answer": "string"\n'
            "}}"
        ),
        (
            "human",
            "Question:\n{question}\n\n"
            "Context:\n{context}"
        )
    ]
)
generate_answer_chain = generate_answer_prompt | llm.with_structured_output(GenerateAnswer)

def generate(state):
    """
    Generate an answer to a user's question based on retrieved documents.

    This function takes the session state containing the user's question and
    the relevant documents, constructs a context string from the documents,
    and passes it to the answer generation chain. It returns the generated
    answer along with the final status indicating the answer quality or
    processing result.

    Args:
        state (dict): A dictionary containing:
            - "question" (str): The user's question.
            - "docs" (list): A list of Document objects retrieved for the question.

    Returns:
        dict: A dictionary containing:
            - "answer" (str): The generated answer based on the provided context.
            - "final_status" (str): Status of the answer, e.g., "clean", 
             or "not_in_docs".
    """
    context = "\n\n".join(d.page_content for d in state['docs'])

    out = generate_answer_chain.invoke({
        "question": state["question"],
        "context": context
    })

    return {
        "answer": out.answer,
        "final_status": out.final_status
    }

class Checker(BaseModel):
    verdict: Literal["Supported", "Not Supported"]
    reason:str
    
checker_llm = ChatGroq(
   
    model="llama-3.3-70b-versatile",
    temperature=0.0   # MUST
)
check_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strict factual evaluator for a RAG system.\n\n"
            "Your task is to verify whether the answer is supported by the provided context.\n\n"
            "STRICT RULES:\n"
            "- Use ONLY the given context.\n"
            "- Do NOT use prior knowledge.\n"
            "- If ANY part of the answer is not found in the context → Not Supported.\n"
            "- If the answer adds extra assumptions → Not Supported.\n"
            "- Be conservative.\n\n"
            "Output JSON only:\n"
            "{{\n"
            '  "verdict": "Supported" or "Not Supported",\n'
            '  "reason": "short explanation"\n'
            "}}"
        ),
        (
            "human",
            "Question:\n{question}\n\n"
            "Context:\n{context}\n\n"
            "Answer:\n{answer}"
        )
    ]
)
check_answer_quality_chain = check_prompt | checker_llm.with_structured_output(Checker)

def check_answer_quality(state):
    """
    Evaluate the quality and support of a generated answer against the retrieved documents.

    This function takes the session state containing the user's question, the
    retrieved documents, and the generated answer. It uses the `check_answer_quality_chain`
    to determine whether the answer is supported by the documents and provides a
    reason for the verdict. It also assigns a `final_status` based on whether the
    answer was regenerated and the verdict outcome.

    Args:
        state (dict): A dictionary containing:
            - "question" (str): The user's original question.
            - "answer" (str): The generated answer to evaluate.
            - "docs" (list): A list of Document objects retrieved for the question.
            - "regenerated" (bool): Indicates if the answer has been regenerated after an initial attempt.

    Returns:
        dict: A dictionary containing:
            - "verdict" (str): One of "Supported" or "Not Supported" indicating answer quality.
            - "reason" (str): Explanation for the verdict.
            - "final_status" (str): Final assessment of the answer; values can be:
                - "clean" (first generation)
                - "corrected" (regenerated and supported)
                - "failed" (regenerated but not supported)
    """
    q = state['question']
    answer = state['answer']

    context = "\n\n".join([d.page_content for d in state['docs']])

    checker_output = check_answer_quality_chain.invoke({
        "question": q,
        "context": context,
        "answer": answer
    })

    verdict = checker_output.verdict
    reason = checker_output.reason

    # ✅ Only assign FINAL status here
    if state['regenerated'] == False:
        final_status = "clean"   # temporary, may change later
    else:
        if verdict == "Supported":
            final_status = "corrected"
        else:
            final_status = "failed"

    return {
        "verdict": verdict,
        "reason": reason,
        "final_status": final_status
    }
rewrite_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a correction assistant for a RAG system.\n\n"
            "A previous answer was found to contain hallucinations.\n\n"
            "Your task:\n"
            "- Fix the answer using ONLY the provided context\n"
            "- Remove any unsupported or incorrect information\n"
            "- Keep the answer accurate and complete\n"
            "- Do NOT add new information outside the context\n"
            "- If the answer cannot be supported, say: 'I don't know'\n"
        ),
        (
            "human",
            "Question:\n{question}\n\n"
            "Context:\n{context}\n\n"
            "Previous Answer:\n{answer}\n\n"
            "Issue Identified:\n{reason}\n\n"
            "Provide a corrected answer."
        )
    ]
)

rewrite_llm = ChatGroq(
    model="llama-3.3-70b-versatile",   # same as generator
    temperature=0.0
)

rewrite_chain = rewrite_prompt | rewrite_llm

def rewrite_answer(state):
    """
    Regenerate or refine an answer based on previous evaluation and reasoning.

    This function takes the current state, including the original question,
    generated answer, reason from the quality check, and retrieved documents.
    It uses the `rewrite_chain` to produce a revised answer that addresses
    any issues highlighted in the reason. Marks the answer as regenerated.

    Args:
        state (dict): A dictionary containing:
            - "question" (str): The original user question.
            - "answer" (str): The previously generated answer.
            - "reason" (str): The explanation or critique from answer quality check.
            - "docs" (list): A list of Document objects used as context for rewriting.

    Returns:
        dict: A dictionary containing:
            - "answer" (str): The rewritten or refined answer.
            - "regenerated" (bool): Always True, indicating that this answer has been regenerated.
    """
    q = state['question']
    answer = state['answer']
    reason = state['reason']

    # Convert docs to text
    context = "\n\n".join([d.page_content for d in state['docs']])

    # Call LLM
    result = rewrite_chain.invoke({
        "question": q,
        "context": context,
        "answer": answer,
        "reason": reason
    })

    return {'answer':result.content,'regenerated':True}

def decide_after_check(state):
    """
    Determine the next step after answer quality verification.

    This function decides whether an answer needs to be rewritten or can
    be considered final based on the verdict from the quality check and
    whether the answer has already been regenerated."""

    if state["verdict"] == "Not Supported" and state["regenerated"] == False:
        return "rewrite"
    else:
        return "final"

def decide_after_generate(state):
    """
    Determine the next step after generating an answer from retrieved documents.

    This function decides whether to proceed to the quality check step
    or terminate the pipeline based on the final_status of the generated answer."""
    if state['final_status'] == 'not_in_docs':
        return "end"
    else:
        return "check"

def build_graph():
    """
    Construct and compile the RAG pipeline as a state graph.

    This function sets up the full workflow for the question-answering system
    using a directed state graph. Nodes represent processing steps, and
    conditional edges control the flow based on intermediate results.

    Nodes included:
        - "init": Initialize the state with default values.
        - "retrieve": Retrieve relevant documents for the question.
        - "generate": Generate an answer based on retrieved documents.
        - "check_answer": Evaluate answer quality and hallucination.
        - "rewrite_answer": Rewrite the answer if it is unsupported.

    Edges:
        - START -> "init"
        - "init" -> "retrieve"
        - "retrieve" -> "generate"
        - "generate" -> "check_answer" or END (conditional via decide_after_generate)
        - "check_answer" -> "rewrite_answer" or END (conditional via decide_after_check)
        - "rewrite_answer" -> "check_answer" (loop for regeneration)

    Returns:
        StateGraph: A compiled state graph representing the full RAG workflow.
    """
    g = StateGraph(State)
    g.add_node("init", initialize_state)
    g.add_node("retrieve", retrieve)
    g.add_node("generate", generate)
    g.add_node('check_answer',check_answer_quality)
    g.add_node('rewrite_answer',rewrite_answer)



    g.add_edge(START, "init")
    g.add_edge('init', "retrieve")
    g.add_edge("retrieve", "generate")

    # ✅ ONLY conditional edges here
    g.add_conditional_edges(
        "generate",
        decide_after_generate,
        {
            "end": END,
            "check": 'check_answer'
        
        }
    )


    # ✅ ONLY conditional edges here
    g.add_conditional_edges(
        "check_answer",
        decide_after_check,
        {
            "rewrite": "rewrite_answer",
            "final": END
        }
    )
    # ✅ Loop back after rewrite
    g.add_edge("rewrite_answer", "check_answer")

    app = g.compile()

    return app        

def format_output(state):
    """
    Format the final output message for the user based on the state's final_status.

    This function generates a user-friendly string response depending on whether
    the answer is clean, corrected, failed, or not present in the documents.

    Parameters:
        state (dict): The current state dictionary containing at least:
            - "final_status" (str): One of ["clean", "corrected", "failed", "not_in_docs"].
            - "answer" (str): The generated answer text.

    Returns:
        str: A formatted message suitable for displaying to the user, with:
            - ✅ for clean or corrected answers
            - 🔄 when an answer has been corrected
            - ❌ if the system failed to generate a reliable answer
            - 📄 if the question is not found in the documents
    """
    status = state["final_status"]

    if status == "clean":
        return f"""
        ✅ Answer:
        {state['answer']}
        """

    elif status == "corrected":
            return f"""
    🔄 Making your answer hallucination free... ✓

    ✅ Answer:
    {state['answer']}
    """

    elif status == "failed":
            return f"""
    ❌ We failed to get a hallucination free output.

    The retrieved documents may not contain enough information 
    to answer your question reliably.
    """

    elif status == "not_in_docs":
            return f"""
    📄 The question you asked is not present in the document.
    """