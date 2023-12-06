import numpy as np
import openai
from prompts import *
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain import LLMChain, PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

embedding = OpenAIEmbeddings()

# from langchain.embeddings import HuggingFaceEmbeddings
# embedding = HuggingFaceEmbeddings()


def _load_split_docs(filename, chunk_size, chunk_overlap):
    r_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
    )

    docs = PyPDFLoader(f"{filename}").load()
    docs_split = r_splitter.split_documents(docs)

    return docs_split


def search_docs(file, query, **kwargs):
    """create vector DB and retrieve chunks per file"""

    chunk_size = kwargs.get("chunk_size", 1000)
    chunk_overlap = kwargs.get("chunk_overlap", 100)
    with_scores = kwargs.get("with_scores", False)
    num_chunks = 30

    docs_split = _load_split_docs(file, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    db = FAISS.from_documents(docs_split, embedding)

    if with_scores:
        docs_and_scores = db.similarity_search_with_score_by_vector(
            embedding.embed_query(query), k=num_chunks
        )
        # docs_and_scores = db.max_marginal_relevance_search_with_score_by_vector(embedding.embed_query(query),k=num_chunks)

    else:
        docs_and_scores = db.similarity_search_by_vector(embedding.embed_query(query), k=num_chunks)
    # docs_and_scores = db.max_marginal_relevance_search_by_vector(embedding.embed_query(query),k=num_chunks)

    return docs_and_scores


def get_response(prompt, llm_model="gpt-4"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=llm_model,
        messages=messages,
        temperature=0,
    )

    return response.choices[0].message["content"]


def get_headers(queries, llm_model="gpt-4"):
    headers = []
    for q in queries:
        header_prompt = f"""Your goal is to identify what this question is asking for. Your answer should be one word. \n
        eg: 
        Q: What is the type of membrane/membrane material used in this work? \n
        A: membrane

        question: {q}
        answer: <provide one word answer here>"""

        chat_answer = get_response(header_prompt, llm_model=llm_model)
        headers.append(chat_answer)
        headers.append("excerpts")
        headers.append("score")

    return headers


def set_answer_query(llm, parser):
    prompt = GPT_QA_PROMPT

    prompt_template = PromptTemplate(
        template=prompt,
        input_variables=["question", "chunk", "previous"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    llmchain = LLMChain(prompt=prompt_template, llm=llm)

    return llmchain


def get_answer_query(evidence, query, previous, llmchain):
    response = llmchain.run({"question": query, "chunk": evidence, "previous": previous})

    return response


def gather_evidence(filename, query, **kwargs):
    """gather evidence from a file for a given query"""

    chunk_size = kwargs.get("chunk_size", 1000)
    top_k = kwargs.get("top_k", 10)
    with_scores = kwargs.get("with_scores", False)

    if with_scores:
        doc_chunks = np.array(
            search_docs(
                filename,
                query,
                chunk_size=chunk_size,
                chunk_overlap=chunk_size * 0.1,
                with_scores=with_scores,
            )
        )

        # lower the score the most similar, already sorted

        if len(doc_chunks) > top_k:
            doc_chunks = doc_chunks[:top_k]

        evidence = ""
        for doc in doc_chunks:
            chunk = doc[0].page_content
            evidence += chunk

    else:
        doc_chunks = np.array(
            search_docs(
                filename,
                query,
                chunk_size=chunk_size,
                chunk_overlap=chunk_size * 0.1,
                with_scores=False,
            )
        )

        evidence = ""
        for doc in doc_chunks:
            chunk = doc.page_content
            evidence += chunk

    return evidence


def set_score_chunks(parser, llm):
    prompt = SCORE_PROMPT

    prompt_template = PromptTemplate(
        template=prompt,
        input_variables=["question", "chunk"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    llmchain = LLMChain(prompt=prompt_template, llm=llm)

    return llmchain


def get_score_chunks(question, doc_chunks, llmchain, parser, cutoff=7):
    evidence = ""
    for doc in doc_chunks:
        chunk = doc.page_content
        response = llmchain.run({"question": question, "chunk": chunk})
        parsed = parser.parse(response)

        if parsed.relevance_score > cutoff:
            evidence += chunk

    return evidence
