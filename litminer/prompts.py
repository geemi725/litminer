QUESTION_PROMPT = """Your goal is to answer the question using the provided text. You are a scientist who is answering a question based on academic literature. \n
Your answer should be precise and concise. No need to answer in sentences. One word or only the value is enough. eg: 1846 m2 \n
text: {text} \n

question: {question} \n
previous Q&A : \n{previous_qa} \n
Format instructions: \n{format_instructions} 

answer:<|your answer|>"""


GPT_QA_PROMPT = """Given the following unformatted text from an academic paper, please find the answer to a question. \n
    
    Question: {question}

    text: {chunk}

    If you find multiple answers, please provide the most relevant answer. \n
    Also provide the text which was used to generate the answer. \n
    Finally, provide a confidence score between 0 and 10 to indicate how confident you are with your answer. \n

    If you couldn't find an answer, reply with 'not found'

    You can also use previous questions and answers to help you generate the answer. \n
    Previous:
    {previous} \n

    answer: <provide answer here>
    literature evidence: <provide text here>
    confidence score: <a value between 0 and 10>

    Format instructions: \n{format_instructions}
    
    """

SCORE_PROMPT = """Given the following text try your best to find the answer to the question. \n  
    Your goal is to see if the given text is useful to answer the question. \n
    If you find the text useful, provide a score between 0 and 10 to indicate how relevant the text is to the question. \n
    If the text is irrelevant, provide a score of 0. \n

    Question: {question}
    text: {chunk}
    
    relevance score: <provide score here>


    Format instructions: \n{format_instructions}
    
    """
