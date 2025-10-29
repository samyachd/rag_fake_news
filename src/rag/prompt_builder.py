from . import retriever

def prompt_build_text(context, user_text):
    # -------------------------
    # Construction du prompt
    # -------------------------
    prompt = f"""
    You are an assistant specialized in verifying information.
    You are given text chunks labeled [TRUE] or [FAKE].

    Context chunks:
    {context}

    Statement:
    \"\"\"{user_text}\"\"\" 

    Instructions:
    - Determine the statement's truthfulness.
    - Follow the majority label among the chunks.
    - Only respond UNCERTAIN if there is no clear majority.
    - Respond strictly in the format:
    VERDICT: TRUE / FAKE / UNCERTAIN
    EXPLANATION: <brief reasoning>
    - Synthetize the answer and do not give me back all the chunks
    """
    return prompt