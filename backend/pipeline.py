from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from backend import llm
from backend.database import Database
import constants

class Pipeline():
    """Pipeline class for processing data"""
    def __init__(self):
        self.prompt = PromptTemplate(
        template="""
        <|begin_of_text|>
        <|start_header_id|>system<|end_header_id|>
        you are an AI assistent
        <|eot_id|>
        <|start_header_id|>user<|end_header_id|>
        {history}
        {context}
        {question}
        <|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>
        """,
        input_variables=["history", "context", "question"],
        )
        self.llm = llm.LLM(local_llm=constants.MODEL_NAME, base_url=constants.INFERENCE_URL)
        self.db = Database()
        self.retriever = self.db.vectorstore.as_retriever()

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        self.rag_chain = (
            {"history":self.get_session_history, "context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm.llm
            | StrOutputParser()
        )
        self.store = {}

    def generate_response(self, query):
        """Generate a response from the LLM model."""
        response = self.rag_chain.invoke(query,
                                         config={
                                             "configurable": {'session_id': '1234'}, 
                                         })

        return response
    
    def stream_response(self, prompt):
        """Stream a response from the LLM model."""
        print(self.retriever.invoke(prompt))
        for chunk in self.rag_chain.stream(prompt):
            yield chunk

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]
    