from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from backend import llm
from backend.database import Database
import constants
from loguru import logger
from langchain.callbacks.tracers import ConsoleCallbackHandler


class Pipeline():
    """Pipeline class for processing data"""
    chat_history = []

    def __init__(self):
        self.prompt = PromptTemplate(
        template="""
        <|begin_of_text|>
        <|start_header_id|>system<|end_header_id|>
        {history}
        <|eot_id|>
        <|start_header_id|>user<|end_header_id|>
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
        
        def get_context(prompt):
            """Get the context of the conversation."""
            docs = self.db.query_document(prompt)
            return format_docs(docs)
            
        # build the pipeline
        self.rag_chain = (
            {"history":self.get_session_history, "context": get_context, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm.llm
            | StrOutputParser()
        )

    def generate_response(self, query):
        """Generate a response from the LLM model."""
        response = self.rag_chain.invoke(query,
                                         config={
                                             "configurable": {'session_id': '1234'}, 
                                         })

        return response
    
    def stream_response(self, prompt):
        """Stream a response from the LLM model."""
        logger.debug(self.retriever.invoke(prompt))
        for chunk in self.rag_chain.stream(prompt,
                                           config={'callbacks': [ConsoleCallbackHandler()]}):
            yield chunk

    
    def get_session_history(self, prompt):
        logger.info(f"length of chat history: {len(self.chat_history)}")
        if len(self.chat_history) > constants.MAX_CHAT_HISTORY:
            logger.info("chat history is greater than max chat history, truncating...")
            self.chat_history = self.chat_history[-constants.MAX_CHAT_HISTORY:]
        return " ".join(self.chat_history)