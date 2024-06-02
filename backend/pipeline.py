from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from backend import llm
from backend.database import Database
import constants

class Pipeline():
    """Pipeline class for processing data"""
    def __init__(self):
        self.prompt = PromptTemplate(
        template="""
            Human: You are an AI assistant, and provides answers to questions by using fact based and statistical information when possible.
            Use the following pieces of information to provide a concise answer to the question enclosed in <question> tags.
            If you don't know the answer, just say that you don't know, don't try to make up an answer.
            <context>
            {context}
            </context>

            <question>
            {question}
            </question>

            The response should be specific and use statistics or numbers when possible.

        Assistant:""",
        input_variables=["context", "question"],
        )
        self.llm = llm.LLM(local_llm=constants.MODEL_NAME, base_url=constants.INFERENCE_URL)
        self.db = Database()
        self.retriever = self.db.vectorstore.as_retriever()

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        self.rag_chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm.llm
            | StrOutputParser()
        )

    def generate_response(self, query):
        """Generate a response from the LLM model."""
        response = self.rag_chain.invoke(query)

        return response
    
    def stream_response(self, prompt):
        """Stream a response from the LLM model."""
        for chunk in self.rag_chain.stream(prompt):
            yield chunk
    