from click import prompt
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import llm
from retriever import Retriever
import constants

class Pipeline():
    """Pipeline class for processing data"""
    def __init__(self):
        self.prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing relevance 
        of a retrieved document to a user question. If the document contains keywords related to the user question, 
        grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
        Provide the binary score as a JSON with a single key 'score' and no premable or explanation.
        <|eot_id|><|start_header_id|>user<|end_header_id|>
        Here is the retrieved document: \n\n {document} \n\n
        Here is the user question: {question} \n <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """,
        input_variables=["question", "document"],
        )
        self.llm = llm.LLM(local_llm=constants.MODEL_NAME, base_url=constants.INFERENCE_URL)
        self.retriver = Retriever()

        self.rag_chain = self.prompt | self.llm.llm | StrOutputParser()
        self.db = self.retriver.create_vector_db()

    def generate_response(self, question, document):
        """Generate a response from the LLM model."""
        docs = self.retriver.query_document(self.db, question)
        response = self.rag_chain.invoke({"content": docs, "question": question})

        return response
    