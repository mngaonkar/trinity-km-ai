from langchain_community.chat_models import ChatOllama

class LLM():
    """The LLM model."""
    def __init__(self, local_llm, base_url="http://localhost:11434"):
        """Initialize the LLM model."""
        self.llm = ChatOllama(model=local_llm, base_url=base_url, temperature=0.6)

    def generate_response(self, prompt, widget):
        text = ""
        """Generate a response from the LLM model."""
        for chunk in self.llm.stream(prompt):
            text += str(chunk.content)
            widget(text)

    def stream_response(self, prompt):
        """Stream a response from the LLM model."""
        for chunk in self.llm.stream(prompt):
            yield chunk.content
    
    
