from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os


class EmbeddingModels:

    @staticmethod
    def gemini_embedding_2_preview():
        """
        Returns Gemini embedding model (gemini-embedding-2-preview).
        Requires GOOGLE_API_KEY in environment variables.
        """

        if not os.getenv("GOOGLE_API_KEY"):
            raise ValueError("GOOGLE_API_KEY environment variable is not set")

        return GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-2-preview"
        )