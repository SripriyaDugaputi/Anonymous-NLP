import pandas as pd

class DataPreprocessor:
    def __init__(self, file_path, chunk_size=None):
        """
        Initializes the DataPreprocessor.

        Parameters:
        - file_path: str, path to the CSV file containing the data.
        - chunk_size: int or None, the size of text chunks (in number of sentences) to split the texts into. 
                      If None, texts are not chunked.
        """
        self.file_path = file_path
        self.chunk_size = chunk_size

    def load_data(self):
        """
        Loads the data from the CSV file.

        Returns:
        - DataFrame containing the loaded data.
        """
        df = pd.read_csv(self.file_path)
        if self.chunk_size is not None:
            df = self.chunk_texts(df)
        return df

    def chunk_texts(self, df):
        """
        Chunks the texts in the DataFrame if chunk_size is specified.

        Parameters:
        - df: DataFrame, the DataFrame containing the texts to chunk.

        Returns:
        - DataFrame with chunked texts.
        """
        chunked_texts = []
        for _, row in df.iterrows():
            text_chunks = self.chunk_text(row['text'])
            for chunk in text_chunks:
                chunked_row = row.copy()
                chunked_row['text'] = chunk
                chunked_texts.append(chunked_row)
        return pd.DataFrame(chunked_texts)

    def chunk_text(self, text):
        """
        Chunks a single text into smaller parts, each with a length of up to chunk_size sentences.

        Parameters:
        - text: str, the text to chunk.

        Returns:
        - List of str, the chunked text parts.
        """
        if not self.chunk_size:
            return [text]
        
        # Split the text into sentences. This is a simplistic approach.
        # For more accurate sentence tokenization, consider using nltk or spacy.
        sentences = text.split('. ')
        chunks = [' '.join(sentences[i:i + self.chunk_size]) for i in range(0, len(sentences), self.chunk_size)]
        return chunks
