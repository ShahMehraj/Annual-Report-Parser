# Annual Report Parser

## Overview

**Annual Report Parser** is a Streamlit-based application designed to extract financial data from PDF annual reports. This tool leverages the power of OpenAI APIs and LangChain for embedding, vector storage, and question-answering, making it easy to analyze financial documents and retrieve key insights in a structured format.

---

## Features

- **PDF Text Extraction**: Parses multiple PDF files and extracts text content efficiently.
- **Text Chunking**: Splits extracted text into manageable chunks for better processing.
- **Vector Store Creation**: Utilizes FAISS for creating and storing vectorized representations of document chunks.
- **OpenAI Integration**: Employs OpenAI models for embeddings and natural language processing.
- **Financial Data Extraction**: Extracts specific financial details like company name, business segments, currency, and revenue.
- **Excel Export**: Generates structured financial data in an Excel file for easy reporting.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ShahMehraj/Annual-Report-Parser.git
   cd Annual-Report-Parser
   ```

2. Set up a Python virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # For Linux/macOS
   venv\Scripts\activate     # For Windows
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set your OpenAI API key in a `.env` file:
   ```plaintext
   OPENAI_API_KEY=your_openai_api_key
   ```

---

## Usage

1. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

2. Upload PDF files containing annual reports using the provided sidebar interface.

3. Process the uploaded reports to extract financial data.

4. Download the extracted data as an Excel file.

---

## File Structure

```
Annual-Report-Parser/
├── app.py                  # Main application script
├── requirements.txt        # Python dependencies
├── .env                    # Environment file for API keys
├── README.md               # Project documentation
├── utils/                  # Helper scripts (e.g., text extraction, vector store management)
├── faiss_index/            # FAISS vector storage
└── outputs/                # Folder for generated Excel files
```

---

## Key Technologies

- **Streamlit**: For building the user interface.
- **OpenAI API**: For embedding and natural language processing.
- **FAISS**: For efficient vector similarity search.
- **PyPDF2**: For PDF text extraction.
- **LangChain**: For chain-based question-answering and embeddings.

---

## Contributing

We welcome contributions to enhance this project! Feel free to:

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature-name`.
3. Commit your changes: `git commit -m "Add new feature"`.
4. Push the branch: `git push origin feature-name`.
5. Submit a pull request.


---

## Contact

If you have any questions or suggestions, feel free to open an issue or reach out to the repository owner. 

Happy Parsing! 🚀
