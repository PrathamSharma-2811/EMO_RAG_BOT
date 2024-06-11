
```markdown
# EMO - Personal Emotional Analyzer

## Project Description
"EMO" is an innovative personal healthcare assistant designed to handle emotions in user text, providing guidance based on the context. Leveraging cutting-edge technologies like RAG chain, Ollama embeddings, Chroma for vector storage, and a Flask-based UI, this project offers a transformative approach to understanding and managing emotions.

## Technologies Used
- **Streamlit**: For creating the user interface.
- **Langchain Community**: For handling document loaders, embeddings, and language models.
- **Chroma**: For vector storage.
- **Ollama**: For embeddings and language models.
- **Python Libraries**: Various supporting libraries like `requests`, `pickle`, etc.

## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Step-by-Step Guide

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/emo-personal-emotional-analyzer.git
   cd emo-personal-emotional-analyzer
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download and Install Ollama**
   - **For Windows**: Download the Ollama installer from the [Ollama Official Website](https://www.ollama.com/download).
   - **For Linux**: Use the following command to install Ollama:
     ```bash
     curl -sSL https://ollama.com/install.sh | sh
     ```

4. **Run the Application**
   ```bash
   streamlit run app.py
   ```

### File Structure
- `app.py`: Main application script.
- `requirements.txt`: List of dependencies.
- `data/`: Directory containing the PDF and other data files.

### Usage
Navigate to `http://localhost:8501` in your web browser to access the application. You can interact with EMO by typing your questions or statements and receiving emotionally intelligent responses.

## References
- **Chroma DB**: [Chroma Documentation](https://docs.trychroma.com/)
- **Ollama**: [Ollama Official Website](https://www.ollama.com)
- **Managing Your Emotions by Joyce Meyer**: [PDF Reference](https://www.joycemeyer.org/ebooks/managing-your-emotions)

## Project Information
This project was developed to enhance emotional intelligence and provide users with a tool to better understand and manage their emotions. It uses advanced NLP techniques to analyze user text and provide meaningful insights and guidance.
