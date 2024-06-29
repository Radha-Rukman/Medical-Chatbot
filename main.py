import streamlit as st
import fitz  # PyMuPDF
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import torch

# Initialize Streamlit app title and other configurations
st.title("Medical private GPT")

# Function to process uploaded files (PDF and text)
def process_file(file):
    if file.type == "application/pdf":
        return extract_text_from_pdf(file)
    elif file.type == "text/plain":
        return extract_text_from_txt(file)
    else:
        raise ValueError("Unsupported file type")

def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def extract_text_from_txt(file):
    content = file.read()
    return content.decode("utf-8")

# Function to load and initialize models
def load_models():
    model_name = "mistralai/Mistral-7B-v0.3"  # Update with your model name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# Chatbot service to handle questions and answers
class ChatbotService:
    def __init__(self):
        self.documents = {}
        self.document_embeddings = {}
        self.tokenizer, self.model = load_models()
        self.embedding_model = load_embedding_model()

    def add_document(self, doc_id, content):
        self.documents[doc_id] = content
        sentences = content.split('\n')
        embeddings = self.embedding_model.encode(sentences, convert_to_tensor=True)
        self.document_embeddings[doc_id] = (sentences, embeddings)

    def answer_question(self, question, document_id=None):
        if document_id:
            if document_id not in self.documents:
                return "Document not found"
            document_text = self.documents[document_id]
            sentences, embeddings = self.document_embeddings[document_id]
            question_embedding = self.embedding_model.encode(question, convert_to_tensor=True)
            scores = torch.nn.functional.cosine_similarity(question_embedding, embeddings)[0]
            best_match_idx = torch.argmax(scores).item()
            best_sentence = sentences[best_match_idx]
            return self.answer_question_with_context(best_sentence, question)
        else:
            return self.answer_general_question(question)

    def answer_question_with_context(self, context, question):
        input_text = context + "\n\nQ: " + question + "\nA:"
        inputs = self.tokenizer(input_text, return_tensors='pt')
        outputs = self.model.generate(inputs['input_ids'], max_length=100)
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer

    def answer_general_question(self, question):
        input_text = "Q: " + question + "\nA:"
        inputs = self.tokenizer(input_text, return_tensors='pt')
        outputs = self.model.generate(inputs['input_ids'], max_length=100)
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer

# Initialize chatbot service
chatbot_service = ChatbotService()

# Streamlit UI
uploaded_file = st.file_uploader("Upload a document", type=["pdf", "txt"])

if uploaded_file:
    try:
        content = process_file(uploaded_file)
        doc_id = uploaded_file.name
        chatbot_service.add_document(doc_id, content)
        st.success(f"Document '{doc_id}' uploaded and processed successfully")
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")

st.header("Chat with the Question Answering Bot")

while True:
    question = st.text_input("You:")
    if st.button("Ask") and question:
        document_id = uploaded_file.name if uploaded_file else None
        if document_id:
            answer = chatbot_service.answer_question(question, document_id)
            st.write("Bot:", answer)
        else:
            st.error("Please upload a document first")
    elif question.strip().lower() == 'done':
        break
