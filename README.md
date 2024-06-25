Intelligent Document-Based Q&A System
Overview
This project is an intelligent Q&A application designed to provide efficient information retrieval and response generation. Utilizing advanced technologies such as Retrieval-Augmented Generation (RAG) with LangChain, Chroma, and OpenAI API, the application supports document uploads in PDF, DOCX, and TXT formats and maintains chat history for context-aware interactions. An interactive dashboard built with Streamlit offers a user-friendly interface for managing documents and Q&A interactions.

Key Features
Efficient Q&A System: Developed using RAG with LangChain, Pinecone, and OpenAI API for effective information retrieval and response generation.
Document Uploads: Supports PDF, DOCX, and TXT formats, allowing users to query based on the content of uploaded files.
Chat History Storage: Maintains conversation context to improve user experience over multiple sessions.
Interactive Dashboard: Built with Streamlit, providing a seamless interface for document management and Q&A interactions.
Technologies Used
LangChain
Pinecone
OpenAI API
Streamlit
PDF, DOCX, TXT (file formats)
RAG (Retrieval-Augmented Generation)
Installation
Clone the Repository

bash
Copy code
git clone (https://github.com/omadkholkar/RAG-Q-A-App-on-Private-Documents-Chroma-.git)
cd qa-system
Install Dependencies

bash
Copy code
pip install -r requirements.txt
Set Up Environment Variables

Create a .env file in the root directory and add your API keys and necessary configuration.
env
Copy code
OPENAI_API_KEY=your_openai_api_key
Run the Application

bash
Copy code
streamlit run app.py
Usage
Upload Documents: Upload your PDF, DOCX, or TXT files via the Streamlit dashboard.
Ask Questions: Enter your questions in the provided text box.
View Responses: Receive answers based on the content of the uploaded documents.
Chat History: Review previous interactions to maintain context.
Contributing
Fork the Repository
Create a New Branch
bash
Copy code
git checkout -b feature-branch
Commit Your Changes
bash
Copy code
git commit -m "Description of changes"
Push to the Branch
bash
Copy code
git push origin feature-branch
Open a Pull Request

Contact
For questions or inquiries, please contact omadkholkar@gmail.com.
