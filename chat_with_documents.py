import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import os


# loading PDF, DOCX and TXT files as LangChain Documents
def load_document(file):
    import os
    name, extension = os.path.splitext(file)

    if extension == '.pdf':
        from langchain.document_loaders import PyPDFLoader
        print(f'Loading {file}')
        loader = PyPDFLoader(file)
    elif extension == '.docx':
        from langchain.document_loaders import Docx2txtLoader
        print(f'Loading {file}')
        loader = Docx2txtLoader(file)
  
    elif extension == '.csv':
        from langchain.document_loaders.csv_loader import CSVLoader
        print(f'Loading {file}')
        loader = CSVLoader(file)
    
    elif extension == '.txt':
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain.document_loaders import TextLoader
        #print(f'Loading {file}')
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        #with open('files/churchill_speech.txt') as data:
                    #text_data=data.read()
        #text_splitter = RecursiveCharacterTextSplitter(chunk_size=100,chunk_overlap=20,length_function=len)
        #data=text_splitter.create_documents([text_data])
        #return data
        loader = TextLoader(file)
    
    else:
        print('Document formt is not supported')
        return None
    
    data = loader.load()
    return data

#wikipedia data loading
def load_from_wikipedia(query,lan='en', load_max_docs=2):
    from langchain.document_loaders import WikipediaLoader
    loader=WikipediaLoader(query=query, lang= lan, load_max_docs=load_max_docs)
    data=loader.load()
    return data

#Chunking
def chunk_data(data, chunk_size=256, chunk_overlap=20):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunk = text_splitter.split_documents(data)
    return chunk

#Embedding
def create_embeddings(chunk):
     embedding = OpenAIEmbeddings()
     vector_store = Chroma.from_documents(chunk, embedding)
     return vector_store

#Ask and get answer
def ask_and_get_answer(vector_store, q, k=3):
    from langchain.chains import RetrievalQA
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=1)

    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': k})

    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    
    answer = chain.invoke(q)
    return answer['result']

def calculate_embedding_cost(texts):
    import tiktoken
    enc = tiktoken.encoding_for_model('text-embedding-ada-002')
    total_tokens=sum([len(enc.encode(page.page_content)) for page in texts])
    #print(f'Total Tokens: {total_tokens}')
    #print(f'Embedding Cost in USD{total_tokens/1000*0.0004:.6f}')
    return total_tokens, total_tokens/1000*0.0004

def clear_history():
   if 'history' in st.session_state:
       del st.session_state['history']
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(),override=True)

    st.image('QA.jpg')
    st.subheader('LLM Question-Answering Application 🤖')
    with st.sidebar:
        api_key=st.text_input('OpenAI API Key :', type='password')
        if api_key:
             os.environ['OPENAI_API_KEY'] = api_key
              
              
        uploaded_file = st.file_uploader('Uploa a file :', type=['pdf','txt','docx'])
        chunk_size =st.number_input('Chunk size:', min_value=100,max_value=2048,value=512,on_change=clear_history)
        k = st.number_input('k',min_value=1,max_value=20,value=3,on_change=clear_history)
        add_data = st.button('ADD DATA',on_click=clear_history)

        if uploaded_file and add_data:
             with st.spinner('Reading, chunking and embedding file ....'):
                bytes_data=uploaded_file.read()
                file_name = os.path.join('./',uploaded_file.name)
                with open(file_name,'wb')as f:
                     f.write(bytes_data)
                
                data=load_document(file_name)
                chunks=chunk_data(data,chunk_size=chunk_size)
                st.write(f'Chunk size: {chunk_size}, Chunk length: {len(chunks)}')
                
                tokens,embedding_cost=calculate_embedding_cost(chunks)
                st.write(f'Embedding cost: $ {embedding_cost: .4f}')

                vector_store = create_embeddings(chunks)
                st.session_state.vs = vector_store
                st.success('File uploaded and chunked and embedded successuflly')

    q=st.text_input('Ask Quesions about the contnent of you file     : ')
    if q:
         if 'vs' in st.session_state:
            vector_store=st.session_state.vs
            st.write(f'k: {k}')
            answer=ask_and_get_answer(vector_store,q,k)
            st.text_area('LLM Answer :',value=answer)
  
            st.divider()
            if 'history' not in st.session_state:
                st.session_state.history = ''
            value = f'Q: {q} \n Answer: {answer}'

            st.session_state.history=f'{value} \n {"-"*100}\n {st.session_state.history}'
            h=st.session_state.history
            st.text_area(label='Chat History',value=h,key='history',height=400)