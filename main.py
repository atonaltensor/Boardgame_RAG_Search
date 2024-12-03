import os
import pandas as pd

from langchain_community.document_loaders import PyPDFLoader
from langchain.document_loaders.pdf import PyPDFDirectoryLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI

from langchain.schema import Document
from typing import List

import streamlit as st

######################################################################
# Environmental Variables for Secrets
# Yes, I know I can use dotenv package, but I like minimizing packages used.
with open (".env", "r") as env_file:
	for line in env_file:
		exec(line)

#########################################################################

def load_and_chunk(file: str = "/shared/rules/Agricola.pdf") -> List[Document]:
	#loading PDF
	loader = PyPDFLoader(file)
	pages = loader.load() #-> List[Document] where each Document is a page of the file
	
	#chunk (yuck!)
	TextSplitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
	chunks = TextSplitter.split_documents(pages) #List[Document] -> List[Document] where each document is a chunk
	
	return chunks
	
def load_dir_and_chunk(folder: str = '/shared/rules/subset') -> List[Document]:
	dir_loader = PyPDFDirectoryLoader(folder)
	pages = dir_loader.load()
	
	TextSplitter = RecursiveCharacterTextSplitter(
		chunk_size=200,
		chunk_overlap=50,
		length_function=len,
		add_start_index=True
	)
	
	chunks = TextSplitter.split_documents(pages)
	return chunks
	
def create_vectordb(chunks: List[Document], db_name: str = 'subset_db') -> Chroma:
	CHROMA_PATH = "maxx_db"
	#retrieve openai embed model
	embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
	#embed the chunks as vectors and load into chromadb
	local_chroma_db = Chroma.from_documents(
		documents = chunks,
		embedding = embeddings, 
		persist_directory = CHROMA_PATH,
		collection_name= db_name
	)
	#local_chroma_db.persist()
	
	return local_chroma_db
	
def load_vectordb(db_path: str = "maxx_db", db_name: str = 'subset_db') -> Chroma:
	embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
	vectordb = Chroma(
		persist_directory=db_path, 
		embedding_function=embeddings,
		collection_name=db_name
	)
	
	####tests
	##Note: by default, "embeddings" is always returned as None for performance reasons
	print(f"keys: {vectordb.get().keys()}")
	print(f"number of records: {len(vectordb.get()['ids'])}")
	
	return vectordb
	
def query_to_context(vectordb, query = "Does this game have separate rules for a solo mode?"):
		
	#get top 5 closest chunks via cosine distance:
	search = vectordb.similarity_search_with_score(query, k=5) #-> List[(Document, Int)]

	context_list = [(f"{Document.page_content} from the game {Document.metadata.get('source', None)}", Document.metadata.get('source', None)) for Document, score in search]

	context_text = "/n/n".join([x[0] for x in context_list])
	context_sources = [x[1] for x in context_list]
	
	return context_text, context_sources
	
def context_to_response(context_text, query):
	
	#prompt engineering
	PROMPT_ENG = """
	Answer the question based only on the following context:
	{context}
	Answer the question based on the above context: {question}.
	Provide a detailed answer.
	"""
	#load retrieved context and user query
	prompt_template = ChatPromptTemplate.from_template(PROMPT_ENG)
	prompt = prompt_template.format(context=context_text, question=query)
	
	#Use LLM to generate answer:
	model = ChatOpenAI(openai_api_key = OPENAI_API_KEY)
	response_text = model.predict(prompt)
	
	return response_text

def main(query = "Which game has separate rules for a solo mode?"):
	#chunks = load_and_chunk(file = "/shared/rules/Agricola.pdf")
	chunks = load_dir_and_chunk(folder = '/shared/rules/subset')
	vectordb = create_vectordb(chunks = chunks, db_name = 'subset_db')
	context_text, context_sources = query_to_context(vectordb, query)
	response_text = context_to_response(context_text)
	
	return reponse_text, context_sources
	
def main2(query = "Can you give me a list of all games that have a specific method of determining the start player, and what that method is?"):
	vectordb = load_vectordb(db_path = "maxx_db", db_name = 'subset_db')
	context_text, context_sources = query_to_context(vectordb, query = query)
	response_text = context_to_response(context_text, query = query)
	return response_text, context_sources
	
if __name__ == '__main__':
	#main()
	#query = "Which game has the most components and how many does it have?"
	#query = "Which games have a setting that takes place in the 20th century?"
	#query = "Can you recommend a game that has low luck and high strategic planning?"
	#query = "Can you give me a list of all games that have a specific method of determining the start player, and what that method is?"
	#main2()
	
	###streamlit
	st.set_page_config(
		page_title="Board Game Rulebook Search", 
		page_icon=":game_die:",
		menu_items={
			'Get help': None,
			'Report a Bug': None,
			'About': None
		})
	st.markdown(
	    """
	<style>
	    [data-testid="collapsedControl"] {
	        display: none
	    }
	</style>
	""",
	    unsafe_allow_html=True,
	)
	st.header("Query Rulebook PDFs")
	
	st.text("This search and retrieval augmented generation uses 100+ board game rulebook PDFs, downloaded from boardgamegeek.com. The full list of games can be found below.")
	
	with st.form("query"):
		form_input = st.text_input('Ask question')
		st.caption("""Sample questions:""")
		st.caption("_Make me a list of games that have an auction mechanism._")
		st.caption("_Which games take place in the middle ages?_")
		st.caption("_Can you outline the setup for the game of Concordia with the Imperium map?_")
		st.caption("_What's the game where players are artists during the Art Noveau period trying to make a living?_")

		submit = st.form_submit_button("Generate")
	
		if submit:
			with st.spinner(text = "Loading Vector DB..."):
				vectordb = load_vectordb(db_path = "maxx_db", db_name = 'full_db')
			with st.spinner(text = "Performing Cosine-similarity search..."):
				context_text, context_sources = query_to_context(vectordb, query = form_input)
			with st.spinner(text = "Generating LLM response..."):
				response_text = context_to_response(context_text, query = form_input)
			st.success(response_text)
			st.write(f"Sources: {context_sources}")
			
			url = "https://home.maxxcho.com/sharing/aMbPWdBGJ"
			st.caption("Note: I'm working on providing a direct link to the source PDFs referenced. For now, you can see all of them at [this link](%s), and download them manually." % url)
			
	st.markdown("""
	### Technical Specifications  
	- Vector Database: _Chroma_  
	- Text Embedding/Vectorization Model: `text-embedding-3-large`  
	- Search Method: _Cosine Similarity_ (Euclidian Metric)  
	- Response Generation Model: `gpt-3.5-turbo-instruct`  
	- Prototype UI: Streamlit
	- Hosting: Synology DSM (Local Server)
	### To-do:  
	- Refine chunk size (Improve accuracy!!!)
	- Use entirely local LLM
	- Ingest more PDFs
	- Prompt-engineering
	""")
	
	with st.expander("See complete list of rulebook PDFs parsed"):
		st.dataframe(pd.read_csv('games.csv'))
		
	st.divider()
	st.caption("Prototype by Maxx Cho")
	
	#streamlit run <FILE_NAME>.py
	#deployment notes to ensure app works remotely: 
	## 1. When setting reverse proxy, NEED to add the 2 heaaders for WebSocket support. Otherwise, page will be stuck on loading screen. 
	## 2. (Unclear if this matters) in ~/.streamlit/config.toml, set server.enableXsrfProtection=false, server.enablesCORS=false, and browser.serverAddress=rag.maxxcho.com