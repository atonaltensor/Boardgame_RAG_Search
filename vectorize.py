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

######################################################################
# Environmental Variables for Secrets
# Yes, I know I can use `dotenv` package, but I like minimizing packages used.
with open (".env", "r") as env_file:
	for line in env_file:
		exec(line)

#########################################################################

def load_and_chunk(file = "/shared/rules/Agricola.pdf"):
	#loading PDF
	loader = PyPDFLoader(file)
	pages = loader.load() #-> List[Document] where each Document is a page of the file
	_ = """
	#print(pages[0])
	Document(metadata={'source': '/shared/rules/Agricola.pdf', 'page': 0}, page_contentplayers (with farmyard spaces as well as 1 example \n       on the reverse side) \nincluding one with an alternative \n       reverse side for the Family game, as welr Improvements (with a summary of scoring on the reverse side)\n360 Cards:\n•   169 1-5 players; \n           41 cards for 3-5 players; 62 cards for 4-5 players)\n•  ncluding 7 upgrades \n           from Major or Minor Improvements)\n•     10 red “MRound cards with possible actions for rounds 1 to 14 \n•     16 green Action cards       on the number of players\n•       5 grey Begging cards\n•       5 Summary carck K)\nWooden playinG pieCes:\n•    5 Family member discs, 4 Stables and 15 Fences olors (blue, green, red, natural wood and purple)\n•    33 round, dark brown Wood c counters\n•    15 round, white Reed counters\n•    18 round, grey Stone counters\n   18 round, orange Vegetable counters\n•    21 Sheep tokens (white cubes)\n•    18 Cattle tokens (brown cubes)\n•      1 Starting player token\nand also:\n•    33 br4 brown/red Wood/Clay hut tiles\n•    36 yellow Food markers labeled “1”\n•      9 mals, goods or Food)\n•      3 Claim markers (with “Guest” on the reverse)\n•      nt game for 1-5 players by Uwe Rosenberg\nPlaying time: Half an hour per player, shars\nCentral Europe, around 1670 AD. The Plague which has raged since 1348 has finas revital-\nized. People are upgrading and renovating their huts. Fields must be plf the previous years \nhas encouraged people to eat more meat (a habit that we cont 17th Century: Not an easy time for farming \n(Agricola is the Latin word for “farmement\nStage 1\nRound 1-4\n2 Wood\nDuring the Feeding phase of the \nHarvest, whenee enough Food \nto feed your family, you must take\n1 Begging card for each missingastures\nGrain*\nVegetables*\nSheep\nWild boar\nCattle\n-1 Point\n0-1\n0\n0\n0\n0\n\n2 Points\n3\n2\n4-5\n2\n4-5\n3-4\n2-3\n3 Points\n4\n3\n6-7\n3\n6-7\n5-6\n4-5\n4 Pted & \nharvested Grain / \nVegetables\n-1 point per unused space in the Farmyard  ut room  \n2 points per Stone house room  \n3 points per Family member\nOccupation\ 139 x 10 x 14 x 16 x\n5 x\n5 x\n2 x\n33 x 24 x\n36 x 9 x 3 x\n')
	>>> 
	"""
	
	#chunk (yuck!)
	TextSplitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
	chunks = TextSplitter.split_documents(pages) #List[Document] -> List[Document] where each document is a chunk
	
	#print(type(chunks[2]))
	#<class 'langchain_core.documents.base.Document'>
	_ = """
	#print(chunks[100])
	page_content='Action space is added into the game; this can be used in the round in which it is turned up
	and in each subsequent round. Each of the 6 stages ends with a Harvest.
	The Action spaces are described here in the order of the game stages.
	Sow and/or bake bread (Stage 1): For a description of Sowing, see Rules, Action C. When
	Sowing, a player need not Sow all her empty fields, some may be left empty. Bake bread' metadata={'source': '/shared/rules/Agricola.pdf', 'page': 8}
	"""
	
	return chunks
	
def load_dir_and_chunk(folder: str = '/shared/rules/subset') -> List[Document]:
	dir_loader = PyPDFDirectoryLoader(folder)
	pages = dir_loader.load()
	_ = """
	#print(pages[0])
	Document(metadata={'source': '/shared/rules/Agricola.pdf', 'page': 0}, page_contentplayers (with farmyard spaces as well as 1 example \n       on the reverse side) \nincluding one with an
	etc...
	"""
	print("Successfully loaded directory!")
	TextSplitter = RecursiveCharacterTextSplitter(
		chunk_size=200,
		chunk_overlap=50,
		length_function=len,
		add_start_index=True
	)
	chunks = TextSplitter.split_documents(pages)
	print("Successfully chunked!")
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
	print("successfully created vectordb!")
	return local_chroma_db
	
def load_vectordb(db_path: str = "maxx_db", db_name: str = 'subset_db') -> Chroma:
	embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
	vectordb = Chroma(
		persist_directory=db_path, 
		embedding_function=embeddings,
		collection_name=db_name
	)
	print("successfully loaded vectordb!")
	####tests
	##Note: by default, "embeddings" is always returned as None for performance reasons
	print(f"keys: {vectordb.get().keys()}")
	print(f"number of records: {len(vectordb.get()['ids'])}")
	print(vectordb.get(ids = ["b9351681-e725-4b5f-bf04-3a79df36939b", "9ab54b1b-5550-4ebd-85e9-7b182d7e082a"], include=["embeddings", "documents"]))
	#print("embeddings: "+vectordb.get()["embeddings"][123])
	#print("documents: "+vectordb.get()["documents"][123])
	#print("uris: "+vectordb.get()["uris"][123])
	#print("data: "+vectordb.get()["data"][123])
	#print("metadatas: "+vectordb.get()["metadatas"][123])
	#print("included: "+vectordb.get()["included"][123])
	
	return vectordb
	
def query_to_context(vectordb, query = "Does this game have separate rules for a solo mode?"):
		
	#get top 5 closest chunks via cosine distance:
	search = vectordb.similarity_search_with_score(query, k=5)
	#-> List[(Document, Int)]
	print("successfully performed similarity search!")
	_ = """
		[(Document(metadata={'page': 38, 'source': '/shared/rules/subset/Civolution.pdf', 'start_index': 1574}, page_content='Other components are not used for V.I.C.I.\nSolo mode: Playing the game\nThe game is played as usual. In each era, go through the 8 phases as always, but with the following adjustments:\nPhase 1: Play as usual.'), 0.34125199913978577),]
	"""

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
	#Don’t justify your answers.
	#Don’t give information not mentioned in the CONTEXT INFORMATION.
	#Do not say "according to the context" or "mentioned in the context" or similar.

	#load retrieved context and user query
	prompt_template = ChatPromptTemplate.from_template(PROMPT_ENG)
	prompt = prompt_template.format(context=context_text, question=query)
	
	#Use LLM to generate answer:
	model = ChatOpenAI(openai_api_key = OPENAI_API_KEY)
	response_text = model.predict(prompt)
	
	return response_text

def main(folder = '/shared/rules/subset2', query = "Which game has separate rules for a solo mode?"):
	#chunks = load_and_chunk(file = "/shared/rules/Agricola.pdf")
	chunks = load_dir_and_chunk(folder)
	vectordb = create_vectordb(chunks = chunks, db_name = 'full_db')
	context_text, context_sources = query_to_context(vectordb, query)
	response_text = context_to_response(context_text)
	
	return reponse_text, context_sources
	
def main2(query = "Can you give me a list of all games that have a specific method of determining the start player, and what that method is?"):
	vectordb = load_vectordb(db_path = "maxx_db", db_name = 'full_db')
	context_text, context_sources = query_to_context(vectordb, query = query)
	response_text = context_to_response(context_text, query = query)
	return response_text, context_sources
	
if __name__ == '__main__':
	#main()
	#query = "Which game has the most components and how many does it have?"
	#query = "Which games have a setting that takes place in the 20th century?"
	#query = "Can you recommend a game that has low luck and high strategic planning?"
	#query = "Can you give me a list of all games that have a specific method of determining the start player, and what that method is?"
	print(main2())
