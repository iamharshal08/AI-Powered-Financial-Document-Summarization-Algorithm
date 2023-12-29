#!/usr/bin/env python
# coding: utf-8

import os
import requests
from urllib.parse import urljoin
from bs4 import BeautifulSoup

from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI

from PyPDF2 import PdfMerger

os.environ['OPENAI_API_KEY'] = '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'

def merge_pdfs(folder_location, output_path):
    merger = PdfMerger()
    for filename in os.listdir(folder_location):
        if filename.endswith('.pdf'):
            file_path = os.path.join(folder_location, filename)
            merger.append(file_path)

    merger.write(output_path)
    merger.close()
    
def generate_context_file(output_path):
    
    url = "https://www.jpmorganchase.com/content/dam/jpmc/jpmorgan-chase-and-co/documents/Supplier-Code-of-Conduct.pdf"

    org_name = url.split(".")[1]

    folder_location = "context_files/"+org_name#"context_files/jpmorganchase"#f"context_files/{org_name}"
    if not os.path.exists(folder_location):
        try:
            os.mkdir(folder_location.split('/')[0])
        except Exception:
            pass
        os.mkdir(folder_location)
        
        
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    for link in soup.select("a[href$='.pdf']"):
        # pdf file name
        filename = os.path.join(folder_location, link['href'].split('/')[-1])
        if not os.path.exists(filename):
            with open(filename, 'wb') as f:
                f.write(requests.get(urljoin(url, link['href'])).content)
                

    # Call the merge_pdfs function with the folder path and output path
    merge_pdfs(folder_location, output_path)


def wrapper_model(pdf_qa,query):
    
    #print("Hello world!")
    
    # Specify the output file path for the merged PDF
    # output_path = 'context_files/qna.pdf'
    # # generate_context_file(output_path)
    # loader=PyPDFLoader(output_path)
    # pages = loader.load_and_split()
    #
    # embeddings = OpenAIEmbeddings()
    # vectordb = FAISS.from_documents(pages, embedding=embeddings)
    # #vectordb.persist()
    # memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    # pdf_qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0.8) , vectordb.as_retriever(), memory=memory)


    result = pdf_qa({"question": query})
    print("Answer:")
    return result["answer"]

# def main():
# print(wrapper('What is my name'))


# if __main__ == '__main__':
#     main()
    
    
