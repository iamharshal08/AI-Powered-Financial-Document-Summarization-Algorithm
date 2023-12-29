import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from wrapper import wrapper_model

from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI




app = dash.Dash(__name__)
output_path = 'context_files/qna.pdf'
# generate_context_file(output_path)
loader = PyPDFLoader(output_path)
pages = loader.load_and_split()

embeddings = OpenAIEmbeddings()
vectordb = FAISS.from_documents(pages, embedding=embeddings)
# vectordb.persist()
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
pdf_qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0.8), vectordb.as_retriever(), memory=memory)

# Append the CSS file
# app.css.append_css({'external_url': 'styles.css'})

# Define the layout
app.layout = dcc.Loading([html.Div(
    className="container",
    children=[
        html.Div(
            className="question-container",
            children=[
                html.H1("Ask a Question"),
                dcc.Input(
                    id="question-input",
                    type="text",
                    placeholder="Type your question here...",
                    className="form-control question-input"
                ),
                html.Button("Ask", id="ask-button", className="btn btn-primary"),
                html.H3(id="answer-output", className="answer-output typewriter",children=['ANSWERS'])
            ]
        )
    ]
)])

# Define the callback to handle question answering
@app.callback(
    Output("answer-output", "children"),
    [Input("ask-button", "n_clicks")],
    [State("question-input", "value")]
)
def ask_question(n_clicks, question):
    if n_clicks is None:
        return ""

    # Implement your question answering logic here
    answer = "This is the answer to your question: {}".format(wrapper_model(pdf_qa,question))
    print(answer)
    # Typing animation effect
    return answer
    # return dcc.Interval(
    #     id="typing-interval",
    #     interval=150,
    #     max_intervals=len(answer) + 2,
    #     n_intervals=0
    # )


# @app.callback(
#     Output("answer-output", "children"),
#     [Input("typing-interval", "n_intervals")],
#     [State("answer-output", "children")]
# )
# def display_answer(n_intervals, current_text):
#     if n_intervals > 0:
#         typing_text = current_text[:-1] if current_text.endswith("|") else current_text + "|"
#         return typing_text
#
#     return current_text


if __name__ == "__main__":
    app.run_server(debug=False)

