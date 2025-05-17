from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import time
from vector import retriever

model = OllamaLLM(model="llama3.2")

template = """
You are a resume re-compiler. first determine which section (metadata) you are pulling from. I have already given you a resume. Based on the "Job Description" query, you will use the given resume to simply select the most impactful points that align with the "Job Description" query and, return them. Do MINOR rephrasing adjustments if needed. You will make sure to rely solely on the given resume for the facts. Do not hallucinate facts.
Here is your resume {resume}
Here is the "Job escription" query {question}
"""

prompt = ChatPromptTemplate.from_template(template)

chain = prompt | model

while True:
    print("\n\n-----------------------------------------------")
    question = input("Ask your question (q to quit): ")
    print("\n\n")
    if question == "q":
        break

    start_time = time.time()

    resume = retriever.invoke(question)
    result = chain.invoke({"resume": resume, "question": question})
    end_time = time.time()
    print(result)

    print(model.model)
    print(f"\nResponse time: {end_time - start_time:.2f} seconds")
