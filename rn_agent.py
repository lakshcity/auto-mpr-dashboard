from services.agent import pdf_agent

while True:
    q = input("Ask PDF: ")
    print(pdf_agent(q))
