from retriever import retrieve

def chatbot():
    print("Simple AI Chatbot (type 'exit' to quit)")

    while True:
        question = input("\nYou: ")

        if question.lower() == "exit":
            print("Chatbot: Goodbye!")
            break

        contexts = retrieve(question)

        print("\nChatbot (retrieved context):")
        for c in contexts:
            print("-", c.strip())

        print("\nChatbot (final answer):")
        answer = contexts[0].strip()
        print("Based on the available data, here is the answer:\n")
        print(answer)

if __name__ == "__main__":
    chatbot()