from search import search_prompt

def main():
    while True:
        question = input("\nPERGUNTA: ").strip()

        if not question.strip():
            break

        answer = search_prompt(question)

        print(f"RESPOSTA: {answer}")

if __name__ == "__main__":
    main()