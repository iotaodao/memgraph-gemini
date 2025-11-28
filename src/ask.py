import sys
from query import search

if __name__ == "__main__":
    question = sys.argv[1] if len(sys.argv) > 1 else "Summarize the content of the uploaded documents."
    search(question)
