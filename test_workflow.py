import os
from dotenv import load_dotenv
from workflow.workflow import Workflow

# Load environment variables
load_dotenv()

if __name__ == "__main__":
    # Create and compile the workflow
    app = Workflow().define_workflow()
    
    # Test question
    question = "What is prompt engineering?"
    
    print(f"\n{'='*60}")
    print(f"Question: {question}")
    print(f"{'='*60}\n")
    
    # Invoke the workflow
    result = app.invoke({"question": question})
    
    # Print the results
    print(f"\n{'='*60}")
    print("Workflow Results:")
    print(f"{'='*60}")
    print(f"\nQuestion: {result.get('question', 'N/A')}")
    print(f"\nGeneration:\n{result.get('generation', 'N/A')}")
    print(f"\nVectorstore Documents Retrieved: {len(result.get('documents', [])) - len(result.get('web_search_docs', []))}")
    print(f"Web Search Documents Retrieved: {len(result.get('web_search_docs', []))}")
    print(f"Total Documents: {len(result.get('documents', []))}")
    print(f"Web Search Used: {result.get('web_search', 'N/A')}")
    print(f"\n{'='*60}\n")
