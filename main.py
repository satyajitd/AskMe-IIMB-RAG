from dotenv import load_dotenv
from workflow.workflow import Workflow

# Load environment variables
load_dotenv()

# Create the workflow app
app = Workflow().define_workflow()

