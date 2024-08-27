import os

from crewai import Agent, Task, Crew
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
# code_interpreter=CodeInterpreterTool()
coding_agent = Agent(
    role = "Senior Python Program",
    goal="Craft syntactically correct python code following object oriented principles",
    backstory="You are a senior python programmer in Meta and expert in developing Python code to drive extreme "
              "automation",
    allow_code_execution=True)

coding_task = Task(description="""Create and execute a python code to convert EBCDIC to ASCII""",
                   agent=coding_agent,
                   # tools=[code_interpreter],
                   expected_output="The result of execution of code")

crew = Crew(
  agents=[coding_agent],
  tasks=[coding_task],
  verbose=2, # You can set it to 1 or 2 to different logging levels
)

# Get your crew to work!
result = crew.kickoff()

print("######################")
print(result)