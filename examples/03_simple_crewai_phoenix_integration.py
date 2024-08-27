from crewai import Agent, Task, Crew, Process
from dotenv import load_dotenv
from openinference.instrumentation.crewai import CrewAIInstrumentor
from openinference.instrumentation.langchain import LangChainInstrumentor
from opentelemetry import trace as trace_api
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

load_dotenv()

tracer_provider = trace_sdk.TracerProvider()
span_exporter = OTLPSpanExporter("http://localhost:6006/v1/traces")
span_processor = SimpleSpanProcessor(span_exporter)
tracer_provider.add_span_processor(span_processor)
trace_api.set_tracer_provider(tracer_provider)

CrewAIInstrumentor().instrument(skip_dep_check=True)
LangChainInstrumentor().instrument()

teacher_manager = Agent(
    role='Manager',
    goal= 'Manages the work of physics and maths teacher',
    backstory= 'You are the manager of the school who helps in directing the question {question}'
               'to the right teacher based on the nature of the question.',
    verbose=True
)

math_teacher= Agent(
    role='Maths Teacher',
    goal='Answer questions related to Mathematics',
    backstory='Experienced Math teacher who helps students answer math questions',
    verbose=True
)

physics_teacher= Agent(
    role='Physics Teacher',
    goal='Answer questions related to Physics',
    backstory='Experienced Physics teacher who helps students answer math questions',
    verbose=True
)


maths_teacher_task = Task(
    description='Only answer questions related to mathematics. The question is {question}',
    expected_output='Correct answer to the math question',
    agent=math_teacher)

physics_teacher_task = Task(
    description='Only answer questions related to physics. The question is {question}',
    expected_output='Correct answer to the physics question',
    agent=physics_teacher)


teacher_crew = Crew(
    tasks=[maths_teacher_task,physics_teacher_task],  # Tasks to be delegated and executed under the manager's supervision
    agents=[math_teacher,physics_teacher],
    process=Process.hierarchical,  # Specifies the hierarchical management approach
    memory=True,  # Enable memory usage for enhanced task execution
    manager_agent=teacher_manager,  # Optional: explicitly set a specific agent as manager instead of the manager_llm
    planning=False,  # Enable planning feature for pre-execution strategy
)

answer = teacher_crew.kickoff({"question":"What is pythagorus theorem?"})

print(answer)