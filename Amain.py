from dotenv import load_dotenv

import os, time

from typing import TypedDict, List, Dict, Any, Annotated
from support import planner_node, assistant_node, save_tool_output_node, hitl_plan_node, hitl_node, update_kg_node, planner_router, validator_router, claim_validator_node
from support import week_today, query_knowledge_graph, suggest_topics, search_web, fact_checker, soldium_imperium

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage

load_dotenv()
tools=[week_today, query_knowledge_graph, suggest_topics, search_web, fact_checker, soldium_imperium]

class AgentState(TypedDict):
    messages:Annotated[list, add_messages]
    reasoning_trace:List[str]
    tool_outputs:Dict[str,Any]
    planning_info:Dict[str,Any]
    generated_post:str
    kg_summary:str
    recent_topics:list
    topic_candidates:list
    selected_topic:str
    retrieved_context:str
    sources:list
    claims:list
    plan_review_status: str
    post_review_status: str

#creazione del grafo
builder=StateGraph(AgentState)

builder.add_node("planner", planner_node)
builder.add_node("assistant", assistant_node)
builder.add_node("tools", ToolNode(tools))
builder.add_node("save_tool_output", save_tool_output_node)
builder.add_node("human_plan_review", hitl_plan_node) 
builder.add_node("human_post_review", hitl_node)
builder.add_node("update_kg", update_kg_node)
builder.add_node("claim_validator", claim_validator_node)

builder.add_edge(START, "planner")
builder.add_conditional_edges("planner", planner_router, {"human_plan_review": "human_plan_review", "assistant": "assistant", "__end__": END})
builder.add_conditional_edges("assistant", tools_condition, {"tools": "tools", "__end__": "claim_validator"})
builder.add_edge("tools", "save_tool_output")
builder.add_edge("save_tool_output", "assistant")
builder.add_conditional_edges("claim_validator", validator_router,{"assistant": "assistant", "human_post_review": "human_post_review"})
builder.add_edge("update_kg", END)

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

output_dir = "./"
file_path = os.path.join(output_dir, "struttura_grafo_FUN.png")
os.makedirs(output_dir, exist_ok=True)
image_data = graph.get_graph(xray=True).draw_mermaid_png()
with open(file_path, "wb") as f:
    f.write(image_data)

config = {"configurable": {"thread_id": "blog_generation_1"}}

time.sleep(5)
print("BENVENUTO SU BLOG-MAKER (ROMA EDITON) MUSEO!\n")
msg = [HumanMessage(content="Avvia la pianificazione editoriale dei post per la settimana.")]

initial_state={
    "messages":msg,
    "reasoning_trace":[],
    "tool_outputs":{},
    "planning_info":{},
    "generated_post":"",
    "kg_summary":"",
    "recent_topics":[],
    "topic_candidates":[],
    "selected_topic":"",
    "retrieved_context":"",
    "sources":[],
    "claims":[],
    "plan_review_status": "pending",
    "post_review_status": "pending",
}

for event in graph.stream(initial_state, config=config, stream_mode="updates"):
    for node_name, data in event.items():
        print(f"nodo [{node_name}]")

while True:
    state_snapshot = graph.get_state(config)
    if not state_snapshot.tasks:
        print("\nIl lavoro quì è terminato, grazie e arrivederci.")
        break

    print("\nIl flusso si è fermato, ecco cosa era stato generato.")

    _decision = input("Cosa ne pensi? ")
    _text = input("Perchè?: ")
    scelta_utente = {"decision": _decision, "text": _text}
    
    last_event = None
    for event in graph.stream(Command(resume=scelta_utente), config, stream_mode="updates"):
        for node_name, data in event.items():
            print(f"nodo: [{node_name}]")

trax = input("Vuoi vedere il Trace finale?: ")
if trax == "si":
    final_state = graph.get_state(config).values
    print("\nREASONING TRACE FINALE\n")
    for t in final_state.get("reasoning_trace", []):
        print(t)
else:
    print("OK ciao")