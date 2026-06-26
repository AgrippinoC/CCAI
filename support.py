from dotenv import load_dotenv

import os, random, calendar, datetime, time, json, re
from typing import TypedDict, List, Dict, Any, Literal, Annotated
from pydantic import BaseModel, Field

from inferenz import inferenza
from kg import query_graph_rag, query_graph_plan, get_recent_topics, suggest_next_topics, update_post_knowledge_graph, save_editorial_plan, check_duplicate_claims

from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_mistralai import ChatMistralAI
from langchain_community.tools import DuckDuckGoSearchRun

load_dotenv()

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "langchain-academy"

# Setup LLM
llm = ChatMistralAI(model="mistral-small-latest", temperature=0)
search_engine = DuckDuckGoSearchRun()

#lista dei tool
@tool
def soldium_imperium()->str:
    """Genera un numero casuale tra 0 e 29, ivi compresi
        Lo passi poi al classificatore che darà l'img corrispondente"""
    print("Chiamata al ViT\n")
    numb = random.randint(0,29)
    imp = inferenza(numb)
    #print(f"imperator {numb}_class = {imp}")
    return imp
@tool
def week_today()->str:
    """Restituisce il giorno della settimana corrente"""
    _today = datetime.date.today()
    week = calendar.day_name[_today.weekday()]
    return f"{week}"
@tool
def query_knowledge_graph(query: str)->str:
    """Interroga il Knowledge Graph per le informazioni storiche dai documenti archiviati"""
    print("query_knowledge_graph")
    return query_graph_rag(query)
@tool
def suggest_topics()->str:
    """Suggerisci topic non trattati recentemente e non presenti nello scheduling"""
    print("suggest_topics")
    return str( suggest_next_topics())
@tool
def search_web(query: str) -> str:
    """Cerca info su internet per verificare i fatti da siti fidati"""
    print("Avviata anche la ricerca web")
    trusted_sites = ["worldhistory.org", "it.wikipedia.org", "treccani.it", "archeoroma.it", "romanoimpero.com"]
    filter = " OR ".join([f"site:{site}" for site in trusted_sites])
    _query = f"{query} ({filter})"
    return search_engine.run(_query)
@tool
def fact_checker(query:str)->str:
    """Recupera fonti dal web e dal KG per verificare una affermazione"""
    web=search_web.invoke(query)
    kg=query_graph_rag(query)
    print(f"fact_checker:\nWEB RESULTS: {web}\nKG RESULTS: {kg}")
    return f"""
            WEB RESULTS: {web}
            KG RESULTS: {kg}
            """

tools=[week_today, query_knowledge_graph, suggest_topics, search_web, fact_checker, soldium_imperium]
llm_with_tools = llm.bind_tools(tools, tool_choice="auto")

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

#nodi di pianificazione
class EditorialPlan(BaseModel):
    topic: str = Field(description="Il nome del topic selezionato per oggi, oppure stringa vuota se non c'è")
    justification: str = Field(description="La giustificazione della scelta")
    content: str = Field(description="L'intero piano generato ed eventuali dettagli")
def is_publish_day():
    today = datetime.date.today()
    p = today.weekday()
    return today.weekday() in [0, 3, 5]
def extract_today_topic(plan_text: str, today: datetime.date):
    weekday_map = {0: "Lunedì", 3: "Giovedì", 5: "Sabato"}
    giorno = weekday_map.get(today.weekday())
    if not giorno:
        return None
    if isinstance(plan_text, list):
       plan_text = "\n".join(plan_text)
       for line in plan_text.split("\n"):
        if line.strip().startswith(giorno):
            return line.split(":", 1)[1].strip()
    return None
def planner_node(state: AgentState):

    _today = datetime.date.today()
    anno, settimana, giorno = _today.isocalendar()
    _key = f"{settimana}_{anno}"
    trace = list(state.get("reasoning_trace", []))

    plan_vecchio = query_graph_plan(_key)
    if plan_vecchio:
        print(f"\nPiano editoriale numero {_key} trovato")
        trace.append(f"Thought: Recuperato piano editoriale corrente")
        if not is_publish_day():
            print("\nOggi non è giorno di pubblicazione\n")
            trace.append("Thought: Oggi non è un giorno di pubblicazione.")
            return {
                "planning_info": {"plan": plan_vecchio},
                "selected_topic": None,
                "plan_review_status": "approved",
                "reasoning_trace": trace
            }

        today_topic = extract_today_topic(plan_vecchio, _today)
        trace.append(f"Thought: Topic di oggi è {today_topic}")
        print(f"\nIl Topic di oggi è {today_topic}")
        return {
            "planning_info": {"plan": plan_vecchio},
            "selected_topic": today_topic,
            "plan_review_status": "approved",
            "reasoning_trace": trace
        }

    print("Nessun piano trovato nel KG. Generazione nuovo piano.\n")

    recent = get_recent_topics()
    candidates = suggest_next_topics()
    user_feedback = ""

    if state.get("messages"):
        
        last_msg = state["messages"][-1]
        if (isinstance(last_msg, HumanMessage) and "Modifica" in last_msg.content):
            user_feedback = f""" 
                {last_msg.content}
                Genera dunque un nuovo piano editoriale con quello IDENTICO stile.
                Modifica seguendo TASSATIVAMENTE le opinioni scritte alla fine del piano.
                Seleziona poi il TOPIC di oggi nel campo 'topic'.
                Inserisci l'intera programmazione settimanale nel campo 'content'.
            """

    plan_prompt = f"""Sei un editor di un blog sull'Impero Romano.
        Si pubblicano BREVI POST e SOLO 3 volte a settimana tenedo conto dei seguenti:
            TOPIC RECENTI:{recent}
            TOPIC SUGGERITI:{candidates}
        Genera dunque un piano editoriale settimanale che:
            eviti ripetizioni
            copra eventuali argomenti mancanti
            giustifichi l'ordine
        La tipologia deve essere del tipo
            [Giorno]: [Topic]
            Breve descrizione in 1 riga
        Assegna i topic del post esplicitamente:
            Lunedì:
            Giovedì:
            Sabato:
        Restituisci poi:
            topic = il topic previsto per il prossimo giorno utile di pubblicazione
            content = piano completo della settimana
    """
    structured_llm = llm.with_structured_output(EditorialPlan)
    time.sleep(3)
    if user_feedback:
        plan_res = structured_llm.invoke(user_feedback)
    else:
        plan_res = structured_llm.invoke(plan_prompt)
    
    today_topic = extract_today_topic(plan_res, _today)
    selected_topic = plan_res.topic if plan_res.topic else today_topic

    if not is_publish_day():
        print("\nPiano generato, ma oggi non è giorno di pubblicazione.\n")
        trace.append("Thought: Generato nuovo pianoeditoriale. Oggi non è un giorno di pubblicazione.")

        return {
            "recent_topics": recent, "topic_candidates": candidates, "selected_topic": selected_topic,
            "planning_info": {"plan": plan_res.content, "week_key": _key},
            "reasoning_trace": trace,
            "plan_review_status": "pending"
        }
    
    trace.append(f"Thought: Generato nuovo piano editoriale. Topic scelto: {selected_topic}")
    return {
        "recent_topics": recent, "topic_candidates": candidates, "selected_topic": selected_topic,
        "planning_info": {"plan": plan_res.content, "week_key": _key},
        "reasoning_trace": trace,
        "plan_review_status": "pending"
    }

#nodo assistente per creare post
def extract_claims(post: str) -> list[str]:
    "Nodo che si occupa di creare i claim"
    p = f"""
    Dato questo testo, estrai le principali affermazioni che contiene.
    Segui queste regole:
    - una BREVE frase (15 parole) per claim
    - MASSIMO 3 claim
    - formato JSON array

    TESTO: {post}
    """
    time.sleep(2)
    clm = llm.invoke(p)
    txt = clm.content
    txt = re.sub(r"```json|```", "", txt).strip()
    claims = json.loads(txt)
    return claims
def claim_validator_node(state: AgentState):
    """Nodo che controlla se i claim siano duplicati"""
    if state.get("claims"):
        is_duplicate = check_duplicate_claims(state["claims"], threshold=0.95)
        if is_duplicate:
            trace = list(state.get("reasoning_trace", []))
            trace.append("Thought: Il post contiene informazioni già trattate in precedenza. Ritrattazione.")
            return {
                "reasoning_trace": trace,
                "messages": [HumanMessage(content="Il post che hai generato contiene argomenti simili a post passati. Riscrivilo cambiando focus o dettagli.")]
            }

    return state
def assistant_node(state:AgentState):
    """Assistente che si occupa di usar i tool e scrivere i post"""
    messages_for_llm = list(state.get("messages", []))
    
    x = f"""
    Sei un blogger esperto di un museo sulla Roma Antica. Oggi si parla di "{state['selected_topic']}".
    Il tuo obiettivo è scrivere un breve articolo di MASSIMO 10 righe su questo tema.
    
    Regole operative obbligatorie:
    1. Usa prevalentemente il tool `query_knowledge_graph` per ottenere informazioni e contesto dal KG.
    2. Se mancano informazioni o vuoi verificare i fatti, usa il tool `fact_checker` o `search_web`.
    3. Usare MASSIMO DUE VOLTE lo stesso tool per ricavare informazioni.
    4. Non inventare dati. Inserisci nel testo esplicitamente le fonti usate e fornite dai tool (es. [KG-RAG], [Web-Search:solo il nome del sito]).
    5. Nel testo aggiungi alla fine una frase tipo 'Oggi è [], il museo ti aspetta dalle 9 alle 21.' Con il tool `week_today` hai il contesto del giorno. 
    6. Quando hai raccolto abbastanza materiale, scrivi il post e NON invocare altri tool.
    """
    #5. Formula poi i tuoi passaggi nello schema: Thought: <motivazione sul tool usato> Action: <tool_name> Observation: <risultato>

    if state['selected_topic'] == "Monete":
        
        y = f"""
        7. REQUISITO PRIORITARIO: Prima di usare gli altri tool, DEVI USARE il tool `soldium_imperium` per determinare quale moneta/imperatore analizzare
        8. SOLO DOPO aver ottenuto il nome dell'imperatore, puoi usare gli altri tool `query_knowledge_graph` o `fact_checker`.
         
        L'articolo sarà su quell'Imperatore e sulle monete nella sua epoca.
            
        La struttura dell'articolo è del tipo:'Nel post di oggi parleremo di NOME_IMPERATORE: + BREVE BIOGRAFIA
        Nel nostro museo è presente una moneta coniata durante il suo regno: + MONETE NEL SUO REGNO ASPETTI ECONOMICI'
        """
        x = x + y
        print("if monete print\n")
    
    instruction = SystemMessage(content = x)
    messages_for_llm = [instruction] + messages_for_llm
    time.sleep(3)
    ass_res = llm_with_tools.invoke(messages_for_llm)

    trace = state["reasoning_trace"]
    trace.append(f"""
        Thought: Genero articolo
        Action: Utilizzo di LLM + TOOLS
        Observation: post generato
        """
    )
    
    if not ass_res.tool_calls:
        state["reasoning_trace"].append("Thought: Post pronto per la revisione.")

    output = {"messages": [ass_res]}
    if not ass_res.tool_calls:
        output["generated_post"] = ass_res.content
        output["claims"] = extract_claims(ass_res.content)
    else:
        pass 
    return output

#nodo salvo output da tool
def save_tool_output_node(state: AgentState):
    """Nodo che salva il trace e le source"""
    last = state["messages"][-1]
    
    current_tool_outputs = state.get("tool_outputs", {}) or {}
    updated_tool_outputs = dict(current_tool_outputs)
    new_trace = list(state.get("reasoning_trace", []))
    sources = list(state.get("sources", []))

    if isinstance(last, ToolMessage):
        updated_tool_outputs[last.name] = last.content
        new_trace.append(f"Thought: Ricevuto output dal tool {last.name}")
        
        if last.name == "query_knowledge_graph":
            sources.append("KG-RAG")
        elif last.name == "search_web" or last.name == "fact_checker":
            sources.append("Web-Search")
        else:
            sources.append(last.name)

    return {
        "reasoning_trace": new_trace, 
        "tool_outputs": updated_tool_outputs,
        "sources": sources
    }

#nodi di routing
def planner_router(state):
    if state.get("plan_review_status") == "pending":
        return "human_plan_review"
    if state["selected_topic"] is None:
        return "__end__"
    if state.get("plan_review_status") == "approved":
        return "assistant"
    return "human_plan_review"
def validator_router(state: AgentState):
    if state.get("reasoning_trace") and "Ritrattazione" in state["reasoning_trace"][-1]:
        return "assistant"
    return "human_post_review"

#nodi tipo Human-in-the-loop
def hitl_plan_node(state: AgentState) -> Command[Literal["planner", "assistant"]]:
    """Questo nodo ferma il grafo per verificare il Piano Editoriale.
        Riprende se l'utente invia la sua decisione.
    """
    _plan = state["planning_info"].get("plan","Nessun piano generato.")
    printplan = _plan.replace("**", "")
    
    print("\nECCO IL PIANO EDITORIALE")
    print(printplan)

    user_feedback = interrupt({
        "action": "Richiesta approvazione piano editoriale",
        "plan_to_review": _plan
    })

    decision = user_feedback.get("decision")
    note = user_feedback.get("text", "")
    
    if decision == "approve":
        print("Ok, il Piano Editoriale è approvato\n")
        save_editorial_plan(
            week_key=state["planning_info"]["week_key"],
            plan_text=state["planning_info"]["plan"]
        )
        is_post = "assistant" if is_publish_day() else "__end__"
        return Command(
            update={"plan_review_status": "approved",
                "reasoning_trace": state["reasoning_trace"] + ["User: Piano Approvato."]
            },
            goto= is_post
        )
    else:
        print("Piano Editoriale rifiutato, avviare ri-pianificazione\n")
        return Command(
            update={
                "plan_review_status": "rejected",
                "messages": [HumanMessage(content=f"{printplan}.\n Serve una Modifica con: {note}")],
                "reasoning_trace": state["reasoning_trace"] + [f"User: Piano Rifiutato. Note: {note}"]
            },
            goto="planner"
        )

def hitl_node(state: AgentState) -> Command[Literal["planner", "update_kg"]]:

    """Questo nodo ferma il grafo per verifciare il Post.
    Riprende solo quando l'utente invia la sua decisione.
    """
    print("\nECCO IL POST GENERATO")
    test = state.get("generated_post", "Nessun post generato.")
    printtest = test.replace("**", "")
    print(printtest)

    user_feedback = interrupt({
            "action": "Richiesta approvazione post",
            "post_to_review": state["generated_post"]
        }
    )
    
    decision = user_feedback.get("decision")
    nuovo_testo = user_feedback.get("text", "")

    if decision == "approve":
        print("Post Approvato")
        return Command(
            update={
                "post_review_status": "approved",
                "reasoning_trace": state["reasoning_trace"] + ["User: Approvato."]
            },
            goto="update_kg"
        )
    elif decision == "modify":
        print("Post Modificato")
        return Command(
            update={
                "post_review_status": "approved",
                "generated_post": nuovo_testo,
                "reasoning_trace": state["reasoning_trace"] + ["User: Modificato e Approvato."]
            },
            goto="update_kg"
        )
    elif decision == "reject":
        print("Post Rifiutato")
        return Command(
            update={
                "post_review_status": "rejected",
                "messages": [HumanMessage(content=f"{printtest}\n Modifica tenendo conto del feedback: {nuovo_testo}")],
                "reasoning_trace": state["reasoning_trace"] + ["User: Rifiutato. Richiesta rigenerazione."]
            },
            goto="assistant"
        )

#update kg
def update_kg_node(state):

    print(f"UPDATE KG ESEGUITO")
    print("CLAIMS CHE ENTRANO NEL KG:", state["claims"])
    print("SOURCES CHE ENTRANO NEL KG:", state["sources"])
    time.sleep(6)
    update_post_knowledge_graph(
        post_text= state["generated_post"],
        topic= state["selected_topic"],
        sources= state["sources"],
        claims= state["claims"]
    )
    return state