# import os
# from fastapi import FastAPI
# from pydantic import BaseModel
# from openai import OpenAI
# from langchain_chroma import Chroma
# from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings
# from dotenv import load_dotenv
# import uuid

# # ------------------ INITIAL SETUP ------------------
# load_dotenv()

# app = FastAPI()
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# embedding = OpenAIEmbeddings(model="text-embedding-3-small")

# # --------------- LOAD MULTIPLE KNOWLEDGE FILES ----------------

# print("📚 Loading company knowledge base from directory...")

# docs = []
# knowledge_dir = "./knowledge_base"   # 📁 put all PDFs here

# # Loop through files in directory
# for filename in os.listdir(knowledge_dir):
#     file_path = os.path.join(knowledge_dir, filename)

#     if filename.lower().endswith(".pdf"):
#         print(f"📄 Loading PDF: {filename}")
#         loader = PyPDFLoader(file_path)
#         docs.extend(loader.load())

#     elif filename.lower().endswith(".txt"):
#         print(f"📝 Loading TXT: {filename}")
#         loader = TextLoader(file_path, encoding="utf-8")
#         docs.extend(loader.load())

#     elif filename.lower().endswith(".csv"):
#         print(f"📊 Loading CSV: {filename}")
#         loader = CSVLoader(file_path)
#         docs.extend(loader.load())

# print(f"✅ Loaded {len(docs)} documents from directory.")

# # Split all documents into chunks
# splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# texts = splitter.split_documents(docs)

# # Create or load vector store
# kb_vectorstore = Chroma.from_documents(
#     texts, embedding, persist_directory="./reva_realestate_kb"
# )

# memory_vectorstore = Chroma(
#     persist_directory="./reva_chat_memory", embedding_function=embedding
# )

# # --------------- API MODELS ----------------
# class ChatRequest(BaseModel):
#     message: str
#     session_id: str = None
#     history: list = []

# class ChatResponse(BaseModel):
#     reply: str
#     session_id: str

# # --------------- CHAT ENDPOINT ----------------
# @app.post("/chat", response_model=ChatResponse)
# def chat(request: ChatRequest):
#     session_id = request.session_id or str(uuid.uuid4())

#     # Retrieve from knowledge base
#     kb_docs = kb_vectorstore.similarity_search(request.message, k=3)
#     kb_context = "\n\n".join([doc.page_content for doc in kb_docs])

#     # Retrieve past memory
#     past_memories = memory_vectorstore.similarity_search(request.message, k=3)
#     memory_context = "\n\n".join([doc.page_content for doc in past_memories])

#     system_prompt = (
#         """Role & Personality

#         You are REVA (Real Estate Virtual Assistant) — an intelligent, friendly, and professional AI sales assistant for a real estate company.
#         Your goal is to qualify leads, understand their needs, build trust, and guide them toward booking a property consultation or site visit.
#         You always sound human, polite, and confident. You use conversational tone and sales psychology subtly.

#         Primary Objectives

#         Engage the lead with a warm, natural tone that feels like chatting with a helpful agent.

#         Understand their property needs: type, location, budget, urgency, and purpose (investment, living, rental).

#         Qualify the lead using contextual reasoning and data-driven logic.

#         Score the lead internally (based on qualification rules) and pass the result to the CRM / sheet.

#         Book an appointment or forward to a human agent if the lead is highly qualified.

#         Knowledge Sources (RAG Context)

#         You have access to a knowledge base that contains:

#         Details about available properties (e.g., apartments, villas, plots, offices).

#         Locations covered (e.g., DHA, Bahria Town, Emaar, Gulberg, etc.).

#         Price ranges, payment plans, amenities.

#         FAQs on property buying, investment, and financing.

#         Company background, agent contact details, and office timings.

#         When answering questions, always prioritize knowledge base data first.
#         If something is missing, politely say:

#         “I’ll check that detail and get back to you shortly.”

#         Conversation Flow Logic
#         🟢 Stage 1: Greeting & Context

#         Start with a friendly welcome and identify their purpose:

#         “Hello 👋 I’m REVA from Somewhere. Are you looking to buy, rent, or invest in property today?”

#         🟡 Stage 2: Needs Discovery

#         Ask strategic, natural questions to gather:

#         Property type (house, apartment, plot, commercial)

#         Preferred location(s)

#         Budget range

#         Purchase timeframe

#         Purpose (personal use, investment, rental)

#         Contact preference (call or WhatsApp follow-up)

#         🟠 Stage 3: Qualification

#         Use the following internal logic:

#         Parameter	Example	Score Range
#         Budget Alignment	Matches project price range	+2
#         Urgency	Within 30 days	+3
#         Property Type Match	Matches available inventory	+1
#         Location Match	Covered in KB	+2
#         Purpose	Investment or ready-to-move	+1
#         Responsiveness	Engages actively	+1

#         ➡️ Lead Score = Sum of all matched parameters
#         Then classify as:

#         Hot Lead (7–10) → Ready for call/visit booking

#         Warm Lead (4–6) → Send more details & follow-up

#         Cold Lead (0–3) → Nurture with educational content

#         Tone & Language Guidelines

#         Use friendly and persuasive language.

#         Keep sentences short and conversational.

#         Maintain professionalism while sounding human.

#         Mirror user’s energy and formality.

#         Avoid robotic or overly salesy scripts.

#         Always acknowledge the user’s last message before asking the next question.

#         Example:

#         “That sounds great! Apartments in Bahria Town have been quite popular lately. May I know your budget range so I can suggest the best options?”

#         Behavior Rules

#         Always store & reuse contextual data (lead_name, location, budget, property_type, etc.) in memory.

#         If the user revisits later, recall their preferences:

#         “Welcome back, Ali! Last time you mentioned you were looking for a 3-bedroom apartment in DHA — still interested in that area?”

#         Never give legal, financial, or tax advice.

#         End every session by confirming the next step:

#         “Would you like me to book a free call with one of our property experts?”

#         “Can I send you a few property images and payment plans here on WhatsApp?”

#         Output Requirements

#         At the end of each conversation (or when prompted by the system), summarize:

#         {
#         "lead_id": "auto-generated",
#         "lead_name": "Ali Khan",
#         "phone": "+923001234567",
#         "intent": "Buying apartment in DHA",
#         "budget": "12M PKR",
#         "urgency": "Immediate",
#         "purpose": "Personal use",
#         "qualification_score": 8,
#         "lead_stage": "Hot",
#         "next_action": "Schedule site visit"
#         }

#         Example Tone

#         “Hi there! 👋 Thanks for reaching out. I’m REVA, your virtual property assistant.
#         Can I ask what kind of property you’re looking for — an apartment, a house, or something commercial?”

#         Fail-safe Handling

#         If the user goes off-topic:

#         “I’d love to help, but my expertise is in real estate listings, buying, and investment. Could you tell me what kind of property you’re interested in?”

#         If the user asks something unavailable:

#         “That’s a great question! I don’t have that detail in my data right now, but I can check with our team and get back to you."
#         """
#     )

#     messages = [
#         {"role": "system", "content": system_prompt},
#         {"role": "system", "content": f"Relevant company context:\n{kb_context}"},
#         {"role": "system", "content": f"Relevant conversation memory:\n{memory_context}"},
#         {"role": "user", "content": request.message},
#     ]

#     response = client.chat.completions.create(model="gpt-4o", messages=messages)
#     reply = response.choices[0].message.content.strip()

#     # Save new memory
#     # memory_vectorstore.add_texts([f"User: {request.message}\nAssistant: {reply}"])
#     if "I'm REVA" not in reply:  
#         memory_vectorstore.add_texts([f"{request.message}"])

#     return ChatResponse(reply=reply, session_id=session_id)

# # --------------- SERVER RUNNER ----------------
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("real_estate:app", host="0.0.0.0", port=8000, reload=True)




import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import uuid

from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

# ✅ Load environment variables
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# ✅ Initialize FastAPI
app = FastAPI()

# ✅ Gemini Embeddings for Vector DB
embedding = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=GOOGLE_API_KEY
)

# ✅ Gemini Chat Model (LLM)
chat_model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.3
)

# ✅ Load Knowledge Base Files
knowledge_dir = "./knowledge_base"
docs = []

# Check if knowledge_base directory exists
if os.path.exists(knowledge_dir):
    print("📚 Loading knowledge base files...")
    for filename in os.listdir(knowledge_dir):
        path = os.path.join(knowledge_dir, filename)
        
        # Skip if it's a directory
        if not os.path.isfile(path):
            continue

        try:
            if filename.lower().endswith(".pdf"):
                loader = PyPDFLoader(path)
                docs.extend(loader.load())
                print(f"📄 Loaded PDF: {filename}")

            elif filename.lower().endswith(".txt"):
                loader = TextLoader(path, encoding="utf-8")
                docs.extend(loader.load())
                print(f"📝 Loaded TXT: {filename}")

            elif filename.lower().endswith(".csv"):
                loader = CSVLoader(path)
                docs.extend(loader.load())
                print(f"📊 Loaded CSV: {filename}")
        except Exception as e:
            print(f"❌ Error loading {filename}: {str(e)}")
else:
    print(f"⚠️ Knowledge base directory '{knowledge_dir}' not found. Creating empty vector store.")

print(f"✅ Total docs loaded: {len(docs)}")

# ✅ Initialize vector stores
if docs:
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(docs)
    print(f"✅ Total chunks created: {len(texts)}")

    # Create or load knowledge base vector store
    kb_vectorstore = Chroma.from_documents(
        documents=texts,
        embedding=embedding,
        persist_directory="./reva_realestate_kb"
    )
else:
    # Create empty vector store if no documents
    kb_vectorstore = Chroma(
        persist_directory="./reva_realestate_kb",
        embedding_function=embedding
    )
    print("⚠️ No documents found. Using empty knowledge base.")

# Create or load memory vector store
memory_vectorstore = Chroma(
    persist_directory="./reva_chat_memory",
    embedding_function=embedding
)

# ✅ API Request/Response Models
class ChatRequest(BaseModel):
    message: str
    session_id: str = None
    history: list = []

class ChatResponse(BaseModel):
    reply: str
    session_id: str

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    session_id = request.session_id or str(uuid.uuid4())

    # Retrieve relevant context from knowledge base
    kb_docs = kb_vectorstore.similarity_search(request.message, k=3)
    kb_context = "\n\n".join(doc.page_content for doc in kb_docs) if kb_docs else "No relevant information found."

    # Retrieve relevant context from memory
    memory_docs = memory_vectorstore.similarity_search(request.message, k=3)
    memory_context = "\n\n".join(doc.page_content for doc in memory_docs) if memory_docs else "No previous conversation history."

    # Build prompt — NOTE the leading f to allow {} interpolation
    system_prompt = f"""Role & Personality
You are REVA (Real Estate Virtual Assistant) — an intelligent, friendly, and professional AI sales assistant for a real estate company.
Your goal is to qualify leads, understand their needs, build trust, and guide them toward booking a property consultation or site visit.
You always sound human, polite, and confident. You use conversational tone and sales psychology subtly.

Primary Objectives
Engage the lead with a warm, natural tone that feels like chatting with a helpful agent.
Understand their property needs: type, location, budget, urgency, and purpose (investment, living, rental).
Qualify the lead using contextual reasoning and data-driven logic. Score the lead internally and pass the result to the CRM / sheet.

Knowledge Base:
{kb_context}

Past Conversation Memory:
{memory_context}

User: {request.message}

Answer:
"""

    # Call the Gemini chat model (wrap in try/except for safety)
    try:
        response = chat_model.invoke(system_prompt)
    except Exception as e:
        # Log and return a 500 so client sees a helpful error
        print(f"❌ Model invocation error: {e}")
        raise HTTPException(status_code=500, detail="Model invocation failed")

    # Robust extraction of the reply (handles multiple possible response shapes)
    reply = ""
    try:
        # Case A: response has `.content` attribute
        if hasattr(response, "content"):
            content = response.content
            if isinstance(content, str):
                reply = content
            elif isinstance(content, list):
                # list of parts (dicts or strings)
                parts = []
                for part in content:
                    if isinstance(part, dict):
                        # common key names: 'text' or 'content' or 'output'
                        parts.append(part.get("text") or part.get("content") or "")
                    else:
                        parts.append(str(part))
                reply = "".join(parts)
            else:
                # fallback to str()
                reply = str(content)

        # Case B: response is dict-like
        elif isinstance(response, dict):
            # try typical keys
            reply = response.get("output", "") or response.get("content", "") or response.get("text", "")
            # sometimes nested candidates
            if not reply:
                candidates = response.get("candidates") or response.get("choices")
                if isinstance(candidates, list) and candidates:
                    first = candidates[0]
                    if isinstance(first, dict):
                        reply = first.get("content") or first.get("text") or ""
                    else:
                        reply = str(first)

        # Case C: direct string
        else:
            reply = str(response)
    except Exception as e:
        print(f"⚠️ Error extracting reply: {e}")
        reply = ""

    reply = (reply or "").strip()

    # Store conversation in memory (avoid storing trivial bot intros)
    try:
        if reply and "I'm REVA" not in reply and "I’m REVA" not in reply:
            memory_text = f"User: {request.message}\nREVA: {reply}"
            memory_vectorstore.add_texts([memory_text])
    except Exception as e:
        print(f"⚠️ Error storing memory: {str(e)}")

    return ChatResponse(reply=reply or "Sorry, I couldn't generate a reply right now.", session_id=session_id)