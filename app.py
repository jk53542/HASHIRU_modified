from typing import List
import gradio as gr

import base64
from src.manager.manager import GeminiManager, Mode
from enum import Enum
import os
import base64
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Depends
from fastapi.responses import RedirectResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware
from authlib.integrations.starlette_client import OAuth
import requests
from src.manager.manager import GeminiManager
import argparse

# 1. Load environment --------------------------------------------------
load_dotenv()
AUTH0_DOMAIN = os.getenv("AUTH0_DOMAIN")
AUTH0_CLIENT_ID = os.getenv("AUTH0_CLIENT_ID")
AUTH0_CLIENT_SECRET = os.getenv("AUTH0_CLIENT_SECRET")
AUTH0_AUDIENCE = os.getenv("AUTH0_AUDIENCE")
SESSION_SECRET_KEY = os.getenv("SESSION_SECRET_KEY", "replace-me")

# 2. Auth0 client ------------------------------------------------------
oauth = OAuth()
oauth.register(
    "auth0",
    client_id=AUTH0_CLIENT_ID,
    client_secret=AUTH0_CLIENT_SECRET,
    client_kwargs={"scope": "openid profile email"},
    server_metadata_url=f"https://{AUTH0_DOMAIN}/.well-known/openid-configuration",
)

# 3. FastAPI app -------------------------------------------------------
app = FastAPI()

# Create static directory if it doesn't exist
os.makedirs("static/fonts/ui-sans-serif", exist_ok=True)
os.makedirs("static/fonts/system-ui", exist_ok=True)

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Add session middleware (no auth requirement)
app.add_middleware(
    SessionMiddleware,
    secret_key=SESSION_SECRET_KEY,
    session_cookie="session",
    max_age=86400,
    same_site="lax",
)

# 4. Auth routes -------------------------------------------------------
# Dependency to get the current user


def get_user(request: Request):
    user = request.session.get('user')
    if user:
        return user['name']
    return None


@app.get('/')
def public(request: Request, user=Depends(get_user)):
    root_url = gr.route_utils.get_root_url(request, "/", None)
    if user:
        return RedirectResponse(url=f'{root_url}/hashiru/')
    else:
        return RedirectResponse(url=f'{root_url}/login-page')


@app.get("/login")
async def login(request: Request):
    print("Session cookie:", request.cookies.get("session"))
    print("Session data:", dict(request.session))
    root_url = gr.route_utils.get_root_url(request, "/login", None)
    redirect_uri = f"{root_url}/auth"
    return await oauth.auth0.authorize_redirect(request, redirect_uri, audience=AUTH0_AUDIENCE, prompt="login")


@app.get("/auth")
async def auth(request: Request):
    try:
        token = await oauth.auth0.authorize_access_token(request)
        print("Token received:", token)
        request.session["user"] = token["userinfo"]
        return RedirectResponse("/hashiru")
    except Exception as e:
        print("Error during authentication:", str(e))
        return RedirectResponse("/")


@app.get("/logout")
async def logout(request: Request):
    root_url = gr.route_utils.get_root_url(request, "/logout", None)
    redirect_uri = f"{root_url}/post-logout"
    auth0_logout_url = (
        f"https://{AUTH0_DOMAIN}/v2/logout"
        f"?client_id={AUTH0_CLIENT_ID}"
        f"&returnTo={redirect_uri}"
    )
    return RedirectResponse(auth0_logout_url)


@app.get("/post-logout")
async def post_logout(request: Request):
    request.session.clear()
    return RedirectResponse("/")


@app.get("/manifest.json")
async def manifest():
    return JSONResponse({
        "name": "HASHIRU AI",
        "short_name": "HASHIRU",
        "icons": [],
        "start_url": "/",
        "display": "standalone"
    })


@app.get("/api/login-status")
async def api_login_status(request: Request):
    if "user" in request.session:
        user_info = request.session["user"]
        user_name = user_info.get("name", user_info.get("email", "User"))
        return {"status": f"Logged in: {user_name}"}
    else:
        return {"status": "Logged out"}


_header_html = f"""
<div style="
    display: flex;
    flex-direction: row;
    align-items: center;
    justify-content: flex-start;
">
  <img src="https://media.githubusercontent.com/media/HASHIRU-AI/HASHIRU/refs/heads/main/HASHIRU_LOGO.png" width="40" class="logo"/>
  <h1>
    HASHIRU AI
  </h1>
</div>
"""
css = """
.logo {
    margin-right: 20px;
}
"""


def _chat_preview_for_gradio(messages):
    """
    Gradio ChatInterface streams (message, history) and diffs consecutive states.
    We must yield on *every* step from the manager. Steps whose last entry is
    `function_call` or `tool` were previously skipped, which left large gaps in
    the stream and could crash Gradio's diff (utils.compare_objects IndexError).
    """
    if not messages:
        return {"role": "assistant", "content": ""}
    last = messages[-1]
    if last.get("role") == "assistant":
        return last
    for m in reversed(messages):
        if m.get("role") == "assistant":
            return m
    return {"role": "assistant", "content": ""}


def run_model(message, history):
    raw_text = ""
    if isinstance(message, dict) and message.get("text"):
        raw_text = message["text"]
    try:
        from src.manager.orchestration_trace import (
            begin_orchestration_user_turn,
            parse_trace_context_from_user_text,
        )

        clean_text, ctx = parse_trace_context_from_user_text(raw_text)
        excerpt = (clean_text or "").strip()
        if len(excerpt) > 8000:
            excerpt = excerpt[:8000]
        begin_orchestration_user_turn(
            ctx if ctx is not None else {},
            user_turn_excerpt=excerpt or None,
        )
    except Exception:
        clean_text = raw_text
        try:
            from src.manager.orchestration_trace import begin_orchestration_user_turn

            ex = (raw_text or "").strip()
            if len(ex) > 8000:
                ex = ex[:8000]
            begin_orchestration_user_turn({}, user_turn_excerpt=ex or None)
        except Exception:
            pass
    if "text" in message:
        if isinstance(message, dict) and clean_text.strip() != "":
            history.append({"role": "user", "content": clean_text})
    if 'files' in message:
        for file in message['files']:
            history.append({
                "role": "user",
                "content": (file,)
            })
    yield "", history
    for messages in model_manager.run(history):
        yield _chat_preview_for_gradio(messages), messages


with gr.Blocks() as login:
    with gr.Column():
        gr.HTML("<h1 style='font-size: 3em; text-align: center;'>Welcome to Hashiru AI</h1>")
        with gr.Column():
            gr.Markdown("## About Us")
            gr.Markdown("Hashiru AI is a startup developing an \"Expert Orchestrator AI\" envisioned as a dynamically adapting, budget-aware, and deeply personalized personal assistant designed to evolve with user needs and financial constraints. It addresses the inflexibility, limited capabilities, and potential costliness of current personal assistants by offering a solution that handles novel requests and complex tasks efficiently. A key strength of Hashiru AI is its commitment to personalization; the system continuously learns from user interactions, storing preferences and learning patterns to adapt its behavior and provide a truly tailored experience. Hashiru AI's product utilizes a dynamic multi-agent system where a \"CEO Agent\" manages specialized agents that are created for specific tasks and retired to optimize resources and costs. The system emphasizes cost-consciousness by prioritizing resource-efficient models, tracking expenses, and leveraging local models for a significant portion of complex tasks. Furthermore, it can autonomously create and integrate new API tools, delivering highly customized tools tailored to individual workflows and dynamically expanding its capabilities without manual intervention. The core value proposition of Hashiru AI lies in its profound adaptability, intelligence, and ability to deliver cost savings, all while maintaining a strong focus on user personalization, targeting power users such as professionals, researchers, and creatives who require more sophisticated and individually attuned AI assistance.")
        with gr.Column():
            gr.HTML("<img src='https://media.githubusercontent.com/media/HASHIRU-AI/HASHIRU/refs/heads/main/HASHIRU_LOGO.png' alt='Company Image' style='width:200px; height:auto; display:block; margin-left: auto; margin-right: auto;'>") # Replace with your image URL
            gr.HTML("<b><p style=\"text-align:center;\">NOTE: If you're an AgentX Judge, please refer to the Pitch Deck for login credentials.</p></b>")
        with gr.Row():
            gr.Markdown("")
            gr.HTML("<a href='/login' style='display: block; margin: 20px auto; padding: 10px 20px; background-color: #7e7932; color: white; text-align: center; text-decoration: none; border-radius: 5px;'>Login</a>")
            gr.Markdown("")

app = gr.mount_gradio_app(app, login, path="/login-page")

parser = argparse.ArgumentParser()
parser.add_argument('--no-auth', action='store_true')
args, unknown = parser.parse_known_args()
no_auth = args.no_auth

with gr.Blocks(title="HASHIRU AI", css=css, fill_width=True, fill_height=True) as demo:
    # Budget modes disabled by default so budget does not block agent creation or invocation.
    _default_modes = [m for m in Mode if m not in (Mode.ENABLE_RESOURCE_BUDGET, Mode.ENABLE_ECONOMY_BUDGET)]
    model_manager = GeminiManager(
        gemini_model="gemini-2.0-flash", modes=_default_modes)
    try:
        from src.manager.orchestration_trace import init_orchestration_trace_session

        init_orchestration_trace_session()
    except Exception:
        pass

    def update_model(modeIndexes: List[int]):
        modes = [Mode(i+1) for i in modeIndexes]
        print(f"Selected modes: {modes}")
        model_manager.set_modes(modes)

    with gr.Column(scale=1):
        with gr.Row(scale=0):
            with gr.Column(scale=0):
                gr.Markdown(_header_html)
                gr.Button("Logout", link="/logout")

            with gr.Column(scale=1):
                with gr.Accordion("Model Settings", open=False):
                    model_dropdown = gr.Dropdown(
                        choices=[mode.name for mode in Mode],
                        value=model_manager.get_current_modes,
                        interactive=True,
                        type="index",
                        multiselect=True,
                        label="Select Modes",
                    )

                    model_dropdown.change(
                        fn=update_model, inputs=model_dropdown, outputs=[])
        with gr.Row(scale=1):
            chatbot = gr.Chatbot(
                avatar_images=("https://media.githubusercontent.com/media/HASHIRU-AI/HASHIRU/refs/heads/main/HASHIRU_2.png", 
                               "https://media.githubusercontent.com/media/HASHIRU-AI/HASHIRU/refs/heads/main/HASHIRU.png"),
                type="messages",
                show_copy_button=True,
                editable="user",
                scale=1,
                render_markdown=True,
                placeholder="Type your message here...",
            )
            gr.ChatInterface(fn=run_model,
                             type="messages",
                             chatbot=chatbot,
                             additional_outputs=[chatbot],
                             save_history=True,
                             editable=True,
                             multimodal=True,
                             show_progress="full")

app = gr.mount_gradio_app(app, demo, path="/hashiru", auth_dependency=get_user, ssr_mode=False,)

if __name__ == "__main__":
    import uvicorn

    if no_auth:
        demo.launch(favicon_path="favicon.ico")
    else:
        uvicorn.run(app)