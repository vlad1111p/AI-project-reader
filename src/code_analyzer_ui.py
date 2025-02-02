import json
import os
import tkinter as tk
from threading import Thread
from tkinter import filedialog, messagebox

from src.service.code_analyzer import analyze

CONFIG_FILE = "config.json"


def save_config():
    """Save the OpenAI API key and last-used model name to a config file."""
    config_data = {
        "openai_api_key": api_key_entry.get(),
        "model_name": model_name_entry.get()
    }
    with open(CONFIG_FILE, "w") as f:
        json.dump(config_data, f)


def load_config():
    """Load the OpenAI API key and last-used model name from a config file."""
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)
    return {}


def update_model_entry(*args):
    """Update the model name entry when a dropdown selection is made."""
    selected_model = model_name_var.get()
    model_name_entry.delete(0, tk.END)
    model_name_entry.insert(0, selected_model)


def run_analysis():
    query = query_entry.get()
    project_path = project_entry.get()
    language = language_var.get()
    model_type = model_type_var.get()
    model_name = model_name_entry.get()
    openai_key = api_key_entry.get()

    if not query or not project_path or not openai_key:
        messagebox.showerror("Error", "Please enter a query, project path, and OpenAI API key.")
        return

    os.environ["OPENAI_API_KEY"] = openai_key

    save_config()

    def analyze_and_display():
        try:
            result = analyze(query, project_path, language, model_type, model_name)
            if not isinstance(result, str):
                result = str(result)

            output_text.config(state=tk.NORMAL)
            output_text.delete("1.0", tk.END)
            output_text.insert(tk.END, result)
            output_text.config(state=tk.DISABLED)
        except Exception as e:
            messagebox.showerror("Error", f"Analysis failed: {str(e)}")

    Thread(target=analyze_and_display, daemon=True).start()


def select_project():
    folder = filedialog.askdirectory()
    project_entry.delete(0, tk.END)
    project_entry.insert(0, folder)


config = load_config()
saved_api_key = config.get("openai_api_key", "")
saved_model_name = config.get("model_name", "gpt-4o")

root = tk.Tk()
root.title("AI Code Analyzer")

tk.Label(root, text="Query:").grid(row=0, column=0, sticky="w")
query_entry = tk.Entry(root, width=50)
query_entry.grid(row=0, column=1, padx=5, pady=5)

tk.Label(root, text="Project Path:").grid(row=1, column=0, sticky="w")
project_entry = tk.Entry(root, width=50)
project_entry.grid(row=1, column=1, padx=5, pady=5)
tk.Button(root, text="Browse", command=select_project).grid(row=1, column=2, padx=5)

tk.Label(root, text="Language:").grid(row=2, column=0, sticky="w")
language_var = tk.StringVar(value="python")
language_dropdown = tk.OptionMenu(root, language_var, "python", "java")
language_dropdown.grid(row=2, column=1, padx=5, pady=5)

tk.Label(root, text="Model Type:").grid(row=3, column=0, sticky="w")
model_type_var = tk.StringVar(value="Chatgpt")
model_type_dropdown = tk.OptionMenu(root, model_type_var, "Chatgpt", "Ollama")
model_type_dropdown.grid(row=3, column=1, padx=5, pady=5)

tk.Label(root, text="Model Name:").grid(row=4, column=0, sticky="w")

model_name_var = tk.StringVar(value=saved_model_name)
model_name_dropdown = tk.OptionMenu(root, model_name_var, "gpt-4o", "gpt-4o-mini", "o1", "o3-mini")
model_name_dropdown.grid(row=4, column=1, padx=5, pady=5)

model_name_entry = tk.Entry(root, width=30)
model_name_entry.grid(row=4, column=2, padx=5, pady=5)
model_name_entry.insert(0, saved_model_name)

model_name_var.trace_add("write", update_model_entry)

tk.Label(root, text="OpenAI API Key:").grid(row=5, column=0, sticky="w")
api_key_entry = tk.Entry(root, width=50, show="*")
api_key_entry.grid(row=5, column=1, padx=5, pady=5)

if saved_api_key:
    api_key_entry.insert(0, saved_api_key)

tk.Button(root, text="Run Analysis", command=run_analysis).grid(row=6, column=1, pady=10)

output_text = tk.Text(root, height=10, width=60, state=tk.DISABLED)
output_text.grid(row=7, column=0, columnspan=3, padx=5, pady=5)

root.mainloop()
