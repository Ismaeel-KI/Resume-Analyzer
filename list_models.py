"""List available Gemini models"""
import os
import google.generativeai as genai

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    print("❌ GEMINI_API_KEY environment variable not set")
    exit(1)

genai.configure(api_key=api_key)

print("\n📋 Available models:\n")
for model in genai.list_models():
    print(f"  - {model.name}")
    if hasattr(model, 'supported_generation_methods'):
        print(f"    Methods: {model.supported_generation_methods}")
