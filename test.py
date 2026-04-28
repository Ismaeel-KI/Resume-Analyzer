import google.generativeai as genai

genai.configure(api_key="AIzaSyAwSr5gsgU7sQhbtL16HppfVJ8UrqdRclI")

for m in genai.list_models():
    print(m.name, m.supported_generation_methods)   