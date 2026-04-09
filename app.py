import gradio as gr
from transformers import pipeline

# Load GPT-2 text generation pipeline
generator = pipeline("text-generation", model="gpt2")

def generate_text(prompt):
    result = generator(
        prompt,
        max_new_tokens=120,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        num_return_sequences=1,
        pad_token_id=generator.tokenizer.eos_token_id
    )
    return result[0]["generated_text"]

demo = gr.Interface(
    fn=generate_text,
    inputs=gr.Textbox(lines=3, placeholder="Enter your prompt here"),
    outputs=gr.Textbox(lines=10),
    title="Text Generation Model",
    description="Generate coherent paragraphs from user prompts using GPT-2."
)

if __name__ == "__main__":
    demo.launch()