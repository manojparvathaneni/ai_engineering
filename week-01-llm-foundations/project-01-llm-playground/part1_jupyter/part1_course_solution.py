"""
LLM Playground - Course Solution (Minimal)
==========================================

This is the quick ~3-5 line solution to complete the course assignment.
Copy this into your Jupyter notebook to fulfill the project requirements.

Prerequisites:
    pip install transformers torch ipywidgets
    
Note: First time running will download models (~500MB for GPT-2, ~1.5GB for Qwen)
"""

import ipywidgets as widgets
from IPython.display import display, Markdown, clear_output
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# =============================================================================
# STEP 1: Load models and tokenizers
# =============================================================================

print("Loading models... (this may take a minute on first run)")

# Load GPT-2 (small, fast, good for demos)
gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
gpt2_model = AutoModelForCausalLM.from_pretrained("gpt2")

# Load Qwen (more capable, instruction-tuned)
# Using the small version for faster loading
qwen_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
qwen_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B")

# Store in dict for easy access
models = {
    "GPT-2": (gpt2_model, gpt2_tokenizer),
    "Qwen-0.5B": (qwen_model, qwen_tokenizer)
}

print("✓ Models loaded!")


# =============================================================================
# STEP 2: Define helper function for text generation
# =============================================================================

def generate_text(prompt, model_name, strategy, temperature, max_tokens=50):
    """Generate text using specified model and decoding strategy."""
    
    model, tokenizer = models[model_name]
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Set up generation parameters based on strategy
    gen_kwargs = {
        "max_new_tokens": max_tokens,
        "pad_token_id": tokenizer.eos_token_id,
    }
    
    if strategy == "Greedy":
        gen_kwargs["do_sample"] = False
    elif strategy == "Top-k":
        gen_kwargs["do_sample"] = True
        gen_kwargs["top_k"] = 50
        gen_kwargs["temperature"] = temperature
    elif strategy == "Top-p":
        gen_kwargs["do_sample"] = True
        gen_kwargs["top_p"] = 0.9
        gen_kwargs["temperature"] = temperature
    elif strategy == "Temperature":
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
    
    # Decode and return
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# =============================================================================
# STEP 3: Create interactive UI elements
# =============================================================================

# Prompt input
prompt_input = widgets.Textarea(
    value="The future of artificial intelligence is",
    placeholder="Enter your prompt here...",
    description="Prompt:",
    layout=widgets.Layout(width="100%", height="80px")
)

# Model selector
model_selector = widgets.Dropdown(
    options=["GPT-2", "Qwen-0.5B"],
    value="GPT-2",
    description="Model:"
)

# Strategy selector
strategy_selector = widgets.Dropdown(
    options=["Greedy", "Top-k", "Top-p", "Temperature"],
    value="Top-p",
    description="Strategy:"
)

# Temperature slider
temperature_slider = widgets.FloatSlider(
    value=0.7,
    min=0.1,
    max=2.0,
    step=0.1,
    description="Temperature:",
    readout_format=".1f"
)

# Max tokens slider
max_tokens_slider = widgets.IntSlider(
    value=50,
    min=10,
    max=200,
    step=10,
    description="Max Tokens:"
)

# Output area
output_area = widgets.Output()


# =============================================================================
# STEP 4: Add generate button
# =============================================================================

generate_button = widgets.Button(
    description="🚀 Generate",
    button_style="primary",
    layout=widgets.Layout(width="150px")
)


# =============================================================================
# STEP 5: Define button behavior
# =============================================================================

def on_generate_clicked(b):
    """Handle generate button click."""
    with output_area:
        clear_output()
        print("Generating...")
        
        result = generate_text(
            prompt=prompt_input.value,
            model_name=model_selector.value,
            strategy=strategy_selector.value,
            temperature=temperature_slider.value,
            max_tokens=max_tokens_slider.value
        )
        
        clear_output()
        display(Markdown(f"**Generated Text:**\n\n{result}"))
        display(Markdown(f"\n---\n*Model: {model_selector.value} | "
                        f"Strategy: {strategy_selector.value} | "
                        f"Temperature: {temperature_slider.value}*"))

generate_button.on_click(on_generate_clicked)


# =============================================================================
# STEP 6: Display the full UI
# =============================================================================

# Layout the controls
controls = widgets.VBox([
    prompt_input,
    widgets.HBox([model_selector, strategy_selector]),
    widgets.HBox([temperature_slider, max_tokens_slider]),
    generate_button,
    widgets.HTML("<hr>"),
    output_area
])

display(Markdown("# 🎮 LLM Playground"))
display(controls)
