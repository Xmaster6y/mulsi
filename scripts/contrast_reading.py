"""Simple showcase of the contrast reading vectors.

Run with:
```
poetry run python -m scripts.contrast_reading
```
"""

from transformers import AutoModelForCausalLM, AutoTokenizer

from mulsi import ContrastReader, LlmWrapper, TdTokenizer
from scripts.utils import viz

####################
# HYPERPARAMETERS
####################
model_name = "gpt2"
pros_inputs = "I love this codebase"
cons_inputs = "I hate this codebase"
####################

love_reader = ContrastReader(pros_inputs=pros_inputs, cons_inputs=cons_inputs)
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
td_tokenizer = TdTokenizer(tokenizer)

wrapper = LlmWrapper(model=model, tokenizer=td_tokenizer)
love_reading_vector = love_reader.compute_reading_vector(wrapper=wrapper)

sentences = [
    "I love this codebase",
    "I hate this codebase",
    "I like this codebase",
    "Doing XAI research is what I love",
    "My girlfriend loves me",
    "I am neutral about this codebase",
    "I like eating ice cream",
    "When is the next train?",
]
head_line = ("Sentence", "Cosine Similarity")
table = []
for sentence in sentences:
    cosim = love_reader.read(
        wrapper=wrapper,
        inputs=sentence,
        reading_vector=love_reading_vector,
    ).item()
    table.append((sentence, f"{cosim:.2f}"))
viz.table_print(headings=head_line, table=table)
