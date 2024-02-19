"""Simple showcase of the contrast reading vectors.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer

from mulsi import RepresentationReader, TdTokenizer
from mulsi.wrapper import LlmWrapper

####################
# HYPERPARAMETERS
####################
model_name = "gpt2"
pros_inputs = "I love this codebase"
cons_inputs = "I hate this codebase"
####################

love_reader = RepresentationReader.from_name(
    "contrast", pros_inputs=pros_inputs, cons_inputs=cons_inputs
)
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
td_tokenizer = TdTokenizer(tokenizer)

wrapper = LlmWrapper(model=model, tokenizer=td_tokenizer)
love_reading_vector = love_reader.compute_reading_vector(wrapper=wrapper)

sentences = [
    "I love this codebase",
    "I hate this codebase",
    "I am neutral about this codebase",
    "I like eating ice cream",
    "When is the next train?",
]
for sentence in sentences:
    print(
        sentence,
        love_reader.read(
            wrapper=wrapper,
            inputs=sentence,
            reading_vector=love_reading_vector,
        ),
    )
