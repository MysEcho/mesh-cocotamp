import os
from openai import OpenAI
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
import numpy as np
import tiktoken

# Generate the prompt
ROOM_PROMPT = """You will be given possible locations of {object} in a house.
Using common sense that any human would have, predict the location of the {object}.
The following options are possible locations: 
{options_text}
Return with {letters_joined} which represents the room, and nothing else.
MAKE SURE your output is one of the {len_letters} characters stated.
Please note that the provided options have been randomly shuffled, so it is essential to consider them fairly and without bias.
"""

# Generate the prompt
SURFACE_PROMPT = """You will be given possible locations of {object} in {room}.
Using common sense that any human would have, predict the location of the {object}.
The following options are possible locations: 
{options_text}
Return with {letters_joined} which represents the surface, and nothing else.
MAKE SURE your output is one of the {len_letters} characters stated.
Please note that the provided options have been randomly shuffled, so it is essential to consider them fairly and without bias.
"""

def get_completion(
    messages: list[dict[str, str]],
    model: str = "gpt-4o",
    max_tokens=1,
    temperature=0.7,
    stop=None,
    seed=123,
    tools=None,
    logprobs=None,
    top_logprobs=None,
    logit_bias=None) -> str:
    params = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stop": stop,
        "seed": seed,
        "logprobs": logprobs,
        "top_logprobs": top_logprobs,
        "logit_bias": logit_bias,
    }
    if tools:
        params["tools"] = tools
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    completion = client.chat.completions.create(**params)
    return completion

def get_gpt4_token_ids(input_string: str):
    # Load the GPT-4 tokenizer
    tokenizer = tiktoken.encoding_for_model("gpt-4o")
    # Get the token IDs
    token_ids = tokenizer.encode(input_string)
    return token_ids

def create_room_belief(target_objects: list, rooms: list[str], class_from_body: dict) -> dict:
    # Assign letters to each location
    # Generates ['A', 'B', 'C', ..., 'H']
    letters = [chr(i) for i in range(65, 65 + len(rooms))]
    rooms_class = [class_from_body[room] for room in rooms]
    options = [f"{letter}) {room}" for letter, room in zip(letters, rooms_class)]
    options_text = "\n".join(options)
    logit_bias = {}
    for letter in letters:
        token_ids = get_gpt4_token_ids(" "+letter)
        logit_bias[token_ids[0]] = 100
    initial_belief = {}
    for target_object in target_objects:
        string_target = class_from_body[target_object]
        API_RESPONSE = get_completion(
        [{"role": "user", "content": ROOM_PROMPT.format(object=string_target, 
                                                        options_text=options_text,
                                                        letters_joined=", ".join(letters),
                                                        len_letters=len(letters))}],
        model="gpt-4o",
        logprobs=True,
        top_logprobs=8,
        logit_bias=logit_bias)  
        top_logprobs = API_RESPONSE.choices[0].logprobs.content[0].top_logprobs
        
        obj_belief = {}
        for logprob in top_logprobs:
            if logprob.token in letters:
                obj_belief[rooms_class[letters.index(logprob.token)]] = max(np.exp(logprob.logprob), 0.01)
        for room in rooms_class:
            if room not in obj_belief:
                obj_belief[room] = 0.01
        initial_belief[string_target] = obj_belief
    return initial_belief

def create_surface_belief(target_objects: list, room: int, surfaces: list[int], class_from_body: dict) -> dict:
    # Assign letters to each location
    # Generates ['A', 'B', 'C', ..., 'H']
    letters = [chr(i) for i in range(65, 65 + len(surfaces))]
    surfaces_class = [class_from_body[surf] for surf in surfaces]
    
    llm_surfaces_class = []
    for surf in surfaces:
        # _, surface = class_from_body[surf].split("|")
        surface, _ = class_from_body[surf].split("|")
        location_class = surface + " in " + class_from_body[room]
        llm_surfaces_class.append(location_class)
    
    options = [f"{letter}) {location}" for letter, location in zip(letters, llm_surfaces_class)]
    options_text = "\n".join(options)
    logit_bias = {}
    for letter in letters:
        token_ids = get_gpt4_token_ids(" "+letter)
        logit_bias[token_ids[0]] = 100
    initial_belief = {}
    for target_object in target_objects:
        string_target = class_from_body[target_object]
        API_RESPONSE = get_completion(
        [{"role": "user", "content": SURFACE_PROMPT.format(object=string_target, 
                                                        room=class_from_body[room],
                                                        options_text=options_text,
                                                        letters_joined=", ".join(letters),
                                                        len_letters=len(letters))}],
        model="gpt-4o",
        logprobs=True,
        top_logprobs=16,
        logit_bias=logit_bias)
        top_logprobs = API_RESPONSE.choices[0].logprobs.content[0].top_logprobs
        
        obj_belief = {}
        for logprob in top_logprobs:
            if logprob.token in letters:
                obj_belief[surfaces_class[letters.index(logprob.token)]] = max(np.exp(logprob.logprob), 0.01)
        print(obj_belief)
        for surf in surfaces_class:
            if surf not in obj_belief:
                obj_belief[surf] = 0.01
        initial_belief[(string_target, class_from_body[room])] = obj_belief
    return initial_belief
