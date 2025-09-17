import os
from openai import OpenAI
from pydantic import BaseModel

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
import numpy as np
import pprint
import tiktoken

from examples.discrete_belief.dist import DDist
from llm_tamp.belief import BeliefState


SYS_PROMPT = """You will receive:
- Current belief about where an object might be located.
- observation_location: the location that was just inspected.
- visibility: how much of the location was visible. 0 means not visible at all, 1 means fully visible.
- result: whether the object was found there.
- co_detected: other objects found at the same location.

Based on this information and common sense, predict where the object is most likely to be now.

Choose the most likely location from the given options."""

USER_PROMPT = """Current belief about {object_name}: {belief_text}
observation_location: {obs_loc} 
visibility: {visibility:.1f}
result: {result}
co_detected: {co_detected_text}

Given this information, where is {object_name} most likely to be?
{options_text}

Return with {letters_joined} which represents the location, and nothing else.
MAKE SURE your output is one of the {len_letters} characters stated."""


def get_gpt4_token_ids(input_string: str):
    # Load the GPT-4 tokenizer
    tokenizer = tiktoken.encoding_for_model("gpt-4o")
    # Get the token IDs
    token_ids = tokenizer.encode(input_string)
    return token_ids


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
    logit_bias=None,
) -> str:
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


class LLMUpdater:
    def __init__(self, target_objs: list, class_from_body: dict, body_from_class: dict):
        self.target_objs = target_objs
        self.class_from_body = class_from_body
        self.body_from_class = body_from_class
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    def update_belief(
        self,
        belief: dict,
        obs_loc: str,
        found: bool,
        visibility: float,
        co_detected: list[str],
    ) -> dict:
        # Extract object name and locations from belief dict
        object_name = list(belief.keys())[0]
        locations = list(belief[object_name].keys())

        # Create multiple choice options
        letters = [chr(i) for i in range(65, 65 + len(locations))]
        options = [
            f"{letter}) {location}" for letter, location in zip(letters, locations)
        ]
        options_text = "\n".join(options)

        # Create belief text description
        belief_items = []
        for loc, prob in belief[object_name].items():
            belief_items.append(f"{loc}: {prob:.2f}")
        belief_text = ", ".join(belief_items)

        # Create co-detected text
        if co_detected:
            co_detected_text = (
                f"co_detected at {obs_loc}: {', '.join(co_detected)}"
            )
        else:
            co_detected_text = f"No other objects were detected at {obs_loc}"

        # Create result text
        result = "Found the object" if found else "Did not find the object"

        # Set up logit bias to favor letter tokens
        logit_bias = {}
        for letter in letters:
            token_ids = get_gpt4_token_ids(" " + letter)
            logit_bias[token_ids[0]] = 100

        # Make API call
        messages = [
            {"role": "system", "content": SYS_PROMPT},
            {
                "role": "user",
                "content": USER_PROMPT.format(
                    object_name=object_name,
                    belief_text=belief_text,
                    obs_loc=obs_loc,
                    result=result,
                    visibility=visibility,
                    co_detected_text=co_detected_text,
                    options_text=options_text,
                    letters_joined=", ".join(letters),
                    len_letters=len(letters),
                ),
            },
        ]

        response = get_completion(
            messages,
            model="gpt-4o",
            logprobs=True,
            top_logprobs=len(letters),
            logit_bias=logit_bias,
        )

        # Extract probabilities from logprobs
        top_logprobs = response.choices[0].logprobs.content[0].top_logprobs

        updated_probs = {}
        for logprob in top_logprobs:
            if logprob.token in letters:
                location = locations[letters.index(logprob.token)]
                updated_probs[location] = max(np.exp(logprob.logprob), 0.01)

        # Add small probability for locations not in top logprobs
        for location in locations:
            if location not in updated_probs:
                updated_probs[location] = 0.01

        # Normalize probabilities
        total = sum(updated_probs.values())
        for location in updated_probs:
            updated_probs[location] /= total

        print("--------------------------------")
        print(USER_PROMPT.format(
                    object_name=object_name,
                    belief_text=belief_text,
                    obs_loc=obs_loc,
                    result=result,
                    visibility=visibility,
                    co_detected_text=co_detected_text,
                    options_text=options_text,
                    letters_joined=", ".join(letters),
                    len_letters=len(letters),))
        print("--------------------------------")
        print(object_name, updated_probs)
        print("--------------------------------")

        return {object_name: updated_probs}

    def correlationUpdate(
        self,
        target_item: int,
        detected_items: set,
        target_loc: int,
        surf_visibility: float,
        room_visibility: float,
        known: set,
        p_fp: float,
        p_fn: float,
        state: BeliefState,
        obs: bool,
    ):
        """Adapter so that this updater can be dropped in anywhere a CorrelationalObsModel is
        currently used (e.g.
        llm_tamp.primitives.CoDetect).  The signature mirrors the one expected by
        that code path.

        Parameters
        ----------
        target_item : int
            PyBullet body‐id of the *query* object whose belief we are updating.
        detected_items : set[int]
            Set of body‐ids detected during the same observation.  Currently
            only used to supply `co_detected` names to the language model.
        target_loc : int
            PyBullet body‐id of the *surface* that was inspected.
        surf_visibility : float
            Fraction \(0–1] describing how much of the inspected surface was
            visible.
        room_visibility : float
            (Not used by this updater but preserved for interface
            compatibility.)
        known : set[int]
            Set of body‐ids whose true poses are already known.  (Not used.)
        p_fp, p_fn : float
            False-positive / false-negative rates.  (Ignored for LLM update.)
        state : BeliefState
            Current belief state which will be updated *in-place*.
        obs : bool
            True if the object was found on `target_loc`, False otherwise.
        """
        # ------------------------------------------------------------------
        # 1. Collect current belief over surfaces for *target_item*
        # ------------------------------------------------------------------
        room_body = state.get_room_body(target_loc)
        b_on = state.b_on[(target_item, room_body)]
        b_in = state.b_in[target_item]
        # Map room body → probability and room name
        curr_room_belief = {}
        for room_id in b_in.support():
            room_name = state.task.class_from_body[room_id]
            curr_room_belief[room_name] = b_in.prob(room_id)
        # Map surface body → probability and surface name
        curr_surf_belief = {}
        for surface_id in b_on.support():
            surf_name = state.task.class_from_body[surface_id]
            curr_surf_belief[surf_name] = b_on.prob(surface_id)
        # ------------------------------------------------------------------
        # 2. Query the LLM for an updated belief
        # ------------------------------------------------------------------
        object_name = state.task.class_from_body[target_item]
        bel_room_input = {object_name: curr_room_belief}
        bel_surf_input = {object_name: curr_surf_belief}
        obs_room_name = state.task.class_from_body[room_body]
        obs_surf_name = state.task.class_from_body[target_loc]
        co_detected_names = [
            state.task.class_from_body[obj]
            for obj in detected_items
            if obj != target_item
        ]

        room_updated = self.update_belief(
            bel_room_input, obs_room_name, obs, room_visibility, co_detected_names
        )
        room_updated_probs = room_updated[object_name]
        surf_updated = self.update_belief(
            bel_surf_input, obs_surf_name, obs, surf_visibility, co_detected_names
        )
        surf_updated_probs = surf_updated[object_name]
        # ------------------------------------------------------------------
        # 3. Write the updated probabilities back into the BeliefState
        # ------------------------------------------------------------------
        for room_id in b_in.support():
            room_name = state.task.class_from_body[room_id]
            b_in.setProb(room_id, room_updated_probs.get(room_name, 0.0))
        b_in.normalize()

        for surface_id in b_on.support():
            surf_name = state.task.class_from_body[surface_id]
            b_on.setProb(surface_id, surf_updated_probs.get(surf_name, 0.0))
        b_on.normalize()


def update_belief(
    belief: dict, obs_loc: str, found: bool, visibility: float, co_detected: list[str]
):
    """Convenience wrapper maintained for backward compatibility so that
    other modules (`llm_tamp.co_model`) can still import this symbol.  It
    internally constructs a temporary `LLMUpdater` instance using dummy
    mappings because they are not required for the flat belief update.
    """
    updater = LLMUpdater(target_objs=[], class_from_body={}, body_from_class={})
    return updater.update_belief(belief, obs_loc, found, visibility, co_detected)


def main():
    baseball_dist = {
        "baseball": {
            "livingroom_table": 0.2,
            "livingroom_shelf": 0.2,
            "kitchen_refrigerator": 0.2,
            "kitchen_sink": 0.2,
            "playroom_cabinet": 0.2,
        }
    }
    fork_dist = {
        "fork": {
            "livingroom_table": 0.2,
            "livingroom_shelf": 0.2,
            "kitchen_refrigerator": 0.2,
            "kitchen_sink": 0.2,
            "playroom_cabinet": 0.2,
        }
    }
    spoon_dist = {
        "spoon": {
            "livingroom_table": 0.2,
            "livingroom_shelf": 0.2,
            "kitchen_refrigerator": 0.2,
            "kitchen_sink": 0.2,
            "playroom_cabinet": 0.2,
        }
    }

    visibility = 0.5
    found = False
    co_detected = ["spoon"]
    obs_loc = "kitchen_sink"

    updated_belief = update_belief(fork_dist, obs_loc, found, visibility, co_detected)
    print(updated_belief)


if __name__ == "__main__":
    main()
