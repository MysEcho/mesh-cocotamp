import os
from openai import OpenAI
from pydantic import BaseModel
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
import numpy as np
import pprint

SYS_PROMPT = """You will receive:
- belief: a dict mapping location names (strings) to prior probabilities (floats summing to 1.0).
- observation_location: a string for the location just inspected.
- found: a boolean, True if the object was found at observation_location, else False.
- visibility: a float between 0 and 1 indicating how much of observation_location was visible.
- co_detected: a list of other object names detected at the same time.

Update the belief as follows:
1. If found is True, set belief[observation_location] = 1.0 and all other locations to 0.0.
2. If found is False, using common sense that any human would have, update the belief based on the visibility and the other objects detected.

Return only the updated belief as a Python dict with each probability as a float. 
The sum of the probabilities must be 1.0. 
Do not include any additional text.

When you respond, output only a valid JSON object that follows this schema: {\"belief\": {<location>: <probability float>}}. Do not wrap the JSON in markdown or provide any explanatory text."""


USER_PROMPT = """The following is the belief: {belief}.
The following is the observation_location: {obs_loc}.
The following is the found: {found}.
The following is the visbility of the location: {visibility}
The following are the detected objects during observation: {co_detected}"""

class UpdatedBelief(BaseModel):
    belief: dict

def update_belief(belief: dict, obs_loc: str, found: bool, visibility: float, co_detected: list[str]) -> UpdatedBelief:
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    messages=[{"role": "system", "content": SYS_PROMPT},
              {"role": "user", "content": USER_PROMPT.format(belief=belief, 
                                                             obs_loc=obs_loc,
                                                             found=found,
                                                             visibility=visibility,
                                                             co_detected=co_detected)}]
    response = client.chat.completions.create(
        model="gpt-4o-2024-08-06",
        messages=messages,
        response_format={"type": "json_object"},  # enforce valid JSON output
    )

    content = response.choices[0].message.content

    parsed = UpdatedBelief.model_validate_json(content)
    return parsed


def main():
    baseball_dist = {'baseball': {'livingroom_table': 0.2, 
                    'livingroom_shelf': 0.2, 
                    'kitchen_refrigerator': 0.2,
                    'kitchen_sink': 0.2,
                    'playroom_cabinet': 0.2}}
    fork_dist = {'fork': {'livingroom_table': 0.2, 
                        'livingroom_shelf': 0.2, 
                        'kitchen_refrigerator': 0.2,
                        'kitchen_sink': 0.2,
                        'playroom_cabinet': 0.2}}
    spoon_dist = {'spoon': {'livingroom_table': 0.2, 
                        'livingroom_shelf': 0.2, 
                        'kitchen_refrigerator': 0.2,
                        'kitchen_sink': 0.2,
                        'playroom_cabinet': 0.2}}

    visibility = 0.5
    found = False
    co_detected = ['spoon']
    obs_loc = 'kitchen_sink'

    updated_belief = update_belief(fork_dist, obs_loc, found, visibility, co_detected)
    print(updated_belief)

if __name__ == "__main__":
    main()