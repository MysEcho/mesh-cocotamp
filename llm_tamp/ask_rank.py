import os
from openai import OpenAI
from pydantic import BaseModel
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
import numpy as np
import pprint

ROOM_SYS_PROMPT = """You will be given possible locations of an object in a house as a list.
Using common sense that any human would have, rank the locations from most likely to least likely.
The most likely location should be the first item in the list and the least likely location should be the last item in the list.
The content of the list should not change but the order can be changed.
"""

SURFACE_SYS_PROMPT = """You will be given possible locations of an object in a {room} as a list.
Using common sense that any human would have, rank the locations from most likely to least likely.
The most likely location should be the first item in the list and the least likely location should be the last item in the list.
The content of the list should not change but the order can be changed.
"""

USER_PROMPT = """The following is the object: {object}.
The following are possible locations of the object: {locations}"""

FIX_PROMPT = """Your previous ranking for {object} was {ranked_locs}.
You are missing the following locations: {missing}.
You have hallucinated the following locations: {hallucinated}.
The following are possible locations of the object: {locations}
Please provide the correct ranking of the locations.
"""

class ObjectPlacementRank(BaseModel):
    obj: str
    surfaces: list[str]

def get_room_ranking(obj: str, surfaces: list[str]) -> ObjectPlacementRank:
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    messages=[{"role": "system", "content": ROOM_SYS_PROMPT},
              {"role": "user", "content": USER_PROMPT.format(object=obj, 
                                                             locations=surfaces)}]
    obj_ranking = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=messages,
        response_format=ObjectPlacementRank).choices[0].message.parsed
    return obj_ranking

def get_surface_ranking(obj: str, room: str, surfaces: list[str]) -> ObjectPlacementRank:
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    messages=[{"role": "system", "content": SURFACE_SYS_PROMPT.format(room=room)},
              {"role": "user", "content": USER_PROMPT.format(object=obj, 
                                                             locations=surfaces)}]
    obj_ranking = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=messages,
        response_format=ObjectPlacementRank).choices[0].message.parsed
    return obj_ranking

def get_fixed_ranking(obj: str, 
                      missing: set,
                      hallucinated: set,
                      ranked_surfaces: list[str],
                      surfaces: list[str]) -> ObjectPlacementRank:
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    messages=[{"role": "system", "content": ROOM_SYS_PROMPT},
              {"role": "user", "content": FIX_PROMPT.format(object=obj,
                                                            ranked_locs=ranked_surfaces,
                                                            missing=missing,
                                                            hallucinated=hallucinated,
                                                            locations=surfaces)}]
    obj_ranking = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=messages,
        response_format=ObjectPlacementRank).choices[0].message.parsed
    return obj_ranking

def create_room_belief(target_objects: list[int], rooms: list[int], class_from_body: dict) -> dict:
    initial_belief = {}
    for target_object in target_objects:
        obj_class = class_from_body[target_object]
        rooms_class = [class_from_body[room] for room in rooms]
        correct_gen = True
        obj_ranking = get_room_ranking(obj_class, rooms_class)
        count = 0
        while correct_gen and count < 10:        
            ranked_rooms = obj_ranking.surfaces
            set_ranked_rooms = set(ranked_rooms)
            set_rooms = set(rooms_class)
            missing = set_rooms - set_ranked_rooms 
            hallucinated = set_ranked_rooms - set_rooms
            print(ranked_rooms)
            count += 1
            if missing or hallucinated:
                obj_ranking = get_fixed_ranking(obj_class, missing, hallucinated, ranked_rooms, rooms_class)
            else:
                correct_gen = False
        ranked_rooms_class = obj_ranking.surfaces
        num_rooms = len(ranked_rooms_class)
        tot = num_rooms * (num_rooms+1) // 2
        obj_belief = {}
        for room in ranked_rooms_class:
            obj_belief[room] = num_rooms / tot
            num_rooms -= 1
        initial_belief[obj_class] = obj_belief
    pprint.pprint(initial_belief)
    return initial_belief

def create_surface_belief(target_objects: list[int], room: int, surfaces: list[int], class_from_body: dict) -> dict:     
    initial_belief = {}
    for target_object in target_objects:
        obj_class = class_from_body[target_object]
        surfaces_class = [class_from_body[surf].split("|")[0] for surf in surfaces]
        correct_gen = True
        obj_ranking = get_surface_ranking(obj_class, class_from_body[room], surfaces_class)
        count = 0
        while correct_gen and count < 10:        
            ranked_surfaces = obj_ranking.surfaces
            set_ranked_surfaces = set(ranked_surfaces)
            set_surfaces = set(surfaces_class)
            missing = set_surfaces - set_ranked_surfaces 
            hallucinated = set_ranked_surfaces - set_surfaces
            print(ranked_surfaces)
            count += 1
            if missing or hallucinated:
                obj_ranking = get_fixed_ranking(obj_class, missing, hallucinated, ranked_surfaces, surfaces_class)
            else:
                correct_gen = False
        ranked_surfaces_class = obj_ranking.surfaces
        num_surfaces = len(ranked_surfaces_class)
        tot = num_surfaces * (num_surfaces+1) // 2
        obj_belief = {}
        for surf in ranked_surfaces_class:
            obj_belief[surf+"|"+class_from_body[room]] = num_surfaces / tot
            num_surfaces -= 1
        initial_belief[(obj_class, class_from_body[room])] = obj_belief
    pprint.pprint(initial_belief)
    return initial_belief