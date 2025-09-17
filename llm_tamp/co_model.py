import sys
import os
from openai import OpenAI
from pydantic import BaseModel

PACKAGES = ['pddlstream']

def add_packages(packages):
    sys.path.extend(os.path.abspath(os.path.join(os.getcwd(), d)) for d in packages)

add_packages(PACKAGES)

from examples.discrete_belief.dist import DDist
from llm_tamp.belief import BeliefState
from llm_tamp.llm_updater import update_belief
import ollama
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import json

EVENLY_DISTRIBUTED_SYS_PROMPT = """You will be given an object that commonly appears in a typical household environment.
Using common sense determine if the object tends to be distributed throughout a typical household, such as doorknobs and light switches.
"""

EVENLY_DISTRIBUTED_USER_PROMPT = """The following is the object: {object}."""

class EvenlyDistributed(BaseModel):
    obj: str
    even: bool

def get_evenly_distributed(obj: str) -> EvenlyDistributed:
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    messages=[{"role": "system", "content": EVENLY_DISTRIBUTED_SYS_PROMPT},
              {"role": "user", "content": EVENLY_DISTRIBUTED_USER_PROMPT.format(object=obj)}]
    evenly_distributed = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=messages,
        response_format=EvenlyDistributed).choices[0].message.parsed
    return evenly_distributed

def remove_suffix(s: str) -> str:
    """Remove trailing underscore and digits from the end of a string."""
    return re.sub(r'_\d+$', '', s)

class CorrelationalObsModel():
    def __init__(self, target_objs: list, objects:list, file_name: str,
                 class_from_body: dict, body_from_class: dict, use_correlation: bool,
                 alpha = 2.0):

        f = open(file_name)
        self.context_dict = json.load(f)
        
        self.class_from_body = class_from_body
        self.body_from_class = body_from_class
        self.target_objs = target_objs
        self.objects = objects
        self.alpha = alpha
        self.use_correlation = use_correlation
        if self.use_correlation:
            self.embedding_map = self.gen_embedding_map()
            self.sim_map = self.gen_map()        

    def gen_embedding_map(self):
        string_objs = []
        for obj in self.objects:
            string_objs.append(remove_suffix(self.class_from_body[obj]))
        embedding_map = {}
        for obj in string_objs:
            embedding_map[obj] = ollama.embed(model='nomic-embed-text', 
                                              input=self.context_dict[obj])['embeddings']
        return embedding_map

    def amplify(self, sim):
        mult_val = sim * self.alpha
        if mult_val >= 0.9: mult_val = 0.9
        elif mult_val <= -0.9: mult_val = -0.9
        return mult_val

    def cal_similarity(self, text: str, context: str):
        text_embedding = self.embedding_map[text]
        vec = np.array(text_embedding).reshape(1, -1)
        context_embedding = self.embedding_map[context]
        context_vec = np.array(context_embedding).reshape(1, -1)
        return cosine_similarity(vec, context_vec)[0][0]

    def gen_map(self):
        minimum, maximum = float('inf'), float('-inf')
        sim_map = {}
        print(self.target_objs)
        string_targets = [remove_suffix(self.class_from_body[target_obj[0]]) for target_obj in self.target_objs]
        string_objs = [remove_suffix(self.class_from_body[obj]) for obj in self.objects]
        for target in string_targets:
            for obj in string_objs:
                if obj == target:
                    sim_map[(target, obj)] = 0.95
                elif obj != target:
                    sim = self.cal_similarity(target, obj)
                    shifted_sim = sim - 0.57
                    amplified_sim = self.amplify(shifted_sim)
                    sim_map[(target, obj)] = amplified_sim
                    if amplified_sim > maximum: maximum = amplified_sim
                    if amplified_sim < minimum: minimum = amplified_sim
        return sim_map
    
    def get_observation_fn(self, loc, p_look_fp, p_look_fn, visibility):
        """
        loc         : the true location of the object
        p_look_fp   : probability of false positive detection
        p_look_fn   : probability of false negative detection
        visibility  : factor in [0, 1], how visible the object is
        """
        # Clamp visibility to [0, 1] just in case
        visibility = max(0.0, min(1.0, visibility))

        def observation_fn(query_loc):
            """
            Returns a distribution over {True, False} for "object detected"
            given that we are 'looking' at query_loc.
            """
            if query_loc == loc:
                # Object is actually here
                # Base detection prob is (1 - p_look_fn),
                # scale by visibility
                p_true = (1 - p_look_fn) * visibility
            else:
                # Object is actually NOT here
                # Base false-positive prob is p_look_fp,
                # scale by visibility
                p_true = p_look_fp * visibility

            # Probability of "detected = False"
            p_false = 1 - p_true

            return DDist({True: p_true, False: p_false})

        return observation_fn
    
    def get_pos_correlation_fn(self, loc, num_loc, sim):
        def fn(l):
            if l == loc:
                P_false, P_true = 1, 0
            else:
                P_false, P_true = 0, 1
            P_true = sim * P_true + (1 - sim) * (1/num_loc)
            P_false = sim * P_false + (1 - sim) * (1/num_loc)
            return DDist({True: P_true,
                        False: P_false})
        return fn
    
    def get_neg_correlation_fn(self, loc, num_loc, sim):
        def fn(l):
            if l == loc:
                P_false, P_true = 0, 1
            else:
                P_false, P_true = 1, 0
            P_true = abs(sim) * P_true + (1 + sim) * (1/num_loc)
            P_false = abs(sim) * P_false + (1 + sim) * (1/num_loc)
            return DDist({True: P_true,
                        False: P_false})
        return fn
    
    def correlationUpdate(self, target_item:int, detected_items: set, target_loc: int, surf_visibility:float, 
                          room_visibility:float, known: set, p_fp: float, p_fn: float, state: BeliefState, obs: bool):
        # Create observation model
        room_body = state.get_room_body(target_loc)
        num_room = len(state.b_in[target_item].support())
        num_surf = len(state.b_on[(target_item, room_body)].support())
        surf_obs_model = self.get_observation_fn(target_loc, p_fp, p_fn, surf_visibility)
        room_obs_model = self.get_observation_fn(room_body, p_fp, p_fn, room_visibility)
        # Update detected object belief
        state.b_on[(target_item, room_body)].obsUpdate(surf_obs_model, obs)
        state.b_in[target_item].obsUpdate(room_obs_model, obs)
        # Update other object belief based on correlation
        if self.use_correlation and not obs:
            for obj in detected_items:
                # if get_evenly_distributed(remove_suffix(self.class_from_body[obj])).even:
                #     continue
                # if obj was not already detected
                if obj not in known and obj != target_item:
                    sim = self.sim_map[tuple((
                        remove_suffix(self.class_from_body[target_item]), 
                        remove_suffix(self.class_from_body[obj])))]
                    # Create correlation model
                    if sim >= 0:
                        print("Positive correlation: ", self.class_from_body[target_item], self.class_from_body[obj], sim)
                        surf_corr_model = self.get_pos_correlation_fn(target_loc, num_surf, sim)
                        room_corr_model = self.get_pos_correlation_fn(room_body, num_room, sim)
                    else:
                        print("Negative correlation: ", self.class_from_body[target_item], self.class_from_body[obj], sim)
                        surf_corr_model = self.get_neg_correlation_fn(target_loc, num_surf, sim)
                        room_corr_model = self.get_neg_correlation_fn(room_body, num_room, sim)
                    state.b_on[(target_item, room_body)].obsUpdate(surf_corr_model, obs)
                    state.b_in[target_item].obsUpdate(room_corr_model, obs)
