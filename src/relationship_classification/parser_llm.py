#!/usr/bin/env python3
# filepath: /root/catkin_ws/src/scene_graph_room_classification/src/relationship_classification/parser_llm.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys

class QwenChatNode:
    def __init__(self):
        self.model_name = "Qwen/Qwen2.5-1.5B-Instruct"
        print("Loading Qwen model... This may take a moment.")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,  # float32 for CPU
            device_map="cpu"            
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        print("Model loaded successfully!")
        print("Qwen Chat Node is ready. Type 'quit' or 'exit' to stop.")
        print("-" * 50)
    
    def generate_response(self, user_prompt):
        """Generate a response from the model"""
        messages = [
            #{"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "system", "content": """You are a scene graph parser that extracts structured information from scene descriptions. 

                Your task is to analyze text descriptions of scenes and extract:
                1. OBJECTS: All physical entities mentioned in the scene
                2. ATTRIBUTES: Properties of objects (color, size, material, shape, etc.)
                3. RELATIONSHIPS: Spatial and semantic connections between objects
                
                Output format should be structured as follows:
                
                **OBJECTS:**
                - object_name (attributes: attribute1, attribute2, ...)
                
                **RELATIONSHIPS:**
                - object1 [relationship] object2
                - object1 [relationship] object2
                
                Use clear, consistent relationship terms like:
                - Spatial: on, under, above, below, next to, beside, in front of, behind, inside, outside, near, far from
                - Support: supports, rests on, hangs from, attached to
                - Containment: contains, holds, filled with
                - Functional: connected to, part of, belongs to
                
                Be precise and only extract information explicitly stated or clearly implied in the text. If attributes or relationships are unclear, omit them rather than guess."""},
            {"role": "user", "content": user_prompt}
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to("cpu")
        
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response
    
    def run(self):
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                # Check for exit conditions
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                # Skip empty input
                if not user_input:
                    continue
                
                # Generate and display response
                print("Qwen: ", end="", flush=True)
                response = self.generate_response(user_input)
                print(response)
                
            except KeyboardInterrupt:
                print("\n\nReceived Ctrl+C, shutting down...")
                break
            except Exception as e:
                print(f"Error: {e}")
                print("Please try again.")

if __name__ == "__main__":
    node = QwenChatNode()
    node.run()
