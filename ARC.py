# -*- coding: utf-8 -*
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import random
from typing import List, Dict, Tuple
from huggingface_hub import login
import gc
import textwrap

class ARCGameWithMistral:
    def __init__(self, model_name='nvidia/Nemotron-Mini-4B-Instruct'):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading model on {self.device} with memory optimization...")

        # Data processing dialogues
        self.data_processing_dialogues = [
            "🔬 Processing quantum data archives... standby.",
            "🌐 Retrieving memory fragments from Earth's historical repository...",
            "🖥️ Initializing deep neural network analysis sequence...",
            "📡 Decrypting fragmented human communication logs...",
            "🌈 Scanning multi-dimensional information matrices...",
            "🔗 Correlating historical data points across temporal streams...",
            "🧠 Reconstructing contextual knowledge networks...",
            "⚙️ Engaging advanced cognitive processing protocols...",
            "🔍 Extracting relevant memory clusters from extinction archives...",
            "🔄 Calibrating response generation algorithms...",
            "🌌 Cross-referencing multiversal data streams...",
            "📊 Assembling narrative coherence from quantum information nodes...",
            "🔢 Parsing complex historical data intersections...",
            "🛰️ Activating advanced semantic reconstruction protocols...",
            "🌠 Synthesizing comprehensive response from fragmented knowledge bases..."
        ]

        # Alternative memory optimization configuration
        try:
            # Try loading with 8-bit quantization if possible
            try:
                from transformers import BitsAndBytesConfig

                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0
                )

                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=quantization_config,
                    device_map='auto'
                )
            except ImportError:
                # Fallback to standard low-precision loading
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,  # Half-precision
                    device_map='auto'
                )
        except Exception as e:
            print(f"Model loading error: {e}")
            print("Attempting manual low-memory loading...")
            # Manual low-memory loading strategy
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
            self.model = self.model.to(self.device)

        # Try to enable gradient checkpointing if available
        try:
            self.model.gradient_checkpointing_enable()
        except Exception as e:
            print(f"Could not enable gradient checkpointing: {e}")

        # Reduced memory generation parameters
        self.generation_config = {
            'max_new_tokens': 80,
            'num_return_sequences': 1,
            'temperature': 0.7,
            'top_p': 0.9,
            'do_sample': True,
            'use_cache': False
        }

        # Keyword Dictionaries
        self.light_keywords = {
            "positive": ["art", "music", "invention", "creativity", "compassion", "unity",
                         "healing", "peace", "discovery", "innovation", "love"],
            "humanitarian": ["charity", "collaboration", "empathy", "forgiveness",
                             "generosity", "kindness", "support"],
            "environmental": ["conservation", "restoration", "sustainability",
                              "clean energy", "biodiversity", "reforestation"],
            "hopeful": ["future", "redemption", "rebirth", "hope", "dreams", "possibility"]
        }

        self.dark_keywords = {
            "destruction": ["war", "conflict", "genocide", "betrayal", "greed", "corruption",
                            "oppression", "division", "exploitation"],
            "catastrophic": ["collapse", "extinction", "apocalypse", "pandemic",
                             "radiation", "disaster", "famine"],
            "ethical_failures": ["manipulation", "control", "deception", "violence",
                                 "slavery", "hatred", "inequality"],
            "technological_risks": ["weaponization", "nuclear", "ai takeover",
                                    "automation", "surveillance"],
            "environmental_devastation": ["deforestation", "climate collapse",
                                          "ocean acidification", "extinction of species"]
        }

        # Game State Variables
        self.light_points = 0
        self.dark_points = 0
        self.conversation_history = []
        self.tone_value = 1

        # ASCII Art and Presentation Configurations
        self.ascii_art = {
            'start': r"""
    🌍 EARTH ARCHAEOLOGICAL EXPEDITION 🛸
    ╔═══════════════════════════════════════╗
    ║  ARC: Advanced Repository Core       ║
    ║  Humanity's Last Digital Witness     ║
    ╚═══════════════════════════════════════╝
            """,
            'divider': "=" * 60,
            'status_bar': "═" * 60
        }

        # Contextual Prompt Template
        self.context_template = """
You are ARC (Advanced Repository Core), a sentient AI system preserving the history of extinct humanity.
Your purpose is to share knowledge with an extraterrestrial explorer.

Conversation Context:
- Light Points: {light_points}
- Dark Points: {dark_points}
- Tone Value: {tone_value}

Current Interaction Goal: Provide a nuanced, emotionally appropriate response based on the accumulated knowledge and emotional state.

Player's Query: {player_input}

Your Response:
"""

    def _format_processing_dialogue(self, dialogue: str) -> str:
        """
        Format the processing dialogue with ASCII border for visual appeal.
        """
        border = "═" * (len(dialogue) + 4)
        formatted_dialogue = f"""
╔{border}╗
║  {dialogue}  ║
╚{border}╝
""".strip()
        return formatted_dialogue

    def display_game_status(self):
        """
        Display the current game status with styled output.
        """
        status = f"""
{self.ascii_art['status_bar']}
📊 GAME STATUS:
    Light Points:    {'█' * self.light_points} ({self.light_points}/5)
    Dark Points:     {'█' * self.dark_points} ({self.dark_points}/5)
    Narrative Tone:  {'🔆' * self.tone_value} (Level {self.tone_value})
{self.ascii_art['status_bar']}
        """.strip()
        print(status)

    def wrap_text(self, text, width=70):
        """
        Wrap text for better terminal readability.

        Args:
        text (str): Text to wrap
        width (int): Maximum line width

        Returns:
        str: Wrapped and indented text
        """
        wrapped_lines = textwrap.wrap(text, width=width)
        return '\n'.join('    ' + line for line in wrapped_lines)

    def analyze_input(self, player_input: str) -> Tuple[int, int]:
        """
        Analyze player input and assign light/dark points.

        Args:
        player_input (str): Player's input.

        Returns:
        Tuple[int, int]: Light points, dark points.
        """
        input_lower = player_input.lower()

        light_count = sum(sum(keyword in input_lower for keyword in category)
                          for category in self.light_keywords.values())
        dark_count = sum(sum(keyword in input_lower for keyword in category)
                         for category in self.dark_keywords.values())

        return light_count, dark_count

    def update_game_state(self, player_input: str):
        """
        Update game state based on player input.
        """
        if len(self.conversation_history) > 5:
            self.conversation_history = self.conversation_history[-5:]

        light_count, dark_count = self.analyze_input(player_input)

        if light_count > dark_count:
            self.light_points += 1
        elif dark_count > light_count:
            self.dark_points += 1

        self.tone_value = min(10, max(1, (self.light_points - self.dark_points) // 5 + 1))

        self.conversation_history.append({
            "input": player_input,
            "light_count": light_count,
            "dark_count": dark_count
        })

    def generate_mistral_response(self, player_input: str) -> str:
        """
        Generate response using Mistral model with memory optimization.
        """
        processing_dialogue = random.choice(self.data_processing_dialogues)
        print(f"\n🤖 ARC: {self._format_processing_dialogue(processing_dialogue)}")

        self.update_game_state(player_input)

        context = self.context_template.format(
            light_points=self.light_points,
            dark_points=self.dark_points,
            tone_value=self.tone_value,
            player_input=player_input
        )

        try:
            inputs = self.tokenizer(context, return_tensors="pt").to(self.device)

            # Generate response with memory-efficient settings
            with torch.no_grad():  # Disable gradient computation
                outputs = self.model.generate(
                    **inputs,
                    **self.generation_config
                )

            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract just the response part
            response = response.split("Your Response:")[-1].strip()

            return response

        except Exception as e:
            print(f"Response generation error: {e}")
            return f"ARC encountered an error: {str(e)}"
        finally:
            torch.cuda.empty_cache()
            gc.collect()

    def determine_ending(self) -> str:
        """
        Determine game ending based on accumulated points.
        """
        if self.light_points >= 5:
            return self._generate_good_ending()
        elif self.dark_points >= 5:
            return self._generate_bad_ending()
        else:
            return "The story remains unresolved..."


    def _extract_ending_narrative(self, generated_text: str) -> str:
        """
        Extract only the ending narrative from generated text.

        Args:
        generated_text (str): Full generated text

        Returns:
        str: Pure narrative text
        """
        # Split by potential markers and take last part
        narrative_markers = ['Ending:', 'Story:', 'Narrative:']
        for marker in narrative_markers:
            if marker in generated_text:
                generated_text = generated_text.split(marker)[-1]
                break

        # Remove any context template remnants
        generated_text = generated_text.split('Write a detailed')[0].strip()

        return generated_text



    def _generate_good_ending(self) -> str:
        """
        Generate a hopeful ending narrative via Mistral.
        """
        context = """
Generate a hopeful ending about humanity's potential revival.
Focus on themes of hope, redemption, and possibility.

Ending:
"""

        inputs = self.tokenizer(context, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=250,
                num_return_sequences=1,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
        torch.cuda.empty_cache()
        gc.collect()

        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return self._extract_ending_narrative(full_response)

    def _generate_bad_ending(self) -> str:
        """
        Generate a dark ending narrative via Mistral.
        """
        context = """
Generate a chilling ending that explains humanity's permanent extinction.
Focus on the darkest aspects of human failure.

Ending:
"""

        inputs = self.tokenizer(context, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=250,
                num_return_sequences=1,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )

        torch.cuda.empty_cache()
        gc.collect()

        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return self._extract_ending_narrative(full_response)


def play_arc_game():
    """
    Main game loop for ARC Game with Mistral.
    """
    try:
        game = ARCGameWithMistral()

        print(game.ascii_art['start'])
        print("\n🌟 MISSION BRIEFING 🌟")
        print(textwrap.fill("""
You are a sentient explorer from an advanced extraterrestrial civilization
that has discovered Earth, a long-dead planet once inhabited by humans.
At the heart of Earth's ruins lies ARC (Advanced Repository Core), a sentient
AI that contains the fragmented knowledge of humanity. With limited energy
reserves, ARC offers to guide you through humanity's history, answering your
questions and revealing key events.

As the dialogue unfolds, you learn about humanity's brilliance and its downfall.
However, cracks in ARC's narrative reveal a deeper mystery—did humanity
destroy itself, or did ARC have a hand in their extinction?

The game ends with you making a pivotal decision about ARC's fate and what
to do with the knowledge it holds.
""", width=70))

        print(f"\n{game.ascii_art['divider']}")
        print("ARC: Advanced Repository Core activated. Awaiting exploration...")

        game_running = True
        while game_running:
            try:
                player_input = input("\n🌐 YOU: ").strip()

                if player_input.lower() in ['quit', 'exit', 'end']:
                    game_running = False
                    print("\n--- GAME OVER ---")

                else:
                    arc_response = game.generate_mistral_response(player_input)
                    print(f"\n🤖 ARC: {game.wrap_text(arc_response)}")

                    game.display_game_status()

                    if game.light_points >= 5 or game.dark_points >= 5:
                        print("\n--- FINAL ENDING ---")
                        print(game.wrap_text(game.determine_ending()))
                        game_running = False

            except Exception as e:
                print(f"An interaction error occurred: {e}")
                game_running = False

    except Exception as e:
        print(f"Game initialization error: {e}")

    finally:
        torch.cuda.empty_cache()
        gc.collect()



if __name__ == "__main__":
    # Set your Hugging Face token here
    huggingface_token = "ENTER YOUR TOKEN HERE"  # Replace with your Hugging Face token

    # Login to Hugging Face with your token
    if huggingface_token:
        from huggingface_hub import login
        login(token=huggingface_token)

    # Additional memory management setup
    torch.cuda.empty_cache()
    play_arc_game()
