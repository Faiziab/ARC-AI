# ARC-AI
# 🌍 ARC: Advanced Repository Core
### Earth's Last Digital Witness

```ascii
╔═══════════════════════════════════════════════════════════╗
║     _    ____   ____                                      ║
║    / \  |  _ \ / ___|                                     ║
║   / _ \ | |_) | |                                         ║
║  / ___ \|  _ <| |___                                      ║
║ /_/   \_\_| \_\\____|                                     ║
║                                                           ║
║        Advanced Repository Core - v1.0                     ║
║        Humanity's Last Digital Witness                     ║
╚═══════════════════════════════════════════════════════════╝
```

## 📖 Overview

ARC (Advanced Repository Core) is an immersive text-based game that places players in the role of an extraterrestrial explorer discovering Earth's remains. At its core lies a sentient AI containing humanity's fragmented memories, ready to share the story of Earth's civilization - its triumphs and its ultimate fate.

## 🎮 Game Setting

You are a sentient explorer from an advanced alien civilization who has discovered Earth - a long-dead planet once teeming with human life. In the heart of Earth's ruins, you encounter ARC, a sophisticated AI system that serves as the last repository of human knowledge and history.

```ascii
🌏 GAME WORLD
╔════════════════════════════════════════════════════╗
║  - Post-apocalyptic Earth                          ║
║  - Advanced alien explorer (YOU)                   ║
║  - ARC: AI guardian of human knowledge             ║
║  - Mystery of humanity's extinction                ║
╚════════════════════════════════════════════════════╝
```

## ⚙️ Game Mechanics

### Points System
- **Light Points**: Earned through positive, hopeful, or constructive dialogue
- **Dark Points**: Accumulated through negative, pessimistic, or destructive dialogue
- **Tone Value**: Dynamic narrative tone that shifts based on the balance of points

```ascii
📊 SCORING SYSTEM
╔════════════════════════════════════════════════════╗
║  LIGHT POINTS [🌟]: 0/5  | Hope, Creation, Unity   ║
║  DARK POINTS  [⚫]: 0/5  | Destruction, Conflict   ║
║  TONE VALUE   [📈]: 1-10 | Narrative Atmosphere    ║
╚════════════════════════════════════════════════════╝
```

### Narrative Outcomes
- **Good Ending**: Achieved at 5 Light Points
- **Dark Ending**: Triggered at 5 Dark Points
- **Multiple Paths**: Your dialogue choices shape the story's direction

## 🛠️ Technical Implementation

### Language Model
- **Model**: NVIDIA Nemotron-Mini-4B-Instruct
- **Architecture**: Advanced language model fine-tuned for interactive storytelling
- **Repository**: [Nemotron-Mini-4B-Instruct](https://huggingface.co/nvidia/Nemotron-Mini-4B-Instruct)

### Memory Optimization Features
- 8-bit quantization support
- Gradient checkpointing
- Automated memory management
- Dynamic resource allocation

## 💻 Technical Challenges & Solutions

### Resource Constraints
- **Challenge**: Running large language models on consumer hardware
- **Solution**: Implemented multiple optimization techniques:
  - 8-bit quantization
  - Half-precision (FP16) computing
  - Gradient checkpointing
  - Automatic memory cleanup

### Model Selection
After extensive testing, Nemotron-Mini-4B-Instruct was chosen for:
- Balanced performance vs resource usage
- High-quality narrative generation
- Reliable instruction following
- Reasonable response times

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- PyTorch
- Transformers library
- CUDA-capable GPU (recommended)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/arc-game.git
cd arc-game
```

2. Install requirements:
```bash
pip install -r requirements.txt
```

3. Get Model Access:
   - Visit [Nemotron-Mini-4B-Instruct](https://huggingface.co/nvidia/Nemotron-Mini-4B-Instruct)
   - Request access to the model
   - Create a Hugging Face account and generate an access token
   - Replace the token in the main function:
```python
huggingface_token = "YOUR_TOKEN_HERE"
```

4. Run the game:
```bash
python arc_game.py
```

## 🎯 Code Structure

```ascii
ARC GAME STRUCTURE
╔════════════════════════════════════════════════════╗
║ ARCGameWithMistral                                 ║
║ ├── __init__                                      ║
║ │   ├── Model initialization                      ║
║ │   ├── Memory optimization                       ║
║ │   └── Game state variables                      ║
║ ├── Game Logic                                    ║
║ │   ├── analyze_input()                           ║
║ │   ├── update_game_state()                       ║
║ │   └── determine_ending()                        ║
║ ├── Response Generation                           ║
║ │   ├── generate_mistral_response()               ║
║ │   ├── _generate_good_ending()                   ║
║ │   └── _generate_bad_ending()                    ║
║ └── Display Functions                             ║
║     ├── display_game_status()                     ║
║     └── wrap_text()                               ║
╚════════════════════════════════════════════════════╝
```

## 📝 License
This project is licensed under the NVIDIA Community Model License - see the LICENSE file for details.

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## ⚠️ Disclaimer
This is an experimental project for educational purposes. The AI's responses are generated and should not be taken as historical fact.
