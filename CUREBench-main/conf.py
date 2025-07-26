class MedGemmaTextModel:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def load(self):
        print("✅ MedGemma model is already loaded in memory.")

    def inference(self, prompt: str, max_tokens: int = 256):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=max_tokens)
        response_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        # Structure attendue par eval_framework
        messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response_text}]
        return response_text, messages
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Authentification Hugging Face
login("+++++++++")

model_id = "google/medgemma-4b-it"  # (Si ce modèle supporte text-only)
local_path = "/content/drive/MyDrive/medgemma_text_model"  # Sauvegarde sur Google Drive

# Télécharger et sauvegarder
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.save_pretrained(local_path)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)


print(f"✅ Modèle texte sauvegardé dans {local_path}")
