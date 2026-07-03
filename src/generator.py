import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List
from src.chunking import Chunk


class RAGGenerator:
    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading {self.model_name} on {self.device} (This might take a minute the first time)...")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Load model efficiently
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True
        )


    def generate_answer(self, question: str, retrieved_chunks: List[Chunk], max_new_tokens: int = 512) -> str:
        # Combine chunks into context, but limit it to avoid running out of memory/tokens
        context_parts = []
        current_len = 0

        for c in retrieved_chunks:
            chunk_str = f"Source: {c.file_path}\n{c.text}\n"
            if current_len + len(chunk_str) > 10000:  # Safe character limit for ~2000 tokens
                break
            context_parts.append(chunk_str)
            current_len += len(chunk_str)

        context_text = "\n".join(context_parts)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful coding assistant. "
                    "Answer the user's question based ONLY on the provided context. "
                    "If the context does not contain enough information to answer "
                    "the question, respond with exactly: "
                    "'I don't have enough information in the provided context to answer this question.' "
                    "Do NOT use your training knowledge to fill gaps. "
                    "Do NOT guess or infer beyond what is explicitly stated in the context. "
                    "Keep your answer clear, self-contained, and faithful to the source."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Context information is below.\n"
                    f"---------------------\n"
                    f"{context_text}\n"
                    f"---------------------\n"
                    f"Given the context information, answer the following question: {question}"
                )
            }
        ]

        # Qwen uses chat templates
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False  # Disable "thinking..." to save tokens and avoid confusion
        )

        # Fix: pass padding=True so attention_mask is generated
        model_inputs = self.tokenizer(
            [text],
            return_tensors="pt",
            truncation=True
        ).to(self.device)

        generated_ids = self.model.generate(
            model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,   # ✅ Fix: pass attention_mask
            max_new_tokens=max_new_tokens,
            temperature=0.3,
            do_sample=False,
        )

        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response.strip()
