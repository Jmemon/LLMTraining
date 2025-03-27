import torch
import torch.nn as nn
from transformers import PreTrainedTokenizer
from vorox.configs import RunConfig

class Pipeline:
    """
    Text generation pipeline for Vorox models.
    
    Provides a high-level interface for text generation using Vorox models,
    handling tokenization, generation loop, and output processing.
    
    Architecture:
        - Implements a stateful wrapper around model and tokenizer with O(1) initialization
        - Autoregressive generation with configurable sampling parameters
        - Handles tokenization, padding, and truncation automatically
    """
    
    def __init__(self, model: nn.Module, tokenizer: PreTrainedTokenizer):
        """
        Initialize the pipeline with a model and tokenizer.
        
        Args:
            model: The Vorox model to use for generation
            tokenizer: The tokenizer to use for encoding/decoding text
        """
        self.model = model
        self.tokenizer = tokenizer
    
    def generate(self, inputs, cfg: RunConfig):
        """
        Generate text from the given inputs.
        
        Args:
            inputs: Text input to generate from
            cfg: Configuration for the generation process
            
        Returns:
            Generated text output
        """
        # Tokenize inputs
        max_seq_len = cfg.train.max_seq_len if hasattr(cfg, 'train') and hasattr(cfg.train, 'max_seq_len') else 1024
        input_ids = self.tokenizer(
            inputs, 
            padding='max_length',
            truncation=True,
            max_length=max_seq_len,
            return_tensors='pt'
        ).input_ids
        
        # Move to the same device as the model
        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Generate tokens until we hit the stop token or max length
        generated_ids = input_ids.clone()
        max_new_tokens = cfg.evaluator.max_new_tokens if hasattr(cfg, 'evaluator') and hasattr(cfg.evaluator, 'max_new_tokens') else 100
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Get model predictions
                outputs = self.model(generated_ids, causal_attn_mask=True, apply_softmax=True)
                next_token_logits = outputs[:, -1, :]
                
                # Sample from the logits
                temperature = cfg.evaluator.temperature if hasattr(cfg, 'evaluator') and hasattr(cfg.evaluator, 'temperature') else 1.0
                if temperature > 0:
                    # If apply_softmax=True is used in the model forward pass, we already have probabilities
                    probs = next_token_logits
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    # Greedy decoding
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Append the new token
                generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                
                # Check if we've hit the end token
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
        
        # Decode the generated tokens
        generated_text = self.tokenizer.decode(
            generated_ids[0], 
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        
        return generated_text
