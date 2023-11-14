import torch
import torch.nn as nn
from typing_extensions import Protocol

class TransformerOutput(Protocol):
    pass

class EncoderTransformer(nn.Module):
    """
    A wrapper class for a predefined transformer encoder that uses custom embeddings.

    This class is designed to take a predefined transformer and apply it to custom embeddings,
    bypassing the transformer model's usual embedding layer.

    Attributes:
        transformer (nn.Module): A Hugging Face transformer model that has been configured to accept custom embeddings.

    Args:
        transformer (nn.Module): An instance of a transformer model from the Hugging Face library.
    """

    def __init__(self, transformer):
        super(EncoderTransformer, self).__init__()
        self.transformer = transformer

    def forward(self, embedding: torch.Tensor, attention_mask: torch.Tensor) -> TransformerOutput:
        """
        Forward pass for the EncoderTransformer using custom embeddings.

        Args:
            embedding (torch.Tensor): The custom embedding tensor to be processed by the predefined transformer.
                                      Shape: (batch_size, seq_length, embedding_dim)
            attention_mask (torch.Tensor): The attention mask for the predefined transformer.
                                           Shape: (batch_size, seq_length)

        Returns:
            TransformerOutput: An output object from the predefined transformer. The exact type and attributes of
                               this object can vary depending on the transformer used. Typical transformers return
                               an object with attributes like 'last_hidden_state', and possibly others such as
                               'pooler_output' or 'hidden_states', depending on the model configuration.
                               Users should refer to the specific Hugging Face transformer documentation for details
                               on the output object.
        """
        # In Hugging Face's implementation, 'inputs_embeds' is used to pass custom embeddings.
        # 'attention_mask' is used as usual.
        output = self.transformer(inputs_embeds=embedding, attention_mask=attention_mask)

        return output
