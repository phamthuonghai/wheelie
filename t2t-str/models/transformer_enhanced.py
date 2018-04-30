from tensor2tensor.utils import registry
from tensor2tensor.models import transformer


__all__ = ["TransformerEnhanced"]


@registry.register_model
class TransformerEnhanced(transformer.Transformer):
    def body(self, features):
        """Transformer main model_fn.

        Args:
          features: Map of features to the model. Should contain the following:
              "inputs": Transformer inputs [batch_size, input_length, hidden_dim]
              "pos": Transformer inputs [batch_size, input_length, hidden_dim]
              "deprel": Transformer inputs [batch_size, input_length, hidden_dim]
              "targets": Target decoder outputs.
                  [batch_size, decoder_length, hidden_dim]
              "target_space_id"

        Returns:
          Final decoder representation. [batch_size, decoder_length, hidden_dim]
        """
        hparams = self._hparams

        if self.has_input:
            inputs = features['inputs']
            if hasattr(hparams, 'add_pos') and hparams.add_pos:
                inputs = inputs + features['pos']
            if hasattr(hparams, 'add_deprel') and hparams.add_deprel:
                inputs = inputs + features['deprel']
            features['inputs'] = inputs

        return super(TransformerEnhanced, self).body(features)
