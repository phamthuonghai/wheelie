# Enhancing Tensor2Tensor with Dependency Parse

Google's [Tensor2Tensor](https://github.com/tensorflow/tensor2tensor/) encodes positional information with each word.
This positional encoding is a vector of sine function with different frequencies (fixed) or can be learned with the model.
Our hypothesis is that dependency parse can add more helpful information and/or replace the current positional encoding scheme.

Our goal is to encode the `tree address` or at least depth of each source word in a dependency parse, then compare the enhanced
model with the current one.
