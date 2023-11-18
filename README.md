# Deep Learning Development

I have recently initiated this project.

Most ideas were inspired by Kaggle Competitions.
Currently, most of the code is related to transformers because I am only participating in competitions involving sequential data.

# Demo Version

You can install a demo version.
Currently, most of modules are transformer-based.
I will upload example usages soon.

use the line to install this project
pip install git+https://github.com/crimson206/DeepLearningDevelopment@0.1.4

useful embedding moduels are in
from CrimsonDeepLearning.transformers.embeddings import ...

# Recommended Notebooks
- Params Shaker: This is to search for the global optimum with randomness.
  https://github.com/crimson206/DeepLearningDevelopment/blob/main/src/global_optimum/shakers/z_ParamsShaker.ipynb

- MultiPositionalEncoder:
  Does your problem have more than one position?
  Do you want to understand the attention mechanism better?
  Read the article.
  
  https://github.com/crimson206/DeepLearningDevelopment/blob/main/src/model_development/transformers/embeddings/z_MultiPositionalEncoders.ipynb
