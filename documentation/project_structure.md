# Project structure

The project concerns extractive QA. The work on it is located in [extractive-qa](./../extractive-qa) directory. 
Its structure is as follows:
* ***data*** directory holds the data referenced in the notebooks
* ***figures*** directory holds figures regarding general data analysis etc.
* ***model-evaluation*** directory holds figures and graphs related to models, their training, and evaluation
* ***notebooks*** directory holds various notebooks for training and examining various models
* ***tf-models*** directory holds best versions of models saved and trained from specific notebooks
* ***training-checkpoints*** directory holds model training checkpoints (typically they are stored there temporarily until the best checkpoint is saved)

Additionally, there is a local package called ***question_answering*** with utility functions, constants and paths used all across the project. 