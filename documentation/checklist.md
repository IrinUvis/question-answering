# Checklist

## To do

### Training

#### SQuAD 1.0
- [ ] Train BERT cased model on SQuAD 1.0 and evaluate it on SQuAD 1.0
- [ ] Train BERT uncased model on SQuAD 1.0 and evaluate it on SQuAD 1.0
- [ ] Train ALBERT cased model on SQuAD 1.0 and evaluate it on SQuAD 1.0

#### SQuAD 2.0
- [ ] Train BERT cased model on SQuAD 2.0 and evaluate in on SQuAD 2.0 and SQuAD 1.0
- [ ] Train BERT uncased model on SQuAD 2.0 and evaluate in on SQuAD 2.0 and SQuAD 1.0
- [ ] Train ALBERT cased model on SQuAD 2.0 and evaluate in on SQuAD 2.0 and SQuAD 1.0

#### Additional tasks
- [ ] Generate figures for training

### Others
- [ ] Generate figures for dataset analysis

## Improvements from last semester
- train 5 final models of each type
- pick best model based on learning rate only (1 model for each)
- while training on SQuAD 1.0 filter samples above max number of tokens
- while training on SQuAD 2.0 either filter samples above max number of tokens or use stride mechanism (shouldn't matter)
- evaluating SQuAD 1.0 on SQuAD 1.0 is easy
- evaluating SQuAD 2.0 on SQuAD 2.0 is easy
- evaluating SQuAD 2.0 on SQuAD 1.0???
