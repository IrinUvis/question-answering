from datasets import Dataset


def preprocess_squad2_training_dataset_no_stride(
    dataset: Dataset,
    tokenizer,
    max_length: int,
    batched: bool = True,
    remove_columns: list[str] = None,
):
    def preprocess_samples(samples):
        questions = [q.strip() for q in samples["question"]]
        contexts = [c.strip() for c in samples["context"]]

        inputs = tokenizer(
            questions,
            contexts,
            max_length=max_length,
            padding="max_length",
            return_offsets_mapping=True,
        )

        offset_mapping = inputs.pop("offset_mapping")
        answer_starts = samples["answer_start"]
        answer_texts = samples["answer_text"]
        start_positions = []
        end_positions = []

        for i, offset in enumerate(offset_mapping):
            answer_start = answer_starts[i]
            answer_text = answer_texts[i]

            # If there is no answer, label is (0, 0)
            if len(answer_start) == 0 and len(answer_text) == 0:
                start_positions.append(0)
                end_positions.append(0)
            # Otherwise, we try to find start and end token indices
            else:
                answer_start = answer_start[0]
                answer_text = answer_text[0]

                start_char = answer_start
                end_char = start_char + len(answer_text)
                sequence_ids = inputs.sequence_ids(i)

                # Find the start and end of the context
                idx = 0
                while sequence_ids[idx] != 1:
                    idx += 1
                context_start = idx
                while sequence_ids[idx] == 1:
                    idx += 1
                context_end = idx - 1

                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)

                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs

    return dataset.map(
        preprocess_samples, batched=batched, remove_columns=remove_columns
    )


def preprocess_squad2_test_dataset_no_stride(
    dataset: Dataset,
    tokenizer,
    max_length: int,
    batched: bool = True,
    remove_columns: list[str] = None,
):
    def preprocess_samples(samples):
        questions = [q.strip() for q in samples["question"]]
        contexts = [c.strip() for c in samples["context"]]

        inputs = tokenizer(
            questions,
            contexts,
            max_length=max_length,
            padding="max_length",
            return_offsets_mapping=True,
        )

        offset_mapping = inputs.pop("offset_mapping")
        answer_starts_batch = samples["answer_start"]
        answer_texts_batch = samples["answer_text"]
        start_positions = []
        end_positions = []
        example_ids = []
        new_offset_mapping = []

        for i, offset in enumerate(offset_mapping):
            example_id = samples["id"][i]
            answer_starts = answer_starts_batch[i]
            answer_texts = answer_texts_batch[i]
            sequence_ids = inputs.sequence_ids(i)
            sample_start_positions = []
            sample_end_positions = []

            # If there is no answer, label is (0, 0)
            if len(answer_starts) == 0 and len(answer_texts) == 0:
                sample_start_positions.append(0)
                sample_end_positions.append(0)
            # Otherwise, we try to find start and end token indices
            else:
                for j, answer_start in enumerate(answer_starts):
                    answer_text = answer_texts[j]
                    start_char = answer_start
                    end_char = start_char + len(answer_text)

                    # Find the start and end of the context
                    idx = 0
                    while sequence_ids[idx] != 1:
                        idx += 1
                    context_start = idx
                    while sequence_ids[idx] == 1:
                        idx += 1
                    context_end = idx - 1

                    idx = context_start
                    while idx <= context_end and offset[idx][0] <= start_char:
                        idx += 1
                    sample_start_positions.append(idx - 1)

                    idx = context_end
                    while idx >= context_start and offset[idx][1] >= end_char:
                        idx -= 1
                    sample_end_positions.append(idx + 1)

            new_offset_mapping.append(
                [o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)]
            )

            start_positions.append(sample_start_positions)
            end_positions.append(sample_end_positions)
            example_ids.append(example_id)

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        inputs["example_id"] = example_ids
        inputs["offset_mapping"] = new_offset_mapping
        return inputs

    return dataset.map(
        preprocess_samples, batched=batched, remove_columns=remove_columns
    )
