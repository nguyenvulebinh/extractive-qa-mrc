import datasets
from datasets import concatenate_datasets
import glob

datasets.temp_seed(101)
datasets.disable_progress_bar()


def assert_sample(sample):
    assert sample['context'][sample['answer_start_idx']: sample['answer_start_idx'] + len(sample['answer_text'])] == \
           sample['answer_text'], sample
    assert len(sample['context']) > 0
    assert len(sample['question']) > 0
    return True


def format_sample(sample):
    context_prev = sample['context'][:sample['answer_start_idx']].split()
    sample['answer_word_start_idx'] = len(context_prev)
    sample['answer_word_end_idx'] = len(context_prev) + len(sample['answer_text'].split()) - 1
    return sample


if __name__ == "__main__":
    train_set = []
    valid_set = []
    for part in glob.glob('../data-bin/unify/*.jsonl'):
        dataset = datasets.load_dataset('json', data_files=[part])['train']
        dataset.filter(assert_sample)
        dataset = dataset.map(format_sample)

        all_data = dataset.train_test_split(test_size=0.1)
        train = all_data['train']
        valid = all_data['test']
        train_set.append(train)
        valid_set.append(valid)

    train_dataset = concatenate_datasets(train_set)
    valid_dataset = concatenate_datasets(valid_set)

    train_dataset.save_to_disk('../data-bin/processed/train.dataset')
    valid_dataset.save_to_disk('../data-bin/processed/valid.dataset')

    print("Train: {} samples".format(len(train_dataset)))
    print("Valid: {} samples".format(len(valid_dataset)))
