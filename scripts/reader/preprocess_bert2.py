import json
import os
import argparse
import datetime
import sys

from hdf5_util import *
from drqa.reader import BERTEmbedder

sys.stdout.flush()

def load_dataset(input_file):
    """Read a list of `InputExample`s from an input file."""
    dataset = []
    with open(input_file) as f:
        data = json.load(f)['data']
        for article in data:
            dataset.extend(article['paragraphs'])
    return dataset

def load_dataset2(input_file):
    """Read a list of `InputExample`s from an input file."""
    dataset = {'contexts': {}, 'qas': {}}
    cid = 1
    with open(input_file) as f:
        data = json.load(f)['data']
        for article in data:
            for para in article['paragraphs']:
                ctx = {'cid': str(cid), 'context': para['context']}
                qas = para['qas']
                for qa in qas:
                    qa['cid'] = str(cid)
                    dataset['qas'][qa['id']] = qa
                dataset['contexts'][str(cid)] = ctx
                cid += 1
            # break
    return dataset

def find_answer(ctx_token_ids, ans_token_ids):
    """Match token offsets with the char begin/end offsets of the answer."""
    # TODO: use KMP to improve performance
    ans_index = 0
    ctx_start = 0
    ctx_index = 0
    while ctx_start < len(ctx_token_ids) - len(ans_token_ids):
        if ans_token_ids[ans_index] == ctx_token_ids[ctx_index]:
            ans_index += 1
            ctx_index += 1
            if ans_index == len(ans_token_ids):
                return [ctx_start, ctx_index]
        else:
            ctx_start += 1
            ctx_index = ctx_start
            answer_index = 0
    return None

def save_hdf5(processed_dataset, filename):
    save_dict_to_hdf5(processed_dataset, filename)

def load_hdf5(filename):
    return load_dict_from_hdf5(filename)

def process_dataset_batch(dataset, bert_path, workers=None):
    """preprocess dataset to BERT representation"""

    bert_embedder = BERTEmbedder(bert_path)

    processed_dataset = {'contexts': {}, 'qas': {}}

    ctxs = list(dataset['contexts'].values())
    batch_size = 64
    for bid in range(0, len(ctxs), batch_size):
        print('batch %d:%d start' % (bid, bid + batch_size))
        start_time = datetime.datetime.now()     #放在程序开始处
        batch = ctxs[bid:(bid + batch_size)]
        docs = [ ctx['context'] for ctx in batch ]
        (ctx_bert_predict, ctx_raw_features) = bert_embedder.embed_documents(docs)
        i = 0
        for pred in ctx_bert_predict:
            cid = batch[i]['cid']
            ctx_tokens = ctx_raw_features[i].tokens
            # processed_ctx = {'bert_features': np.asarray(pred['layer_output_0'].tolist()), 'cid': cid, 'tokens': ctx_tokens, 'token_ids': np.asarray(ctx_raw_features[i].input_ids)}
            processed_ctx = {'bert_features': np.asarray(pred['layer_output_0'].tolist()), 'cid': cid, 'token_ids': ctx_raw_features[i].input_ids}
            processed_dataset['contexts'][cid] = processed_ctx
            i += 1
        end_time = datetime.datetime.now()      #放在程序结尾处
        interval = (end_time-start_time).seconds    #以秒的形式
        print('batch %d end, time spent: %d' % (bid, interval))

    qas = list(dataset['qas'].values())
    for qid in range(0, len(qas), batch_size):
        batch = qas[bid:(bid + batch_size)]
        ques = [ qa['question'] for qa in batch ]
        (que_bert_predicts, que_raw_features) = bert_embedder.embed_questions(ques)
        i = 0
        for pred in que_bert_predicts:
            qa = batch[i]
            i += 1
            cid = qa['cid']
            ans_text = qa['answers'][0]['text']
            answer_token_ids = bert_embedder.convert_txt_to_token_ids(qa['answers'][0]['text'])
            answer_offsets = find_answer(processed_dataset['contexts'][cid]['token_ids'], answer_token_ids)
            if answer_offsets is not None:
                processed_qa = {
                    'bert_features': np.asarray(pred['layer_output_0'].tolist()), 
                    'qid': qa['id'], 
                    'cid': cid, 
                    'answer_offsets': np.asarray(answer_offsets)
                }
                processed_dataset['qas'][qa['id']] = processed_qa

    for ctx in processed_dataset['contexts']:
        del processed_dataset['contexts'][ctx]['token_ids']

    return processed_dataset


def process_dataset(dataset, bert_path, workers=None):
    """Iterate processing (tokenize, parse, etc) dataset multithreaded."""
    bert_embedder = BERTEmbedder(bert_path)

    processed_dataset = {'contexts': {}, 'qas': {}}
    cid = 1
    for ctx_and_qas in dataset:
        context = ctx_and_qas['context']
        qas = ctx_and_qas['qas']
        (ctx_bert_predict, ctx_raw_features) = bert_embedder.embed_document(context)
        ctx_bert_features = None
        for result in ctx_bert_predict:
            ctx_bert_features = result['layer_output_0']
        ctx_token_ids = ctx_raw_features[0].input_ids
        ctx_tokens = ctx_raw_features[0].tokens
        processed_ctx = {'bert_features': np.asarray(ctx_bert_features.tolist()), 'cid': cid, 'tokens': json.dumps(ctx_tokens)}
        processed_dataset['contexts'][cid] = processed_ctx
        for qa in qas:
            question = qa['question']
            (q_bert_predict, q_raw_features) = bert_embedder.embed_question(question)
            q_bert_features = None
            for result in q_bert_predict:
                q_bert_features = result['layer_output_0']
            answer_token_ids = bert_embedder.convert_txt_to_token_ids(qa['answers'][0]['text'])
            answer_offsets = find_answer(ctx_token_ids, answer_token_ids)
            if answer_offsets is not None:
                processed_qa = {'bert_features': np.asarray(q_bert_features.tolist()), 'qid': qa['id'], 'cid': cid, 'answer_offsets': answer_offsets}
                processed_dataset['qas'][qa['id']] = processed_qa
        cid += 1
    return processed_dataset


if __name__ == '__main__':
    # processed_dataset = load_hdf5('test.hdf5')
    # print(processed_dataset.keys())
    # exit()
    # save_dict_to_hdf5({'cid': '1212', 'bert_features': np.asarray([[1, 2, 3]])}, 'test.hdf5')
    # exit()
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, help='Path to SQuAD data directory')
    parser.add_argument('--out_dir', type=str, help='Path to output file dir')
    parser.add_argument('--bert_path', type=str, help='Path to output file dir')
    # parser.add_argument('--split', type=str, help='Filename for train/dev split',
    #                     default='SQuAD-v1.1-train')
    # # parser.add_argument('--workers', type=int, default=None)
    # # parser.add_argument('--tokenizer', type=str, default='corenlp')
    args = parser.parse_args()
    # bert_path = '/Users/weize/Workspace/VENV-3.6/workspace/bert/uncased_L-12_H-768_A-12'
    dataset = load_dataset2(os.path.join(args.data_dir, 'SQuAD-v1.1-train.json'))
    print('SQuAD-v1.1 dataset context count: %d, question count: %d ' % (len(dataset['contexts']), len(dataset['qas'])))
    processed_dataset = process_dataset_batch(dataset, args.bert_path)
    # print(processed_dataset)
    print(len(processed_dataset['contexts']))
    print(len(processed_dataset['qas']))
    save_hdf5(processed_dataset, 'bert.hdf5')
    # with open(os.path.join(args.out_dir, 'SQUAD-v1.1-train-processed-bert.json'), 'w') as file:
    #     json.dump(processed_dataset, file)


    