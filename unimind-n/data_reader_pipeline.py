import os
import logging
import torch
import pickle
from tqdm import tqdm
logger = logging.getLogger(__name__)

def process_pipeline_know(args, tokenizer, data, all_preds, task='know'): # Topic
    sid = 21128
    new_source_ids = []
    count = 0

    for source_id, pred in tqdm(zip(data['know']['source_ids'], all_preds['goal']), desc="pipeline_know",
                                bar_format=' {percentage:3.0f} % | {bar:23} {r_bar}',
                                total=len(data['know']['source_ids'])):
        # assert source_id.count(sid) == 1 ## Default TODO:HJ no_goal_seq에서 Error 터졌음
        if source_id.count(sid) == 1:
            old_source_id = source_id.copy()
            source_id = source_id[1:source_id.index(sid)]
        else:  ## HJ Error 수정시도
            # logger.info("source_id.count(sid) == 1 294번째 줄 Goal 시 에러")
            old_source_id = source_id.copy()
            source_id = source_id[1:list(filter(lambda x: x[1] == 102, enumerate(source_id)))[-2][
                                        0] + 1]  # HJ: Goal Seq가 사라진 후, 다음 주제 예측: [SEP]에서 해당부분 삭제
        source_id += tokenizer.encode('[goal]' + ''.join(pred.split(' ')))[1:] + tokenizer.encode('预测下一个话题：')[1:]
        # print(old_source_id, source_id[source_id.index(sid):])
        new_source_ids.append([101] + source_id[-args.max_seq_length + 1:])  # [CLS] + source_id~
        if old_source_id == new_source_ids[-1]:
            count += 1
        else:
            pass
            # print(tokenizer.decode(old_source_id, skip_special_tokens=True, clean_up_tokenization_spaces=True))
            # print(tokenizer.decode(new_source_ids[-1], skip_special_tokens=True, clean_up_tokenization_spaces=True))
    print(float(count) / len(new_source_ids))
    data['know']['source_ids'] = new_source_ids
    return data['know']

def process_pipeline_item(args, tokenizer, data, all_preds, task='item'):
    rec_index = data['rec_index']
    filtered_preds = []
    filtered_knows = []
    for i, pred in enumerate(all_preds['goal']):
        if i in rec_index:
            filtered_preds.append(pred)
    for i, pred in enumerate(all_preds['know']):
        if i in rec_index:
            filtered_knows.append(pred)
    assert len(filtered_preds) == len(data['item']['source_ids'])
    assert len(filtered_knows) == len(data['item']['source_ids'])
    sid = 21128
    new_source_ids = []
    count = 0
    for source_id, pred, pred_know in tqdm(zip(data['item']['source_ids'], filtered_preds, filtered_knows),
                                           desc="pipeline_item", bar_format=' {percentage:3.0f} % | {bar:23} {r_bar}',
                                           total=len(data['item']['source_ids'])):
        # assert source_id.count(sid) == 1 # Default (TODO:HJ no_goal_seq에서 Error 터졌음)
        if source_id.count(sid) == 1:
            old_source_id = source_id.copy()
            source_id = source_id[1:source_id.index(sid)]
        else:  ## HJ Error 수정시도
            # logger.info("source_id.count(sid) == 1 311번째 줄 item 시 에러")
            old_source_id = source_id.copy()
            source_id = source_id[1:list(filter(lambda x: x[1] == 102, enumerate(source_id)))[-2][
                                        0] + 1]  # HJ: Goal Seq가 사라진 후, 다음 주제 예측: [SEP]에서 해당부분 삭제
        source_id += tokenizer.encode('[goal]' + ''.join(pred.split(' ')))[1:] + tokenizer.encode(
            '[knowledge]' + ''.join(pred_know.split(' ')))[1:] + tokenizer.encode('推荐：')[1:]
        new_source_ids.append([101] + source_id[-args.max_seq_length + 1:])
        if old_source_id == new_source_ids[-1]:
            count += 1
        else:
            pass
            # print(tokenizer.decode(old_source_id, skip_special_tokens=True, clean_up_tokenization_spaces=True))
            # print(tokenizer.decode(new_source_ids[-1], skip_special_tokens=True, clean_up_tokenization_spaces=True))
    print("count가 old_source_id == new_source_ids[-1] 와 연관되어있는데 확인필요",
          float(count) / len(new_source_ids))  # HJ: Goal seq 사라진 후 처리에서 count 확인
    data['item']['source_ids'] = new_source_ids
    return data['item']

def process_pipeline_resp(args, tokenizer, data, all_preds, task='resp'):
    if args.data_name == 'durecdial':
        path = os.path.join(args.data_dir, 'kb_{}.jsonl'.format(args.data_name))
        kbs = []
        with open(path, 'r', encoding='utf-8') as infile:
            for line in infile:
                kbs.append(eval(line.strip('\n')))
        assert len(kbs) == len(data['resp']['source_ids'])  # HJ kb_{}.jsonl 이 뭔파일이지? User Profile 같기도한데
    sid = 21128  # ([101, 21128, 102], '[CLS] [goal] [SEP]')
    new_source_ids = []
    count = 0
    rec_index = data['rec_index']
    item_dict = data['item_dict']
    i = 0
    j = 0
    error_resp_count = 0
    for source_id, goal_pred, know_pred in tqdm(zip(data['resp']['source_ids'], all_preds['goal'], all_preds['know']),
                                                desc="pipeline_resp",
                                                bar_format=' {percentage:3.0f} % | {bar:23} {r_bar}',
                                                total=len(data['resp']['source_ids'])):
        # assert source_id.count(sid) <= 1 ## Default HJ Error 수정시도
        if source_id.count(sid) <= 1:
            pass
        else:  ## HJ Error 수정시도
            error_resp_count += 1
        old_source_id = source_id.copy()
        uid = source_id[-6:]
        if sid in source_id:
            source_id = source_id[1:source_id.index(sid)]
        else:
            source_id = []
        goal_pred = ''.join(goal_pred.split(' '))
        if args.data_name == 'durecdial':
            kb = kbs[j]
            know_text = []
            knows = ''.join(know_pred.split(' ')).split('|')
            for obj in knows:
                if obj not in kb:
                    continue
                tup = kb[obj]
                if type(tup) is str:
                    know_text.append(obj + '：' + tup)
                elif type(tup) is dict:
                    flag = True
                    for key in tup:
                        if key in knows:
                            know_text.append(obj + '，' + key + '，' + '、'.join(tup[key]))
                            flag = False
                    if flag:
                        for key in tup:
                            know_text.append(obj + '，' + key + '，' + '、'.join(tup[key]))
            if len(know_text) == 0 and knows != ['']:
                for obj in kb:
                    tup = kb[obj]
                    if type(tup) is str:
                        continue
                    else:
                        for key in tup:
                            know_text.append(obj + '，' + key + '，' + '、'.join(tup[key]))
            know_pred = '|'.join(know_text)
        else:
            know_pred = ''.join(know_pred.split(' '))

        if j in rec_index:
            item_pred = item_dict[all_preds['item'][i][0]]
            i += 1
        else:
            item_pred = ''
        j += 1

        know_len = int(args.max_seq_length / 2)
        source_id += (tokenizer.encode('[goal]' + goal_pred)[1:] + tokenizer.encode('[knowledge]')[
                                                                   1:-1] + tokenizer.encode(know_pred)[1:][
                                                                           -know_len:] + tokenizer.encode(
            '[item]' + item_pred)[1:] + uid)
        new_source_ids.append([101] + source_id[-args.max_seq_length + 1:])
        if old_source_id == new_source_ids[-1]:
            count += 1
        else:
            pass
            # print(know_pred)
            # print(tokenizer.decode(old_source_id, skip_special_tokens=True, clean_up_tokenization_spaces=True))
            # print(tokenizer.decode(new_source_ids[-1], skip_special_tokens=True, clean_up_tokenization_spaces=True))
    print(float(count) / len(new_source_ids))
    data['resp']['source_ids'] = new_source_ids
    logger.info(f"파이프라인 task resp : {error_resp_count} 만큼 resp 에서 sid로 인한 에러 발생(without goalseq 관련")
    return data['resp']
