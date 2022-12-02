import os
import logging
import torch
import pickle
from tqdm import tqdm
import data_util
logger = logging.getLogger(__name__)

def write_pkl(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)


def read_pkl(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def load_and_cache_examples(args, tokenizer, evaluate=False):
    from train import get_time_kst
    mode = 'test' if evaluate else 'train'
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, '{}_{}_cached_nl_{}_{}_{}_{}_{}'.format(
        args.time,
        args.gpu,
        args.data_name, # durecdial
        mode, # train/test
        list(filter(None, args.model_name_or_path.split('/'))).pop(), # bart-base-chinese
        str(args.max_seq_length), # 512
        str(args.max_target_length))) # 100

    if os.path.exists(cached_features_file) and args.use_cached_data: # 파일있으면, 그대로 pkl load
        logger.info("Loading features from cached file %s", cached_features_file)
        features = read_pkl(cached_features_file)
        print("\nLoaded number of instance:", len(features['resp']['source_ids']))
    else: # 없으면 convert_to_features 함수를 통해 파일 생성 & pkl 파일로 저장
        logger.info("Creating features from dataset file at %s", args.data_dir)
        if args.goal_prompt_idea >= 1 : features = data_util.convert_to_features_goalPromptidea1_2_3(args,tokenizer,mode)
        # elif args.goal_prompt_idea == 2: features = data_util.convert_to_features_goalPromptidea2_instruction(args,tokenizer,mode) # HJ mode = evaluate일때 test, 아니면 train
        # elif args.goal_prompt_idea == 3: features = data_util.convert_to_features_goalPromptidea1(args,tokenizer,mode) # HJ mode = evaluate일때 test, 아니면 train
        else:  features = convert_to_features(args, tokenizer, mode) # HJ Default

        if args.save_tokenized_data:
            print("Loaded number of instance:", len(features['resp']['source_ids']))
            logger.info("Saving features into cached file %s", cached_features_file)
            write_pkl(features, cached_features_file)
    return features

def convert_to_features(args, tokenizer, mode):
    path = os.path.join(args.data_dir, '{}/item2id.txt'.format(args.data_name)) # item2id
    with open(path, 'r', encoding='utf-8') as infile:
        #item_dict = {0:'PAD'}
        item_dict = {}
        for line in infile:
            items = line.strip().split('\t')
            #item_dict[int(items[1])+1] = items[0]
            item_dict[int(items[1])] = items[0]
        item_dict[len(item_dict)] = '<PAD>' # item_dict의 맨 마지막에 <PAD> 추가

    if args.data_name == 'durecdial': # DuRecDial dataset에 대하여
        path = os.path.join(args.data_dir, 'kb_{}.jsonl'.format(args.data_name)) # HJ: outfile = kb_{}.jsonl 인데, 여기에 user profile이 담긴는거같음
        outfile = open(path, 'w', encoding='utf-8')
    path = os.path.join(args.data_dir, '{}/{}.jsonl'.format(args.data_name, mode)) # durecdial/test.jsonl
    print('tokenizing {}'.format(path))
    #print(tokenizer.SPECIAL_TOKENS_ATTRIBUTES)
    data_dict = {'resp':{'source_ids':[], 'target_ids':[], 'item_ids':[]}, 'item':{'source_ids':[], 'target_ids':[], 'item_ids':[]}, 'goal':{'source_ids':[], 'target_ids':[], 'item_ids':[]}, 'know':{'source_ids':[], 'target_ids':[], 'item_ids':[]}}

    logger.info('In Goal pred -- With Goal Sequence :: {}'.format(args.goal_input))
    logger.info('In Topic pred -- With Goal Sequence :: {}'.format(args.in_topic_with_goal_seq))
    logger.info('In Topic pred -- With Topic Sequence :: {}'.format(args.in_topic_with_topic_seq))
    with open(path, 'r', encoding='utf-8') as infile:
        max_dia_len,avg_dia_len,max_res_len,avg_res_len = 0,[],0,[]
        source_ids, target_ids, item_ids, hist_ids, rec_index = [],[],[],[],[]
        i = 0
        for line in tqdm(infile, desc="convert_to_features", bar_format=' {percentage:3.0f} % | {bar:23} {r_bar}'):
            d = eval(line.strip())
            know = d['knowledge']
            conv = d['conversation']
            source_id = []
            source_know_id = []
            source_goal_id = []
            target_id = []
            hist_id = know['item_history'] if len(know['item_history'])>0 else [len(item_dict)-1]
            #hist_id = tokenizer.encode('[history]' + '|'.join(['<'+str(x)+'>' for x in know['item_history']]))[1:]
            profile_id = tokenizer.encode('[profile]' + '|'.join(know['user_profile']))[1:]
            ## HJ 첫번째 발화
            first_utt = conv[0]
            if first_utt['role'] == 'user' and args.data_name == 'durecdial': pass # user가 먼저 말했고 durecdial이라면? pass
            else: # # System 이 먼저 말했다면
                if type(first_utt['goal']) is list:
                    first_utt['goal'] = '|'.join(first_utt['goal'])
                source_goal_id += tokenizer.encode('[goal]' + first_utt['goal'])[1:]
                source_know_id += tokenizer.encode('[knowledge]' + '|'.join(first_utt['knowledge']))[1:]

            source_id += tokenizer.encode('[{}]'.format(first_utt['role']) + first_utt['utterance'])[1:] # [USER]/[SYS] blablabla [SEP]
            if 'dialog' in args.goal_input:
                source_goal_id += tokenizer.encode('[{}]'.format(first_utt['role']) + first_utt['utterance'])[1:]  # HJ GoalSeq관련처리-1 [USER]/[SYS] blablabla [SEP]
            source_know_id += tokenizer.encode('[{}]'.format(first_utt['role']) + first_utt['utterance'])[1:] # HJ : goal id 와 know id 가 같게된다?
            ## HJ 두번째 발화부터
            for utt in conv[1:]: # 2번째 대화부터
                if utt['role'] == 'user':# and args.data_name == 'durecdial': # User가 말했다면
                    source_id += tokenizer.encode('[user]' + utt['utterance'])[1:]
                    if args.data_name == 'tgredial':
                        source_know_id += tokenizer.encode('[knowledge]' + '|'.join(utt['knowledge']))[1:]
                        source_goal_id += tokenizer.encode('[goal]' + '|'.join(utt['goal']))[1:]
                    source_know_id += tokenizer.encode('[user]' + utt['utterance'])[1:] # [0] == [CLS] 빼고
                    if 'dialog' in args.goal_input: # HJ GoalSeq관련처리-1 [USER]/[SYS] blablabla [SEP]
                        source_goal_id += tokenizer.encode('[user]' + utt['utterance'])[1:]
                    continue
                if type(utt['goal']) is list: # 골이 여러개라면? 한문장으로 붙여넣기
                    utt['goal'] = '|'.join(utt['goal'])

                ### prepare response generation data
                target_id = tokenizer.encode(utt['utterance']) # System 발화
                know_len = int(args.max_seq_length/2)
                if args.data_name == 'tgredial': # 生成回复 == 응답생성 (GPT한테 응답생성하라고 키워드넣는느낌)
                    new_source_id = source_id + tokenizer.encode('[goal]' + utt['goal'])[1:] + tokenizer.encode('[knowledge]')[1:-1] + tokenizer.encode('|'.join(utt['knowledge']))[1:][-know_len:] + tokenizer.encode('[item]' + '|'.join(utt['item']))[1:] + tokenizer.encode('生成回复：')[1:]
                else:
                    new_source_id = source_id + tokenizer.encode('[goal]' + utt['goal'])[1:] + tokenizer.encode('[knowledge]')[1:-1] + tokenizer.encode('|'.join(utt['know_text']))[1:][-know_len:] + tokenizer.encode('[item]' + '|'.join(utt['item']))[1:] + tokenizer.encode('生成回复：')[1:]
                    if mode == 'test':
                        outfile.write(str(know['knowledge']) + '\n') # HJ: outfile = kb_{}.jsonl 인데, 여기에 user profile이 담긴는거같음


                source_ids.append([101] + new_source_id[-args.max_seq_length+1:])
                target_ids.append([101] + target_id[-args.max_target_length+1:])
                item_ids.append([len(item_dict)-1]) # [PAD]
                data_dict['resp']['source_ids'].append(source_ids[-1])
                data_dict['resp']['target_ids'].append(target_ids[-1])
                data_dict['resp']['item_ids'].append(item_ids[-1])

                avg_dia_len.append(len(new_source_id))
                max_dia_len = max(max_dia_len, len(new_source_id))
                avg_res_len.append(len(target_id))
                max_res_len = max(max_res_len, len(target_id))

                ### prepare goal selection data
                target_id = tokenizer.encode(utt['goal'])
                new_source_id = source_goal_id + tokenizer.encode('计划下一个目标：')[1:] # HJ Natural Language Prompt -- plan the next goal

                if 'dialog' in args.goal_input and 'goal' in args.goal_input: # HJ GoalSeq관련처리-1
                    source_goal_id += (tokenizer.encode('[goal]' + utt['goal'])[1:] + tokenizer.encode('[{}]'.format(utt['role']) + utt['utterance'])[1:])  # HJ Goal Sequence 를 넣고, Next Goal 예측할때
                elif 'dialog' in args.goal_input:  # dialog 만 받을때
                    source_goal_id += (tokenizer.encode('[{}]'.format(utt['role']) + utt['utterance'])[1:])
                elif 'goal' in args.goal_input:  # goal 만 받을 때
                    source_goal_id += (tokenizer.encode('[goal]' + utt['goal'])[1:])
                else:
                    print('goal 인풋 체크')
                    assert 0

                source_ids.append([101] + new_source_id[-args.max_seq_length+1:])
                target_ids.append([101] + target_id[-args.max_target_length+1:])
                item_ids.append([len(item_dict)-1])
                data_dict['goal']['source_ids'].append(source_ids[-1])
                data_dict['goal']['target_ids'].append(target_ids[-1])
                data_dict['goal']['item_ids'].append(item_ids[-1])

                ### prepare topic prediction data
                target_id = tokenizer.encode('|'.join(utt['knowledge']))
                if args.in_topic_with_goal_seq=='T':
                    new_source_id = profile_id + source_know_id + tokenizer.encode('[goal]' + utt['goal'])[1:] + tokenizer.encode('预测下一个话题：')[1:] # HJ Topic예측시 Goal 빼고 다 넣기 Default # HJ prompt 다음주제예측
                else:
                    new_source_id = profile_id + source_know_id + tokenizer.encode('预测下一个话题：')[1:] # # HJ Topic예측시 Goal 빼고 다 넣기
                #new_source_id = profile_id + source_know_id + tokenizer.encode('[knowledge]')[1:] # HJ Topic예측시 Goal 빼고 다 넣기
                # source_know_id += (tokenizer.encode('[knowledge]' + '|'.join(utt['knowledge']))[1:] + tokenizer.encode('[{}]'.format(utt['role']) + utt['utterance'])[1:]) # HJ Default 토픽예측에서 토픽시퀀스 살아있는상태
                if args.in_topic_with_topic_seq=='T':
                    source_know_id += (tokenizer.encode('[knowledge]' + '|'.join(utt['knowledge']))[1:] + tokenizer.encode('[{}]'.format(utt['role']) + utt['utterance'])[1:])
                else:
                    source_know_id += (tokenizer.encode('[{}]'.format(utt['role']) + utt['utterance'])[1:]) # HJ: Topic 예측시 Topic Seq 빼고 보기
                source_ids.append([101] + new_source_id[-args.max_seq_length+1:])
                target_ids.append([101] + target_id[-args.max_target_length+1:])
                item_ids.append([len(item_dict)-1])
                data_dict['know']['source_ids'].append(source_ids[-1])
                data_dict['know']['target_ids'].append(target_ids[-1])
                data_dict['know']['item_ids'].append(item_ids[-1])

                ### prepare item recommendation data
                if len(utt['item_id']) > 0:
                    target_text = []
                    for item, item_id in zip(utt['item'], utt['item_id']):
                        target_text.append('<'+str(item_id)+'>'+item)
                    target_id = tokenizer.encode('|'.join(target_text))
                    new_source_id = profile_id + source_id + tokenizer.encode('[goal]' + utt['goal'])[1:] + tokenizer.encode('[knowledge]' + '|'.join(utt['knowledge']))[1:] + tokenizer.encode('推荐：')[1:] # HJ: 상품 prompt
                    item_id = utt['item_id']
                    source_ids.append([101] + new_source_id[-args.max_seq_length+1:])
                    target_ids.append([101] + target_id[-args.max_target_length+1:])
                    item_ids.append(item_id)
                    data_dict['item']['source_ids'].append(source_ids[-1])
                    data_dict['item']['target_ids'].append(target_ids[-1])
                    data_dict['item']['item_ids'].append(item_ids[-1])
                    rec_index.append(i)
                i += 1

                source_id += tokenizer.encode('[{}]'.format(utt['role']) + utt['utterance'])[1:]

                #hist_ids.append(hist_id)
                #hist_id.extend(item_id)

        print('{} set, max_res_len: {}, max_dia_len: {}, avg_res_len: {}, avg_dia_len: {}'.format(mode, max_res_len, max_dia_len, float(sum(avg_res_len))/len(avg_res_len), float(sum(avg_dia_len))/len(avg_dia_len)))

    if mode == 'train':
        #return {'source_ids':source_ids, 'target_ids':target_ids, 'item_ids':item_ids, 'item_dict':item_dict}
        data_dict['item_dict'] = item_dict
        return data_dict
    else: # eval, test??
        data_dict['item_dict'] = item_dict
        data_dict['rec_index'] = rec_index
        return data_dict

def merge_dataset(ft_dataset):
    source_ids = []
    target_ids = []
    item_ids = []
    item_dict = ft_dataset['item_dict']
    for task in ['resp','goal','know','item']:
        task_dataset = ft_dataset[task]

        for source_id, target_id, item_id in tqdm(zip(task_dataset['source_ids'], task_dataset['target_ids'], task_dataset['item_ids']), desc="merge_dataset", bar_format=' {percentage:3.0f} % | {bar:23} {r_bar}', total=len(task_dataset['source_ids'])):
            source_ids.append(source_id)
            target_ids.append(target_id)
            item_ids.append(item_id)
    return {'source_ids':source_ids, 'target_ids':target_ids, 'item_ids':item_ids, 'item_dict':item_dict}

def process_pipeline_data(args, tokenizer, data, all_preds, task):
    if task == 'resp':
        if args.data_name == 'durecdial':
            path = os.path.join(args.data_dir, 'kb_{}.jsonl'.format(args.data_name))
            kbs = []
            with open(path, 'r', encoding='utf-8') as infile:
                for line in infile:
                    kbs.append(eval(line.strip('\n')))
            assert len(kbs) == len(data['resp']['source_ids']) #HJ kb_{}.jsonl 이 뭔파일이지? User Profile 같기도한데
        sid = 21128 # ([101, 21128, 102], '[CLS] [goal] [SEP]')
        new_source_ids = []
        count = 0
        rec_index = data['rec_index']
        item_dict = data['item_dict']
        i = 0
        j = 0
        error_resp_count=0
        for source_id, goal_pred, know_pred in tqdm(zip(data['resp']['source_ids'], all_preds['goal'], all_preds['know']), desc="pipeline_resp", bar_format=' {percentage:3.0f} % | {bar:23} {r_bar}', total=len(data['resp']['source_ids'])):
            # assert source_id.count(sid) <= 1 ## Default HJ Error 수정시도
            if source_id.count(sid) <= 1 : pass
            else: ## HJ Error 수정시도
                error_resp_count+=1
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
                        know_text.append(obj+'：'+tup)
                    elif type(tup) is dict:
                        flag = True
                        for key in tup:
                            if key in knows:
                                know_text.append(obj+'，'+key+'，'+'、'.join(tup[key]))
                                flag = False
                        if flag:
                            for key in tup:
                                know_text.append(obj+'，'+key+'，'+'、'.join(tup[key]))
                if len(know_text) == 0 and knows != ['']:
                    for obj in kb:
                        tup = kb[obj]
                        if type(tup) is str:
                            continue
                        else:
                            for key in tup:
                                know_text.append(obj+'，'+key+'，'+'、'.join(tup[key]))
                know_pred = '|'.join(know_text)
            else:
                know_pred = ''.join(know_pred.split(' '))

            if j in rec_index:
                item_pred = item_dict[all_preds['item'][i][0]]
                i += 1
            else:
                item_pred = ''
            j += 1

            know_len = int(args.max_seq_length/2)
            source_id += (tokenizer.encode('[goal]' + goal_pred)[1:] + tokenizer.encode('[knowledge]')[1:-1] + tokenizer.encode(know_pred)[1:][-know_len:] + tokenizer.encode('[item]' + item_pred)[1:] + uid)
            new_source_ids.append([101] + source_id[-args.max_seq_length+1:])
            if old_source_id == new_source_ids[-1]:
                count += 1
            else:
                pass
                # print(know_pred)
                # print(tokenizer.decode(old_source_id, skip_special_tokens=True, clean_up_tokenization_spaces=True))
                #print(tokenizer.decode(new_source_ids[-1], skip_special_tokens=True, clean_up_tokenization_spaces=True))
        print(float(count)/len(new_source_ids))
        data['resp']['source_ids'] = new_source_ids
        logger.info(f"파이프라인 task resp : {error_resp_count} 만큼 resp 에서 sid로 인한 에러 발생(without goalseq 관련")
        return data['resp']
    elif task == 'know': ### HJ: Topic 예측 pipeline :: UserProfile + Dialog + Topic Thread + goal(gt) + [Topic]
        sid = 21128
        new_source_ids = []
        count = 0

        for source_id, pred in tqdm(zip(data['know']['source_ids'], all_preds['goal']), desc="pipeline_know", bar_format=' {percentage:3.0f} % | {bar:23} {r_bar}', total=len(data['know']['source_ids'])):
            # assert source_id.count(sid) == 1 ## Default TODO:HJ no_goal_seq에서 Error 터졌음
            if source_id.count(sid) == 1 :
                old_source_id = source_id.copy()
                source_id = source_id[1:source_id.index(sid)]
            else: ## HJ Error 수정시도
                # logger.info("source_id.count(sid) == 1 294번째 줄 Goal 시 에러")
                old_source_id = source_id.copy()
                source_id = source_id[1:list(filter(lambda x : x[1]==102, enumerate(source_id)))[-2][0]+1] # HJ: Goal Seq가 사라진 후, 다음 주제 예측: [SEP]에서 해당부분 삭제
            source_id += tokenizer.encode('[goal]' + ''.join(pred.split(' ')))[1:] + tokenizer.encode('预测下一个话题：')[1:]
            #print(old_source_id, source_id[source_id.index(sid):])
            new_source_ids.append([101] + source_id[-args.max_seq_length+1:]) # [CLS] + source_id~
            if old_source_id == new_source_ids[-1]:
                count += 1
            else:
                pass
                #print(tokenizer.decode(old_source_id, skip_special_tokens=True, clean_up_tokenization_spaces=True))
                #print(tokenizer.decode(new_source_ids[-1], skip_special_tokens=True, clean_up_tokenization_spaces=True))
        print(float(count)/len(new_source_ids))
        data['know']['source_ids'] = new_source_ids
        return data['know']
    elif task == 'item':
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
        for source_id, pred, pred_know in tqdm(zip(data['item']['source_ids'], filtered_preds, filtered_knows), desc="pipeline_item",bar_format=' {percentage:3.0f} % | {bar:23} {r_bar}', total=len(data['item']['source_ids'])):
            # assert source_id.count(sid) == 1 # Default (TODO:HJ no_goal_seq에서 Error 터졌음)
            if source_id.count(sid) == 1 :
                old_source_id = source_id.copy()
                source_id = source_id[1:source_id.index(sid)]
            else: ## HJ Error 수정시도
                # logger.info("source_id.count(sid) == 1 311번째 줄 item 시 에러")
                old_source_id = source_id.copy()
                source_id = source_id[1:list(filter(lambda x : x[1]==102, enumerate(source_id)))[-2][0]+1] # HJ: Goal Seq가 사라진 후, 다음 주제 예측: [SEP]에서 해당부분 삭제
            source_id += tokenizer.encode('[goal]' + ''.join(pred.split(' ')))[1:] + tokenizer.encode('[knowledge]' + ''.join(pred_know.split(' ')))[1:] + tokenizer.encode('推荐：')[1:]
            new_source_ids.append([101] + source_id[-args.max_seq_length+1:])
            if old_source_id == new_source_ids[-1]:
                count += 1
            else:
                pass
                #print(tokenizer.decode(old_source_id, skip_special_tokens=True, clean_up_tokenization_spaces=True))
                #print(tokenizer.decode(new_source_ids[-1], skip_special_tokens=True, clean_up_tokenization_spaces=True))
        print("count가 old_source_id == new_source_ids[-1] 와 연관되어있는데 확인필요",float(count)/len(new_source_ids)) # HJ: Goal seq 사라진 후 처리에서 count 확인
        data['item']['source_ids'] = new_source_ids
        return data['item']