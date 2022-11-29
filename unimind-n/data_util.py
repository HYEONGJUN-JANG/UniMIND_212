import os, logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

def convert_to_features_goalPromptidea1_2_3(args, tokenizer, mode):
    '''
    dialog + goal seq를 섞어넣는것이 아닌, dialog만 넣고, goal seq를 뒤에 쭈르륵 넣는식
    '''
    path = os.path.join(args.data_dir, '{}/item2id.txt'.format(args.data_name)) # item2id
    with open(path, 'r', encoding='utf-8') as infile:
        item_dict = {}
        for line in infile:
            items = line.strip().split('\t')
            item_dict[int(items[1])] = items[0]
        item_dict[len(item_dict)] = '<PAD>' # item_dict의 맨 마지막에 <PAD> 추가

    if args.data_name == 'durecdial': # DuRecDial dataset에 대하여
        path = os.path.join(args.data_dir, 'kb_{}.jsonl'.format(args.data_name)) # HJ: outfile = kb_{}.jsonl 인데, 여기에 user profile이 담긴는거같음
        outfile = open(path, 'w', encoding='utf-8')
    path = os.path.join(args.data_dir, '{}/{}.jsonl'.format(args.data_name, mode)) # durecdial/test.jsonl
    print('tokenizing {}'.format(path))
    data_dict = {'resp':{'source_ids':[], 'target_ids':[], 'item_ids':[]}, 'item':{'source_ids':[], 'target_ids':[], 'item_ids':[]}, 'goal':{'source_ids':[], 'target_ids':[], 'item_ids':[]}, 'know':{'source_ids':[], 'target_ids':[], 'item_ids':[]}}

    logger.info('convert_to_features_goal Prompt idea1 사용')
    if args.goal_instruction: logger.info('Goal instruction 사용')
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
            source_id,source_know_id,source_goal_id = [],[],[]
            target_id = []
            hist_id = know['item_history'] if len(know['item_history'])>0 else [len(item_dict)-1]
            profile_id = tokenizer.encode('[profile]' + '|'.join(know['user_profile']))[1:]
            # HJ: Goal sequence 분리용
            gs_goal_list=[]
            gs_utt_list=[]
            ## HJ 첫번째 발화
            first_utt = conv[0]
            if first_utt['role'] == 'user' and args.data_name == 'durecdial': pass # user가 먼저 말했고 durecdial이라면? pass
            else: # # System 이 먼저 말했다면
                if type(first_utt['goal']) is list:
                    first_utt['goal'] = '|'.join(first_utt['goal'])
                source_goal_id += tokenizer.encode('[goal]' + first_utt['goal'])[1:]
                source_know_id += tokenizer.encode('[knowledge]' + '|'.join(first_utt['knowledge']))[1:]

            source_id += tokenizer.encode('[{}]'.format(first_utt['role']) + first_utt['utterance'])[1:] # [USER]/[SYS] blablabla [SEP]
            gs_utt_list += tokenizer.encode('[{}]'.format(first_utt['role']) + first_utt['utterance'])[1:] # [USER]/[SYS] blablabla [SEP]
            if 'dialog' in args.goal_input:
                source_goal_id += tokenizer.encode('[{}]'.format(first_utt['role']) + first_utt['utterance'])[1:]  # HJ GoalSeq관련처리-1 [USER]/[SYS] blablabla [SEP]
            source_know_id += tokenizer.encode('[{}]'.format(first_utt['role']) + first_utt['utterance'])[1:] # HJ : goal id 와 know id 가 같게된다?
            ## HJ 두번째 발화부터
            for utt in conv[1:]: # 2번째 대화부터
                if utt['role'] == 'user': # and args.data_name == 'durecdial': # User가 말했다면
                    source_id += tokenizer.encode('[user]' + utt['utterance'])[1:]
                    if args.data_name == 'tgredial':
                        source_know_id += tokenizer.encode('[knowledge]' + '|'.join(utt['knowledge']))[1:]
                        source_goal_id += tokenizer.encode('[goal]' + '|'.join(utt['goal']))[1:]
                    source_know_id += tokenizer.encode('[user]' + utt['utterance'])[1:] # [0] == [CLS] 빼고
                    if 'dialog' in args.goal_input: # HJ GoalSeq관련처리-1 [USER]/[SYS] blablabla [SEP]
                        source_goal_id += tokenizer.encode('[user]' + utt['utterance'])[1:]
                        gs_utt_list += tokenizer.encode('[user]' + utt['utterance'])[1:]
                    continue
                if type(utt['goal']) is list: # 골이 여러개라면? 한문장으로 붙여넣기
                    utt['goal'] = '|'.join(utt['goal'])

                ### prepare response generation data
                target_id = tokenizer.encode(utt['utterance']) # System 발화
                know_len = int(args.max_seq_length/2)
                if args.data_name == 'tgredial': # 生成回复 == 응답생성 (GPT한테 응답생성하라고 키워드넣는느낌)
                    new_source_id = source_id + tokenizer.encode('[goal]' + utt['goal'])[1:] + tokenizer.encode('[knowledge]')[1:-1] + tokenizer.encode('|'.join(utt['knowledge']))[1:][-know_len:] + tokenizer.encode('[item]' + '|'.join(utt['item']))[1:] + tokenizer.encode('生成回复：')[1:]
                else: # DuRecDial
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
                # new_source_id = source_goal_id + tokenizer.encode('计划下一个目标：')[1:] # HJ Natural Language Prompt -- plan the next goal
                if args.goal_instruction:
                    if args.goal_prompt_idea1_order == 'ug':
                        if gs_goal_list: new_source_id = gs_utt_list + + gs_goal_list + tokenizer.encode('计划下一个目标：')[1:]  # HJ Utterance~~ + Goal~~ + plan the next goal
                        else: new_source_id = gs_utt_list + tokenizer.encode('计划下一个目标：')[1:]  # HJ Utterance~~ + Goal~~ + plan the next goal
                    elif args.goal_prompt_idea1_order == 'gu':
                        if gs_goal_list: new_source_id = gs_goal_list + gs_utt_list + tokenizer.encode('计划下一个目标：')[1:]  # HJ Utterance~~ + Goal~~ + plan the next goal
                        else: new_source_id = gs_utt_list + tokenizer.encode('计划下一个目标：')[1:]  # HJ Utterance~~ + Goal~~ + plan the next goal
                    else:
                        print("Check goal prompt idea1 order"); assert 0
                else:
                    if args.goal_prompt_idea1_order == 'ug':
                        if gs_goal_list: new_source_id = tokenizer.encode('对话如此时：')[1:] + gs_utt_list + tokenizer.encode('目标顺序如此时：')[1:] + gs_goal_list + tokenizer.encode('计划下一个目标：')[1:] #HJ Utterance~~ + Goal~~ + plan the next goal
                        else: new_source_id = tokenizer.encode('对话如此时：')[1:] + gs_utt_list +  tokenizer.encode('计划下一个目标：')[1:] #HJ Utterance~~ + Goal~~ + plan the next goal
                    elif args.goal_prompt_idea1_order =='gu':
                        if gs_goal_list: new_source_id = tokenizer.encode('目标顺序如此时：')[1:] + gs_goal_list + tokenizer.encode('对话如此时：')[1:] + gs_utt_list + tokenizer.encode('计划下一个目标：')[1:] #HJ Utterance~~ + Goal~~ + plan the next goal
                        else: new_source_id = tokenizer.encode('对话如此时：')[1:] + gs_utt_list + tokenizer.encode('计划下一个目标：')[1:] #HJ Utterance~~ + Goal~~ + plan the next goal
                    else: print("Check goal prompt idea1 order"); assert 0
                    pass
                # HJ 현재까지 진행된 utt, goal 처리해서 new_source_id 생성

                if 'dialog' in args.goal_input and 'goal' in args.goal_input: # HJ GoalSeq관련처리-1
                    gs_goal_list += tokenizer.encode('[goal]' + utt['goal'])[1:]
                    gs_utt_list += tokenizer.encode('[{}]'.format(utt['role']) + utt['utterance'])[1:]
                    # source_goal_id += (tokenizer.encode('[goal]' + utt['goal'])[1:] + tokenizer.encode('[{}]'.format(utt['role']) + utt['utterance'])[1:])  # HJ Goal Sequence 를 넣고, Next Goal 예측할때

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

