import numpy as np
import os
import pandas as pd


def data_process(args):
    # process data
    train_data_directory = os.path.join(args.data_dir, args.dataset, args.dataset + '_train.csv')
    valid_data_directory = os.path.join(args.data_dir, args.dataset, args.dataset + '_test.csv')
    test_data_directory = os.path.join(args.data_dir, args.dataset, args.dataset + '_test.csv')
    Skill_Diff_directory = os.path.join(args.data_dir, args.dataset, args.dataset + '_Sdiff.csv')
    args.train_seqs, train_student_num, train_max_skill_id, train_max_question_id, feature_answer_id = load_data(
        train_data_directory, args.field_size, args.max_step)
    args.test_seqs, test_student_num, test_max_skill_id, test_max_question_id, _ = load_data(test_data_directory,
                                                                                             args.field_size,
                                                                                             args.max_step)
    args.valid_seqs, _, _, _, _ = load_data(valid_data_directory, args.field_size, args.max_step)
    S_Diff_dic = get_S_diff_dic(Skill_Diff_directory)

    print("original test seqs num:%d" % len(args.test_seqs))
    lens = []
    for i in range(len(args.test_seqs)):
        lens.append(len(args.test_seqs[i]))

    student_num = train_student_num + test_student_num
    args.skill_num = max(train_max_skill_id, test_max_skill_id) + 1
    args.qs_num = max(train_max_question_id, test_max_question_id) + 1
    args.question_num = args.qs_num - args.skill_num
    args.feature_answer_size = feature_answer_id + 1
    print("skill_num: ", args.skill_num)
    print('question_num: ', args.question_num)
    print('train_student_num: ', train_student_num)

    skill_matrix_directory = os.path.join(args.data_dir, args.dataset, args.dataset + '_skill_matrix.txt')
    args.skill_matrix = np.loadtxt(skill_matrix_directory)  # multi-skill

    Qmatrix_directory = os.path.join(args.data_dir, args.dataset, args.dataset + '_Qmatrix.txt')
    args.Qmatrix = np.loadtxt(Qmatrix_directory)  # multi-skill

    qs_adj_list, interactions = build_adj_list(args.train_seqs, args.test_seqs, args.Qmatrix,
                                               args.qs_num)  # [[neighbor skill/question] for all qs]]

    args.question_neighbors, args.skill_neighbors = extract_qs_relations(qs_adj_list, args.skill_num, args.qs_num,
                                                                         args.question_neighbor_num,
                                                                         args.skill_neighbor_num)

    args.Q_Modes = get_Modes(args.skill_matrix, args.question_neighbors, S_Diff_dic)

    return args


def get_S_diff_dic(srcFile):
    with open(srcFile) as f:
        data = pd.read_csv(f)
        return data.to_dict(orient="records")[0]


def get_Modes(skill_matrix, question_neighbors, S_diff_dic):
    Q_Modes = []

    row, col = np.diag_indices_from(skill_matrix)
    skill_matrix[row, col] = 1

    SSList = np.array(skill_matrix.tolist())

    x = SSList.flatten().tolist()
    x = list(filter(lambda number: number > 0, x))
    percentile = np.percentile(x, (0, 25, 50, 75, 100), interpolation='linear')
    Q1 = percentile[0]
    Q2 = percentile[1]
    Q3 = percentile[2]
    Q4 = percentile[3]
    Q5 = percentile[4]

    for i in range(SSList.shape[0]):
        for j in range(SSList.shape[1]):
            if i != j:
                if SSList[i][j] < Q2:
                    SSList[i][j] = 0
                elif SSList[i][j] >= Q2 and SSList[i][j] < Q3:
                    SSList[i][j] = 0.35
                elif SSList[i][j] >= Q3 and SSList[i][j] < Q4:
                    SSList[i][j] = 0.65
                else:
                    SSList[i][j] = 1

    for i in range(len(question_neighbors)):
        question_neighbors[i].sort()
        question_neighbors[i] = list(question_neighbors[i])
        question_neighbors_diff = [{"skill": j, "diff": S_diff_dic[str(j)]} for j in question_neighbors[i]]
        question_neighbors_diff.sort(key=lambda x: x["diff"])
        question_neighbors[i] = [j["skill"] for j in question_neighbors_diff]

    for i in range(len(question_neighbors)):
        SubGraph = SSList[question_neighbors[i], :][:, question_neighbors[i]]
        Q_Modes.append(SubGraph.flatten().tolist())

    return Q_Modes


def select_part_seqs(min_len, max_len, seqs):
    temp_seqs = []
    for seq in seqs:
        if len(seq) >= min_len and len(seq) <= max_len:
            temp_seqs.append(seq)

    print("seq num is: %d" % len(temp_seqs))
    return temp_seqs


def build_adj_list(train_seqs, test_seqs, Qmatrix, qs_num):
    interactions = 0
    single_skill = []
    qs_num = int(qs_num)
    adj_list = [[] for _ in range(qs_num)]
    num_skill = Qmatrix.shape[1]

    adj_num = [0 for _ in range(qs_num)]

    for seqs in [train_seqs, test_seqs]:
        for seq in seqs:
            interactions += len(seq)
            for step in seq:
                step[1] = int(step[1])
                adj_list[step[1]] = np.reshape(np.argwhere(Qmatrix[int(step[1]) - num_skill] == 1), [-1]).tolist()
                adj_num[step[1]] += 1
                for skill_index in np.reshape(np.argwhere(Qmatrix[int(step[1]) - num_skill] == 1), [-1]).tolist():
                    adj_num[skill_index] += 1
                    if skill_index not in single_skill:
                        single_skill.append(skill_index)
                    if step[1] not in adj_list[skill_index]:
                        adj_list[skill_index].append(step[1])

    return adj_list, interactions


def extract_qs_relations(qs_list, s_num, qs_num, q_neighbor_size, s_neighbor_size):
    question_neighbors = np.zeros([int(qs_num), int(q_neighbor_size)], dtype=np.int32)  # the first s_num rows are 0
    skill_neighbors = np.zeros([int(s_num), int(s_neighbor_size)], dtype=np.int32)
    s_num_dic = {}
    q_num_dic = {}
    for index, neighbors in enumerate(qs_list):
        if index < s_num:  # s  QSlist前面的是知识点
            if len(neighbors) not in q_num_dic:
                q_num_dic[len(neighbors)] = 1
            else:
                q_num_dic[len(neighbors)] += 1
            if len(neighbors) > 0:
                if len(neighbors) >= s_neighbor_size:
                    # skill_neighbors[index] = np.random.choice(neighbors, s_neighbor_size, replace=False)
                    skill_neighbors[index] = neighbors[:s_neighbor_size]
                else:
                    quotient = int(s_neighbor_size / len(neighbors))
                    remainder = s_neighbor_size % len(neighbors)
                    if remainder:
                        temp = neighbors * quotient
                        temp.extend(neighbors[:remainder])
                        skill_neighbors[index] = temp
                    else:
                        skill_neighbors[index] = neighbors * quotient

        else:
            if len(neighbors) not in s_num_dic:
                s_num_dic[len(neighbors)] = 1
            else:
                s_num_dic[len(neighbors)] += 1
            if len(neighbors) > 0:
                if len(neighbors) >= q_neighbor_size:
                    question_neighbors[index] = np.random.choice(neighbors, q_neighbor_size, replace=False)
                else:
                    question_neighbors[index] = np.random.choice(neighbors, q_neighbor_size, replace=True)

    return question_neighbors, skill_neighbors


def load_data(dataset_path, field_size, max_seq_len):
    seqs = []
    student_id = 0
    max_skill = -1
    max_question = -1
    feature_answer_size = -1
    with open(dataset_path, 'r') as f:
        feature_answer_list = []
        for line_id, line in enumerate(f):
            fields = line.strip().strip(',')
            i = line_id % (field_size + 1)
            if i != 0:
                feature_answer_list.append(list(map(float, fields.split(","))))
            if i == 1:
                if max(feature_answer_list[-1]) > max_skill:
                    max_skill = max(feature_answer_list[-1])
            elif i == 2:
                if max(feature_answer_list[-1]) > max_question:
                    max_question = max(feature_answer_list[-1])
            elif i == field_size:
                student_id += 1
                if max(feature_answer_list[-1]) > feature_answer_size:
                    feature_answer_size = max(feature_answer_list[-1])
                if len(feature_answer_list[0]) > max_seq_len:
                    n_split = len(feature_answer_list[0]) // max_seq_len
                    if len(feature_answer_list[0]) % max_seq_len:
                        n_split += 1
                else:
                    n_split = 1
                for k in range(n_split):

                    if k == n_split - 1:
                        end_index = len(feature_answer_list[0])
                    else:
                        end_index = (k + 1) * max_seq_len
                    split_list = []

                    for i in range(len(feature_answer_list)):
                        split_list.append(feature_answer_list[i][k * max_seq_len:end_index])

                    split_list = np.stack(split_list, 1).tolist()

                    seqs.append(split_list)
                feature_answer_list = []

    return seqs, student_id, max_skill, max_question, feature_answer_size


def pad_sequences(sequences, maxlen=None, dtype='int32', padding='pre', truncating='pre', value=0.):
    lengths = [len(s) for s in sequences]
    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    sample_shape = tuple()

    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]  # 3
            break

    x = (np.ones((nb_samples, maxlen) + sample_shape) * value).astype(dtype)

    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        if truncating == 'pre':  # maxlen!=none may need to truncating
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen + 1]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))
        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x


# select same skill index
def sample_hist_neighbors(seqs_size, max_step, hist_num, skill_index):
    hist_neighbors_index = []

    for i in range(seqs_size):
        seq_hist_index = []
        seq_skill_index = skill_index[i]
        # [max_step,M]
        for j in range(1, max_step):
            same_skill_index = [k for k in range(j) if seq_skill_index[k] == seq_skill_index[j]]

            if hist_num != 0:
                # [0,j] select M
                if len(same_skill_index) >= hist_num:
                    seq_hist_index.append(np.random.choice(same_skill_index, hist_num, replace=False))
                else:
                    if len(same_skill_index) != 0:
                        seq_hist_index.append(np.random.choice(same_skill_index, hist_num, replace=True))
                    else:
                        seq_hist_index.append(([max_step - 1 for _ in range(hist_num)]))
            else:
                seq_hist_index.append([])
        hist_neighbors_index.append(seq_hist_index)
    return hist_neighbors_index


def format_data(seqs, max_step, feature_size, hist_num):
    seqs = seqs
    seq_lens = np.array(list(map(lambda seq: len(seq), seqs)))

    # [batch_size,max_len,feature_size]
    features_answer_index = pad_sequences(seqs, maxlen=max_step, padding='post', value=0)
    target_answers = pad_sequences(np.array([[j[-1] - feature_size for j in i[1:]] for i in seqs]), maxlen=max_step - 1,
                                   # feature_size = 17904
                                   padding='post', value=0)
    skills_index = features_answer_index[:, :, 0]
    hist_neighbor_index = sample_hist_neighbors(len(seqs), max_step, hist_num, skills_index)  # [batch_size,max_step,M]
    # arg_parser.add_argument('--hist_neighbor_num', type=int, default=0)  # history neighbor num
    return features_answer_index, target_answers, seq_lens, hist_neighbor_index


class DataGenerator(object):

    def __init__(self, seqs, max_step, batch_size, feature_size, hist_num):  # feature_dkt
        np.random.seed(42)
        self.seqs = seqs
        self.max_step = max_step
        self.batch_size = batch_size
        self.batch_i = 0
        self.end = False
        self.feature_size = feature_size
        self.n_batch = int(np.ceil(len(seqs) / batch_size))
        self.hist_num = hist_num

    def next_batch(self):
        batch_seqs = self.seqs[self.batch_i * self.batch_size:(
                                                                          self.batch_i + 1) * self.batch_size]  # size为[batch_size,raw_time_steps,3]
        self.batch_i += 1

        if self.batch_i == self.n_batch:
            self.end = True

        format_data_list = format_data(batch_seqs, self.max_step, self.feature_size,
                                       self.hist_num)  # [feature_index,target_answers,sequences_lens,hist_neighbor_index]
        return format_data_list

    def shuffle(self):
        self.pos = 0
        self.end = False
        np.random.shuffle(self.seqs)

    def reset(self):
        self.pos = 0
        self.end = False
