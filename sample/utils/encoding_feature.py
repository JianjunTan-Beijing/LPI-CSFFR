import math
import string
from functools import reduce
import numpy as np
import os
import sys


WINDOW_P_UPLIMIT = 3
WINDOW_P_STRUCT_UPLIMIT = 3
WINDOW_R_UPLIMIT = 4
WINDOW_R_STRUCT_UPLIMIT = 4
CODING_FREQUENCY = True
VECTOR_REPETITION_CNN = 1

script_dir, script_name = os.path.split(os.path.abspath(sys.argv[0]))
parent_dir = os.path.dirname(script_dir)
DATA_BASE_PATH = parent_dir + '/data/'
SEQ_BASE_PATH = DATA_BASE_PATH + '/sequence/'
STR_BASE_PATH = DATA_BASE_PATH + '/structure/'
PSE_BASE_PATH = DATA_BASE_PATH + '/pse/'

def sum_power(num, bottom, top):
    return reduce(lambda x, y: x + y, map(lambda x: num ** x, range(bottom, top + 1)))

PRO_CODING_LENGTH = sum_power(20, 1, WINDOW_P_UPLIMIT)
RNA_CODING_LENGTH = sum_power(4, 1, WINDOW_R_UPLIMIT)

# 蛋白质编码
class ProEncoder:
    # elements = 'AIYHRDC'
    elements = 'AGVILFPYMTSHNQWRKDEC'
    structs = 'HEC'


    element_number = 20
    struct_kind = 3



    def __init__(self, WINDOW_P_UPLIMIT, WINDOW_P_STRUCT_UPLIMIT, CODING_FREQUENCY, VECTOR_REPETITION_CNN,
                 TRUNCATION_LEN=None, PERIOD_EXTENDED=None):

        self.WINDOW_P_UPLIMIT = WINDOW_P_UPLIMIT
        self.WINDOW_P_STRUCT_UPLIMIT = WINDOW_P_STRUCT_UPLIMIT
        self.CODING_FREQUENCY = CODING_FREQUENCY
        self.VECTOR_REPETITION_CNN = VECTOR_REPETITION_CNN

        self.TRUNCATION_LEN = TRUNCATION_LEN
        self.PERIOD_EXTENDED = PERIOD_EXTENDED
# 序列

        k_mers = ['']
        self.k_mer_list = []
        self.k_mer_map = {}
        for T in range(self.WINDOW_P_UPLIMIT):
            temp_list = []
            for k_mer in k_mers:
                for x in self.elements:
                    temp_list.append(k_mer + x)
            k_mers = temp_list
            self.k_mer_list += temp_list
        for i in range(len(self.k_mer_list)):
            self.k_mer_map[self.k_mer_list[i]] = i
# 结构

        k_mers = ['']
        self.k_mer_struct_list = []
        self.k_mer_struct_map = {}
        for T in range(self.WINDOW_P_STRUCT_UPLIMIT):
            temp_list = []
            for k_mer in k_mers:
                for s in self.structs:
                    temp_list.append(k_mer + s)
            k_mers = temp_list
            self.k_mer_struct_list += temp_list
        for i in range(len(self.k_mer_struct_list)):
            self.k_mer_struct_map[self.k_mer_struct_list[i]] = i

        print('1')


    def encode_conjoint(self, seq):
        seq = ''.join([x for x in seq if x in self.elements])
        seq_len = len(seq)
        if seq_len == 0:
            return 'Error'
        result = []
        offset = 0
        for K in range(1, self.WINDOW_P_UPLIMIT + 1):
            vec = [0.0] * (self.element_number ** K)
            counter = seq_len - K + 1
            for i in range(seq_len - K + 1):
                k_mer = seq[i:i + K]
                vec[self.k_mer_map[k_mer] - offset] += 1
            vec = np.array(vec)
            offset += vec.size
            if self.CODING_FREQUENCY:
                vec = vec / counter
            result += list(vec)
        return result

    def encode_conjoint_struct(self, struct):
        struct = ''.join([x for x in struct if x in self.structs])
        seq_len = len(struct)

        result_struct = []
        offset_struct = 0
        for K in range(1, self.WINDOW_P_STRUCT_UPLIMIT + 1):
            vec_struct = [0.0] * (self.struct_kind ** K)
            counter = seq_len - K + 1
            if counter <= 0:
                print('error')
            for i in range(seq_len - K + 1):
                k_mer_struct = struct[i:i + K]
                vec_struct[self.k_mer_struct_map[k_mer_struct] - offset_struct] += 1
            vec_struct = np.array(vec_struct)
            offset_struct += vec_struct.size
            if self.CODING_FREQUENCY:
                vec_struct = vec_struct / counter
            result_struct += list(vec_struct)
        return result_struct



# RNA编码
class RNAEncoder:
    elements = 'AUCG'
    structs = '.('

    element_number = 4
    struct_kind = 2

    def __init__(self, WINDOW_R_UPLIMIT, WINDOW_R_STRUCT_UPLIMIT, CODING_FREQUENCY, VECTOR_REPETITION_CNN,
                 TRUNCATION_LEN=None, PERIOD_EXTENDED=None):

        self.WINDOW_R_UPLIMIT = WINDOW_R_UPLIMIT
        self.WINDOW_R_STRUCT_UPLIMIT = WINDOW_R_STRUCT_UPLIMIT
        self.CODING_FREQUENCY = CODING_FREQUENCY
        self.VECTOR_REPETITION_CNN = VECTOR_REPETITION_CNN

        self.TRUNCATION_LEN = TRUNCATION_LEN
        self.PERIOD_EXTENDED = PERIOD_EXTENDED


        k_mers = ['']
        self.k_mer_list = []
        self.k_mer_map = {}
        for T in range(self.WINDOW_R_UPLIMIT):
            temp_list = []
            for k_mer in k_mers:
                for x in self.elements:
                    temp_list.append(k_mer + x)
            k_mers = temp_list
            self.k_mer_list += temp_list
        for i in range(len(self.k_mer_list)):
            self.k_mer_map[self.k_mer_list[i]] = i


        k_mers = ['']
        self.k_mer_struct_list = []
        self.k_mer_struct_map = {}
        for T in range(self.WINDOW_R_STRUCT_UPLIMIT):
            temp_list = []
            for k_mer in k_mers:
                for s in self.structs:
                    temp_list.append(k_mer + s)
            k_mers = temp_list
            self.k_mer_struct_list += temp_list
        for i in range(len(self.k_mer_struct_list)):
            self.k_mer_struct_map[self.k_mer_struct_list[i]] = i



    def encode_conjoint(self, seq):
        seq = seq.replace('T', 'U')
        seq = ''.join([x for x in seq if x in self.elements])
        seq_len = len(seq)
        if seq_len == 0:
            return 'Error'
        result = []
        offset = 0
        for K in range(1, self.WINDOW_R_UPLIMIT + 1):
            vec = [0.0] * (self.element_number ** K)
            counter = seq_len - K + 1
            for i in range(seq_len - K + 1):
                k_mer = seq[i:i + K]
                vec[self.k_mer_map[k_mer] - offset] += 1
            vec = np.array(vec)
            offset += vec.size
            if self.CODING_FREQUENCY:
                vec = vec / counter
            result += list(vec)
        return result

    def encode_conjoint_struct(self, struct):
        struct = struct.replace(')', '(')
        struct = ''.join([x for x in struct if x in self.structs])
        seq_len = len(struct)


        result_struct = []
        offset_struct = 0
        for K in range(1, self.WINDOW_R_STRUCT_UPLIMIT + 1):
            vec_struct = [0.0] * (self.struct_kind ** K)
            counter = seq_len - K + 1
            for i in range(seq_len - K + 1):
                k_mer_struct = struct[i:i + K]
                vec_struct[self.k_mer_struct_map[k_mer_struct] - offset_struct] += 1
            vec_struct = np.array(vec_struct)
            offset_struct += vec_struct.size
            if self.CODING_FREQUENCY:
                vec_struct = vec_struct / counter
                if counter == 0:
                    print('!')
            result_struct += list(vec_struct)
        return result_struct

#
def read_data_seq(path):
    seq_dict = {}
    with open(path, 'r') as f:
        name = ''
        for line in f:
            line = line.strip()
            if line[0] == '>':
                name = line[1:]
                seq_dict[name] = ''
            else:
                if line.startswith('XXX'):
                    seq_dict.pop(name)
                else:
                    seq_dict[name] = line
    return seq_dict

def read_data_pair(path):
    pos_pairs = []
    neg_pairs = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            p, r, label = line.split('\t')
            if label == '1':
                pos_pairs.append((p, r))
            elif label == '0':
                neg_pairs.append((p, r))
    return pos_pairs, neg_pairs

def encode_data(data_set):
    pro_seqs = read_data_seq(SEQ_BASE_PATH +   data_set +'_rdprotein_seq.txt')
    rna_seqs = read_data_seq(SEQ_BASE_PATH +   data_set +'_rdrna_seq.txt')
    pro_structs = read_data_seq(STR_BASE_PATH +  data_set + '_rdprotein_struct0.txt')
    rna_structs = read_data_seq(STR_BASE_PATH  +  data_set +'_rdrna_struct0.txt')
    return pro_seqs, rna_seqs, pro_structs, rna_structs


def write_file(data_set):
    pro_seqs, rna_seqs, pro_structs, rna_structs = encode_data(data_set)
    PE = ProEncoder(WINDOW_P_UPLIMIT, WINDOW_P_STRUCT_UPLIMIT, CODING_FREQUENCY, VECTOR_REPETITION_CNN)
    RE = RNAEncoder(WINDOW_R_UPLIMIT, WINDOW_R_STRUCT_UPLIMIT, CODING_FREQUENCY, VECTOR_REPETITION_CNN)
    f1 = open(SEQ_BASE_PATH + data_set + '_rdprotein_3upmer.txt', 'w', encoding='utf-8')
    f2 = open(SEQ_BASE_PATH + data_set + '_rdrna_4upmer.txt', 'w', encoding='utf-8')
    f3 = open(STR_BASE_PATH + data_set + '_rdprotein_struct.txt', 'w', encoding='utf-8')
    f4 = open(STR_BASE_PATH + data_set + '_rdrna_struct.txt', 'w', encoding='utf-8')

    for p_id in pro_seqs:
        p_conjoint = PE.encode_conjoint(pro_seqs[p_id])
        for i in p_conjoint:
            f1.write(str(i) + '\t')
        f1.write('\n')

    for r_id in rna_seqs:
        r_conjoint = RE.encode_conjoint(rna_seqs[r_id])

        for i in r_conjoint:
            f2.write(str(i) + '\t')
        f2.write('\n')

    for p_id in pro_structs:
        p_conjoint_struct = PE.encode_conjoint_struct(pro_structs[p_id])

        for i in p_conjoint_struct:
            f3.write(str(i) + '\t')
        f3.write('\n')

    for r_id in rna_structs:
        r_conjoint_struct = RE.encode_conjoint_struct(rna_structs[r_id])

        for i in r_conjoint_struct:
            f4.write(str(i) + '\t')
        f4.write('\n')

    f1.close()
    f2.close()
    f3.close()
    f4.close()



