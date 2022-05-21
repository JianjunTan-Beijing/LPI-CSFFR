import math
import os
import sys
import time
import numpy as np
import argparse
import keras
import tensorflow as tf
from functools import reduce
from keras import optimizers
from keras.utils import np_utils
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from utils.callbacks import AccHistoryPlot, EarlyStopping
from utils.model_conjoint_dense import conjoint_cnn_denseblock, conjoint_cnn_sep_denseblock
from numpy import random
import datetime
from utils.encoding_feature import write_file

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def main(args):
    random.seed(1)
    DATA_SET = args.d
    TIME_FORMAT = "-%y-%m-%d-%H-%M-%S"
    K_FOLD = 5
    BATCH_SIZE = 50
    PATIENCES = 10
    OPTIMIZER = 'adam'
    WINDOW_P_UPLIMIT = 3
    WINDOW_R_UPLIMIT = 4
    WINDOW_P_STRUCT_UPLIMIT = 3
    WINDOW_R_STRUCT_UPLIMIT = 4
    ADAM_LEARNING_RATE = 0.001
    MONITOR = 'acc'
    MIN_DELTA = 0.0
    SHUFFLE = True
    VERBOSE = 2
    TRAIN_EPOCHS = 30
    # 设置路径
    script_dir, script_name = os.path.split(os.path.abspath(sys.argv[0]))
    parent_dir = os.path.dirname(script_dir)
    # 结果存放
    DATA_BASE_PATH = parent_dir + '/data/'
    RESULT_BASE_PATH = parent_dir + '/result/'


    # 放评价指标的
    method = 'seq&struct&pse'
    model_metrics = {method: np.zeros(7)}
    metrics_whole = {method: np.zeros(7)}

    result_save_path = RESULT_BASE_PATH + DATA_SET + "/" + DATA_SET + args.sample + args.mode  +  time.strftime(TIME_FORMAT, time.localtime()) +  "/"
    if not os.path.exists(result_save_path):
        os.mkdir(result_save_path)
    out = open(result_save_path + 'result.txt', 'w', encoding='utf-8')


    # 返回正样本和负样本对
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


    def read_data_index(path):
        seq_dict = {}
        ids = 0
        with open(path, 'r') as f:
            name = ''
            for line in f:
                line = line.strip()
                if line[0] == '>':
                    name = line[1:]
                    seq_dict[name] = ids
                    ids += 1

        return seq_dict

    # 评价指标
    def calc_metrics(y_label, y_proba):
        con_matrix = confusion_matrix(y_label, [1 if x >= 0.5 else 0 for x in y_proba])
        TN = float(con_matrix[0][0])
        FP = float(con_matrix[0][1])
        FN = float(con_matrix[1][0])
        TP = float(con_matrix[1][1])
        P = TP + FN
        N = TN + FP
        Sn = TP / P if P > 0 else 0
        Sp = TN / N if N > 0 else 0
        Acc = (TP + TN) / (P + N) if (P + N) > 0 else 0
        Pre = (TP) / (TP + FP) if (TP + FP) > 0 else 0
        Rec = (TP) / (TP + FN) if (TP + FN) > 0 else 0
        F1 = (2 * Pre * Rec) / (Pre + Rec) if (Pre + Rec) > 0 else 0
        MCC = 0
        tmp = math.sqrt((TP + FP) * (TP + FN)) * math.sqrt((TN + FP) * (TN + FN))
        if tmp != 0:
            MCC = (TP * TN - FP * FN) / tmp
        fpr, tpr, thresholds = roc_curve(y_label, y_proba)
        AUC = auc(fpr, tpr)
        return Acc, Sn, Sp, Pre, MCC, F1, AUC

    def read_feature(path):
        features = []
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                line = line.split('\t')
                features.append(line)
        return features

    # 加载数据
    write_file(DATA_SET)

    def load_data(data_set):
        positive_pairs = {}
        negtive_pairs = {}
        pro_seqs_pse_id = read_data_index(DATA_BASE_PATH + "sequence/" + data_set + '_rdprotein_seq.txt')
        rna_seqs_pse_id = read_data_index(DATA_BASE_PATH + "sequence/" + data_set + '_rdrna_seq.txt')
        pro_structs_id = read_data_index(DATA_BASE_PATH + "structure/" + data_set + '_rdprotein_struct0.txt')
        rna_structs_id = read_data_index(DATA_BASE_PATH + "structure/" + data_set + '_rdrna_struct0.txt')
        pro_seq_features = read_feature(DATA_BASE_PATH + "sequence/" + data_set + '_rdprotein_3upmer.txt')
        rna_seq_features = read_feature(DATA_BASE_PATH + "sequence/" + data_set + '_rdrna_4upmer.txt')
        pro_struct_features = read_feature(DATA_BASE_PATH + "structure/" + data_set + '_rdprotein_struct.txt')
        rna_struct_features = read_feature(DATA_BASE_PATH + "structure/" + data_set + '_rdrna_struct.txt')
        pro_pse_features = read_feature(DATA_BASE_PATH + "pse/" + data_set + '_rdprotein_pse.txt')
        rna_pse_features = read_feature(DATA_BASE_PATH + "pse/" + data_set + '_rdrna_pse.txt')
        print(args.sample)
        if args.sample == 'random':
            pos_pairs, neg_pairs = read_data_pair(DATA_BASE_PATH + data_set + '_rdpairs.txt')
        elif args.sample == 'swrandom':
            pos_pairs, neg_pairs = read_data_pair(DATA_BASE_PATH + data_set + '_swrandompairs.txt')
        elif args.sample == 'swsort':
            pos_pairs, neg_pairs = read_data_pair(DATA_BASE_PATH + data_set + '_swsortpairs.txt')
        else:
            print('error sample')
            sys.exit(0)
        for i in pos_pairs:
            positive_pairs[i] = 1
        for j in neg_pairs:
            negtive_pairs[j] = 0
        return positive_pairs, negtive_pairs, pro_seqs_pse_id, rna_seqs_pse_id, pro_structs_id, rna_structs_id, pro_seq_features, rna_seq_features, pro_struct_features, rna_struct_features, pro_pse_features, rna_pse_features

    pos_pairs, neg_pairs, pro_seqs_pse_id, rna_seqs_pse_id, pro_structs_id, rna_structs_id, pro_seq_features, rna_seq_features, pro_struct_features, rna_struct_features, pro_pse_features, rna_pse_features = load_data(
        DATA_SET)


    def random_dic(dicts):
        dict_key_ls = list(dicts.keys())
        random.shuffle(dict_key_ls)
        new_dic = {}
        for key in dict_key_ls:
            new_dic[key] = dicts.get(key)
        return new_dic


    def coding_pairs(pairs, pro_id, rna_id, pro_feature, rna_feature):
        # pairs = random_dic(pairs)
        p_fea = []
        r_fea = []
        for pr in pairs:
            if pr[0] in pro_id and pr[1] in rna_id:
                p_fea_tmp = np.array(pro_feature[pro_id[pr[0]]])
                r_fea_tmp = np.array(rna_feature[rna_id[pr[1]]])
                p_fea.append(p_fea_tmp)
                r_fea.append(r_fea_tmp)
            else:
                print('Skip pair {} according to sequence dictionary.'.format(pr))
        return p_fea, r_fea

    def sum_power(num, bottom, top):
        return reduce(lambda x, y: x + y, map(lambda x: num ** x, range(bottom, top + 1)))

    def merge_method(method, kind):
        samples = []
        print('choose feature type ' + method)
        if kind == 1:
            p_seqfea, r_seqfea = coding_pairs(pos_pairs, pro_seqs_pse_id, rna_seqs_pse_id, pro_seq_features,
                                              rna_seq_features)
            p_structfea, r_structfea = coding_pairs(pos_pairs, pro_structs_id, rna_structs_id, pro_struct_features,
                                                    rna_struct_features)
            p_psefea, r_psefea = coding_pairs(pos_pairs, pro_seqs_pse_id, rna_seqs_pse_id, pro_pse_features,
                                              rna_pse_features)
            for i in range(len(p_seqfea)):
                p_seqfea0 = p_seqfea[i]
                r_seqfea0 = r_seqfea[i]
                p_structfea0 = p_structfea[i]
                r_structfea0 = r_structfea[i]
                p_psefea0 = p_psefea[i]
                r_psefea0 = r_psefea[i]
                samples.append([[p_seqfea0, r_seqfea0, p_structfea0, r_structfea0, p_psefea0, r_psefea0], kind])
        elif kind == 0:
            p_seqfea, r_seqfea = coding_pairs(neg_pairs, pro_seqs_pse_id, rna_seqs_pse_id, pro_seq_features,
                                              rna_seq_features)
            p_structfea, r_structfea = coding_pairs(neg_pairs, pro_structs_id, rna_structs_id, pro_struct_features,
                                                    rna_struct_features)
            p_psefea, r_psefea = coding_pairs(neg_pairs, pro_seqs_pse_id, rna_seqs_pse_id, pro_pse_features,
                                              rna_pse_features)
            for i in range(len(p_seqfea)):
                p_seqfea0 = p_seqfea[i]
                r_seqfea0 = r_seqfea[i]
                p_structfea0 = p_structfea[i]
                r_structfea0 = r_structfea[i]
                p_psefea0 = p_psefea[i]
                r_psefea0 = r_psefea[i]
                samples.append([[p_seqfea0, r_seqfea0, p_structfea0, r_structfea0, p_psefea0, r_psefea0], kind])

        p_pse_coding_length = 22
        p_seq_coding_length = sum_power(20, 1, WINDOW_P_UPLIMIT)
        p_struct_coding_length = sum_power(3, 1, WINDOW_P_STRUCT_UPLIMIT)
        r_pse_coding_length = 18
        r_seq_coding_length = sum_power(4, 1, WINDOW_R_UPLIMIT)
        r_struct_coding_length = sum_power(2, 1, WINDOW_R_STRUCT_UPLIMIT)

        return samples, p_pse_coding_length, p_seq_coding_length, p_struct_coding_length, r_pse_coding_length, r_seq_coding_length, r_struct_coding_length

    def standardization(X):
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)
        return X, scaler

    # write program parameter settings to result file
    settings = (
        """# Analyze data set {}\n
    Program parameters:
    K_FOLD = {},
    BATCH_SIZE = {},
    TRAIN_EPOCHS = {},
    PATIENCES = {},
    OPTIMIZER = {},
    ADAM_initialLEARNING_RATE = {},
    MONITOR = {},
    MIN_DELTA = {},
        """.format(DATA_SET, K_FOLD, BATCH_SIZE, TRAIN_EPOCHS, PATIENCES, OPTIMIZER, ADAM_LEARNING_RATE,
                   MONITOR, MIN_DELTA)
    )
    out.write(settings)


    # 标准化
    def pre_process_data(samples):
        p_seq_feature = np.array([x[0][0] for x in samples])
        r_seq_feature = np.array([x[0][1] for x in samples])
        p_struct_feature = np.array([x[0][2] for x in samples])
        r_struct_feature = np.array([x[0][3] for x in samples])
        p_pse_feature = np.array([x[0][4] for x in samples])
        r_pse_feature = np.array([x[0][5] for x in samples])
        p_seq_conjoint, scaler_pseq = standardization(p_seq_feature)
        p_struct_conjoint, scaler_pstruct = standardization(p_struct_feature)
        p_pse_conjoint, scaler_ppse = standardization(p_pse_feature)
        r_seq_conjoint, scaler_rseq = standardization(r_seq_feature)
        r_struct_conjoint, scaler_rstruct = standardization(r_struct_feature)
        r_pse_conjoint, scaler_rpse = standardization(r_pse_feature)
        p_seq_conjoint_cnn = np.array([list(map(lambda e: [e], x)) for x in p_seq_conjoint])
        p_struct_conjoint_cnn = np.array([list(map(lambda e: [e], x)) for x in p_struct_conjoint])
        p_pse_conjoint_cnn = np.array([list(map(lambda e: [e], x)) for x in p_pse_conjoint])
        r_seq_conjoint_cnn = np.array([list(map(lambda e: [e], x)) for x in r_seq_conjoint])
        r_struct_conjoint_cnn = np.array([list(map(lambda e: [e], x)) for x in r_struct_conjoint])
        r_pse_conjoint_cnn = np.array([list(map(lambda e: [e], x)) for x in r_pse_conjoint])

        y_samples = np.array([x[1] for x in samples])

        X_samples = [[p_seq_conjoint_cnn, r_seq_conjoint_cnn, p_struct_conjoint_cnn, r_struct_conjoint_cnn, p_pse_conjoint_cnn, r_pse_conjoint_cnn]]

        return X_samples, y_samples

    print("Coding positive protein-rna pairs.\n")
    samples1, p_pse_coding_length, p_seq_coding_length, p_struct_coding_length, r_pse_coding_length, r_seq_coding_length, r_struct_coding_length = merge_method(method, kind=1)
    positive_sample_number = len(samples1)
    print("Coding negative protein-rna pairs.\n")
    samples0, _, _, _, _, _, _ = merge_method(method, kind=0)
    samples = samples1 + samples0
    negtive_sample_number = len(samples0)
    sample_train_num = len(samples)

    print('\nPos samples: {}, Neg samples: {}.\n'.format(positive_sample_number, negtive_sample_number))
    out.write('\nPos samples: {}, Neg samples: {}.\n'.format(positive_sample_number, negtive_sample_number))

    X_train_al_sep, y_train_al = pre_process_data(samples=samples)


    def get_callback_list(patience, result_path, stage, fold, X_test, y_test):
        earlystopping = EarlyStopping(monitor=MONITOR, min_delta=MIN_DELTA, patience=patience, verbose=1,
                                      mode='auto', restore_best_weights=True)
        acchistory = AccHistoryPlot([stage, fold], [X_test, y_test], data_name=DATA_SET,
                                    result_save_path=result_path, validate=0, plot_epoch_gap=10)
        return [acchistory, earlystopping]

    def get_optimizer(opt_name):
        if opt_name == 'adam':
            return optimizers.adam(lr=ADAM_LEARNING_RATE)
        else:
            return opt_name

    print('\n\nK-fold cross validation processes:\n')
    out.write('\n\nK-fold cross validation processes:\n')
    for fold in range(K_FOLD):

        train_id = [i for i in range(sample_train_num) if i % K_FOLD != fold]
        test_id = [i for i in range(sample_train_num) if i % K_FOLD == fold]

        X_train_conjoint = [np.array(X_train_al_sep[0][0][train_id]), np.array(X_train_al_sep[0][1][train_id]), np.array(X_train_al_sep[0][4][train_id]), np.array(X_train_al_sep[0][5][train_id]), np.array(X_train_al_sep[0][2][train_id]), np.array(X_train_al_sep[0][3][train_id])]
        X_test_conjoint = [np.array(X_train_al_sep[0][0][test_id]), np.array(X_train_al_sep[0][1][test_id]), np.array(X_train_al_sep[0][4][test_id]), np.array(X_train_al_sep[0][5][test_id]), np.array(X_train_al_sep[0][2][test_id]), np.array(X_train_al_sep[0][3][test_id])]

        y_train_mono = y_train_al[train_id]
        y_train = np_utils.to_categorical(y_train_mono, 2)
        y_test_mono = y_train_al[test_id]
        y_test = np_utils.to_categorical(y_test_mono, 2)

        print(str('\n# ' + '=' * 10 + " Fold {} " + "=" * 10 + '\n').format(fold))
        out.write(str('\n# ' + '=' * 10 + " Fold {} " + "=" * 10 + '\n').format(fold))

        # =================================================================

        stage = method
        print("\n# Module LPI-CSFFR part #\n")

        print('start train ' + args.mode)
        if args.mode == 'cnn_denseblock':
            model_conjoint_cnn_sep = conjoint_cnn_denseblock(p_seq_coding_length,r_seq_coding_length,p_pse_coding_length,r_pse_coding_length, p_struct_coding_length, r_struct_coding_length, 
                                                             pro_seq_coding = True, pro_struct_coding = True, pro_pse_coding = True, rna_seq_coding = True, rna_struct_coding = True, rna_pse_coding = True)

        elif args.mode == 'cnnsep_denseblock':
            model_conjoint_cnn_sep = conjoint_cnn_sep_denseblock( p_seq_coding_length, r_seq_coding_length,p_pse_coding_length, r_pse_coding_length,p_struct_coding_length, r_struct_coding_length,
                                                                 pro_seq_coding=True, pro_struct_coding=True,
                                                                 pro_pse_coding=True, rna_seq_coding=True,
                                                                 rna_struct_coding=True, rna_pse_coding=True)
        else:
            print('error mode!')
            sys.exit(0)
        callbacks = get_callback_list(PATIENCES, result_save_path, stage, fold, X_test_conjoint,
                                      y_test)

        model_conjoint_cnn_sep.compile(loss='categorical_crossentropy', optimizer=get_optimizer(OPTIMIZER),
                                       metrics=['accuracy'])

        model_conjoint_cnn_sep.fit(x=X_train_conjoint,
                                   y=y_train,
                                   epochs=TRAIN_EPOCHS,
                                   batch_size=BATCH_SIZE,
                                   verbose=VERBOSE,
                                   shuffle=SHUFFLE,
                                   callbacks=callbacks, validation_data=(X_test_conjoint, y_test))
        # model_path = result_save_path + f'model-{fold:}.h5'
        # modelsave = model_conjoint_cnn_sep.save(model_path)
        # test
        y_test_predict = model_conjoint_cnn_sep.predict(X_test_conjoint)
        model_metrics[method] = np.array(calc_metrics(y_test[:, 1], y_test_predict[:, 1]))
        print('Best performance for module LPI-CSFFR:\n {}\n'.format(model_metrics[method].tolist()))

        # =================================================================

        for key in model_metrics:
            print(key + " : " + str(model_metrics[key].tolist()) + "\n")
            out.write(key + " : " + str(model_metrics[key].tolist()) + "\n")

        for key in model_metrics:
            metrics_whole[key] += model_metrics[key]

    print('\nMean metrics in {} fold:\n'.format(K_FOLD))
    out.write('\nMean metrics in {} fold:\n'.format(K_FOLD))
    for key in metrics_whole.keys():
        if key == method:
            metrics_whole[key] /= K_FOLD
            print(key + " : " + str(metrics_whole[key].tolist()) + "\n")
            out.write(key + " : " + str(metrics_whole[key].tolist()) + "\n")

    out.flush()
    out.close()


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Master script.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument("-d", nargs="?", type=str, choices=['RPI1460','NPInter', 'RPI1807'], default='RPI1460', help='dataset of lncRNA-Protein interaction.\n')
    argparser.add_argument("-sample", nargs='?', type=str,choices=['random', 'swrandom'],default='random', help='different negative samples.\n')
    argparser.add_argument("-mode", nargs='?', type=str, choices=['cnn_denseblock', 'cnnsep_denseblock'], default='cnnsep_denseblock', help='training mode for lncRNA-Protein interaction.\n')
    args = argparser.parse_args()
    start_time = datetime.datetime.now()
    main(args)
    print("Done.")
    end_time = datetime.datetime.now()
    meal_time = (end_time - start_time)
    print("\n一共耗时： %s" % (str(meal_time)))

