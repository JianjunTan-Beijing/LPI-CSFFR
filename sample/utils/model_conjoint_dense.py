from keras.layers import Dense, Concatenate, Conv1D,  BatchNormalization, MaxPooling1D
from keras.layers import Dropout, Flatten, Input,  Activation
from keras.models import Model
from keras.layers import concatenate

Blocks = 3
Block_size = 18
Filter_num = 24
Kernel_sizes = 6

def dense_block(x, blocks, size):
    if blocks == 0:
        x = x
    elif blocks > 0:
        for i in range(blocks):
            x = conv_block(x, size)
    return x

def conv_block(x, growth_rate):
    x1 = Conv1D(growth_rate, 3, padding='same')(x)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x = Concatenate(axis=-1)([x, x1])
    return x

def conjoint_cnn_denseblock(pro_seq_coding_length, rna_seq_coding_length,  pro_struct_coding_length, rna_struct_coding_length,pro_pse_coding_length, rna_pse_coding_length,  pro_seq_coding = True, pro_struct_coding = True, pro_pse_coding = True, rna_seq_coding = True, rna_struct_coding = True, rna_pse_coding = True):
    if pro_seq_coding == True and rna_seq_coding == True and pro_struct_coding == True and rna_struct_coding == True and pro_pse_coding == True and rna_pse_coding == True:
        # sequence拼接
        xp_seq_in_conjoint_cnn = Input(shape=(pro_seq_coding_length, 1))
        xr_seq_in_conjoint_cnn = Input(shape=(rna_seq_coding_length, 1))
        seq_in_conjoint = Concatenate(axis=1)(
            [xp_seq_in_conjoint_cnn, xr_seq_in_conjoint_cnn])
        seq_in_conjoint = Conv1D(filters=Filter_num + Blocks * Block_size, kernel_size=1)(seq_in_conjoint)
        seq_in_conjoint = BatchNormalization()(seq_in_conjoint)
        seq_in_conjoint = Activation('relu')(seq_in_conjoint)
        seq_cnn = Conv1D(filters=Filter_num, kernel_size=Kernel_sizes, padding='same')(
            seq_in_conjoint)
        seq_cnn = BatchNormalization()(seq_cnn)
        seq_cnn = Activation('relu')(seq_cnn)
        seq_cnn = MaxPooling1D(pool_size=Kernel_sizes)(seq_cnn)
        seq_cnn = dense_block(seq_cnn, Blocks, Block_size)
        seq_cnn = BatchNormalization()(seq_cnn)

        # structure拼接
        xp_struct_in_conjoint_cnn = Input(shape=(pro_struct_coding_length, 1))
        xr_struct_in_conjoint_cnn = Input(shape=(rna_struct_coding_length, 1))
        struct_in_conjoint = Concatenate(axis=1)(
            [xp_struct_in_conjoint_cnn, xr_struct_in_conjoint_cnn])
        struct_in_conjoint = Conv1D(filters=Filter_num + Blocks * Block_size, kernel_size=1)(struct_in_conjoint)
        struct_in_conjoint = BatchNormalization()(struct_in_conjoint)
        struct_in_conjoint = Activation('relu')(struct_in_conjoint)
        struct_cnn = Conv1D(filters=Filter_num, kernel_size=Kernel_sizes, padding='same')(
            struct_in_conjoint)
        struct_cnn = BatchNormalization()(struct_cnn)
        struct_cnn = Activation('relu')(struct_cnn)
        struct_cnn = MaxPooling1D(pool_size=Kernel_sizes)(struct_cnn)
        struct_cnn = dense_block(struct_cnn, Blocks, Block_size)
        struct_cnn = BatchNormalization()(struct_cnn)

        # pse拼接
        xp_pse_in_conjoint_cnn = Input(shape=(pro_pse_coding_length, 1))
        xr_pse_in_conjoint_cnn = Input(shape=(rna_pse_coding_length, 1))
        pse_in_conjoint = Concatenate(axis=1)(
            [xp_pse_in_conjoint_cnn, xr_pse_in_conjoint_cnn])
        pse_in_conjoint = Conv1D(filters=Filter_num + Blocks * Block_size, kernel_size=1)(pse_in_conjoint)
        pse_in_conjoint = BatchNormalization()(pse_in_conjoint)
        pse_in_conjoint = Activation('relu')(pse_in_conjoint)
        pse_cnn = Conv1D(filters=Filter_num, kernel_size=Kernel_sizes, padding='same')(
            pse_in_conjoint)
        pse_cnn = BatchNormalization()(pse_cnn)
        pse_cnn = Activation('relu')(pse_cnn)
        pse_cnn = MaxPooling1D(pool_size=Kernel_sizes)(pse_cnn)
        pse_cnn = dense_block(pse_cnn, Blocks, Block_size)
        pse_cnn = BatchNormalization()(pse_cnn)


        x_in_conjoint = Concatenate(axis=1)([seq_cnn, struct_cnn, pse_cnn])
        x_out_conjoint = Flatten()(x_in_conjoint)
        x_out_conjoint = Dense(192, activation='relu', use_bias=False)(
            x_out_conjoint)
        x_out_conjoint = Dropout(0.2)(x_out_conjoint)
        x_out_conjoint = Dense(64, activation='relu', use_bias=False)(
            x_out_conjoint)
        x_out_conjoint = Dropout(0.2)(x_out_conjoint)
        y_conjoint = Dense(2, activation='softmax')(x_out_conjoint)

        model_conjoint = Model(inputs=[xp_seq_in_conjoint_cnn, xr_seq_in_conjoint_cnn, xp_struct_in_conjoint_cnn,
                                       xr_struct_in_conjoint_cnn, xp_pse_in_conjoint_cnn, xr_pse_in_conjoint_cnn],
                               outputs=y_conjoint)
        return model_conjoint


def conjoint_cnn_sep_denseblock(pro_seq_coding_length, rna_seq_coding_length,  pro_struct_coding_length, rna_struct_coding_length,pro_pse_coding_length, rna_pse_coding_length,  pro_seq_coding = True, pro_struct_coding = True, pro_pse_coding = True, rna_seq_coding = True, rna_struct_coding = True, rna_pse_coding = True): #其实这个效果也还可以,结果表明这样的效果很好
    if pro_seq_coding == True and rna_seq_coding == True and pro_struct_coding == True and rna_struct_coding == True and pro_pse_coding == True and rna_pse_coding == True:

        xp_seq_in_conjoint_cnn = Input(shape=(pro_seq_coding_length, 1))
        xr_seq_in_conjoint_cnn = Input(shape=(rna_seq_coding_length, 1))
        seq_in_conjoint = Concatenate(axis=1)(
            [xp_seq_in_conjoint_cnn, xr_seq_in_conjoint_cnn])
        seq_in_conjoint = Conv1D(filters=Filter_num + Blocks * Block_size, kernel_size=1)(seq_in_conjoint)
        seq_in_conjoint = BatchNormalization()(seq_in_conjoint)
        seq_in_conjoint = Activation('relu')(seq_in_conjoint)
        seq_cnn = Conv1D(filters=Filter_num, kernel_size=Kernel_sizes, padding='same')(
            seq_in_conjoint)
        seq_cnn = BatchNormalization()(seq_cnn)
        seq_cnn = Activation('relu')(seq_cnn)
        seq_cnn = MaxPooling1D(pool_size=Kernel_sizes)(seq_cnn)
        seq_cnn = dense_block(seq_cnn, Blocks, Block_size)
        seq_cnn = BatchNormalization()(seq_cnn)


        xp_struct_in_conjoint_cnn = Input(shape=(pro_struct_coding_length, 1))
        xr_struct_in_conjoint_cnn = Input(shape=(rna_struct_coding_length, 1))
        struct_in_conjoint = Concatenate(axis=1)(
            [xp_struct_in_conjoint_cnn, xr_struct_in_conjoint_cnn])
        struct_in_conjoint = Conv1D(filters=Filter_num + Blocks * Block_size, kernel_size=1)(struct_in_conjoint)
        struct_in_conjoint = BatchNormalization()(struct_in_conjoint)
        struct_in_conjoint = Activation('relu')(struct_in_conjoint)

        seqstruct_in_conjoint = Concatenate(axis=1)(
            [seq_cnn, struct_in_conjoint])
        seqstruct_cnn = Conv1D(filters=Filter_num, kernel_size=Kernel_sizes, padding='same')(
            seqstruct_in_conjoint)
        seqstruct_cnn = BatchNormalization()(seqstruct_cnn)
        seqstruct_cnn = Activation('relu')(seqstruct_cnn)
        seqstruct_cnn = MaxPooling1D(pool_size=Kernel_sizes)(seqstruct_cnn)
        seqstruct_cnn = dense_block(seqstruct_cnn, Blocks, Block_size)
        seqstruct_cnn = BatchNormalization()(seqstruct_cnn)


        xp_pse_in_conjoint_cnn = Input(shape=(pro_pse_coding_length, 1))
        xr_pse_in_conjoint_cnn = Input(shape=(rna_pse_coding_length, 1))
        pse_in_conjoint = Concatenate(axis=1)(
            [xp_pse_in_conjoint_cnn, xr_pse_in_conjoint_cnn])
        pse_in_conjoint = Conv1D(filters=Filter_num + Blocks * Block_size, kernel_size=1)(pse_in_conjoint)
        pse_in_conjoint = BatchNormalization()(pse_in_conjoint)
        pse_in_conjoint = Activation('relu')(pse_in_conjoint)


        seqstructpse_in_conjoint = Concatenate(axis=1)(
            [seqstruct_cnn, pse_in_conjoint])
        seqstructpse_in_conjoint = Conv1D(filters=Filter_num, kernel_size=Kernel_sizes, padding='same')(
            seqstructpse_in_conjoint)
        seqstructpse_cnn = BatchNormalization()(seqstructpse_in_conjoint)
        seqstructpse_cnn = Activation('relu')(seqstructpse_cnn)
        seqstructpse_cnn = MaxPooling1D(pool_size=Kernel_sizes)(seqstructpse_cnn)
        seqstructpse_cnn = dense_block(seqstructpse_cnn, Blocks, Block_size)
        seqstructpse_cnn = BatchNormalization()(seqstructpse_cnn)

        x_out_conjoint = Flatten()(seqstructpse_cnn)
        x_out_conjoint = Dense(192, activation='relu', use_bias=False)(
            x_out_conjoint)
        x_out_conjoint = Dropout(0.2)(x_out_conjoint)
        x_out_conjoint = Dense(64, activation='relu', use_bias=False)(
            x_out_conjoint)
        x_out_conjoint = Dropout(0.2)(x_out_conjoint)
        y_conjoint = Dense(2, activation='softmax')(x_out_conjoint)
        model_conjoint = Model(inputs=[xp_seq_in_conjoint_cnn, xr_seq_in_conjoint_cnn, xp_struct_in_conjoint_cnn,
                                       xr_struct_in_conjoint_cnn, xp_pse_in_conjoint_cnn, xr_pse_in_conjoint_cnn],
                               outputs=y_conjoint)
        return model_conjoint
