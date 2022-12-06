# ====================================================
# CFG
# ====================================================
class CFG:
    use_cuda_if_available = True


    ##### MODEL

    embedding_dim=128

    ii_neighbor_num=10
    model_save_path="./weight/best_model.pt"
    max_epoch=2000
    # enable_tensorboard=yes
    initial_weight=1e-3


    ################
    ##### Training

    train_file_path="/opt/ml/level2_dkt_recsys-level2-recsys-11/data/"
    fe_num = '09'

    #need to specify the avaliable gpu index. If gpu is not avaliable, we will use cpu.
    gpu=1
    num_workers=1

    learning_rate=1e-3
    batch_size=1024
    early_stop_epoch=50

    #L = -(w1 + w2*\beta)) * log(sigmoid(e_u e_i)) - \sum_{N-} (w3 + w4*\beta) * log(sigmoid(e_u e_i'))
    w1=1e-7
    w2=1
    w3=1e-7
    w4=1

    negative_num=200
    negative_weight=200

    #weight of l2 normalization
    GAMMA=1e-4 
    #weight of L_I
    LAMBDA=1e-3

    #whether to sift the pos item when doing negative sampling
    sampling_sift_pos=False

    #################
    #### Testing

    #can be customized to your gpu size
    test_batch_size=2048
    topk=20
    test_file_path="/opt/ml/level2_dkt_recsys-level2-recsys-11/data/"