import tensorflow as tf


cfg = tf.contrib.training.HParams(
    #cfg
            out_dir='./out/dbnCaps',
            input_size=166,
out_size=2,
batch_size=148,

    # caps net
primaryCaps_out_num = 10,
            primaryCaps_vec_num = 8,
            outCaps_vec_num = 16,
            mask_with_y = True,
            # caps train param
            caps_startLr=0.04,
            caps_decay_steps=1500,
            caps_decay_rate=0.96,
            caps_epochs=50000,
    #caps train show
            print_frq=1000,
            val_frq=100,
# decoder net
decoder_sizes=[128],
decoder_size= 1,
    # loss param
m_plus=0.9,
m_minus=0.1,
lambda_val=0.5,
regularization_scale=0.392,
stddev=0.01,
iter_routing=3
)


