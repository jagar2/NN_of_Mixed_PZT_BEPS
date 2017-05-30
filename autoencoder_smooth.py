import os
import numpy as np
from sklearn import (decomposition, preprocessing, cluster,
                     preprocessing, cluster, metrics)
import scipy.io as io
import h5py
import tensorflow as tf
from keras.models import Sequential
from keras.layers import (Dense, Conv1D, GRU, LSTM, Recurrent, Bidirectional,
                          TimeDistributed, Dropout, Flatten, RepeatVector, Reshape)


def load_data(data_path, smooth = False, components = 15):
    f = io.matlab.loadmat(data_path)
    loop_data = f['Loopdata_mixed'][()]
    X = loop_data.reshape(loop_data.shape[0]*loop_data.shape[0],-1)
    X -= np.nanmean(X)
    X[np.where(np.isfinite(X)==0)] = 0
    X /= np.std(X)

    if smooth:
        # Sets the number of componets to save
        pca_model = decomposition.PCA(n_components=components)

        # Computes the PCA of the piezoelectric hysteresis loops
        PCA_data = pca_model.fit(X)

        # Does the inverse tranform - creates a low rank representation of the data
        X = pca_model.inverse_transform(pca_model.transform(X))

    X = np.atleast_3d(X)
    return X

def get_run_id(layer_type, size, num_layers, embedding, lr, drop_frac, batch_size, **kwargs):
    return (f"{layer_type}{size:03d}_x{num_layers}_emb{embedding:03d}_{lr:1.0e}"
            f"_drop{int(100 * drop_frac)}_batch{batch_size}").replace('e-', 'm')


def rnn_auto(layer, size, num_layers, embedding, n_step, drop_frac=0., bidirectional=True,
             **kwargs):
    if bidirectional:
        wrapper = Bidirectional
    else:
        wrapper = lambda x: x
    model = Sequential()
    model.add(wrapper(layer(size, return_sequences=(num_layers > 1)),
                        input_shape=(n_step, 1)))
    for i in range(1, num_layers):
        model.add(wrapper(layer(size, return_sequences=(i < num_layers - 1))))
        if drop_frac > 0.:
            model.add(Dropout(drop_frac))
    model.add(Dense(embedding, activation='linear', name='encoding'))
    model.add(RepeatVector(n_step))
    for i in range(num_layers):
        model.add(wrapper(layer(size, return_sequences=True)))
        if drop_frac > 0.:
            model.add(Dropout(drop_frac))
    model.add(TimeDistributed(Dense(1, activation='linear')))

    return model


def main(arg_dict=None):
    from keras.optimizers import Adam
    import shutil
    from keras.callbacks import TensorBoard, ModelCheckpoint
    from keras_tqdm import TQDMCallback

    from argparse import ArgumentParser, Namespace
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, default='data.mat')
    parser.add_argument("--size", type=int)
    parser.add_argument("--num_layers", type=int)
    parser.add_argument('--embedding', type=int)
    parser.add_argument("--drop_frac", type=float, default=0.)
    parser.add_argument("--batch_size", type=int, default=265)
    parser.add_argument("--n_cycles", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--layer_type", type=str)
    parser.add_argument("--N_train", type=int)
#    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument('--bidirectional', dest='bidirectional', action='store_true')
    parser.add_argument('--overwrite', dest='overwrite', action='store_true')
    parser.add_argument('--log_dir', type=str, default='log')
    parser.add_argument('--smooth', type=bool, default=False)
    parser.add_argument('--Components', type=int, default=15)
    parser.set_defaults(bidirectional=True, overwrite=False)
    args = parser.parse_args(None if arg_dict is None else [])  # don't read argv if arg_dict present
    if arg_dict:  # merge additional arguments w/ defaults
        args = Namespace(**{**args.__dict__, **arg_dict})

#    if args.layer_type == 'conv' and args.filter_length is None:
#        parser.error("--layer_type {} requires --filter_length".format(args.layer_type))

    X = load_data(args.data_path, smooth = args.smooth, components = args.Components)
    if args.N_train:
        train = np.arange(args.N_train)
    else:
        train = np.arange(len(X))

    run = get_run_id(**args.__dict__)
    log_dir = os.path.join(args.log_dir, run)
    print("Logging to {}".format(os.path.abspath(log_dir)))
    weights_path = os.path.join(log_dir, 'weights.h5')
    if os.path.exists(weights_path):
        if args.overwrite:
            print(f"Overwriting {log_dir}")
            shutil.rmtree(log_dir, ignore_errors=True)
        else:
            raise ValueError("Model file already exists")

    layer = {'lstm': LSTM, 'gru': GRU, 'conv': Conv1D}[args.layer_type]
    if issubclass(layer, Recurrent):
        model = rnn_auto(layer, args.size, args.num_layers, args.embedding, n_step=X.shape[1],
                         drop_frac=args.drop_frac)
    else:
        raise NotImplementedError("TODO convolutional")
    model.compile(Adam(args.lr), loss='mse')

    history = model.fit(X[train], X[train], epochs=args.epochs, batch_size=args.batch_size,
                        callbacks=[TQDMCallback(),
                                   TensorBoard(log_dir=log_dir, write_graph=False),
                                   ModelCheckpoint(weights_path)],
                        verbose=False)

    return X, model, history


if __name__ == '__main__':
    X, model, history = main()
