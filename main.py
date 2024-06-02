import argparse
import numpy as np
import torch
from torchinfo import summary
from src.data import load_data
from src.methods.pca import PCA
from src.methods.deep_network import MLP, CNN, Trainer, MyViT
from src.utils import normalize_fn, accuracy_fn, macrof1_fn, get_n_classes

def main(args):
    """
    The main function of the script. Do not hesitate to play with it
    and add your own code, visualization, prints, etc!
    Arguments:
        args (Namespace): arguments that were parsed from the command line (see at the end 
                          of this file). Their value can be accessed as "args.argument".
    """
    ## 1. First, we load our data and flatten the images into vectors
    device = torch.device('cuda' if torch.cuda.is_available() else args.device)
    print(f"Using device: {device}")

    xtrain, xtest, ytrain = load_data(args.data)
    
    xtrain = xtrain.reshape(xtrain.shape[0], -1)
    xtest = xtest.reshape(xtest.shape[0], -1)

    ## 2. Then we must prepare it. This is where you can create a validation set,
    #  normalize, add bias, etc.

    # Make a validation set
    if not args.test:
        print("Creating validation set from training data")
        n = xtrain.shape[0]
        indices = np.random.permutation(n)
        xtrain = xtrain[indices]
        ytrain = ytrain[indices]
        split = int(0.8 * n)
        xval = xtrain[split:]
        yval = ytrain[split:]
        xtrain = xtrain[:split]
        ytrain = ytrain[:split]

    xtrain_mean = np.mean(xtrain)
    xtrain_std = np.std(xtrain)

    xtest_mean = np.mean(xtest)
    xtest_std = np.std(xtest)

    xtrain = normalize_fn(xtrain, xtrain_mean, xtrain_std)
    xtest = normalize_fn(xtest, xtest_mean, xtest_std)
    if not args.test:
        xval_mean = np.mean(xval)
        xval_std = np.std(xval)
        xval = normalize_fn(xval, xval_mean, xval_std)

    # Dimensionality reduction (MS2)
    if args.use_pca:
        print("Using PCA")
        pca_obj = PCA(d=args.pca_d)
        xtrain = pca_obj.reduce_dimension(xtrain.reshape(xtrain.shape[0], -1))
        xtest = pca_obj.reduce_dimension(xtest.reshape(xtest.shape[0], -1))
        if not args.test:
            xval = pca_obj.reduce_dimension(xval.reshape(xval.shape[0], -1))

    ## 3. Initialize the method you want to use.

    # Neural Networks (MS2)

    # Prepare the model (and data) for Pytorch
    n_classes = get_n_classes(ytrain)
    if args.nn_type == "transformer":
        
        xtrain = xtrain.reshape(xtrain.shape[0], 1, 28, 28)
        xtest = xtest.reshape(xtest.shape[0], 1, 28, 28)
        if not args.test:
            xval = xval.reshape(xval.shape[0], 1, 28, 28)

        chw = (1, 28, 28)  # MNIST images are 1 channel, 28x28 pixels
        n_patches = 7  # Split each dimension into 7 patches, so each patch is 4x4 pixels
        n_blocks = 8  # Number of transformer blocks
        hidden_d = 32  # Hidden dimension of the model
        n_heads = 16  # Number of attention heads
        out_d = n_classes  # Number of output classes (digits 0-9)

        model = MyViT(chw, n_patches, n_blocks, hidden_d, n_heads, out_d)
    elif args.nn_type == "mlp":
        model = MLP(input_size=xtrain.shape[1], n_classes=n_classes)
    elif args.nn_type == "cnn":
        
        xtrain = xtrain.reshape(xtrain.shape[0], 1, 28, 28)
        xtest = xtest.reshape(xtest.shape[0], 1, 28, 28)
        if not args.test:
            xval = xval.reshape(xval.shape[0], 1, 28, 28)

        model = CNN(input_channels=1, n_classes=n_classes)
    else:
        raise ValueError("Invalid nn_type, it can be 'mlp' | 'transformer' | 'cnn'")
    summary(model)

    # Trainer object
    method_obj = Trainer(model, lr=args.lr, epochs=args.max_iters, batch_size=args.nn_batch_size, device=device)

    ## 4. Train and evaluate the method

    # Fit (:=train) the method on the training data
    preds_train = method_obj.fit(xtrain, ytrain)

    # Predict on unseen data
    # preds = method_obj.predict(xtest)

    ## Report results: performance on train and valid/test sets
    acc = accuracy_fn(preds_train, ytrain)
    macrof1 = macrof1_fn(preds_train, ytrain)
    print(f"\nTrain set: accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")


    ## As there are no test dataset labels, check your model accuracy on validation dataset.
    # You can check your model performance on test set by submitting your test set predictions on the AIcrowd competition.
    if not args.test:
        preds_val = method_obj.predict(xval)
        acc = accuracy_fn(preds_val, yval)
        macrof1 = macrof1_fn(preds_val, yval)
        print(f"Validation set: accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")


    ### WRITE YOUR CODE HERE if you want to add other outputs, visualization, etc.


if __name__ == '__main__':
    # Definition of the arguments that can be given through the command line (terminal).
    # If an argument is not given, it will take its default value as defined below.
    parser = argparse.ArgumentParser()
    # Feel free to add more arguments here if you need!

    # MS2 arguments
    parser.add_argument('--data', default="dataset", type=str, help="path to your dataset")
    parser.add_argument('--nn_type', default="mlp",
                        help="which network architecture to use, it can be 'mlp' | 'transformer' | 'cnn'")
    parser.add_argument('--nn_batch_size', type=int, default=64, help="batch size for NN training")
    parser.add_argument('--device', type=str, default="cpu",
                        help="Device to use for the training, it can be 'cpu' | 'cuda' | 'mps'")
    parser.add_argument('--use_pca', action="store_true", help="use PCA for feature reduction")
    parser.add_argument('--pca_d', type=int, default=100, help="the number of principal components")
    parser.add_argument('--lr', type=float, default=1e-1, help="learning rate for methods with learning rate")
    parser.add_argument('--max_iters', type=int, default=15, help="max iters for methods which are iterative")
    parser.add_argument('--test', action="store_true",
                        help="train on whole training data and evaluate on the test data, otherwise use a validation set")

    # "args" will keep in memory the arguments and their values,
    # which can be accessed as "args.data", for example.
    args = parser.parse_args()
    main(args)
