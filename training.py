import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import pandas as pd
from sklearn.metrics import confusion_matrix
from collections import deque
import numpy as np
import os
import os.path as osp 
import sys
import matplotlib.pyplot as plt
import tensorboard
import tensorboardX
from tensorboardX import SummaryWriter
import pandas as pd
from functools import partial
from sklearn.metrics import pairwise_distances, pairwise


class EarlyStopping(object):
    '''EarlyStopping handler can be used to stop the training if no improvement after a given number of events
    Args:
        patience (int):
            Number of events to wait if no improvement and then stop the training
    '''
    def __init__(self, patience):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.meter = deque(maxlen=patience)

    def is_stop_training(self, score):
        stop_sign = False

        self.meter.append(score)

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score:
            self.best_score = score
        elif score - self.best_score >= .4:
            self.counter += 1
        elif .3 <= score - self.best_score < .4:
            self.counter += .50
        elif .2 <= score - self.best_score < .3:
            self.counter += .25
        elif .1 <= score - self.best_score < .2:   
            self.counter += .1
        elif score - self.best_score <.1:
            self.counter = self.counter

        if self.counter > self.patience:
            print("counter is greater than patience")
            stop_sign = True

        return stop_sign


def gaussian_kernel_matrix(x, y, sigmas):
    '''
    Gaussian RBF kernel to be used in MMD
    Args:
    x,y: latent features
    sigmas: free parameter that determins the width of the kernel
    Returns:
    '''
    beta = 1. / (2. * (torch.unsqueeze(sigmas, 1)))
    dist = compute_pairwise_distances(x, y)
    s = torch.matmul(beta, dist.contiguous().view(1, -1))
    return torch.sum(torch.exp(-s), 0).view(*dist.size())


def compute_pairwise_distances(x, y):
    if not x.dim() == y.dim() == 2:
        raise ValueError('Both inputs should be matrices.')
    if x.size(1) != y.size(1):
        raise ValueError('The number of features should be the same.')

    norm = lambda x: torch.sum(torch.pow(x, 2), 1)
    return torch.transpose(norm(torch.unsqueeze(x, 2) - torch.transpose(y, 0, 1)), 0, 1)


def maximum_mean_discrepancy(x, y, kernel=gaussian_kernel_matrix):
    ''' 
    Calculate the matrix that includes all kernels k(xx), k(y,y) and k(x,y)
    '''
    cost = torch.mean(kernel(x, x))
    cost += torch.mean(kernel(y, y))
    cost -= 2 * torch.mean(kernel(x, y))
    # We do not allow the loss to become negative. 
    cost = torch.clamp(cost, min=0.0)
    return cost


def mmd_distance(hs, ht):
    '''
    Maximum Mean Discrepancy - MMD
    Args:
        hs: source domain embeddings
        ht: target domain embeddings
    Returns:
        MMD value
    '''
    sigmas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5,
              10, 15, 20, 25, 30, 35, 100, 1e3, 1e4, 1e5, 1e6]
    gaussian_kernel = partial(gaussian_kernel_matrix,
                              sigmas=torch.Tensor(sigmas).double().cuda())
    loss_value = maximum_mean_discrepancy(hs, ht, kernel=gaussian_kernel)
    return torch.clamp(loss_value, min=1e-4)


def train(base_network, dset_loaders, weights, epochs, early_stop_patience, output_path, use_gpu):
 
    classifier_loss, best_acc, temp_acc, train_acc = 0.0, 0.0, 0.0, 0.0

    opt = optim.Adam(base_network.parameters(), lr = .00001, betas= (0.7, 0.8), weight_decay = .001, amsgrad = False, eps = 1e-8) 
    class_weight = torch.from_numpy(np.array(weights, dtype = np.float64))
    class_criterion = nn.CrossEntropyLoss(weight=class_weight) 
    early_stop_engine = EarlyStopping(early_stop_patience)
    train_examples = len(dset_loaders["train"])
    num_train_iterations = train_examples*epochs
    snapshot_interval = num_train_iterations * 0.25
    num_valid_iterations = len(dset_loaders["valid"])

    if use_gpu:
        base_network = base_network.cuda()

    # set up summary writer
    writer = SummaryWriter(output_path)

    # Set log file
    if not osp.exists(output_path):
        os.makedirs(osp.join(output_path))
        out_file = open(osp.join(output_path, "log.txt"), "w")
    else:
        out_file = open(osp.join(output_path, "log.txt"), "w")

    #################
    # Training step
    #################

    for i in range(0, num_train_iterations):
        if i % train_examples == 0:
            base_network.train(False)

            snapshot_obj = {'epoch': i/train_examples, 
                            "base_network": base_network.state_dict(), 
                            'valid accuracy': temp_acc,
                            'train accuracy' : train_acc,
                            }

            snapshot_obj['class_criterion'] = class_criterion.state_dict()

            if (i+1) % snapshot_interval == 0:
                torch.save(snapshot_obj, osp.join(output_path, "epoch_{}_model.pth.tar".format(i/train_examples)))
                    
            if temp_acc > best_acc:
                best_acc = temp_acc

                # Save best model
                torch.save(snapshot_obj, osp.join(output_path, "best_model.pth.tar"))

            log_str = "epoch: {}, validation accuracy: {:.5f}, training accuracy: {:.5f}\n".format(i/train_examples, temp_acc, train_acc)
            out_file.write(log_str)
            out_file.flush()
            writer.add_scalar("Validation Accuracy", temp_acc, i/train_examples)
            writer.add_scalar("Training Accuracy", train_acc, i/train_examples)

        ## Train one iteration
        base_network.train(True)

        try:
            inputs_train, labels_train = iter(dset_loaders["train"]).next()
        except StopIteration:
            iter(dset_loaders["train"])

        if use_gpu:
            inputs_train, labels_train = Variable(inputs_train).cuda(), Variable(labels_train).cuda()
        else:
            inputs_train, labels_train = Variable(inputs_train), Variable(labels_train)
            
        inputs = inputs_train
        train_batch_size = inputs_train.size(0)

        features, logits = base_network(inputs)
        train_logits = logits.narrow(0, 0, train_batch_size)

        classifier_loss = class_criterion(train_logits, torch.argmax(labels_train.long(), axis = 1))
        classifier_loss.backward()

        opt.step()

        if i % train_examples == 0:

            train_acc, _ = classification_test(dset_loaders, 'train', base_network, gpu=use_gpu, verbose = False, save_where = None)

            # Logging:
            out_file.write('epoch {}: train classifier loss={:0.4f}\n'.format(i/train_examples, classifier_loss.data.cpu().float().item(),))
            out_file.flush()

            # Logging for tensorboard
            writer.add_scalar("Training Total Loss", classifier_loss.data.cpu().float().item(), i/train_examples)
                
            #################
            # Validation step
            #################
            for j in range(0, num_valid_iterations):
                base_network.train(False)
                with torch.no_grad():

                    try:
                        inputs_valid, labels_valid = iter(dset_loaders["valid"]).next()
                    except StopIteration:
                        iter(dset_loaders["valid"])

                    if use_gpu:
                        inputs_valid, labels_valid = Variable(inputs_valid).cuda(), Variable(labels_valid).cuda()
                    else:
                        inputs_valid,labels_valid = Variable(inputs_valid), Variable(labels_valid)
                        
                    inputs = inputs_valid
                    valid_batch_size = inputs_valid.size(0)

                    features, logits = base_network(inputs)
                    valid_logits = logits.narrow(0, 0, valid_batch_size)
                
                    classifier_loss = class_criterion(valid_logits, torch.argmax(labels_valid.long(), axis = 1))

                if j % num_valid_iterations == 0:
                    temp_acc, _ = classification_test(dset_loaders, 'valid', base_network, gpu=use_gpu, verbose = False, save_where = None)

                    out_file.write('epoch {}: valid classifier loss={:0.4f}\n'.format(i/train_examples, classifier_loss.data.cpu().float().item(),))
                    out_file.flush()

                    writer.add_scalar("Validation Total Loss", classifier_loss.data.cpu().float().item(), i/train_examples)
            
                    if early_stop_engine.is_stop_training(classifier_loss.cpu().float().item()):
                        out_file.write("overfitting after {}, stop training at epoch {}\n".format(early_stop_patience, i/train_examples))
                        out_file.write("finish training! \n")
                        out_file.close()
                        torch.save(snapshot_obj, osp.join(output_path, "final_model.pth.tar"))

                        sys.exit()

    out_file.write("finish training! \n")
    out_file.close()
    torch.save(snapshot_obj, osp.join(output_path, "final_model.pth.tar"))



def train_da(base_network, dset_loaders, weights, epochs, early_stop_patience, output_path, use_gpu):

    best_acc, temp_acc, train_acc, transfer_loss, classifier_loss, total_loss  = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    opt = optim.Adam(base_network.parameters(), lr = .00001, betas= (0.7, 0.8), weight_decay = .0001, amsgrad = False, eps = 1e-8) 
    class_weight = torch.from_numpy(np.array(weights, dtype = np.float64))
    tr_loss_weights = 0.01
    class_criterion = nn.CrossEntropyLoss(weight=class_weight) 
    early_stop_engine = EarlyStopping(early_stop_patience)
    train_examples = len(dset_loaders["train"])
    num_train_iterations = train_examples*epochs
    train_examples_noisy = len(dset_loaders["train_noisy"])
    num_valid_iterations_noisy = len(dset_loaders["valid_noisy"])
    num_train_iterations = train_examples*epochs
    snapshot_interval = num_train_iterations * 0.25
    num_valid_iterations = len(dset_loaders["valid"])

    if use_gpu:
        base_network = base_network.cuda()

    # set up summary writer
    writer = SummaryWriter(output_path)

    # Set log file
    if not osp.exists(output_path):
        os.makedirs(osp.join(output_path))
        out_file = open(osp.join(output_path, "log.txt"), "w")
    else:
        out_file = open(osp.join(output_path, "log.txt"), "w")

    #################
    # Training step
    #################


    for i in range(0, num_train_iterations):
        if i % train_examples == 0:
            base_network.train(False)

            snapshot_obj = {'epoch': i/train_examples, 
                            "base_network": base_network.state_dict(), 
                            'valid accuracy': temp_acc,
                            'train accuracy' : train_acc,
                            }
            snapshot_obj['class_criterion'] = class_criterion.state_dict()

            if (i+1) % snapshot_interval == 0:
                torch.save(snapshot_obj, 
                        osp.join(output_path, "epoch_{}_model.pth.tar".format(i/train_examples)))
                    
            if temp_acc > best_acc:
                best_acc = temp_acc
                # Save best model
                torch.save(snapshot_obj, 
                            osp.join(output_path, "best_model.pth.tar"))
            log_str = "epoch: {}, validation accuracy: {:.5f}, training accuracy: {:.5f}\n".format(i/train_examples, temp_acc, train_acc)
            out_file.write(log_str)
            out_file.flush()
            writer.add_scalar("Validation Accuracy", temp_acc, i/train_examples)
            writer.add_scalar("Training Accuracy", train_acc, i/train_examples)

        ## Train one iteration
        base_network.train(True)

        try:
            inputs_train, labels_train = iter(dset_loaders["train"]).next()
            inputs_train_noisy, labels_train_noisy = iter(dset_loaders["train_noisy"]).next()
        except StopIteration:
            iter(dset_loaders["train"])
            iter(dset_loaders["train_noisy"])

        if use_gpu:
            inputs_train, inputs_train_noisy, labels_train = Variable(inputs_train).cuda(), Variable(inputs_train_noisy).cuda(), Variable(labels_train).cuda()
        else:
            inputs_train, inputs_train_noisy, labels_train = Variable(inputs_train), Variable(inputs_train_noisy), Variable(labels_train)
            
        inputs = torch.cat((inputs_train, inputs_train_noisy), dim=0)
        train_batch_size = inputs_train.size(0)

        features, logits = base_network(inputs)
        train_logits = logits.narrow(0, 0, train_batch_size)

        #transfer loss
        transfer_loss = mmd_distance(features[:train_batch_size], features[train_batch_size:])

        # Source domain classification task loss
        classifier_loss = class_criterion(train_logits, torch.argmax(labels_train.long(), axis = 1))

        #classifier_loss.backward()
        total_loss = tr_loss_weights * transfer_loss + classifier_loss
        total_loss.backward()

        opt.step()

        if i % train_examples == 0:

            train_acc, _ = classification_test(dset_loaders, 'train', base_network, gpu=use_gpu, verbose = False, save_where = None)

            # Logging:
            out_file.write('epoch {}: train total loss={:0.4f}, train transfer loss={:0.4f}, train classifier loss={:0.4f}\n'.format(
                        i/train_examples, total_loss.data.cpu().float().item(), transfer_loss.data.cpu().float().item(), classifier_loss.data.cpu().float().item(),))
            out_file.flush()

            # Logging for tensorboard
            writer.add_scalar("Training total loss", total_loss.data.cpu().float().item(), i/train_examples)
            writer.add_scalar("Training classifier loss", classifier_loss.data.cpu().float().item(), i/train_examples)
            writer.add_scalar("Training transfer loss", transfer_loss.data.cpu().float().item(), i/train_examples)

                
            #################
            # Validation step
            #################
            for j in range(0, num_valid_iterations):
                base_network.train(False)
                with torch.no_grad():

                    try:
                        inputs_valid, labels_valid = iter(dset_loaders["valid"]).next()
                        inputs_valid_noisy, labels_valid_noisy = iter(dset_loaders["valid_noisy"]).next()

                    except StopIteration:
                        iter(dset_loaders["valid"])
                        iter(dset_loaders["valid_noisy"])

                    if use_gpu:
                        inputs_valid, inputs_valid_noisy, labels_valid = Variable(inputs_valid).cuda(), Variable(inputs_valid_noisy).cuda(), Variable(labels_valid).cuda()
                    else:
                        inputs_valid,inputs_valid_noisy, labels_valid = Variable(inputs_valid), Variable(inputs_valid_noisy).cuda(), Variable(labels_valid)
                        
                    valid_inputs = torch.cat((inputs_valid, inputs_valid_noisy), dim=0)
                    valid_batch_size = inputs_valid.size(0)

                    features, logits = base_network(valid_inputs)
                    valid_logits = logits.narrow(0, 0, valid_batch_size)

                    # Transfer loss - MMD
                    transfer_loss = mmd_distance(features[:valid_batch_size], features[valid_batch_size:])
                
                    # Source domain classification task loss
                    classifier_loss = class_criterion(valid_logits, torch.argmax(labels_valid.long(), axis = 1))

                    # Final loss in case we do not want to add Fisher loss and Entropy minimization
                    total_loss = tr_loss_weights * transfer_loss + classifier_loss

                # Logging:
                if j % num_valid_iterations == 0:
                    temp_acc, _ = classification_test(dset_loaders, 'valid', base_network, gpu=use_gpu, verbose = False, save_where = None)

                    out_file.write('epoch {}: valid total loss={:0.4f}, valid transfer loss={:0.4f}, valid classifier loss={:0.4f}\n'.format(
                        i/train_examples, total_loss.data.cpu().float().item(), transfer_loss.data.cpu().float().item(), classifier_loss.data.cpu().float().item(),))
                    out_file.flush()


                    # Logging for tensorboard:
                    writer.add_scalar("Validation total loss", total_loss.data.cpu().float().item(), i/train_examples)
                    writer.add_scalar("Validation classifier loss", classifier_loss.data.cpu().float().item(), i/train_examples)
                    writer.add_scalar("Validation transfer loss", transfer_loss.data.cpu().float().item(), i/train_examples)


                    # Early stop in case we see overfitting
                    if early_stop_engine.is_stop_training(classifier_loss.cpu().float().item()):
                        out_file.write("overfitting after {}, stop training at epoch {}\n".format(early_stop_patience, i/train_examples))
                        out_file.write("finish training! \n")
                        out_file.close()
                        torch.save(snapshot_obj, osp.join(output_path, "final_model.pth.tar"))

                        sys.exit()

    out_file.write("finish training! \n")
    out_file.close()
    torch.save(snapshot_obj, osp.join(output_path, "final_model.pth.tar"))


def classification_test(loader, dictionary_val, model, gpu=True, verbose = False, save_where = None):
    start_test = True

    with torch.no_grad():
        iter_test = iter(loader[str(dictionary_val)])
        for i in range(len(loader[str(dictionary_val)])):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            if gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()
            
            _, outputs = model(inputs)
            softmax_outputs = nn.Softmax(dim=1)(1.0 * outputs)

            if start_test:
                all_softmax_output = softmax_outputs.data.float()
                all_output = outputs.data.float()
                all_label = labels.data.float()
                start_test = False
            else:
                all_softmax_output = torch.cat((all_softmax_output, softmax_outputs.data.float()), 0)
                all_output = torch.cat((all_output, outputs.data.float()), 0)
                all_label = torch.cat((all_label, labels.data.float()), 0)

    #print(all_softmax_output)
    predict = torch.argmax(all_softmax_output, axis = 1)
    #print(predict)
    all_label = torch.argmax(all_label, axis = 1)
    #print(all_label)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).float() / float(all_label.size()[0])
    #print(accuracy)

    conf_matrix = confusion_matrix(all_label.cpu().numpy(), predict.cpu().numpy())

    if verbose:
        output = pd.DataFrame()

        df = pd.DataFrame(all_softmax_output.cpu().detach().numpy(), columns=['elliptical', 'spiral']) #i think? spiral = 1, elliptical = 0
        output['model output'] = pd.Series(torch.max(all_output, 1)[1].cpu().detach().numpy())
        output['labels'] = pd.Series(all_label.cpu().detach().numpy())

        output.to_csv(str(save_where)+"/model_results_"+str(dictionary_val)+".csv")
        df.to_csv(str(save_where)+"/model_predictions_"+str(dictionary_val)+".csv")

    return accuracy, conf_matrix


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 22})
    import numpy as np
    import itertools

    accuracy = (np.trace(cm) / float(np.sum(cm)))*100
    misclass = 100 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('RdPu')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=0)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.2f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('\n Predicted label \n \naccuracy={:0.1f}%\n misclass={:0.1f}%'.format(accuracy,misclass))
    plt.show()