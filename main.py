import os
import sys
import json
import time
import torch
import argparse

from tools.loader    import *
from tools.log_utils import *
from tools.models    import *

# ---------------------------- Model Training, Validation, and Evaluation ---------------------------- #

def __train(args, net, device, criterion, optimizer, trainloader):
    net.train()
    train_loss = 0.0

    sample_count = 0
    for _, _, _, img_feat, q_id_split, targets in trainloader:
        img_feat = img_feat.to(device)
        inputs = q_id_split.to(device)
        targets = targets.squeeze(1).to(device)

        # reset optimization at the beginning of each batch
        if sample_count == 0: 
            batch_loss = 0
            optimizer.zero_grad()

        # optimization
        outputs = net(inputs, img_feat)
        loss = criterion(outputs, targets)
        batch_loss += loss
        train_loss += loss.item()

        # optimize at the end of each batch
        if sample_count == args.batch_size-1:
            batch_loss.backward() 
            optimizer.step()
        sample_count = (sample_count + 1) % args.batch_size

        if args.debug: break
    log_metrics(train_loss, len(trainloader))
    
def __validate(args, net, device, id2answer, testloader):
    net.eval()
    total = 0.0
    correct = 0.0
    sample_count = 0
    log_freq = len(testloader) // 20

    with torch.no_grad():
        for _, _, q_str, img_feat, q_id_split, targets in testloader:
            img_feat = img_feat.to(device)
            inputs = q_id_split.to(device)
            targets = targets.squeeze(1).to(device)

            outputs = net(inputs, img_feat)
            total += targets.size(0)    
            _, predicted = outputs.max(1)   
            correct += predicted.eq(targets).sum().item()

            if sample_count % log_freq == 0:
                print(q_str, id2answer[predicted.squeeze().item()])
            sample_count += 1

            if args.debug: break
    
    acc = 100.0 * correct / total
    print('Accuracy: %.2f' % (acc))
    print_border()
    sys.stdout.flush()

    return acc

# ---------------------------- Exposed API ---------------------------- #

def train(args, ckpt, net, device, id2answer, trainloader, testloader):
    print("TRAINING"); print_border()

    # loss function
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weightdecay)

    # setup optimization
    epoch = 1
    best_val_acc   = 0
    best_val_epoch = 0
    time_all = time.time()

    # optimization loop 
    while best_val_epoch >= epoch - args.epochs:
        print('Epoch: %d' % epoch)

        # optimize and validate
        __train(args, net, device, criterion, optimizer, trainloader)
        epoch_acc = __validate(args, net, device, id2answer, testloader)

        # checkpoint the best net seen so far
        if epoch_acc > best_val_acc:
            # track best metrics
            best_val_acc   = epoch_acc
            best_val_epoch = epoch

            # update saved model
            best_net_path = os.path.join('models', ckpt)
            torch.save(net.state_dict(), best_net_path)
            print('Updated {}.'.format(ckpt))
            print_border()

        time_elapsed = time.time() - time_all
        print('Time elapsed: {:.0f}m {:.0f}s.'.format(
              time_elapsed // 60, time_elapsed % 60))

        epoch += 1
        if args.debug: break

    # load the best net
    print('Best Validation Accuracy: {:.2f} attained at epoch {}'.format(best_val_acc, best_val_epoch))
    best_net_filename = os.path.join('models', ckpt)
    net.load_state_dict(torch.load(best_net_filename))

def evaluate(args, net, device, id2answer, testloader):
    print("EVALUATION"); print_border()

    net.eval()
    results = []
    sample_count = 0
    log_freq = len(testloader) // 20

    with torch.no_grad():
        for img_id, q_id, q_str, img_feat, q_id_split, _ in testloader:
            img_feat = img_feat.to(device)
            inputs = q_id_split.to(device)

            outputs = net(inputs, img_feat, only_img=args.only_img, only_q=args.only_q)
            _, predicted = outputs.max(1)   

            answer = id2answer[predicted.squeeze().cpu().numpy()]
            q_id = int(q_id.cpu().numpy()[0])
            results.append({"answer":answer, "question_id":q_id})

            if sample_count % log_freq == 0:
                img_id = int(img_id.cpu().numpy()[0])
                print(img_id, q_str, id2answer[predicted.squeeze().item()])
            sample_count += 1

            if args.debug: break
    create_file('results/{}.json'.format(args.model), json.dumps(results))

# ---------------------------- Core Pipeline ---------------------------- #

def pipeline(args):
    # load GloVe
    glove_vocab, glove_word2idx, glove = load_GloVe()
    print("{} words in GloVe including {}.".format(len(glove_vocab), glove_vocab[-1]))
    print_border()

    # load data
    effective_batch_size = 1
    trainset, testset = get_data(glove_word2idx, args.top_n_answers)
    trainloader, testloader = get_dataloaders(effective_batch_size, trainset, testset)
    print_border()
    
    # load network
    assert(args.only_img + args.only_q < 1)
    n_classes = args.top_n_answers
    hidden_dim = trainset.img_feat_dim // 4
    net = LSTM(torch.FloatTensor(glove), trainset.img_feat_dim, hidden_dim, 
               n_classes, effective_batch_size, num_layers=1)
    net, device = load_to_device(net)
    net.module.set_device(device)
    print_border()    

    # log
    ckpt = "vqa"
    if args.debug: ckpt += "_debug"
    if args.model is not None: ckpt = args.model
    print_mode_info(args, ckpt, device, n_classes)

    if args.model is None:
        train(args, ckpt, net, device, trainset.id2answer, trainloader, testloader)  
    else:
        net.load_state_dict(torch.load(os.path.join("models", args.model), map_location=device))
        evaluate(args, net, device, trainset.id2answer, testloader)

# Python version: 3.6.7

if __name__ == '__main__':
    description = 'Train classifier on activity classification'
    parser = argparse.ArgumentParser(description)

    help_str = 'Filename of model to evaluate; leave None if you want to train model from scratch'
    parser.add_argument("--model", help=help_str)
    help_str = 'set to evaluate using just the image features'
    parser.add_argument("--only_img", action='store_true', default=False, help=help_str)
    help_str = 'set to evaluate using just the question features'
    parser.add_argument("--only_q", action='store_true', default=False, help=help_str)

    help_str = 'number of epochs to test until convergence'
    parser.add_argument('--epochs', default=5, type=int, help=help_str)
    help_str = 'batch size'
    parser.add_argument('--batch_size', default=64, type=int, help=help_str)
    help_str = 'train on top_n_answers'
    parser.add_argument('--top_n_answers', default=1000, type=int, help=help_str)

    help_str = 'learning rate'
    parser.add_argument('--lr', default=1e-5, type=float, help=help_str)
    help_str = 'weight decay'
    parser.add_argument('--weightdecay', default=0, type=float, help=help_str)

    help_str = 'set to enable debug mode'
    parser.add_argument("--debug", action='store_true', default=False, help=help_str)

    print_border()
    ARGS = parser.parse_args()
    pipeline(ARGS)
