import sys

def create_file(filepath, content):
    file = open(filepath, 'w')
    file.write(content)
    file.close()

def print_border():
    print("-" * 20)

def print_mode_info(args, ckpt, device, n_classes):
    if args.model is None: print("Mode: Training {}".format(ckpt))
    else: print("Mode: Evaluating {}".format(ckpt)) 
    
    print("Python {}".format(sys.version))
    print("Device: {}".format(device))
    print("Number of Classes: {}".format(n_classes))
    print("Parameters: ", args)

    print_border()
    sys.stdout.flush()

def log_metrics(loss, n_batches, correct=None, total=None):
    stats = 'Loss: %.4f ' % (loss / (n_batches))
    if correct is not None:
        acc = 100.0 * correct / total
        stats += '| Accuracy: %.2f' % (acc) 

    print(stats); print_border()
    sys.stdout.flush()
