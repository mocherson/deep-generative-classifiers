import argparse
from dcnn import *


parser = argparse.ArgumentParser(description='baseline classifier for chestXray')
parser.add_argument('--batch-size', type=int, default=16, metavar='N', help='input batch size for training (default: 16)')
parser.add_argument('--path', default='/home/hddraid/shared_data/chest_xray8/', type=str, help='data path')
parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: 10)')
parser.add_argument('--seed', type=int, default=1, metavar='N', help='random seed (default: 1)')
parser.add_argument('-e','--encoder', default='alex', type=str, help='the encoder')
parser.add_argument('--gpu', type=int, default=-1, metavar='N', help='the GPU number (default auto schedule)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',  help='manual epoch number (useful on restarts)')
args = parser.parse_args()

seed = args.seed
torch.manual_seed(args.seed)
batch_size = args.batch_size
enc = args.encoder

train_set = ChestXray_Dataset(path=args.path, use='train',transform=transform)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=16)
validation_set=ChestXray_Dataset(path=args.path,use='validation',transform=transform)
validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=16)
test_set=ChestXray_Dataset(path=args.path,use='test',transform=transform)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=16)
# bbox_set=ChestXray_Dataset(use='bboxtest',transform=transform)
# bbox_loader = DataLoader(bbox_set, batch_size=1, shuffle=True, pin_memory=True, num_workers=0)

torch.cuda.set_device(args.gpu)
path=join(args.path, 'models/baselines')  

if enc=='alex':
    model = MyAlexNet(14)
elif enc=='res50':
    model = MyResNet50(14)
elif enc=='vgg16bn':
    model = MyVggNet16_bn(14)
elif enc=='vgg16':
    model = MyVggNet16(14)
elif enc=='dens161':
    model=MyDensNet161(14)
elif enc=='dens201':
    model=MyDensNet201(14)
elif enc=='dens121':
    model=MyDensNet121(14)
    
print('running model %s...'%(enc))

model =   model.cuda()
criterion = W_BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5)
bestloss = np.inf 
val_perf=[] 
test_perf=[]  
if  args.start_epoch>0:   
    cp = torch.load(join(path,'checkpoint_%s_s%d.pth.tar'%(enc,seed))) 
    model.load_state_dict(cp['state_dict'])
    optimizer.load_state_dict(cp['optimizer'])
    s=cp['epoch']+1
    bestloss = cp['best_loss']
for epoch in range(args.start_epoch, args.epochs):                  
    train(train_loader, model, criterion, optimizer, epoch, 20)
    loss_val, auc_val = validate(validation_loader, model, criterion)
    auc,avgauc,_ = test(test_loader, model)
    val_perf.append(auc_val)
    test_perf.append(auc)

    isbest = loss_val < bestloss
    bestloss = min(bestloss,loss_val)
    save_checkpoint({
                      'epoch': epoch + 1,
                      'state_dict': model.state_dict(),
                      'loss_val': loss_val,
                      'best_loss': bestloss,
                      'auc_test': auc,
                      'avgauc_test': avgauc,
                      'optimizer' : optimizer.state_dict(),
                    }, isbest, filename = join(path,'checkpoint_%s_s%d.pth.tar'%(enc,seed)))

pd.DataFrame(test_perf, index=val_perf, columns= train_set.classes.keys()).to_csv(join(path,'results_%s_s%d.csv'%(enc,seed)) )                   

