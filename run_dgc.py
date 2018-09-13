import argparse
from dcnn import *


parser = argparse.ArgumentParser(description='deep generative classifier for chestXray')
parser.add_argument('--batch-size', type=int, default=16, metavar='N', help='input batch size for training (default: 16)')
parser.add_argument('--path', default='/home/hddraid/shared_data/chest_xray8/', type=str, help='data path')
parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: 10)')
parser.add_argument('--seed', type=int, default=1, metavar='N', help='random seed (default: 1)')
parser.add_argument('--gpu', type=int, default=-1, metavar='N', help='the GPU number (default auto schedule)')
parser.add_argument('-e','--encoder', default='alex', type=str, help='the encoder')
parser.add_argument('-w','--lossweight', default=0, metavar='N', type=float, help='weight of KL divergence (default: 0)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',  help='manual epoch number (useful on restarts)')
args = parser.parse_args()

seed = args.seed
torch.manual_seed(seed)
batch_size = args.batch_size
enc = args.encoder
weight = args.lossweight


train_set = ChestXray_Dataset(path=args.path, use='train',transform=transform)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=16)
validation_set=ChestXray_Dataset(path=args.path,use='validation',transform=transform)
validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=16)
test_set=ChestXray_Dataset(path=args.path,use='test',transform=transform)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=16)


if enc=='alex':
    encoder = MyAlexNet(14).features[:-2]
    hc = 256
elif enc=='res50':
    encoder = MyResNet50(14).features[:-2]
    hc = 2048
elif enc=='vgg16bn':
    encoder = MyVggNet16_bn(14).features[:-2]
    hc = 512
elif enc=='vgg16':
    encoder = MyVggNet16(14).features[:-2]
    hc = 512
elif enc=='dens161':
    encoder=MyDensNet161(14).features[:-2]
    hc = 2208
elif enc=='dens201':
    encoder=MyDensNet201(14).features[:-2]
    hc = 1920
elif enc=='dens121':
    encoder=MyDensNet121(14).features[:-2]
    hc = 1024

if args.gpu>=0:
    torch.cuda.set_device(args.gpu)
folder = join(args.path, 'models')   

model = VClassifier(encoder, hid_channel=hc, outnum=14).cuda()
# model =  nn.DataParallel(model)
criterion = VLoss(w=weight)
optimizer = optim.Adam(model.parameters(), lr=1e-5)

print('training %s with weight=%s...'%(enc, weight))
bestloss = np.inf 
bestauc = 0
s=args.start_epoch
if  args.start_epoch>0:   
    cp = torch.load(join(folder, enc, 'checkpoint_loss_w%s_seed%d.pth.tar'%(weight, seed))) 
    model.load_state_dict(cp['state_dict'])
    optimizer.load_state_dict(cp['optimizer'])
    s=cp['epoch']+1
    bestauc, bestloss = cp['bestauc'], cp['best_loss']
val_perf=[] 
test_perf=[] 
for epoch in range(s, args.epochs):    		
    loss_tr = train2(train_loader, model, criterion, optimizer, epoch, 20)
    checkpoint = {'epoch': epoch, 'state_dict': model.state_dict(), 'loss_tr': loss_tr, 'optimizer': optimizer.state_dict()}
    loss_val, auc_val = validate2(validation_loader, model, criterion)
    auc,avgauc,_=test2(test_loader, model)
    val_perf.append(auc_val)
    test_perf.append(auc)
    
    checkpoint.update({'loss_val': loss_val, 'auc_val': auc_val, 'best_loss': min(bestloss,loss_val),  \
                       'bestauc': max(bestauc,auc_val), 'auc_test': auc, 'avgauc_test': avgauc,}) 
    torch.save(checkpoint, join(folder, enc, 'checkpoint_loss_w%s_seed%d.pth.tar'%(weight, seed))   )
    # if loss_val < bestloss:
    #     bestloss = loss_val
    #     torch.save(checkpoint, join(folder, enc, 'best_loss_w%s_seed%d.pth.tar'%(weight, seed)))
    if bestauc<auc_val:
        bestauc=auc_val
        torch.save(checkpoint, join(folder, enc, 'best_auc_w%s_seed%d.pth.tar'%(weight, seed)))
        
pd.DataFrame(test_perf, index=val_perf, columns= train_set.classes.keys()).to_csv(join(folder, enc, 'results_w%s_seed%d.csv'%(weight, seed)))  
    

