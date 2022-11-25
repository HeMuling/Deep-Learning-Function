import torch
import torch.nn as nn


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)

def dl_train(net, trainer, train_iter, test_iter, num_epoch, device=None):

    '''
    code adapted from https://www.zhihu.com/question/406133826/answer/1334319004
    '''

    device = torch.device('cpu') if device == None else device

    net.apply(init_weights)
    net.to(device)

    loss_function = nn.CrossEntropyLoss()

    # begin to train
    net.train()
    for epoch in range(num_epoch):
        running_loss = 0.0
        for x, y in enumerate(train_iter):
            
            # get data and label, send them into training device (e.g. cuda or cpu)
            data, label = y
            data = data.to(device)
            label = label.to(device)

            # set the grad to zero
            trainer.zero_grad()

            # forward prop
            y_hat = net(data)
            loss = loss_function(y_hat, label)

            # backward prop
            loss.backward()

            # update
            trainer.step()

            running_loss += loss.item()
            
            # show loss every 100 batches
            if x % 100 == 99:
                print(f'epoch {epoch+1}, batch {x+1}, loss {running_loss/100:.3f}')
                running_loss = 0.0
    # begin evaluation
    net.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for x, y in enumerate(test_iter):

            # get data and label, send them into training device (e.g. cuda or cpu)
            data, label = y
            data = data.to(device)
            label = label.to(device)
            
            # compute the prediction
            y_hat = net(data)

            # make the prediction
            _, predicted = torch.max(y_hat.data, 1)

            # make the count
            total += label.size(0)
            correct += (predicted == label).sum().item()

        print(f'Accuracy of the network on the {total} test images: {100 * correct / total}%')
        

'''
an example:
'''

'''
net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
    nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.Sigmoid(),
    nn.Linear(84, 10)
)

path = os.path.join('mnist_data')
trans = torchvision.transforms.ToTensor()

mnistdata_train = torchvision.datasets.MNIST(
    path, train=True,
    transform=trans,
    download=True
)
mnistdata_test = torchvision.datasets.MNIST(
    path, train=False,
    transform=trans, 
    download=True
)

trainer = torch.optim.SGD(net.parameters(), lr=0.9)


train_iter = torch.utils.data.DataLoader(mnistdata_train, batch_size=256, shuffle=True)
test_iter = torch.utils.data.DataLoader(mnistdata_test, batch_size=256, shuffle=True)


dl_train(net, trainer, train_iter, test_iter, 10, torch.device('mps'))
'''