from . import preprocessing
from . import model
#import pickle

def main():  
    # hyperparameters from paper: 
    batch_size = 100
    layers = 1
    encoder = BinaryTreeLSTM()
    #Decoder = LSTM
    lr = 0.005
    h_size = 256
    enbedding_size = 256
    dropout = 0.5
    #Weights initialization : [-0.1,0.1]
    epochs = 30

    # call encoder-decoder model
    model = TreeLSTM(encoder, decoder)
    # define optimizer
    # initialize epochs
    # train network and start timer and end timer for each epoch
    # start counter
    start = time.time()
    model.train()
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                        (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')
    # end timer
    end = time.time()
    print ('Total time in seconds:')
    print (end-start)

    # save model for web app
     # model = pickle.load(open('model.pkl','rb'))
    torch.save(model, 'model.pt')