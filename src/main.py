from dl_model import model
import matplotlib.pyplot as plt
from dl_dataset import load_dataset
from dl_dataset import write_label

def plot_results(y_label_list, y_predict_list, acc):
    for i in range(max(len(y_label_list),len(y_predict_list))):
        plt.title('Result of prediction ' + str(test_set[i]) + ' Acc: ' + str(acc[i]))
        plt.xlabel('Time')
        plt.ylabel('Label')
        if y_label_list is not None:
            y_label = y_label_list[i]
            plt.plot([f/30 for f in range(len(y_label))], y_label, c='r', label='target')
        if y_predict_list is not None:
            y_predict = y_predict_list[i]
            plt.plot([f/30 for f in range(len(y_predict))], y_predict, c='b', label='prediction')
        # plt.legend(loc='upper right', frameon=False)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig(str(test_set[i]) + '_prediction.png')
        plt.show()


if __name__ == '__main__':
    # load the dataset info file
    dataset = load_dataset('rolling.csv')
    # build the model through the model name
    cnnlstm = model('cnnlstm')

    #############################
    # configure here to change the train set and test set
    # set the train set and test set through clip_id in the dataset info file 'rolling.csv'
    train_set = dataset.id[:100]
    test_set = dataset.id[119:129]
    ##############################

    # Automatically generate the train set and test set
    x_train, y_train, x_test, y_test = dataset.train_test_split(train_set, test_set)

    ##############################
    # # uncomment the section if train the model
    # history = cnnlstm.model.fit(x_train, y_train, epochs=50, batch_size=1)
    # cnnlstm.save()
    ##############################

    ##############################
    # uncomment the section if load the trained model
    cnnlstm.load()
    ##############################

    # predict the test set and calculate the accuracy
    predict = cnnlstm.predict(x_test)
    accuracy = cnnlstm.evaluate(predict, y_test)
    for i in range(len(accuracy)):
        print('clip id: ', test_set[i], ' acc: ', accuracy[i])
        write_label(str(test_set[i]) + '_prediction.json', predict[i])

    # plot the target and the prediction
    plot_results(y_test, predict, accuracy)
