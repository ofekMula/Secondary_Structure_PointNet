import matplotlib.pyplot as plt

LOG_FILE_PATH_3 = "./train3.txt"
LOG_FILE_PATH_8 = "./train8.txt"
LOG_FILE_PATH_local = "./local.txt"
EPOCH_PREFIX_STR = "****"
EVAL_MEAN_LOSS = "eval mean loss"
EVAL_ACCURACY = "eval accuracy"
EVAL_AVG_CLASS_ACC = "eval avg class acc"
EPOCH_VAL_DELIMETER = ": "
EPOCH_NUMBER_LABEL = "Epoch number"
EPOCH_ACCURACY_LABEL = "accuracy"
EPOCH_AVG_CLASS_ACC_LABEL = "avg class acc"
EPOCH_MEAN_LOSS_LABEL = "mean loss"

AVG_ACCURACY_PLOT_NAME = "acc vs epoch number.png"
MEAN_LOSS_PLOT_NAME = "mean loss vs epoch number.png"
AVG_CLASS_ACCURACY_PLOT_NAME = "avg class acc vs epoch number.png"
IN_EPOCH = False


#
def visualize_training_log_2(train_results_3, train_results_8, train_results_local):
    epoches_indexes = [i for i in range(0, len(train_results_3))]
    eval_accuracy_per_ecpoch_3 = [train_results_3[i][1] for i in range(0, len(train_results_3))]
    eval_loss_per_ecpoch_3 = [train_results_3[i][0] for i in range(0, len(train_results_3))]
    eval_avg_class_accuracy_per_ecpoch_3 = [train_results_3[i][2] for i in range(0, len(train_results_3))]

    eval_accuracy_per_ecpoch_8 = [train_results_8[i][1] for i in range(0, len(train_results_8))]
    eval_loss_per_ecpoch_8 = [train_results_8[i][0] for i in range(0, len(train_results_8))]
    eval_avg_class_accuracy_per_ecpoch_8 = [train_results_8[i][2] for i in range(0, len(train_results_8))]

    eval_accuracy_per_ecpoch_local = [train_results_local[i][1] for i in range(0, len(train_results_local))]
    eval_loss_per_ecpoch_local = [train_results_local[i][0] for i in range(0, len(train_results_local))]
    eval_avg_class_accuracy_per_ecpoch_local = [train_results_local[i][2] for i in range(0, len(train_results_local))]

    fig1, ax1 = plt.subplots(1)  # Creates figure fig and add an axes, ax.
    fig2, ax2 = plt.subplots(1)  # Another figure
    fig3, ax3 = plt.subplots(1)  # Another figure

    ax1.set_xlabel(EPOCH_NUMBER_LABEL)
    ax1.set_ylabel(EPOCH_ACCURACY_LABEL)
    ax1.plot(epoches_indexes[0:28], eval_accuracy_per_ecpoch_3[0:28])
    ax1.plot(epoches_indexes[0:28], eval_accuracy_per_ecpoch_8[0:28])
    ax1.plot(epoches_indexes[0:28], eval_accuracy_per_ecpoch_local[0:28])

    ax1 = ax1.legend(["train3", "train8", "local"])

    ax2.set_xlabel(EPOCH_NUMBER_LABEL)
    ax2.set_ylabel(EPOCH_MEAN_LOSS_LABEL)
    ax2.plot(epoches_indexes[0:28], eval_loss_per_ecpoch_3[0:28])
    ax2.plot(epoches_indexes[0:28], eval_loss_per_ecpoch_8[0:28])
    ax2.plot(epoches_indexes[0:28], eval_loss_per_ecpoch_local[0:28])

    ax2 = ax2.legend(["train3", "train8", "local"])

    ax3.set_xlabel(EPOCH_NUMBER_LABEL)
    ax3.set_ylabel(EPOCH_AVG_CLASS_ACC_LABEL)
    ax3.plot(epoches_indexes[0:28], eval_avg_class_accuracy_per_ecpoch_3[0:28])
    ax3.plot(epoches_indexes[0:28], eval_avg_class_accuracy_per_ecpoch_8[0:28])
    ax3.plot(epoches_indexes[0:28], eval_avg_class_accuracy_per_ecpoch_local[0:28])

    ax2 = ax3.legend(["train3", "train8", "local"])

    fig1.savefig(AVG_ACCURACY_PLOT_NAME)
    fig2.savefig(MEAN_LOSS_PLOT_NAME)
    fig3.savefig(AVG_CLASS_ACCURACY_PLOT_NAME)


def process_epoch_log(log_file, line):
    epoch_eval_loss = 0
    epoch_eval_accuracy = 0
    epoch_eval_avg_class = 0
    counter = 0
    flag = False

    while True:
        line = log_file.readline()
        if line.startswith(EVAL_MEAN_LOSS):
            counter += 1
            epoch_eval_loss = float(line.split(EPOCH_VAL_DELIMETER)[1][:-2])
        elif line.startswith(EVAL_ACCURACY):
            counter += 1
            epoch_eval_accuracy = float(line.split(EPOCH_VAL_DELIMETER)[1][:-2])
        elif line.startswith(EVAL_AVG_CLASS_ACC):
            counter += 1
            epoch_eval_avg_class = float(line.split(EPOCH_VAL_DELIMETER)[1][:-2])
            if counter == 3:
                flag = True
                break
    if flag:
        return (epoch_eval_loss, epoch_eval_accuracy, epoch_eval_avg_class)
    else:
        return (-1, -1, -1)


if __name__ == "__main__":
    epoch_counter = 0
    log_file_3 = open(LOG_FILE_PATH_3)
    log_file_8 = open(LOG_FILE_PATH_8)
    log_file_local = open(LOG_FILE_PATH_local)

    train_results_3 = []
    train_results_8 = []
    train_results_local = []

    while True:
        line_3 = log_file_3.readline();
        line_8 = log_file_8.readline()
        line_local = log_file_local.readline()

        if not line_8 or not line_3 or not line_local:
            break
        else:
            if line_3.startswith(EPOCH_PREFIX_STR):
                processed_epoch_values_3 = process_epoch_log(log_file_3, line_3)
                print(processed_epoch_values_3)
                if processed_epoch_values_3[0] >= 0:
                    print(epoch_counter)
                    train_results_3.append(processed_epoch_values_3)
                    epoch_counter = +1
                else:
                    break
            if line_8.startswith(EPOCH_PREFIX_STR):
                processed_epoch_values_8 = process_epoch_log(log_file_8, line_8)
                print(processed_epoch_values_8)
                if processed_epoch_values_8[0] >= 0:
                    print(epoch_counter)
                    train_results_8.append(processed_epoch_values_8)
                else:
                    break
            if line_local.startswith(EPOCH_PREFIX_STR):
                processed_epoch_values_local = process_epoch_log(log_file_local, line_local)
                print(processed_epoch_values_local)
                if processed_epoch_values_local[0] >= 0:
                    print(epoch_counter)
                    train_results_local.append(processed_epoch_values_local)
                else:
                    break

log_file_3.close()
log_file_8.close()
log_file_local.close()
visualize_training_log_2(train_results_3, train_results_8, train_results_local)