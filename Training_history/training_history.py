import matplotlib.pyplot as plt
import numpy as np
import os.path as pth
import sys

# Pull data from log file
def GetData(solver_file, log_file):
    if not pth.exists(solver_file): # checking of the existence of log file
        print('Cannot read solver file!')
        return []
    if not pth.exists(log_file): # checking of the existence of log file
        print('Cannot read log file!')
        return []
    with open(solver_file) as f: # reading of data from log file
        solver_data = f.read()
    with open(log_file) as f: # reading of data from log file
        log_data = f.read()
    return solver_data.splitlines(), log_data.splitlines()

def GetIters(data):
    test_iter, max_iter = 0, 0
    pattern_test = 'test_interval: '
    pattern_max = 'max_iter: '
    for st in data:
        if pattern_test in st:
             test_iter = int(st[len(pattern_test):])
        if pattern_max in st:
             max_iter = int(st[len(pattern_max):])
    return test_iter, max_iter

# Get values of losses and accuracies during the training process
def GetFeatures(data, test_iter, max_iter):
    Xacc = list(np.arange(0, max_iter, test_iter))
    Xacc.append(max_iter)
    # Y values for dependency of iterations and losses
    Yacc, loss = [], []
    for st in data:
        if ' accuracy@1 = ' in st:
            Yacc.append(float(st.split()[-1]))
        if ' loss/loss = ' in st:
            loss.append(float(st.split()[-2]))
    return Xacc, Yacc, loss

# Build the dependency of iterations and accuracies
def BuildAccuracyHistoryImage(Xacc, Yacc, loss):
    plt.plot(Xacc, Yacc)
    plt.xlabel('Number of iteration')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.savefig('accuracy.png')
    plt.clf()
    plt.plot(loss)
    plt.ylabel('Loss')
    # hide X tricks
    plt.tick_params(
        axis='x',           # changes apply to the x-axis
        which='both',       # both major and minor ticks are affected
        bottom=False,       # ticks along the bottom edge are off
        top=False,          # ticks along the top edge are off
        labelbottom=False)  # labels along the bottom edge are off
    plt.grid()
    plt.savefig('loss.png')

if (len(sys.argv) != 3): # checking of right call
    print('The call of this script looks like this:\n' +
          '     python trainhistory.py log_file')
else:
    solver_file = sys.argv[1]
    log_file = sys.argv[2]
    solver_data, log_data = GetData(solver_file, log_file)
    if solver_data and log_data: # main part of script
        test_iter, max_iter = GetIters(solver_data)
        Xacc, Yacc, loss = GetFeatures(log_data, test_iter, max_iter)
        BuildAccuracyHistoryImage(Xacc, Yacc, loss)