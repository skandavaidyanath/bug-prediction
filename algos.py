import pandas as pd
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold



EMBEDDING_DIMS = 300

def apply_algo(X_train, X_test, y_train, y_test, algo):
    if algo == 'l':
        clf = LogisticRegression()
        clf.fit(X_train, y_train)
        y_preds = clf.predict(X_test)
        return accuracy_score(y_test, y_preds)

    if algo == 'r':
        clf = RandomForestClassifier(n_estimators=100)
        clf.fit(X_train, y_train)
        y_preds = clf.predict(X_test)
        return accuracy_score(y_test, y_preds)

    if algo == 's':
        clf = SVC(gamma='auto')
        clf.fit(X_train, y_train)
        y_preds = clf.predict(X_test)
        return accuracy_score(y_test, y_preds)
    
    if algo == 'v':
        clf1 = LogisticRegression()
        clf2 = RandomForestClassifier(n_estimators=100)
        clf3 = SVC(gamma='auto')
        bases = [('m1', clf1), ('m2', clf2), ('m3', clf3)]
        model = VotingClassifier(bases, voting = 'hard')
        model.fit(X_train, y_train)
        y_preds = model.predict(X_test)
        return accuracy_score(y_test, y_preds)

    
	

def run(fp, count):
    fp.write('RUN ' + str(count) + '\n')
    data = pd.read_csv('./camel-1.6.csv')	
    data = data.iloc[:,3:]
    X = data.drop(['bug'], axis=1)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    y = data.iloc[:,-1]
    y[y > 0] = 1
    kf = KFold(n_splits=5)
    old_mean_l = 0
    old_mean_r = 0
    old_mean_s = 0
    old_mean_v = 0
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        old_mean_l += apply_algo(X_train, X_test, y_train, y_test, algo='l')/5
        old_mean_r += apply_algo(X_train, X_test, y_train, y_test, algo='r')/5
        old_mean_s += apply_algo(X_train, X_test, y_train, y_test, algo='s')/5
        old_mean_v += apply_algo(X_train, X_test, y_train, y_test, algo='v')/5
    fp.write('L ' + str(old_mean_l) + '\n')
    fp.write('R ' + str(old_mean_r) + '\n')
    fp.write('S ' + str(old_mean_s) + '\n')
    fp.write('V ' + str(old_mean_v) + '\n')
    fp.write('#########################################################\n')
    new_data = pd.read_csv('new_data_camel_{}.csv'.format(EMBEDDING_DIMS))
    y = new_data['bug']
    y[y > 0] = 1
    X = new_data.drop(['bug'], axis=1)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    kf = KFold(n_splits=5)
    new_mean_l = 0
    new_mean_r = 0
    new_mean_s = 0
    new_mean_v = 0
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        new_mean_l += apply_algo(X_train, X_test, y_train, y_test, algo='l')/5
        new_mean_r += apply_algo(X_train, X_test, y_train, y_test, algo='r')/5
        new_mean_s += apply_algo(X_train, X_test, y_train, y_test, algo='s')/5
        new_mean_v += apply_algo(X_train, X_test, y_train, y_test, algo='v')/5
    fp.write('L ' + str(new_mean_l) + '\n')
    fp.write('R ' + str(new_mean_r) + '\n')
    fp.write('S ' + str(new_mean_s) + '\n')
    fp.write('V ' + str(new_mean_s) + '\n')
    fp.write('#########################################################\n')
    return old_mean_l, old_mean_r, old_mean_s, old_mean_v, new_mean_l, new_mean_r, new_mean_s, new_mean_v
    
def main():
    fp = open('embedding_plus_data_results_camel_{}.txt'.format(EMBEDDING_DIMS), 'w')
    old_mean_l = 0
    old_mean_r = 0
    old_mean_s = 0
    old_mean_v = 0
    new_mean_l = 0
    new_mean_r = 0
    new_mean_s = 0
    new_mean_v = 0
    old_l_values = []
    old_r_values = []
    old_s_values = []
    old_v_values = []
    new_l_values = []
    new_r_values = []
    new_s_values = []
    new_v_values = []
    num_iterations = 100
    for i in range(num_iterations):
        temp_old_mean_l, temp_old_mean_r, temp_old_mean_s, temp_old_mean_v, temp_new_mean_l, temp_new_mean_r, temp_new_mean_s, temp_new_mean_v = run(fp, i+1)
        old_mean_l += temp_old_mean_l
        old_mean_r += temp_old_mean_r
        old_mean_s += temp_old_mean_s
        old_mean_v += temp_old_mean_v
        new_mean_l += temp_new_mean_l
        new_mean_r += temp_new_mean_r
        new_mean_s += temp_new_mean_s
        new_mean_v += temp_new_mean_v
        old_l_values.append(temp_old_mean_l)
        old_r_values.append(temp_old_mean_r)
        old_s_values.append(temp_old_mean_s)
        old_v_values.append(temp_old_mean_v)
        new_l_values.append(temp_new_mean_l)
        new_r_values.append(temp_new_mean_r)
        new_s_values.append(temp_new_mean_s)
        new_v_values.append(temp_new_mean_v)
        print(i+1)
    old_mean_l /= num_iterations
    old_mean_r /= num_iterations
    old_mean_s /= num_iterations
    old_mean_v /= num_iterations
    new_mean_l /= num_iterations
    new_mean_r /= num_iterations
    new_mean_s /= num_iterations
    new_mean_v /= num_iterations
    var_old_l = 0
    var_old_r = 0
    var_old_s = 0
    var_old_v = 0
    var_new_l = 0
    var_new_r = 0
    var_new_s = 0
    var_new_v = 0
    for j in range(num_iterations):
        var_old_l += ((old_l_values[j] - old_mean_l) ** 2)/num_iterations
        var_old_r += ((old_r_values[j] - old_mean_r) ** 2)/num_iterations
        var_old_s += ((old_s_values[j] - old_mean_s) ** 2)/num_iterations
        var_old_v += ((old_v_values[j] - old_mean_v) ** 2)/num_iterations
        var_new_l += ((new_l_values[j] - new_mean_l) ** 2)/num_iterations
        var_new_r += ((new_r_values[j] - new_mean_r) ** 2)/num_iterations
        var_new_s += ((new_s_values[j] - new_mean_s) ** 2)/num_iterations
        var_new_v += ((new_v_values[j] - new_mean_v) ** 2)/num_iterations
    std_old_l = var_old_l ** 0.5
    std_old_r = var_old_r ** 0.5
    std_old_s = var_old_s ** 0.5
    std_old_v = var_old_v ** 0.5
    std_new_l = var_new_l ** 0.5
    std_new_r = var_new_r ** 0.5
    std_new_s = var_new_s ** 0.5
    std_new_v = var_new_v ** 0.5
    fp.write('FINAL MEAN OLD L: ' + str(old_mean_l) + '\n')
    fp.write('FINAL MEAN OLD R: ' + str(old_mean_r) + '\n')
    fp.write('FINAL MEAN OLD S: ' + str(old_mean_s) + '\n')
    fp.write('FINAL MEAN OLD V: ' + str(old_mean_v) + '\n')
    fp.write('FINAL MEAN NEW L: ' + str(new_mean_l) + '\n')
    fp.write('FINAL MEAN NEW R: ' + str(new_mean_r) + '\n')
    fp.write('FINAL MEAN NEW S: ' + str(new_mean_s) + '\n')
    fp.write('FINAL MEAN NEW V: ' + str(new_mean_v) + '\n')
    fp.write('FINAL STD OLD L: ' + str(std_old_l) + '\n')
    fp.write('FINAL STD OLD R: ' + str(std_old_r) + '\n')
    fp.write('FINAL STD OLD S: ' + str(std_old_s) + '\n')
    fp.write('FINAL STD OLD V: ' + str(std_old_v) + '\n')
    fp.write('FINAL STD NEW L: ' + str(std_new_l) + '\n')
    fp.write('FINAL STD NEW R: ' + str(std_new_r) + '\n')
    fp.write('FINAL STD NEW S: ' + str(std_new_s) + '\n')
    fp.write('FINAL STD NEW V: ' + str(std_new_v) + '\n')
    fp.close()
    


if __name__ == '__main__':
	main()
