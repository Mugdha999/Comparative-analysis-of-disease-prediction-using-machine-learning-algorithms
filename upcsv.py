import sys
import subprocess
import os
from flask import *
from werkzeug.utils import secure_filename
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.cm import rainbow
import warnings
warnings.filterwarnings('ignore')
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

UPLOAD_FOLDER = 'C:/Users/Mugdha/Desktop/project/dataset/'
ALLOWED_EXTENSIONS = set(['csv','jpg'])

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.static_folder = 'static'
app.secret_key = os.urandom(24)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')  
def upload():  
    return render_template("id.html") 

@app.route('/uploadajax', methods = ['POST'])  
def uploadajax():
    if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
    file = request.files['file']
    # if user does not select file, browser also
    # submit an empty part without filename
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        #Result=random_forest(filename)
        #res=[filename,Result]
        return jsonify({'data': render_template('success.html', filename=filename)})


@app.route('/random_forest/', methods=['POST'])
def random_forest():
    ''''import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    from matplotlib.cm import rainbow
    import warnings
    warnings.filterwarnings('ignore')'''

    filename=None
    if request.method == "POST":
          filename=request.form['data']
    # Importing the dataset
    dataset = pd.read_csv('C:/Users/Mugdha/Desktop/project/dataset/'+filename)
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values
    
    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    from sklearn.ensemble import RandomForestClassifier
    rf_scores = []
    estimators = [10, 100, 200, 500, 1000]
    for i in estimators:
        rf_classifier = RandomForestClassifier(n_estimators = i, random_state = 0)
        rf_classifier.fit(X_train, y_train)
        rf_scores.append(round(rf_classifier.score(X_test, y_test),4)*100)


    colors = rainbow(np.linspace(0, 1, len(estimators)))
    plt.bar([i for i in range(len(estimators))], rf_scores, color = colors, width = 0.8)
    for i in range(len(estimators)):
        plt.text(i, rf_scores[i], rf_scores[i])
    plt.xticks(ticks = [i for i in range(len(estimators))], labels = [str(estimator) for estimator in estimators])
    plt.xlabel('Number of estimators')
    plt.ylabel('Scores')
    plt.title('Random Forest Classifier scores for different number of estimators')
    path='C:/Users/Mugdha/Desktop/project/static/graph/'
    plt.savefig(path+filename+'-rf.png')
    path=filename+"-rf.png"
    plt.clf()
    
    
    result=max(rf_scores)

    response=['Algorithm : Random Forest',result,path]

    return jsonify({'data': render_template('result.html', result=response)})


@app.route('/decision_tree/', methods=['POST'])
def decision_tree():
    '''# Importing the libraries
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    from matplotlib.cm import rainbow
    import warnings
    warnings.filterwarnings('ignore')'''

    filename=None
    if request.method == "POST":
          filename=request.form['data']
    # Importing the dataset
    dataset = pd.read_csv('C:/Users/Mugdha/Desktop/project/dataset/'+filename)
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values
    
    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
   
    # Fitting Random Forest Classification to the Training set
    from sklearn.tree import DecisionTreeClassifier
    max_depth_range = list(range(1, 10))
    accuracy = []
    for depth in max_depth_range:
        clf = DecisionTreeClassifier(max_depth = depth, random_state = 0)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        accuracy.append( round(score,3)*100)
       
    plt.plot(max_depth_range,accuracy)
    for i in range(1,10):
        plt.text(i, accuracy[i-1], (i, accuracy[i-1]))
    plt.xlabel('Max features')
    plt.ylabel('Scores')
    plt.title('Decision Tree Classifier scores for different number of maximum features')
    path='C:/Users/Mugdha/Desktop/project/static/graph/'
    plt.savefig(path+filename+'-dtree.png')
    path=filename+"-dtree.png"
    plt.clf()
    
    result=max(accuracy)

    response=['Algorithm : Decision Tree',result,path]

    return jsonify({'data': render_template('result.html', result=response)})


@app.route('/knn/', methods=['POST'])
def knn():
    '''# Importing the libraries
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    from matplotlib.cm import rainbow
    import warnings
    warnings.filterwarnings('ignore')'''

    filename=None
    if request.method == "POST":
          filename=request.form['data']
    # Importing the dataset
    dataset = pd.read_csv('C:/Users/Mugdha/Desktop/project/dataset/'+filename)
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values
    
    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
        
    # Fitting Random Forest Classification to the Training set
    from sklearn.neighbors import KNeighborsClassifier
    knn_scores = []
    for k in range(1,21):
        knn_classifier = KNeighborsClassifier(n_neighbors = k)
        knn_classifier.fit(X_train,y_train)
        knn_scores.append(round(knn_classifier.score(X_test, y_test),4)*100)
    
    plt.plot([k for k in range(1, 21)], knn_scores, color = 'red')
    for i in range(1,21):
        plt.text(i, knn_scores[i-1], (i, knn_scores[i-1]))
    plt.xticks([i for i in range(1, 21)])
    plt.xlabel('Number of Neighbors (K)')
    plt.ylabel('Scores')
    plt.title('K Neighbors Classifier scores for different K values')
    path='C:/Users/Mugdha/Desktop/project/static/graph/'
    plt.savefig(path+filename+'-knn.png')
    path=filename+"-knn.png"
    plt.clf()
    
    
    result=max(knn_scores)

    response=['Algorithm : knn',result,path]

    return jsonify({'data': render_template('result.html', result=response)})
    

@app.route('/svm/', methods=['POST'])
def svm():
    '''# Importing the libraries
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.cm import rainbow
    import pandas as pd'''

    filename=None
    if request.method == "POST":
          filename=request.form['data']
    # Importing the dataset
    dataset = pd.read_csv('C:/Users/Mugdha/Desktop/project/dataset/'+filename)
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values
    
    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    
    # Fitting Random Forest Classification to the Training set
    from sklearn.svm import SVC
    svc_scores = []
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    for i in range(len(kernels)):
        svc_classifier = SVC(kernel = kernels[i])
        svc_classifier.fit(X_train, y_train)
        svc_scores.append(round(svc_classifier.score(X_test, y_test),4)*100)
    
    colors = rainbow(np.linspace(0, 1, len(kernels)))
    plt.bar(kernels, svc_scores, color = colors)
    for i in range(len(kernels)):
        plt.text(i, svc_scores[i], svc_scores[i])
    plt.xlabel('Kernels')
    plt.ylabel('Scores')
    plt.title('Support Vector Classifier scores for different kernels')
    path='C:/Users/Mugdha/Desktop/project/static/graph/'
    plt.savefig(path+filename+'-svm.png')
    path=filename+"-svm.png"
    plt.clf()
    
    
    result=max(svc_scores)

    response=['Algorithm : svm',result,path]

    return jsonify({'data': render_template('result.html', result=response)})

@app.route('/Predict/', methods=['POST'])
def Predict():
    filename=None
    if request.method == "POST":
          filename=request.form['data']
    # Importing the dataset
    dataset = pd.read_csv('C:/Users/Mugdha/Desktop/project/dataset/'+filename)
    scaled_features = dataset.copy()
    col_names = ['Age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    features = scaled_features[col_names]
    scaler = StandardScaler().fit(features.values)
    features = scaler.transform(features.values)
    scaled_features[col_names] = features
    Y = scaled_features['target']
    X = scaled_features.drop(['target'], axis = 1)
    scaled_features_d = dataset.copy()
    col_names_d = ['Age', 'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction']
    features_d = scaled_features_d[col_names_d]
    scaler = StandardScaler().fit(features_d.values)
    features_d = scaler.transform(features_d.values)
    scaled_features_d[col_names_d] = features_d
    y1 = scaled_features_d['Outcome']
    X1 = scaled_features_d.drop(['Outcome'], axis = 1)
    scaled_features_b = dataset.copy()
    col_names_b = ['clump_thickness',	'uniformity_of_cell_size','uniformity_of_cell_shape','marginal_adhesion','single_epithelial_cell_size','bare_nuclei','bland_chromatin','normal_nucleoli','mitosis']
    features_b = scaled_features_b[col_names_b]
    scaler = StandardScaler().fit(features_b.values)
    features_b = scaler.transform(features_b.values)
    scaled_features_b[col_names_b] = features_b
    y2 = scaled_features_b['Outcome']
    X2 = scaled_features_b.drop(['Outcome'], axis = 1)
    print(X2)

    file = 'C:/Users/Mugdha/Desktop/project/randomforest_model_h.sav'
    loaded_model = pickle.load(open(file, 'rb'))
    y=loaded_model.predict(X)

    dataset = pd.read_csv('C:/Users/Mugdha/Desktop/project/dataset/'+filename)
    dataset['target']=y
    if(y==0):
        result1='No'
    else:
        result1='Yes'

    filename_d = 'C:/Users/Mugdha/Desktop/project/randomforest_model_d.sav'
    loaded_model_d = pickle.load(open(filename_d, 'rb'))
    y1=loaded_model_d.predict(X1)

    dataset = pd.read_csv('C:/Users/Mugdha/Desktop/project/dataset/'+filename)
    dataset['Outcome']=y1
    if(y1==0):
        result2='No'
    else:
        result2='Yes'

    filename_b = 'C:/Users/Mugdha/Desktop/project/randomforest_model_b.sav'
    loaded_model = pickle.load(open(filename_b, 'rb'))
    y2=loaded_model.predict(X)

    dataset = pd.read_csv('C:/Users/Mugdha/Desktop/project/dataset/'+filename)
    dataset['Outcome']=y2
    if(y2==0):
        result3='No'
    else:
        result3='Yes'

    result=[result1,result2,result3]

    return jsonify({'data': render_template('suc.html', res=result)})

if __name__ == '__main__':  
    app.run(debug = True)
